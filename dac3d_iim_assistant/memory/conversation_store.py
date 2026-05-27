"""JSON-backed multi-layer conversation memory for DAC-3D chat sessions."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from config import AppConfig
from security.secrets import assert_no_secrets


_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.:/-]*")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_session_id(session_id: str | None) -> str:
    normalized = (session_id or "default").strip() or "default"
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", normalized)
    return normalized[:120] or "default"


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text for lightweight JSON memory search."""
    lowered = text.lower()
    tokens = _ASCII_TOKEN_RE.findall(lowered)
    for cjk_group in _CJK_RE.findall(text):
        tokens.extend(
            cjk_group[index : index + 2]
            for index in range(max(0, len(cjk_group) - 1))
        )
        if len(cjk_group) <= 8:
            tokens.append(cjk_group)
    return [token for token in tokens if token]


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


@dataclass(slots=True)
class ConversationMemoryHit:
    """One scored conversation memory item returned from JSON search."""

    layer: str
    session_id: str
    turn_id: str
    created_at: str
    score: float
    snippet: str
    user: str = ""
    assistant: str = ""
    intent: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "created_at": self.created_at,
            "score": round(self.score, 4),
            "snippet": self.snippet,
            "user": self.user,
            "assistant": self.assistant,
            "intent": self.intent,
            "metadata": dict(self.metadata or {}),
        }

    def to_context_line(self) -> str:
        intent = f" | intent={self.intent}" if self.intent else ""
        return (
            f"- [{self.layer} | session={self.session_id} | score={self.score:.2f}"
            f"{intent}] {self.snippet}"
        )


class ConversationMemoryStore:
    """Persist conversation history as JSON and expose layered search."""

    def __init__(
        self,
        *,
        root_dir: Path,
        max_turns_per_session: int = 200,
        max_index_items: int = 2000,
        max_field_chars: int = 1800,
    ) -> None:
        self.root_dir = root_dir
        self.sessions_dir = root_dir / "sessions"
        self.index_path = root_dir / "index.json"
        self.max_turns_per_session = max(1, int(max_turns_per_session))
        self.max_index_items = max(1, int(max_index_items))
        self.max_field_chars = max(200, int(max_field_chars))
        self._lock = RLock()

    @classmethod
    def from_config(cls, config: AppConfig) -> "ConversationMemoryStore":
        """Create the JSON memory store configured for this app instance."""
        return cls(
            root_dir=config.conversation_memory_dir,
            max_turns_per_session=config.memory_max_turns,
            max_index_items=config.memory_index_limit,
            max_field_chars=config.memory_max_field_chars,
        )

    def ensure_directories(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def append_turn(
        self,
        *,
        session_id: str | None,
        user: str,
        assistant: str,
        intent: str = "",
        structured_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one completed user/assistant turn to the session JSON file."""
        normalized = _safe_session_id(session_id)
        if not str(user or "").strip() and not str(assistant or "").strip():
            return self.load_session(normalized)
        assert_no_secrets(
            {
                "user": user,
                "assistant": assistant,
                "structured_data": structured_data or {},
            },
            location="conversation_memory",
        )

        with self._lock:
            now = _utc_now_iso()
            session = self.load_session(normalized)
            turns = list(session.get("turns") or [])
            turn_id = f"{normalized}-{len(turns) + 1:06d}"
            turn = {
                "id": turn_id,
                "created_at": now,
                "user": _clip_text(user, self.max_field_chars),
                "assistant": _clip_text(assistant, self.max_field_chars),
                "intent": str(intent or ""),
                "structured_data": structured_data if isinstance(structured_data, dict) else {},
            }
            turns.append(turn)
            if len(turns) > self.max_turns_per_session:
                turns = turns[-self.max_turns_per_session :]

            session["session_id"] = normalized
            session.setdefault("created_at", now)
            session["updated_at"] = now
            session["turns"] = turns
            session["summary"] = self._build_session_summary(turns)
            _write_json(self._session_path(normalized), session)
            self._append_index_item(normalized, turn)
            return session

    def load_session(self, session_id: str | None) -> dict[str, Any]:
        """Load one JSON session file."""
        normalized = _safe_session_id(session_id)
        default = {
            "session_id": normalized,
            "created_at": "",
            "updated_at": "",
            "summary": {},
            "turns": [],
        }
        with self._lock:
            session = _read_json(self._session_path(normalized), default)
        if not isinstance(session.get("turns"), list):
            session["turns"] = []
        return session

    def recent_turns(
        self,
        session_id: str | None,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """Return recent JSON-backed turns for short-term continuity."""
        session = self.load_session(session_id)
        turns = [turn for turn in session.get("turns", []) if isinstance(turn, dict)]
        return turns[-max(0, int(limit)) :]

    def search(
        self,
        query: str,
        *,
        session_id: str | None = None,
        limit: int = 5,
        include_global: bool = True,
    ) -> list[ConversationMemoryHit]:
        """Search recent, summary, and long-term JSON memory layers."""
        clean_query = str(query or "").strip()
        if not clean_query:
            return []

        normalized = _safe_session_id(session_id)
        hits: list[ConversationMemoryHit] = []
        hits.extend(self._search_session_summary(clean_query, normalized))
        hits.extend(self._search_recent_turns(clean_query, normalized))
        hits.extend(
            self._search_index(
                clean_query,
                session_id=normalized,
                include_global=include_global,
            )
        )
        hits.sort(key=lambda item: (item.score, item.created_at), reverse=True)

        deduped: list[ConversationMemoryHit] = []
        seen: set[tuple[str, str, str]] = set()
        for hit in hits:
            key = (hit.layer, hit.session_id, hit.turn_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
            if len(deduped) >= max(1, int(limit)):
                break
        return deduped

    def format_context(
        self,
        query: str,
        *,
        session_id: str | None,
        history: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
        recent_limit: int = 4,
        search_limit: int = 5,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return a prompt-ready multi-layer memory block and selected hits."""
        sections: list[str] = []
        hits_payload: list[dict[str, Any]] = []

        if history:
            history_lines: list[str] = []
            for user_message, assistant_message in history[-recent_limit:]:
                history_lines.append(f"用户: {_clip_text(user_message, 500)}")
                history_lines.append(f"助手: {_clip_text(assistant_message, 500)}")
            if history_lines:
                sections.append("短期记忆（当前页面历史）:\n" + "\n".join(history_lines))

        persisted_recent = self.recent_turns(session_id, limit=recent_limit)
        if persisted_recent:
            lines: list[str] = []
            for turn in persisted_recent:
                lines.append(
                    "- "
                    f"用户: {_clip_text(turn.get('user'), 320)} / "
                    f"助手: {_clip_text(turn.get('assistant'), 320)}"
                )
            sections.append("会话记忆（JSON 最近对话）:\n" + "\n".join(lines))

        hits = self.search(query, session_id=session_id, limit=search_limit, include_global=True)
        if hits:
            sections.append(
                "长期记忆检索（JSON 索引命中）:\n"
                + "\n".join(hit.to_context_line() for hit in hits)
            )
            hits_payload = [hit.to_dict() for hit in hits]

        if not sections:
            return "", []
        return (
            "以下记忆来自本地 JSON 历史对话，只用于理解指代、用户偏好"
            "和前文目标；涉及 DAC-3D 状态、结果或控制动作时仍以工具返回为准。\n"
            + "\n\n".join(sections),
            hits_payload,
        )

    def describe(self) -> dict[str, Any]:
        """Return diagnostic information for the runtime panel."""
        index = self._load_index()
        return {
            "enabled": True,
            "backend": "json",
            "path": str(self.root_dir),
            "session_count": len(list(self.sessions_dir.glob("*.json"))) if self.sessions_dir.exists() else 0,
            "index_items": len(index.get("items", [])),
            "layers": [
                "short_term_history",
                "session_recent_json",
                "session_summary",
                "long_term_json_search",
            ],
        }

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{_safe_session_id(session_id)}.json"

    def _load_index(self) -> dict[str, Any]:
        default = {"version": 1, "updated_at": "", "items": []}
        index = _read_json(self.index_path, default)
        if not isinstance(index.get("items"), list):
            index["items"] = []
        return index

    def _append_index_item(self, session_id: str, turn: dict[str, Any]) -> None:
        index = self._load_index()
        now = _utc_now_iso()
        text = self._turn_text(turn)
        item = {
            "session_id": session_id,
            "turn_id": str(turn.get("id") or ""),
            "created_at": str(turn.get("created_at") or now),
            "user": str(turn.get("user") or ""),
            "assistant": str(turn.get("assistant") or ""),
            "intent": str(turn.get("intent") or ""),
            "text": text,
            "keywords": self._keywords(text),
        }
        items = [entry for entry in index.get("items", []) if isinstance(entry, dict)]
        items.append(item)
        if len(items) > self.max_index_items:
            items = items[-self.max_index_items :]
        index["version"] = 1
        index["updated_at"] = now
        index["items"] = items
        _write_json(self.index_path, index)

    def _search_session_summary(self, query: str, session_id: str) -> list[ConversationMemoryHit]:
        session = self.load_session(session_id)
        summary = session.get("summary")
        if not isinstance(summary, dict):
            return []
        summary_text = json.dumps(summary, ensure_ascii=False)
        score = self._score(query, summary_text)
        if score <= 0:
            return []
        return [
            ConversationMemoryHit(
                layer="session_summary",
                session_id=session_id,
                turn_id="summary",
                created_at=str(session.get("updated_at") or ""),
                score=score + 0.2,
                snippet=_clip_text(summary_text, 700),
                metadata={"source": "session_json_summary"},
            )
        ]

    def _search_recent_turns(self, query: str, session_id: str) -> list[ConversationMemoryHit]:
        hits: list[ConversationMemoryHit] = []
        for recency_rank, turn in enumerate(reversed(self.recent_turns(session_id, limit=8)), start=1):
            text = self._turn_text(turn)
            score = self._score(query, text)
            if score <= 0:
                continue
            hits.append(
                ConversationMemoryHit(
                    layer="session_recent_json",
                    session_id=session_id,
                    turn_id=str(turn.get("id") or ""),
                    created_at=str(turn.get("created_at") or ""),
                    score=score + max(0.0, 0.35 - (recency_rank * 0.03)),
                    snippet=self._snippet(turn),
                    user=str(turn.get("user") or ""),
                    assistant=str(turn.get("assistant") or ""),
                    intent=str(turn.get("intent") or ""),
                    metadata={"source": "session_json"},
                )
            )
        return hits

    def _search_index(
        self,
        query: str,
        *,
        session_id: str,
        include_global: bool,
    ) -> list[ConversationMemoryHit]:
        index = self._load_index()
        hits: list[ConversationMemoryHit] = []
        for entry in index.get("items", []):
            if not isinstance(entry, dict):
                continue
            entry_session_id = str(entry.get("session_id") or "")
            if not include_global and entry_session_id != session_id:
                continue
            text = str(entry.get("text") or "")
            score = self._score(query, text)
            if score <= 0:
                continue
            if entry_session_id == session_id:
                score += 0.18
                layer = "long_term_session_search"
            else:
                layer = "long_term_global_search"
            hits.append(
                ConversationMemoryHit(
                    layer=layer,
                    session_id=entry_session_id,
                    turn_id=str(entry.get("turn_id") or ""),
                    created_at=str(entry.get("created_at") or ""),
                    score=score,
                    snippet=self._snippet(entry),
                    user=str(entry.get("user") or ""),
                    assistant=str(entry.get("assistant") or ""),
                    intent=str(entry.get("intent") or ""),
                    metadata={"source": "conversation_memory_index"},
                )
            )
        return hits

    def _score(self, query: str, text: str) -> float:
        normalized_query = query.strip().lower()
        normalized_text = text.lower()
        if not normalized_query or not normalized_text:
            return 0.0

        score = 0.0
        if normalized_query in normalized_text:
            score += 3.0
        query_tokens = _tokenize(normalized_query)
        if not query_tokens:
            return score

        text_counts = Counter(_tokenize(normalized_text))
        matched = 0
        for token in query_tokens:
            count = text_counts.get(token, 0)
            if count:
                matched += 1
                score += min(3, count) * (1.0 if len(token) > 1 else 0.35)
        coverage = matched / max(1, len(set(query_tokens)))
        return score + coverage

    def _turn_text(self, turn: dict[str, Any]) -> str:
        return " ".join(
            [
                str(turn.get("user") or ""),
                str(turn.get("assistant") or ""),
                str(turn.get("intent") or ""),
                json.dumps(turn.get("structured_data") or {}, ensure_ascii=False),
            ]
        ).strip()

    def _snippet(self, turn: dict[str, Any]) -> str:
        user = _clip_text(turn.get("user"), 260)
        assistant = _clip_text(turn.get("assistant"), 260)
        if user and assistant:
            return f"用户: {user} / 助手: {assistant}"
        return user or assistant or _clip_text(turn.get("text"), 520)

    def _keywords(self, text: str) -> list[str]:
        counts = Counter(_tokenize(text))
        return [token for token, _count in counts.most_common(20)]

    def _build_session_summary(self, turns: list[dict[str, Any]]) -> dict[str, Any]:
        recent_turns = turns[-12:]
        combined_user_text = " ".join(str(turn.get("user") or "") for turn in recent_turns)
        keywords = self._keywords(combined_user_text)
        last_turn = recent_turns[-1] if recent_turns else {}
        intents = [
            str(turn.get("intent") or "")
            for turn in recent_turns
            if str(turn.get("intent") or "")
        ]
        return {
            "updated_at": _utc_now_iso(),
            "turn_count": len(turns),
            "recent_user_topics": keywords[:12],
            "recent_intents": list(dict.fromkeys(intents[-8:])),
            "last_user_goal": _clip_text(last_turn.get("user"), 500),
            "last_assistant_answer": _clip_text(last_turn.get("assistant"), 500),
        }
