"""File-backed Context Tree for DAC-Agent Runtime.

The tree is a small, human-readable Context OS layer inspired by ByteRover and
agent workspace projects. It stores durable operating context as Markdown nodes
and retrieves the most relevant nodes for a task before the LLM/tool loop runs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RE = re.compile(r"[a-z0-9][a-z0-9_.:/-]*")


DEFAULT_CONTEXT_NODES: tuple[dict[str, str], ...] = (
    {
        "relative_path": "machine/dac-runtime.md",
        "id": "machine/dac-runtime",
        "kind": "machine",
        "title": "DAC-3D runtime profile",
        "tags": "runtime,status,inspection,progress,设备,状态,检测",
        "body": "\n".join(
            [
                "# DAC-3D Runtime Profile",
                "",
                "DAC-3D 主系统通过本地 bridge 提供状态、进度、最近检测结果和命令回执。",
                "状态类问题优先读取 runtime snapshot；结果类问题优先读取 latest result。",
            ]
        ),
    },
    {
        "relative_path": "project/dac-agent-runtime.md",
        "id": "project/dac-agent-runtime",
        "kind": "project",
        "title": "DAC-Agent Runtime project context",
        "tags": "agent,runtime,memory,skills,workflow,项目,助手",
        "body": "\n".join(
            [
                "# DAC-Agent Runtime",
                "",
                "本项目不是通用聊天机器人，而是 DAC-3D 工业软件的本地 Agent Runtime。",
                "核心能力包括统一 LLM 入口、多 Agent 分工、技能选择、多层记忆、工具调用和 trace/eval 闭环。",
            ]
        ),
    },
    {
        "relative_path": "operator/default-workflow.md",
        "id": "operator/default-workflow",
        "kind": "operator",
        "title": "Default operator workflow preferences",
        "tags": "operator,preview,confirmation,操作员,预览,流程",
        "body": "\n".join(
            [
                "# Default Operator Workflow",
                "",
                "常规操作希望先看到命令预览、相关参数和预计影响，再继续后续步骤。",
                "状态查询、结果解释和参数说明应尽量给出简洁中文回复。",
            ]
        ),
    },
    {
        "relative_path": "procedure/offline-inspection.md",
        "id": "procedure/offline-inspection",
        "kind": "procedure",
        "title": "Offline inspection procedure",
        "tags": "offline,inspection,folder,pre_fusion_images,离线检测,图片,目录",
        "body": "\n".join(
            [
                "# Offline Inspection Procedure",
                "",
                "离线检测通常从选择图片目录开始，然后生成命令预览，确认输入目录、检测模式和输出位置。",
                "完成后读取最新检测结果，按缺陷类型、严重程度和建议动作解释给操作员。",
            ]
        ),
    },
    {
        "relative_path": "tool/dac-tools.md",
        "id": "tool/dac-tools",
        "kind": "tool",
        "title": "DAC tool and skill routing notes",
        "tags": "tool,gateway,skill,status,result,command,工具,技能",
        "body": "\n".join(
            [
                "# DAC Tool Routing Notes",
                "",
                "`dac3d_status` 读取当前状态；`dac3d_latest_result` 读取最近检测结果；",
                "`dac3d_preview_command` 生成命令预览；`dac_skill_select` 选择任务相关技能。",
            ]
        ),
    },
)


def _clip(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _tokens(text: str) -> set[str]:
    lowered = str(text or "").lower()
    tokens = set(_ASCII_RE.findall(lowered))
    for group in _CJK_RE.findall(str(text or "")):
        tokens.update(group[index : index + 2] for index in range(max(0, len(group) - 1)))
        if len(group) <= 10:
            tokens.add(group)
    return {token for token in tokens if token}


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if "," in stripped:
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return stripped


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        return {}, text

    metadata: dict[str, Any] = {}
    for line in lines[1:end_index]:
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        metadata[key.strip()] = _parse_scalar(raw_value)
    return metadata, "\n".join(lines[end_index + 1 :]).strip()


def _render_frontmatter(node: dict[str, str]) -> str:
    return "\n".join(
        [
            "---",
            f"id: {node['id']}",
            f"kind: {node['kind']}",
            f"title: {node['title']}",
            f"tags: {node['tags']}",
            "---",
            "",
            node["body"],
            "",
        ]
    )


@dataclass(slots=True)
class ContextTreeNode:
    """One durable context node stored as a Markdown file."""

    id: str
    kind: str
    title: str
    path: str
    tags: list[str] = field(default_factory=list)
    content_preview: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContextTreeMatch:
    """One scored Context Tree retrieval result."""

    node: ContextTreeNode
    score: float
    reasons: list[str] = field(default_factory=list)
    content: str = ""

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        payload = {
            "node": self.node.to_dict(),
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }
        if include_content:
            payload["content"] = self.content
        return payload


class FileBackedContextTree:
    """Human-readable local Context Tree with task-based retrieval."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def ensure_defaults(self) -> None:
        """Create starter context nodes when the tree is empty."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if any(self.root_dir.glob("**/*.md")):
            return
        for node in DEFAULT_CONTEXT_NODES:
            path = self.root_dir / node["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_render_frontmatter(node), encoding="utf-8")

    def discover(self) -> list[ContextTreeNode]:
        """Read all Context Tree Markdown nodes."""
        self.ensure_defaults()
        nodes: list[ContextTreeNode] = []
        for path in sorted(self.root_dir.glob("**/*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            metadata, body = _parse_frontmatter(text)
            node_id = str(metadata.get("id") or path.relative_to(self.root_dir).with_suffix(""))
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [part.strip() for part in tags.split(",") if part.strip()]
            nodes.append(
                ContextTreeNode(
                    id=node_id,
                    kind=str(metadata.get("kind") or path.parent.name or "general"),
                    title=str(metadata.get("title") or path.stem),
                    path=str(path),
                    tags=[str(tag) for tag in tags],
                    content_preview=_clip(body, 700),
                    metadata={key: value for key, value in metadata.items() if key not in {"id", "kind", "title", "tags"}},
                )
            )
        return nodes

    def read(self, node_id: str) -> dict[str, Any]:
        """Read one node by id."""
        wanted = str(node_id or "").strip()
        for node in self.discover():
            if node.id == wanted:
                path = Path(node.path)
                metadata, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
                return {"node": node.to_dict(), "metadata": metadata, "content": body}
        raise ValueError(f"Unknown context node: {node_id}")

    def search(self, task: str, *, limit: int = 4, include_content: bool = False) -> list[ContextTreeMatch]:
        """Retrieve relevant context nodes for a task."""
        query = str(task or "")
        query_tokens = _tokens(query)
        matches: list[ContextTreeMatch] = []
        for node in self.discover():
            haystack = " ".join(
                [
                    node.id,
                    node.kind,
                    node.title,
                    " ".join(node.tags),
                    node.content_preview,
                ]
            )
            haystack_tokens = _tokens(haystack)
            overlap = query_tokens & haystack_tokens
            score = len(overlap) * 0.5
            reasons: list[str] = []
            if overlap:
                reasons.append("token_overlap")
            for tag in node.tags:
                if tag and tag.lower() in query.lower():
                    score += 2.5
                    reasons.append(f"tag:{tag}")
            if node.kind and node.kind.lower() in query.lower():
                score += 1.5
                reasons.append(f"kind:{node.kind}")
            if node.title and node.title.lower() in query.lower():
                score += 1.5
                reasons.append("title_match")
            if score <= 0:
                continue
            content = self.read(node.id).get("content", "") if include_content else ""
            matches.append(ContextTreeMatch(node=node, score=score, reasons=reasons, content=content))
        matches.sort(key=lambda item: (item.score, item.node.id), reverse=True)
        return matches[: max(1, int(limit))]

    def build_context(self, task: str, *, limit: int = 3, char_limit: int = 2400) -> tuple[str, list[dict[str, Any]]]:
        """Build prompt-ready task context from selected tree nodes."""
        matches = self.search(task, limit=limit, include_content=True)
        if not matches:
            return "", []
        sections: list[str] = []
        payload: list[dict[str, Any]] = []
        for match in matches:
            node = match.node
            sections.append(
                "\n".join(
                    [
                        f"## {node.title}",
                        f"id: {node.id}",
                        f"kind: {node.kind}",
                        f"tags: {', '.join(node.tags)}",
                        _clip(match.content, 900),
                    ]
                )
            )
            payload.append(match.to_dict(include_content=False))
        return (
            "Context Tree（本地可读工作记忆，按任务检索）:\n"
            + _clip("\n\n".join(sections), char_limit),
            payload,
        )

    def describe(self) -> dict[str, Any]:
        """Return Context Tree diagnostics."""
        nodes = self.discover()
        kinds: dict[str, int] = {}
        for node in nodes:
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
        return {
            "enabled": True,
            "backend": "file_context_tree",
            "root_dir": str(self.root_dir),
            "node_count": len(nodes),
            "kinds": kinds,
            "nodes": [node.to_dict() for node in nodes],
            "workflow": "markdown_nodes -> task_search -> context_builder -> agent_turn",
        }
