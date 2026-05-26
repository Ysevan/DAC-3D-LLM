"""Rule-based industrial equipment information-management Agent."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from machine_agent.analysis import detect_abnormal_patterns, summarize_machine_condition
from machine_agent.data_provider import MachineDataProvider, MockMachineDataProvider
from machine_agent.models import TimeRange, ToolCallRecord


DEMO_QUESTIONS = [
    "现在设备状态怎么样？",
    "上个月运行情况怎么样？",
    "最近有哪些异常？",
    "为什么最近温度报警变多了？",
    "E102 错误代码是什么意思？",
    "帮我总结一下这台机器最近三个月的问题。",
]


class MachineAgentService:
    """Small Agent runtime that chooses tools and composes grounded answers."""

    def __init__(
        self,
        *,
        provider: MachineDataProvider | None = None,
        now: datetime | None = None,
    ) -> None:
        self.provider = provider or MockMachineDataProvider(now=now)
        self.now = getattr(self.provider, "now", now or datetime.now()).replace(microsecond=0)

    def chat(self, message: str) -> dict[str, Any]:
        """Answer a natural-language machine-management question with tool evidence."""
        question = message.strip()
        if not question:
            return {
                "intent": "machine_agent",
                "answer": "请输入设备状态、历史、报警或文档相关问题。",
                "tool_calls": [],
                "demo_questions": DEMO_QUESTIONS,
            }

        time_range = self.resolve_time_range(question)
        tool_calls: list[ToolCallRecord] = []
        lowered = question.lower()

        wants_status = _contains_any(question, ("现在", "当前", "实时", "状态", "status"))
        wants_alarm = _contains_any(question, ("报警", "告警", "异常", "alarm", "为什么", "老是", "变多"))
        wants_history = _contains_any(question, ("上个月", "最近三个月", "三个月", "历史", "运行情况", "产量", "趋势", "情况怎么样"))
        wants_docs = bool(_extract_error_code(question)) or _contains_any(question, ("错误代码", "故障代码", "是什么意思", "手册", "规范", "维护"))
        wants_abnormal = _contains_any(question, ("为什么", "变多", "老是", "模式", "规律", "高发", "差异"))
        wants_summary = _contains_any(question, ("总结", "运行情况", "情况怎么样", "问题"))

        status_payload: dict[str, Any] | None = None
        history_payload: dict[str, Any] | None = None
        alarms_payload: dict[str, Any] | None = None
        docs_payload: dict[str, Any] | None = None
        summary_payload: dict[str, Any] | None = None
        abnormal_payload: dict[str, Any] | None = None

        if wants_status and not wants_history:
            status_payload = self.get_current_machine_status()
            tool_calls.append(
                self._tool_call(
                    "get_current_machine_status",
                    {},
                    "用户询问当前设备状态。",
                    status_payload,
                )
            )

        if wants_history or wants_summary or wants_abnormal:
            history_payload = self.get_machine_history(time_range)
            tool_calls.append(
                self._tool_call(
                    "get_machine_history",
                    time_range.to_dict(),
                    "需要读取时间范围内的设备历史数据。",
                    _compact_history_result(history_payload),
                )
            )

        if wants_alarm or wants_summary or wants_abnormal:
            alarms_payload = self.get_alarm_records(time_range)
            tool_calls.append(
                self._tool_call(
                    "get_alarm_records",
                    time_range.to_dict(),
                    "需要读取报警/异常记录进行归因。",
                    _compact_alarm_result(alarms_payload),
                )
            )

        if wants_docs:
            docs_payload = self.search_machine_docs(question)
            tool_calls.append(
                self._tool_call(
                    "search_machine_docs",
                    {"query": question, "limit": 4},
                    "用户询问错误代码、维护规范或文档解释。",
                    docs_payload,
                )
            )

        if wants_summary:
            if history_payload is None:
                history_payload = self.get_machine_history(time_range)
                tool_calls.append(
                    self._tool_call(
                        "get_machine_history",
                        time_range.to_dict(),
                        "总结运行情况需要历史数据。",
                        _compact_history_result(history_payload),
                    )
                )
            if alarms_payload is None:
                alarms_payload = self.get_alarm_records(time_range)
                tool_calls.append(
                    self._tool_call(
                        "get_alarm_records",
                        time_range.to_dict(),
                        "总结运行问题需要报警记录。",
                        _compact_alarm_result(alarms_payload),
                    )
                )
            summary_payload = self.summarize_machine_condition(time_range)
            tool_calls.append(
                self._tool_call(
                    "summarize_machine_condition",
                    time_range.to_dict(),
                    "聚合历史数据和报警记录形成运行总结。",
                    summary_payload,
                )
            )

        if wants_abnormal:
            abnormal_payload = self.detect_abnormal_patterns(time_range)
            tool_calls.append(
                self._tool_call(
                    "detect_abnormal_patterns",
                    time_range.to_dict(),
                    "分析报警高发类型、时间段和异常/正常数据差异。",
                    abnormal_payload,
                )
            )

        if not tool_calls:
            status_payload = self.get_current_machine_status()
            alarms_payload = self.get_alarm_records(time_range)
            docs_payload = self.search_machine_docs(question)
            tool_calls.extend(
                [
                    self._tool_call(
                        "get_current_machine_status",
                        {},
                        "默认先读取设备实时状态作为上下文。",
                        status_payload,
                    ),
                    self._tool_call(
                        "search_machine_docs",
                        {"query": question, "limit": 4},
                        "尝试从设备文档中检索答案。",
                        docs_payload,
                    ),
                ]
            )

        answer = self._compose_answer(
            question=question,
            time_range=time_range,
            status=status_payload,
            history=history_payload,
            alarms=alarms_payload,
            docs=docs_payload,
            summary=summary_payload,
            abnormal=abnormal_payload,
        )

        if status_payload is None:
            status_payload = self.get_current_machine_status()
        if alarms_payload is None:
            alarms_payload = self.get_alarm_records(self.resolve_time_range("最近30天"))

        return {
            "intent": "machine_agent",
            "answer": answer,
            "tool_calls": [record.to_dict() for record in tool_calls],
            "time_range": time_range.to_dict(),
            "current_status": status_payload,
            "alarm_records": (alarms_payload or {}).get("records", []),
            "summary_result": summary_payload,
            "abnormal_result": abnormal_payload,
            "doc_results": (docs_payload or {}).get("documents", []),
            "demo_questions": DEMO_QUESTIONS,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return dashboard data for the frontend."""
        time_range = self.resolve_time_range("最近30天")
        return {
            "current_status": self.get_current_machine_status(),
            "alarm_records": self.get_alarm_records(time_range)["records"][:8],
            "summary_result": self.summarize_machine_condition(time_range),
            "abnormal_result": self.detect_abnormal_patterns(time_range),
            "demo_questions": DEMO_QUESTIONS,
            "time_range": time_range.to_dict(),
        }

    def get_current_machine_status(self) -> dict[str, Any]:
        """Agent tool: query latest equipment state."""
        return self.provider.get_current_machine_status().to_dict()

    def get_machine_history(self, time_range: TimeRange) -> dict[str, Any]:
        """Agent tool: query historical telemetry."""
        records = self.provider.get_machine_history(time_range.start, time_range.end)
        return {
            "time_range": time_range.to_dict(),
            "records": [record.to_dict() for record in records],
            "record_count": len(records),
        }

    def get_alarm_records(self, time_range: TimeRange) -> dict[str, Any]:
        """Agent tool: query alarm and abnormal records."""
        records = self.provider.get_alarm_records(time_range.start, time_range.end)
        return {
            "time_range": time_range.to_dict(),
            "records": [record.to_dict() for record in sorted(records, key=lambda item: item.timestamp, reverse=True)],
            "record_count": len(records),
        }

    def search_machine_docs(self, query: str, limit: int = 4) -> dict[str, Any]:
        """Agent tool: retrieve equipment documents."""
        docs = self.provider.search_machine_docs(query, limit=limit)
        return {
            "query": query,
            "documents": [doc.to_dict() for doc in docs],
            "record_count": len(docs),
        }

    def summarize_machine_condition(self, time_range: TimeRange) -> dict[str, Any]:
        """Agent tool: summarize condition from history and alarms."""
        history = self.provider.get_machine_history(time_range.start, time_range.end)
        alarms = self.provider.get_alarm_records(time_range.start, time_range.end)
        return summarize_machine_condition(history, alarms, time_range)

    def detect_abnormal_patterns(self, time_range: TimeRange) -> dict[str, Any]:
        """Agent tool: detect explainable abnormal patterns."""
        history = self.provider.get_machine_history(time_range.start, time_range.end)
        alarms = self.provider.get_alarm_records(time_range.start, time_range.end)
        return detect_abnormal_patterns(history, alarms, time_range)

    def resolve_time_range(self, message: str) -> TimeRange:
        """Resolve common Chinese relative time expressions."""
        text = message.lower()
        now = self.now
        if "上个月" in text:
            first_this_month = now.replace(day=1, hour=0, minute=0, second=0)
            end = first_this_month - timedelta(seconds=1)
            start = end.replace(day=1, hour=0, minute=0, second=0)
            return TimeRange(start=start, end=end, label="上个月")
        if "最近三个月" in text or "近三个月" in text or "3个月" in text or "三个月" in text:
            return TimeRange(start=now - timedelta(days=90), end=now, label="最近三个月")
        match = re.search(r"最近\s*(\d{1,3})\s*天", text)
        if match:
            days = int(match.group(1))
            return TimeRange(start=now - timedelta(days=days), end=now, label=f"最近{days}天")
        if "今天" in text:
            start = now.replace(hour=0, minute=0, second=0)
            return TimeRange(start=start, end=now, label="今天")
        if "最近" in text or "近期" in text:
            return TimeRange(start=now - timedelta(days=30), end=now, label="最近30天")
        return TimeRange(start=now - timedelta(days=30), end=now, label="最近30天")

    def _tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        reason: str,
        result: dict[str, Any],
    ) -> ToolCallRecord:
        return ToolCallRecord(
            name=name,
            arguments=arguments,
            reason=reason,
            result=result,
        )

    def _compose_answer(
        self,
        *,
        question: str,
        time_range: TimeRange,
        status: dict[str, Any] | None,
        history: dict[str, Any] | None,
        alarms: dict[str, Any] | None,
        docs: dict[str, Any] | None,
        summary: dict[str, Any] | None,
        abnormal: dict[str, Any] | None,
    ) -> str:
        lines: list[str] = []
        if summary:
            lines.append(summary["summary_text"])
        if abnormal:
            lines.append(abnormal["summary_text"])
        if status and not summary:
            alarm_codes = status.get("active_alarm_codes") or []
            alarm_text = "、".join(alarm_codes) if alarm_codes else "无未关闭报警"
            lines.append(
                f"当前设备 {status.get('machine_name')} 处于 {status.get('state')}，"
                f"温度 {status.get('temperature_c')}C，压力 {status.get('pressure_mpa')}MPa，"
                f"转速 {status.get('rpm')}rpm，电流 {status.get('current_a')}A，当前报警：{alarm_text}。"
            )
        if alarms and not summary and not abnormal:
            records = alarms.get("records", [])
            if records:
                first = records[0]
                lines.append(
                    f"{time_range.label}共有 {alarms.get('record_count', len(records))} 条报警/异常记录，"
                    f"最近一条是 {first.get('code')}（{first.get('alarm_type')}，{first.get('severity')}），状态为 {first.get('status')}。"
                )
            else:
                lines.append(f"{time_range.label}没有报警记录。")
        if docs and docs.get("documents"):
            doc = docs["documents"][0]
            lines.append(
                f"文档检索命中《{doc.get('title')}》的“{doc.get('section')}”：{doc.get('text')}"
            )
        if not lines and history:
            lines.append(f"{time_range.label}读取到 {history.get('record_count', 0)} 条历史记录。")
        if not lines:
            lines.append("我已尝试查询设备状态和文档，但当前 mock 数据没有找到足够依据。")
        lines.append("本次回答基于已展示的 Agent 工具调用结果，不是静态问答。")
        return "\n".join(lines)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.lower() in text.lower() for keyword in keywords)


def _extract_error_code(text: str) -> str:
    match = re.search(r"\b[A-Z]\d{3,4}\b", text.upper())
    return match.group(0) if match else ""


def _compact_history_result(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    return {
        "time_range": payload.get("time_range"),
        "record_count": payload.get("record_count", 0),
        "first_record": records[0] if records else None,
        "last_record": records[-1] if records else None,
    }


def _compact_alarm_result(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    return {
        "time_range": payload.get("time_range"),
        "record_count": payload.get("record_count", 0),
        "latest_records": records[:5],
    }
