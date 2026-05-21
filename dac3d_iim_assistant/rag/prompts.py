"""Prompt builders for the DAC-3D assistant."""

from __future__ import annotations

from collections.abc import Sequence

from rag.retriever import RetrievalItem


def format_retrieval_context(items: Sequence[RetrievalItem]) -> str:
    """Render retrieved chunks into a readable context block."""
    if not items:
        return "未检索到可用的 DAC-3D 文档片段。"
    lines: list[str] = [
        "以下为本次检索命中的全部文档片段。",
        "这些片段的展示顺序不代表优先级，请综合全部证据再回答。",
        "",
    ]
    for item in items:
        lines.append(
            f"[source={item.source} | title={item.title} | section={item.section} | "
            f"type={item.document_type} | score={item.score:.3f}] {item.text}"
        )
    return "\n".join(lines)


def format_history(history: Sequence[tuple[str, str]] | None) -> str:
    """Render conversation history into a prompt-friendly string."""
    if not history:
        return "无历史对话。"
    lines: list[str] = []
    for user_message, assistant_message in history:
        lines.append(f"用户: {user_message}")
        lines.append(f"助手: {assistant_message}")
    return "\n".join(lines)


def build_qa_prompt(
    question: str,
    items: Sequence[RetrievalItem],
    history: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Build the grounded Q&A prompt."""
    return (
        "你是 DAC-3D 的工业检测助手。\n"
        "回答必须只依据检索上下文和历史对话，不得捏造手册内容。\n"
        "如果证据不足，明确说“依据不足”，并建议补充手册或 FAQ。\n"
        "回答要简洁，并在结尾列出引用来源。\n\n"
        f"历史对话:\n{format_history(history)}\n\n"
        f"检索上下文:\n{format_retrieval_context(items)}\n\n"
        f"当前问题:\n{question}"
    )


def build_guidance_prompt(
    question: str,
    items: Sequence[RetrievalItem],
    history: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Build the operator-guidance prompt."""
    return (
        "你是 DAC-3D 的操作指导助手。\n"
        "输出必须是可执行的操作建议，且每一条建议都要来自检索上下文。\n"
        "如果没有足够依据，不要猜测现场操作方法。\n"
        "回答中优先强调风险控制、校准、确认区域和再扫描前的检查。\n\n"
        "输出格式要求：先给 3 到 5 条优先操作步骤，每条不超过 2 句话；"
        "除非用户继续追问，不要展开长篇背景、完整目录或过多文件路径。\n\n"
        f"历史对话:\n{format_history(history)}\n\n"
        f"检索上下文:\n{format_retrieval_context(items)}\n\n"
        f"当前问题:\n{question}"
    )


def build_interpretation_prompt(
    question: str,
    items: Sequence[RetrievalItem],
    parsed_result: dict[str, object],
    history: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Build the defect-interpretation prompt."""
    return (
        "你是 DAC-3D 的检测结果解读助手。\n"
        "必须优先使用结构化结果字段，再结合检索到的判定规则解释严重度。\n"
        "解释中需要说明缺陷类型、位置、严重度、触发阈值或判定理由，并附来源。\n"
        "如果结构化结果包含 is_qualified、is_ignored、sample_quality 或 sample_quality_label，"
        "必须说明这些字段对合格性判断的影响，不能把已提供字段说成缺失。\n"
        "如果检索不到判定依据，只能说明结构化结果，不得扩展结论。\n\n"
        f"历史对话:\n{format_history(history)}\n\n"
        f"结构化结果:\n{parsed_result}\n\n"
        f"检索上下文:\n{format_retrieval_context(items)}\n\n"
        f"当前问题:\n{question}"
    )


def build_command_clarification_prompt(
    command: dict[str, object],
    history: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Build the clarification prompt for incomplete scan commands."""
    return (
        "你正在帮助操作员补全 DAC-3D 扫描命令。\n"
        "你只能指出缺失字段和风险提示，不得自动猜测缺失的危险参数。\n"
        "请直接告诉用户还缺哪些字段，必要时提醒确认当前选区。\n\n"
        f"历史对话:\n{format_history(history)}\n\n"
        f"当前命令草稿:\n{command}"
    )


def build_greeting_prompt(
    *,
    provider: str,
    model_name: str,
) -> str:
    """Build the prompt for a plain hello model introduction."""
    return (
        "你是 DAC-3D 助手。\n"
        "用户只输入了 hello。\n"
        "请你只回答下面三项内容，不要添加其他解释：\n"
        "1. 你调用的模型商\n"
        "2. 你调用的模型名\n"
        "3. 你当前可使用的功能\n"
        "回答必须简洁，使用中文，控制在 3 句话以内，不要引用来源，不要展开技术细节。\n\n"
        f"模型商: {provider}\n"
        f"模型名: {model_name}\n"
        "当前可使用功能: 文档问答、操作指导、检测结果解读、扫描命令预览、运行状态查询。"
    )
