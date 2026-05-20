"""End-to-end and smoke tests for the DAC-3D assistant."""

from __future__ import annotations

from pathlib import Path
import zipfile
from xml.sax.saxutils import escape

from app import DAC3DAssistant
from config import AppConfig
from knowledge_base.build_kb import build_knowledge_base
from rag.retriever import Retriever

SAMPLE_DOCUMENTS = {
    "parameter_notes.md": """# DAC-3D 参数说明

## 分辨率
分辨率决定扫描点间距。更小的点间距意味着更高细节，但也会增加扫描时间。

## 扫描区域
扫描区域决定需要检测的宽度和高度，通常以毫米表示。
""",
    "troubleshooting.md": """# DAC-3D 故障排查

## 样品反光
如果样品太反光，应先降低照明或曝光，再做小范围校准扫描。
""",
    "defect_interpretation.md": """# DAC-3D 缺陷判定

## 划痕严重度
如果划痕深度大于 15 um 或长度大于 0.50 mm，应视为高严重度。
""",
    "manual_overview.md": """# DAC-3D 工作流

## 检测状态
DAC-3D 运行状态通常包括 idle、queued、running、completed 和 error。
""",
}


def _write_minimal_docx(path: Path, paragraphs: list[tuple[str, str | None]]) -> None:
    body_parts: list[str] = []
    for text, style in paragraphs:
        style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body_parts.append(
            f"<w:p>{style_xml}<w:r><w:t>{escape(text)}</w:t></w:r></w:p>"
        )

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body_parts)}<w:sectPr/></w:body>"
        "</w:document>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
        '<w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
        "</w:styles>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        "</Types>"
    )
    relationships_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", relationships_xml)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml)


def make_config(tmp_path: Path, *, provider: str = "mock") -> AppConfig:
    """Create a temporary configuration with sample knowledge-base documents."""
    config = AppConfig(
        base_dir=tmp_path,
        mock_mode=True,
        provider=provider,
        vector_store_type="manifest",
        embedding_download_allowed=False,
    )
    config.ensure_directories()
    for filename, content in SAMPLE_DOCUMENTS.items():
        (config.documents_dir / filename).write_text(content, encoding="utf-8")
    return config


def make_complex_docs_config(tmp_path: Path, *, provider: str = "mock") -> AppConfig:
    """Create a richer document set that matches the current repository manuals."""
    config = AppConfig(
        base_dir=tmp_path,
        mock_mode=True,
        provider=provider,
        vector_store_type="manifest",
        embedding_download_allowed=False,
    )
    config.ensure_directories()
    (config.documents_dir / "manual_overview.md").write_text(
        """# DAC-3D 工作流概览 / Workflow Overview

## 扫描配置 / Scan Setup
开始采集前，操作员需要先定义扫描区域，单位通常为毫米。`10 mm x 10 mm` 是常见的小范围验证区域，适合做快速设置检查和参数确认。

## 区域选择 / Region Selection
扫描区域应绑定到当前可见选区或已保存的 ROI。若没有明确的命名区域，应优先使用 current selection，而不是凭经验扩大扫描范围。

## 扫描模式 / Scan Modes
Standard mode 用于大多数常规扫描，平衡速度和细节。Precision mode 适合怀疑存在缺陷时做高细节确认。Fast mode 适合预览和快速检查。

## 检测状态 / Inspection Status
DAC-3D 运行状态通常包括 `idle`、`queued`、`running`、`completed` 和 `error`。做重新扫描判断前，操作员应同时查看状态消息和进度百分比。
""",
        encoding="utf-8",
    )
    (config.documents_dir / "parameter_notes.md").write_text(
        """# DAC-3D 参数说明 / Parameter Notes

## 分辨率 / Resolution
Resolution 控制扫描时的点间距。点间距越小，细节越丰富，但扫描时间和数据量也会增加。若只是做预览或初步确认，不一定需要高精度分辨率。

## 扫描区域 / Scan Area
Scan area 定义被检测区域的物理长宽，单位通常为毫米。扫描区域越大，循环时间越长，只有在缺陷位置不明确时才应扩大范围。

## 区域 / Region
Region 表示当前要采集的样品区域。安全做法是绑定到可见选区或已保存的感兴趣区域，不要在没有确认选区的情况下执行整片扫描。

## 模式 / Mode
Fast mode 适合预览，standard mode 适合常规检测，precision mode 用于需要更高细节的缺陷确认。切换到 precision mode 前，应先确认 standard mode 的结果是否已经足够。
""",
        encoding="utf-8",
    )
    (config.documents_dir / "troubleshooting.md").write_text(
        """# DAC-3D 故障与操作建议 / Troubleshooting Notes

## 样品反光 / Reflective Samples
如果样品表面过于反光，应先降低照明强度或曝光时间，避免镜面反射造成眩光。如果工艺允许，可以临时做消光处理，或者轻微调整样品倾角。随后应先做小范围校准扫描，再决定是否执行完整扫描。

## 扫描不稳定 / Unstable Scans
如果预览画面不稳定，应确认样品是否固定、当前选区是否正确，以及模式是否匹配检测目标。通常建议先使用 standard mode 完成基础确认，再考虑切换到 precision mode。
""",
        encoding="utf-8",
    )
    (config.documents_dir / "defect_interpretation.md").write_text(
        """# DAC-3D 缺陷判定指南 / Defect Interpretation Guide

## 划痕严重度 / Scratch Severity
如果 scratch 的深度大于 `15 um`，或者长度大于 `0.50 mm`，通常应判定为 high severity。若深度在 `8 um` 到 `15 um` 之间，或者长度在 `0.20 mm` 到 `0.50 mm` 之间，可判定为 medium severity。

## 点蚀严重度 / Pit Severity
如果 pit 的深度大于 `10 um`，通常可视为 high severity。较浅的 pit 可能是 low 或 medium severity，具体还要结合受影响区域和工艺容忍度。

## 报告建议 / Reporting Guidance
当操作员询问缺陷是否严重时，回答中应至少包含缺陷类型、严重度、位置，以及触发该判断的测量值或阈值依据。
""",
        encoding="utf-8",
    )
    return config


def test_build_knowledge_base_and_retrieve_parameter_docs(tmp_path: Path) -> None:
    """The retriever should load persisted chunks and find parameter guidance."""
    config = make_config(tmp_path)
    build_knowledge_base(config)
    retriever = Retriever.from_config(config)
    items = retriever.retrieve("这个参数是什么意思？")

    assert config.vector_store_manifest_path.exists()
    assert items
    assert any(item.source == "parameter_notes.md" for item in items)


def test_retriever_filters_by_document_type(tmp_path: Path) -> None:
    """The retriever should allow document-type filtering."""
    config = make_config(tmp_path)
    build_knowledge_base(config)
    retriever = Retriever.from_config(config)

    items = retriever.retrieve("样品太反光怎么办？", document_type="guidance")

    assert items
    assert all(item.document_type == "guidance" for item in items)


def test_hybrid_retrieval_prioritizes_exact_section_matches_and_dedupes_sections(tmp_path: Path) -> None:
    """Hybrid retrieval should surface the exact matching section instead of repeated background chunks."""
    config = make_config(tmp_path)
    config.chunk_size = 24
    config.chunk_overlap = 6
    (config.documents_dir / "design_reference.md").write_text(
        """# DAC-3D 课题规划

## 课题背景
DAC-3D 是一个三维检测系统，用于理解课程背景和整体结构。
DAC-3D 是一个三维检测系统，用于理解课程背景和整体结构。
DAC-3D 是一个三维检测系统，用于理解课程背景和整体结构。

## 设计原则
可集成：每个课题都是 DAC-3D 系统的一个模块。
可考核：每个模块都能形成独立验收结果。
利就业：技术栈与就业方向相关。
难度适中：16 周内可以交付。
""",
        encoding="utf-8",
    )

    build_knowledge_base(config)
    retriever = Retriever.from_config(config)
    items = retriever.retrieve("DAC-3D 的设计原则是什么？", top_k=4)

    assert items
    assert items[0].section == "设计原则"
    assert any("可集成" in item.text for item in items)
    assert len({(item.source, item.section) for item in items}) == len(items)


def test_query_flow_prefers_top_section_evidence_in_mock_answer(tmp_path: Path) -> None:
    """The mock answer should quote evidence from the top reranked section instead of generic overview text."""
    config = make_config(tmp_path)
    (config.documents_dir / "design_reference.md").write_text(
        """# DAC-3D 课题规划

## 设计原则
可集成：每个课题都是 DAC-3D 系统的一个模块。
可考核：每个模块都能形成独立验收结果。
""",
        encoding="utf-8",
    )
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)

    response = assistant.handle_message("DAC-3D 的设计原则是什么？")

    assert response.intent == "query"
    assert response.source_items[0]["section"] == "设计原则"
    assert "可集成" in response.answer or "可考核" in response.answer


def test_query_flow_returns_grounded_answer(tmp_path: Path) -> None:
    """The assistant should answer Q&A requests with sources."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)

    response = assistant.handle_message("这个参数是什么意思？")

    assert response.intent == "query"
    assert response.sources
    assert "来源" in response.answer


def test_query_flow_sends_all_retrieved_items_to_llm(tmp_path: Path) -> None:
    """The application should pass all matched retrieval chunks to the LLM, not only the display top-k."""
    config = make_config(tmp_path)
    config.retrieval_top_k = 1
    (config.documents_dir / "design_reference.md").write_text(
        """# DAC-3D 课题规划

## 设计原则
可集成：每个课题都是 DAC-3D 系统的一个模块。

## 实现路径
实现路径需要从多波段原始图像重建三维点云。
""",
        encoding="utf-8",
    )
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    observed: dict[str, int] = {}

    class RecordingLLM:
        def generate(self, prompt, *, task, question, retrieval_items=None, parsed_result=None, command=None):
            del prompt, task, question, parsed_result, command
            observed["count"] = len(list(retrieval_items or []))
            return "recorded"

    assistant.llm_client = RecordingLLM()
    response = assistant.handle_message("DAC-3D 的设计原则是什么？")

    assert response.intent == "query"
    assert observed["count"] > config.retrieval_top_k
    assert len(response.source_items) == config.retrieval_top_k


def test_operation_flow_builds_command_preview(tmp_path: Path) -> None:
    """The assistant should convert scan language into a structured preview."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)

    response = assistant.handle_message("扫描 10mm x 10mm 区域")

    assert response.intent == "operation"
    assert response.command_preview is not None
    assert response.command_preview["scan_area_mm"] == {"width": 10.0, "height": 10.0}
    assert response.command_preview["missing_fields"] == []
    assert response.command_preview["warnings"]


def test_operation_flow_requests_missing_fields(tmp_path: Path) -> None:
    """The assistant should refuse to guess required fields."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)

    response = assistant.handle_message("开始扫描")

    assert response.intent == "operation"
    assert response.command_preview is not None
    assert "scan_area_mm" in response.command_preview["missing_fields"]
    assert "缺少" in response.answer or "还缺少" in response.answer


def test_interpretation_flow_uses_mock_result(tmp_path: Path) -> None:
    """The assistant should interpret the latest mock DAC-3D result."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)

    response = assistant.handle_message("这个缺陷严重吗？")

    assert response.intent == "interpretation"
    assert response.parsed_result is not None
    assert response.parsed_result["severity"] == "high"
    assert "判定依据" in response.answer or "阈值" in response.answer


def test_guidance_and_status_flows(tmp_path: Path) -> None:
    """The assistant should cover troubleshooting and status scenarios."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)

    guidance_response = assistant.handle_message("样品太反光了应该怎么办？")
    status_response = assistant.handle_message("当前检测状态是什么？")
    realtime_status_response = assistant.handle_message("当前系统在做什么，运行到哪一步了？")

    assert guidance_response.intent == "guidance"
    assert "照明" in guidance_response.answer or "曝光" in guidance_response.answer
    assert status_response.intent == "status"
    assert status_response.status_summary is not None
    assert status_response.status_summary["state"] == "idle"
    assert realtime_status_response.intent == "status"
    assert realtime_status_response.status_summary is not None
    assert "运行模式" in realtime_status_response.answer


def test_complex_guidance_combines_reflective_and_unstable_steps(tmp_path: Path) -> None:
    """Guidance should combine multiple troubleshooting sections into one ordered response."""
    config = make_complex_docs_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)

    response = assistant.handle_message("如果样品反光，而且预览也不稳定，我应该先做什么，再做什么？")

    assert response.intent == "guidance"
    assert "降低照明" in response.answer or "曝光" in response.answer
    assert "样品是否固定" in response.answer or "standard mode" in response.answer
    assert response.source_items[0]["section"] == "样品反光 / Reflective Samples"
    assert any(item["section"] == "扫描不稳定 / Unstable Scans" for item in response.source_items)


def test_guidance_question_about_scan_policy_is_not_routed_to_command_generation(tmp_path: Path) -> None:
    """Scan policy questions should answer from manuals instead of returning command clarification."""
    config = make_complex_docs_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)

    response = assistant.handle_message("没有明确命名区域时，能不能直接扩大整片扫描范围？")

    assert response.intent == "guidance"
    assert response.command_preview is None
    assert "current selection" in response.answer or "不要" in response.answer


def test_guidance_query_about_rescan_mentions_status_message_and_progress(tmp_path: Path) -> None:
    """Pre-rescan guidance should mention both status messages and progress percentage."""
    config = make_complex_docs_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)

    response = assistant.handle_message("重新扫描前除了状态，还要看什么？")

    assert response.intent == "guidance"
    assert response.command_preview is None
    assert "状态消息" in response.answer
    assert "进度百分比" in response.answer


def test_interpretation_can_use_user_supplied_pit_measurement(tmp_path: Path) -> None:
    """Explicit defect measurements from the user should override the live mock result."""
    config = make_complex_docs_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)

    response = assistant.handle_message("pit 深度 11 um 严重吗？回答里要包含阈值依据。")

    assert response.intent == "interpretation"
    assert response.parsed_result is not None
    assert response.parsed_result["defect_type"] == "pit"
    assert response.parsed_result["severity"] == "high"
    assert response.parsed_result["trigger_measurement"] == "depth_um"
    assert "10 um" in response.answer


def test_web_upload_can_rebuild_knowledge_base_and_record_build_history(tmp_path: Path) -> None:
    """Uploading files through the assistant should rebuild the KB and persist build dates."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    upload_dir = tmp_path / "incoming"
    upload_dir.mkdir()
    uploaded_file = upload_dir / "fixture_guide.md"
    uploaded_file.write_text(
        """# 夹具稳定性说明

## 夹具振动
如果夹具出现振动，应先降低扫描速度并重新固定工件，然后再进行校准扫描。
""",
        encoding="utf-8",
    )

    summary = assistant.build_knowledge_base_from_uploads([str(uploaded_file)])
    items = assistant.retriever.retrieve("夹具振动应该怎么处理？")

    assert (config.documents_dir / "fixture_guide.md").exists()
    assert summary["latest_build_at"]
    assert summary["history"]
    assert summary["history"][0]["trigger"] == "web_upload"
    assert summary["history"][0]["uploaded_files"] == ["fixture_guide.md"]
    assert any(item.source == "fixture_guide.md" for item in items)


def test_web_upload_accepts_non_utf8_text_documents(tmp_path: Path) -> None:
    """Web-managed KB builds should tolerate common Chinese text encodings."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    upload_dir = tmp_path / "incoming_gbk"
    upload_dir.mkdir()
    uploaded_file = upload_dir / "gbk_manual.txt"
    uploaded_file.write_bytes("夹具振动时先固定工件再校准扫描。".encode("gb18030"))

    summary = assistant.build_knowledge_base_from_uploads([str(uploaded_file)])
    items = assistant.retriever.retrieve("夹具振动时应该先做什么？")

    assert summary["latest_build_at"]
    assert any(item.source == "gbk_manual.txt" for item in items)


def test_web_upload_accepts_docx_content_even_with_non_docx_suffix(tmp_path: Path) -> None:
    """Web-managed KB builds should validate by content and accept misnamed DOCX uploads."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    upload_dir = tmp_path / "incoming_docx"
    upload_dir.mkdir()
    uploaded_file = upload_dir / "fixture_payload.data"
    _write_minimal_docx(
        uploaded_file,
        [
            ("夹具稳定性说明", "Heading1"),
            ("1.2 夹具振动", None),
            ("夹具振动时应先降低扫描速度，再重新固定工件。", None),
        ],
    )

    summary = assistant.build_knowledge_base_from_uploads([str(uploaded_file)])
    items = assistant.retriever.retrieve("夹具振动时应该先做什么？")

    assert (config.documents_dir / "fixture_payload.data").exists()
    assert summary["latest_build_at"]
    assert any(item.source == "fixture_payload.data" for item in items)
