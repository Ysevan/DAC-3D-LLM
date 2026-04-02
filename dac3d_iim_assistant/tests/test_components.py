"""Component-level tests for classification, command generation, and parsing."""

from __future__ import annotations

import sys
import types
import zipfile
from xml.sax.saxutils import escape

from config import AppConfig
from intent.classifier import IntentClassifier
from intent.command_generator import CommandGenerator
from intent.parser import IntentParser
from integration.result_parser import parse_result
from knowledge_base.build_kb import chunk_documents, load_documents


def _write_minimal_docx(path, paragraphs: list[tuple[str, str | None]]) -> None:
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


def test_intent_classifier_handles_chinese_and_english() -> None:
    """Intent classification should cover the main Chinese and English flows."""
    classifier = IntentClassifier()

    assert classifier.classify("当前检测状态是什么？").label == "status"
    assert classifier.classify("what should I do if the sample is too reflective?").label == "guidance"
    assert classifier.classify("这个缺陷严重吗？").label == "interpretation"
    assert classifier.classify("scan a 10mm x 10mm area").label == "operation"


def test_intent_classifier_distinguishes_policy_questions_from_scan_commands() -> None:
    """Questions about scan policy should not be misrouted as executable operations."""
    classifier = IntentClassifier()

    assert classifier.classify("没有明确命名区域时，能不能直接扩大整片扫描范围？").label == "guidance"
    assert classifier.classify("重新扫描前除了状态，还要看什么？").label == "guidance"


def test_command_generator_extracts_resolution_mode_and_region() -> None:
    """Command generation should parse the main operator fields."""
    generator = CommandGenerator()

    command = generator.generate("扫描 10mm x 12mm 区域，分辨率 5um，左上区域，精细模式")
    payload = command.to_dict()

    assert payload["scan_area_mm"] == {"width": 10.0, "height": 12.0}
    assert payload["resolution"] == {"value": 5.0, "unit": "um"}
    assert payload["region"] == "top_left"
    assert payload["mode"] == "precision"


def test_intent_parser_returns_unified_operation_structure() -> None:
    """The new parser should return one normalized structure for operation requests."""
    parser = IntentParser()

    parsed = parser.parse("scan a 10mm x 10mm area with 5um resolution in top left precision mode")

    assert parsed.intent == "operation"
    assert parsed.command is not None
    assert parsed.command.action == "scan"
    assert parsed.command.scan_area_mm == {"width": 10.0, "height": 10.0}
    assert parsed.command.resolution == {"value": 5.0, "unit": "um"}
    assert parsed.command.region == "top_left"
    assert parsed.command.mode == "precision"
    assert parsed.missing_fields == []
    assert parsed.needs_clarification is False


def test_intent_parser_requires_clarification_when_scan_area_is_missing() -> None:
    """Operation requests without scan area must remain non-actionable."""
    parser = IntentParser()

    parsed = parser.parse("开始扫描")

    assert parsed.intent == "operation"
    assert "scan_area_mm" in parsed.missing_fields
    assert parsed.needs_clarification is True
    assert parsed.clarification_question is not None
    assert parsed.command is not None
    assert parsed.command.region == "current_selection"
    assert parsed.command.mode == "standard"


def test_result_parser_generates_threshold_reason_for_pit() -> None:
    """Result parsing should produce structured severity rationale."""
    parsed = parse_result(
        {
            "defect_type": "pit",
            "location": "roi-2",
            "confidence": 0.83,
            "measurements": {"depth_um": 11.2},
        }
    )

    assert parsed.severity == "high"
    assert parsed.trigger_measurement == "depth_um"
    assert "10 um" in (parsed.threshold_reference or "")
    assert "11.20" in parsed.rule_reason


def test_config_loads_dotenv_and_anthropic_aliases(tmp_path) -> None:
    """Configuration loading should support .env files and Anthropic-compatible aliases."""
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "DAC3D_LLM_PROVIDER=anthropic",
                "ANTHROPIC_API_KEY=test-key",
                "ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic",
            ]
        ),
        encoding="utf-8",
    )

    config = AppConfig.from_env(tmp_path)

    assert config.provider == "anthropic"
    assert config.api_key == "test-key"
    assert config.api_base_url == "https://api.minimaxi.com/anthropic"


def test_default_chunk_size_is_larger_for_richer_context() -> None:
    """The default chunk window should be large enough to preserve more document context."""
    config = AppConfig()

    assert config.chunk_size == 320
    assert config.chunk_overlap == 80


def test_loader_parses_pdf_content_even_if_extension_is_md(tmp_path) -> None:
    """A file with PDF bytes but a .md suffix should still be parsed as PDF text."""
    fake_pdf = tmp_path / "misnamed_manual.md"
    fake_pdf.write_bytes(b"%PDF-1.7\nfake")

    class FakePage:
        def extract_text(self) -> str:
            return "DAC-3D 本科毕业设计课题规划\n1.2 设计原则\n可集成：每个课题都是 DAC-3D 系统的一个模块。"

    class FakePdfReader:
        def __init__(self, path: str) -> None:
            self.path = path
            self.pages = [FakePage()]

    fake_module = types.ModuleType("pypdf")
    fake_module.PdfReader = FakePdfReader
    original_module = sys.modules.get("pypdf")
    sys.modules["pypdf"] = fake_module
    try:
        documents = load_documents(tmp_path, ("**/*.md",))
        chunks = chunk_documents(documents, chunk_size=80, chunk_overlap=10)
    finally:
        if original_module is None:
            sys.modules.pop("pypdf", None)
        else:
            sys.modules["pypdf"] = original_module

    assert documents
    assert documents[0].title == "DAC-3D 本科毕业设计课题规划"
    assert any(chunk["section"] == "设计原则" for chunk in chunks)
    assert any("可集成" in str(chunk["text"]) for chunk in chunks)


def test_loader_parses_docx_content_even_if_extension_is_bin(tmp_path) -> None:
    """A file with DOCX container bytes but a random suffix should still be parsed as Word content."""
    fake_docx = tmp_path / "misnamed_manual.bin"
    _write_minimal_docx(
        fake_docx,
        [
            ("DAC-3D 结构设计", "Heading1"),
            ("1.2 设计原则", None),
            ("设计原则要求模块可集成、难度适中，并保留良好的考核边界。", None),
        ],
    )

    documents = load_documents(tmp_path, ("**/*",))
    chunks = chunk_documents(documents, chunk_size=80, chunk_overlap=10)

    assert documents
    assert documents[0].title == "DAC-3D 结构设计"
    assert any(chunk["section"] == "设计原则" for chunk in chunks)
    assert any("模块可集成" in str(chunk["text"]) for chunk in chunks)
