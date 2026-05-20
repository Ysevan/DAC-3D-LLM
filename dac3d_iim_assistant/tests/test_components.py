"""Component-level tests for classification, command generation, and parsing."""

from __future__ import annotations

import sys
import types
import zipfile
import json
import os
from xml.sax.saxutils import escape

from config import AppConfig
from intent.classifier import IntentClassifier
from intent.command_generator import CommandGenerator
from intent.parser import IntentParser
from intent.structured_commands import (
    DEFAULT_REQUIRED_CAMERAS,
    CommandAction,
    default_payload,
    default_safety,
    missing_required_fields,
)
from integration.dac3d_client import DAC3DClient
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
    assert classifier.classify("当前系统在做什么？").label == "status"
    assert classifier.classify("现在运行到哪一步了？").label == "status"
    assert classifier.classify("我想查看系统所有信息和当前进度").label == "status"


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


def test_command_generator_supports_dac3d_runtime_actions() -> None:
    """Command generation should expose DAC-3D-specific runtime actions."""
    generator = CommandGenerator()

    online = generator.generate("start online 144 point scan")
    stop = generator.generate("stop detection")
    latest = generator.generate("get latest inspection result")

    assert online.action == "start_online_scan"
    assert online.payload["func"] == "Scan"
    assert online.payload["total_positions"] == 144
    assert online.safety["hardware_required"] is True
    assert stop.action == "stop_detection"
    assert stop.payload["func"] == "Stop"
    assert latest.action == "get_latest_result"
    assert latest.payload["files"] == ["defect_detail.csv", "defect_summary.csv"]


def test_structured_command_contract_defines_actions_payload_and_safety() -> None:
    """The command layer should centralize action payloads and safety policy."""
    online_payload = default_payload(CommandAction.START_ONLINE_SCAN)
    offline_payload = default_payload(CommandAction.START_OFFLINE_DETECTION)
    validation_safety = default_safety(CommandAction.VALIDATE_OFFLINE_FOLDER)

    assert online_payload["func"] == "Scan"
    assert online_payload["total_positions"] == 144
    assert offline_payload["required_cameras"] == DEFAULT_REQUIRED_CAMERAS
    assert validation_safety["safe_to_auto_execute"] is True
    assert missing_required_fields(
        CommandAction.START_OFFLINE_DETECTION,
        {"payload": {"image_folder": None}},
    ) == ["payload.image_folder"]


def test_command_generator_requires_offline_folder_for_offline_detection() -> None:
    """Offline detection should not be executable without the image folder."""
    generator = CommandGenerator()

    command = generator.generate("start offline detection")

    assert command.action == "start_offline_detection"
    assert "payload.image_folder" in command.missing_fields
    assert command.payload["required_cameras"] == ["焦前", "焦面", "焦后"]
    assert command.payload["required_surfaces"] == ["surface1", "surface2"]


def test_command_generator_extracts_offline_folder_for_validation() -> None:
    """Offline folder validation should extract a Windows path into the payload."""
    generator = CommandGenerator()

    command = generator.generate(r"validate offline folder C:\dac3d\pre_fusion_images")

    assert command.action == "validate_offline_folder"
    assert command.payload["image_folder"] == r"C:\dac3d\pre_fusion_images"
    assert command.missing_fields == []
    assert command.safety["safe_to_auto_execute"] is True


def test_command_generator_extracts_step_length_for_scan_estimate() -> None:
    """Scan planning should parse Chinese step length into resolution."""
    generator = CommandGenerator()
    message = (
        "\u6211\u60f3\u626b\u63cf\u4e00\u4e2a25mm\u00d725mm"
        "\u7684\u6954\u5f62\u6ee4\u5149\u7247\uff0c\u6b65\u957f10\u5fae\u7c73"
    )

    command = generator.generate(message)

    assert command.action == "scan"
    assert command.scan_area_mm == {"width": 25.0, "height": 25.0}
    assert command.resolution == {"value": 10.0, "unit": "um"}
    assert command.missing_fields == []


def test_command_generator_extracts_chinese_offline_test_folder() -> None:
    """Chinese offline-test requests should become executable offline detection commands."""
    generator = CommandGenerator()
    message = (
        "选择C:\\Users\\xecat\\DAC-3D-LLM\\福特科\\pre_fusion_images"
        "下的图片，进行离线测试"
    )

    command = generator.generate(message)

    assert command.action == "start_offline_detection"
    assert command.payload["image_folder"] == (
        "C:\\Users\\xecat\\DAC-3D-LLM\\福特科\\pre_fusion_images"
    )
    assert command.payload["message_type"] == "offline_detect_folder"
    assert command.missing_fields == []


def test_dac3d_client_accepts_new_structured_commands(tmp_path) -> None:
    """The mock adapter should validate and simulate DAC-3D command actions."""
    image_dir = tmp_path / "pre_fusion_images"
    image_dir.mkdir()
    for surface in ("surface1", "surface2"):
        for camera in ("焦前", "焦面", "焦后"):
            (image_dir / f"pos1_{surface}_{camera}.jpg").write_bytes(b"fake")

    client = DAC3DClient(mock_mode=True)
    validation = client.submit_scan_command(
        {
            "action": "validate_offline_folder",
            "payload": {"image_folder": str(image_dir)},
            "safety": {"safe_to_auto_execute": True},
        }
    )
    online = client.submit_scan_command(
        {
            "action": "start_online_scan",
            "payload": {"func": "Scan", "total_positions": 144},
            "safety": {"hardware_required": True},
        }
    )

    assert validation["validation"]["ready"] is True
    assert online["status"]["state"] == "queued"
    assert client.runtime_snapshot()["last_command_action"] == "start_online_scan"


def test_dac3d_client_reads_status_file_endpoint(tmp_path) -> None:
    """Standalone web assistant should read DAC-3D host status from a local file bridge."""
    status_file = tmp_path / "dac3d_runtime_status.json"
    status_file.write_text(
        json.dumps(
            {
                "status": {
                    "state": "running",
                    "progress": 42,
                    "message": "检测中...(60/144)",
                    "step": "sample_detection_result",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client = DAC3DClient(endpoint=status_file.as_uri())

    status = client.query_current_status()
    snapshot = client.runtime_snapshot()

    assert status["state"] == "running"
    assert status["progress"] == 42
    assert status["step"] == "sample_detection_result"
    assert snapshot["mode"] == "status_file"


def test_dac3d_client_reads_latest_result_from_status_file(tmp_path) -> None:
    """File bridge should expose the latest DAC-3D inspection result to the assistant."""
    status_file = tmp_path / "dac3d_runtime_status.json"
    status_file.write_text(
        json.dumps(
            {
                "status": {
                    "state": "running",
                    "progress": 10,
                    "message": "检测中...(1/144)",
                    "latest_result": {
                        "result_root": "C:/dac3d/results/latest",
                        "tray_id": 12,
                        "position": 1,
                        "quality": False,
                        "quality_label": "不合格",
                        "defects_num": 1,
                        "files": ["surface1.jpg", "surface2.jpg"],
                        "defects": [
                            {
                                "defect_type": "scratch",
                                "position": [120.0, 240.0],
                                "size": 42.0,
                                "reason": "区域C中发现超标划痕",
                            }
                        ],
                        "parsed_result": {
                            "defect_type": "scratch",
                            "location": "pos1",
                            "confidence": 1.0,
                            "measurements": {"size_px": 42.0},
                            "severity": "medium",
                            "rule_reason": "区域C中发现超标划痕",
                        },
                    },
                    "result_history": [
                        {
                            "position": 1,
                            "quality": False,
                            "quality_label": "不合格",
                            "defects_num": 1,
                            "defects": [{"defect_type": "scratch"}],
                        },
                        {
                            "position": 2,
                            "quality": True,
                            "quality_label": "合格",
                            "defects_num": 0,
                            "defects": [],
                        },
                    ],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client = DAC3DClient(endpoint=status_file.as_uri())

    summary = client.get_latest_result_summary()
    parsed = client.get_recent_inspection_result()

    assert summary["quality_label"] == "不合格"
    assert summary["defects_num"] == 1
    assert summary["files"] == ["surface1.jpg", "surface2.jpg"]
    assert summary["checked_samples"] == 2
    assert len(summary["result_history"]) == 2
    assert parsed["defect_type"] == "scratch"
    assert parsed["rule_reason"] == "区域C中发现超标划痕"


def test_dac3d_client_writes_command_file_for_status_endpoint(tmp_path) -> None:
    """Submitting through a file endpoint should publish a command for the DAC-3D host UI."""
    status_file = tmp_path / "dac3d_runtime_status.json"
    command_file = tmp_path / "dac3d_assistant_command.json"
    status_file.write_text(
        json.dumps({"status": {"state": "idle", "progress": 0}}, ensure_ascii=False),
        encoding="utf-8",
    )
    old_command_path = os.environ.get("DAC3D_COMMAND_PATH")
    os.environ["DAC3D_COMMAND_PATH"] = str(command_file)
    try:
        client = DAC3DClient(endpoint=status_file.as_uri())
        result = client.submit_scan_command(
            {
                "action": "scan",
                "scan_area_mm": {"width": 10.0, "height": 10.0},
                "region": "current_selection",
                "mode": "standard",
                "payload": {},
                "safety": {"needs_confirmation": True},
            }
        )
    finally:
        if old_command_path is None:
            os.environ.pop("DAC3D_COMMAND_PATH", None)
        else:
            os.environ["DAC3D_COMMAND_PATH"] = old_command_path

    payload = json.loads(command_file.read_text(encoding="utf-8"))
    assert result["mode"] == "command_file_bridge"
    assert result["status"]["state"] == "command_sent"
    assert payload["status"] == "pending"
    assert payload["command"]["action"] == "scan"
    assert payload["command"]["scan_area_mm"] == {"width": 10.0, "height": 10.0}


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
