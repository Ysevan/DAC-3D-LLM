from __future__ import annotations

import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def xml_header() -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("*", "")
    return text


def is_ordered_item(line: str) -> bool:
    return bool(re.match(r"^\d+\.\s", line.strip()))


def is_special(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or stripped.startswith("#")
        or stripped.startswith("|")
        or stripped.startswith("- ")
        or is_ordered_item(stripped)
    )


def parse_table(table_lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in table_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [clean_text(cell) for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {"-", " ", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def parse_markdown(source: Path) -> list[tuple[str, object]]:
    lines = source.read_text(encoding="utf-8").splitlines()
    elements: list[tuple[str, object]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if line.startswith("# "):
            elements.append(("report_title", clean_text(line[2:])))
            i += 1
            continue

        if line.startswith("## "):
            heading = clean_text(line[3:])
            if heading == "题目":
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines):
                    elements.append(("document_title", clean_text(lines[i])))
                    i += 1
                continue
            elements.append(("h1", heading))
            i += 1
            continue

        if line.startswith("### "):
            elements.append(("h2", clean_text(line[4:])))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = parse_table(table_lines)
            if rows:
                elements.append(("table", rows))
            continue

        if stripped.startswith("- "):
            items: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(clean_text(lines[i].strip()[2:]))
                i += 1
            elements.append(("bullets", items))
            continue

        if is_ordered_item(stripped):
            items = []
            while i < len(lines) and is_ordered_item(lines[i].strip()):
                items.append(clean_text(lines[i].strip()))
                i += 1
            elements.append(("ordered", items))
            continue

        parts = [clean_text(stripped)]
        i += 1
        while i < len(lines) and lines[i].strip() and not is_special(lines[i]):
            parts.append(clean_text(lines[i]))
            i += 1
        elements.append(("p", " ".join(parts)))

    return elements


def run_xml(text: str, *, bold: bool = False) -> str:
    text = escape(text)
    r_pr = ""
    if bold:
        r_pr = "<w:rPr><w:b/></w:rPr>"
    return f'<w:r>{r_pr}<w:t xml:space="preserve">{text}</w:t></w:r>'


def paragraph_xml(text: str, style: str, *, bold: bool = False) -> str:
    return (
        f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
        f"{run_xml(text, bold=bold)}</w:p>"
    )


def table_xml(rows: list[list[str]], caption: str) -> str:
    col_count = len(rows[0])
    col_width = int(9000 / max(col_count, 1))
    grid = "".join(f'<w:gridCol w:w="{col_width}"/>' for _ in range(col_count))
    trs: list[str] = []
    for row_index, row in enumerate(rows):
        cells_xml = []
        for cell in row:
            tc_pr = ""
            if row_index == 0:
                tc_pr = (
                    "<w:tcPr>"
                    '<w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>'
                    "</w:tcPr>"
                )
            cell_para = (
                "<w:p><w:pPr><w:pStyle w:val=\"TableText\"/></w:pPr>"
                f"{run_xml(cell, bold=row_index == 0)}</w:p>"
            )
            cells_xml.append(f"<w:tc>{tc_pr}{cell_para}</w:tc>")
        trs.append(f"<w:tr>{''.join(cells_xml)}</w:tr>")

    tbl = (
        "<w:tbl>"
        "<w:tblPr>"
        '<w:tblW w:w="0" w:type="auto"/>'
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:left w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:right w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        "</w:tblBorders>"
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        f"{''.join(trs)}"
        "</w:tbl>"
    )
    return paragraph_xml(caption, "TableCaption") + tbl


def build_document_xml(elements: list[tuple[str, object]]) -> str:
    body_parts: list[str] = []
    current_h1 = ""
    table_index = 0

    for kind, value in elements:
        if kind == "report_title":
            body_parts.append(paragraph_xml(str(value), "ReportTitle", bold=True))
            continue

        if kind == "document_title":
            body_parts.append(paragraph_xml(str(value), "DocumentTitle", bold=True))
            body_parts.append("<w:p/>")
            continue

        if kind == "h1":
            current_h1 = str(value)
            body_parts.append(paragraph_xml(current_h1, "Heading1", bold=True))
            continue

        if kind == "h2":
            body_parts.append(paragraph_xml(str(value), "Heading2", bold=True))
            continue

        if kind == "p":
            text = str(value)
            style = "Normal"
            if text.startswith("关键词："):
                style = "Keyword"
            elif "参考文献" in current_h1:
                style = "Reference"
            body_parts.append(paragraph_xml(text, style))
            continue

        if kind == "bullets":
            for item in value:  # type: ignore[assignment]
                body_parts.append(paragraph_xml(f"• {item}", "ListBody"))
            continue

        if kind == "ordered":
            for item in value:  # type: ignore[assignment]
                style = "ListBody"
                if "参考文献" in current_h1:
                    style = "Reference"
                body_parts.append(paragraph_xml(item, style))
            continue

        if kind == "table":
            table_index += 1
            caption = f"表{table_index}"
            rows = value  # type: ignore[assignment]
            if table_index == 1:
                caption = "表1 研究内容与预期产出"
            elif table_index == 2:
                caption = "表2 工作计划和时间安排"
            body_parts.append(table_xml(rows, caption))
            body_parts.append("<w:p/>")

    sect_pr = (
        "<w:sectPr>"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1800" '
        'w:header="720" w:footer="720" w:gutter="0"/>'
        "</w:sectPr>"
    )

    return (
        xml_header()
        + f'<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}"><w:body>'
        + "".join(body_parts)
        + sect_pr
        + "</w:body></w:document>"
    )


def build_styles_xml() -> str:
    return (
        xml_header()
        + f'''
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
        <w:sz w:val="24"/>
        <w:szCs w:val="24"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:line="360" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:line="360" w:lineRule="auto" w:before="0" w:after="0"/>
      <w:ind w:firstLine="420"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
      <w:sz w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ReportTitle">
    <w:name w:val="ReportTitle"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="240" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:sz w:val="36"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="DocumentTitle">
    <w:name w:val="DocumentTitle"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="120" w:after="240"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:sz w:val="32"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="Heading1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="240" w:after="120"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:sz w:val="28"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="Heading2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="120" w:after="60"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:sz w:val="26"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Keyword">
    <w:name w:val="Keyword"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:ind w:firstLine="0"/>
      <w:spacing w:after="120"/>
    </w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListBody">
    <w:name w:val="ListBody"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:ind w:left="420" w:hanging="210"/>
    </w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TableCaption">
    <w:name w:val="TableCaption"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="120" w:after="60"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TableText">
    <w:name w:val="TableText"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:ind w:firstLine="0"/>
      <w:spacing w:line="300" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:sz w:val="22"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Reference">
    <w:name w:val="Reference"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:ind w:left="420" w:hanging="420" w:firstLine="0"/>
      <w:spacing w:line="300" w:lineRule="auto" w:after="60"/>
    </w:pPr>
    <w:rPr>
      <w:sz w:val="22"/>
    </w:rPr>
  </w:style>
</w:styles>
'''.strip()
    )


def build_settings_xml() -> str:
    return (
        xml_header()
        + f'''
<w:settings xmlns:w="{W_NS}">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="420"/>
  <w:characterSpacingControl w:val="doNotCompress"/>
</w:settings>
'''.strip()
    )


def build_font_table_xml() -> str:
    return (
        xml_header()
        + f'''
<w:fonts xmlns:w="{W_NS}">
  <w:font w:name="Times New Roman"/>
  <w:font w:name="宋体"/>
  <w:font w:name="黑体"/>
</w:fonts>
'''.strip()
    )


def build_content_types_xml() -> str:
    return (
        xml_header()
        + """
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/word/fontTable.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml"/>
</Types>
""".strip()
    )


def build_root_rels_xml() -> str:
    return (
        xml_header()
        + """
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
""".strip()
    )


def build_document_rels_xml() -> str:
    return (
        xml_header()
        + """
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable" Target="fontTable.xml"/>
</Relationships>
""".strip()
    )


def build_app_xml() -> str:
    return (
        xml_header()
        + """
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>1.0</AppVersion>
</Properties>
""".strip()
    )


def build_core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        xml_header()
        + f"""
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>基于大语言模型的 DAC-3D 智能交互助手设计与实现</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
""".strip()
    )


def build_docx(source: Path, output: Path) -> None:
    elements = parse_markdown(source)
    document_xml = build_document_xml(elements)
    styles_xml = build_styles_xml()
    settings_xml = build_settings_xml()
    font_table_xml = build_font_table_xml()

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", build_content_types_xml())
        docx.writestr("_rels/.rels", build_root_rels_xml())
        docx.writestr("docProps/app.xml", build_app_xml())
        docx.writestr("docProps/core.xml", build_core_xml())
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles_xml)
        docx.writestr("word/settings.xml", settings_xml)
        docx.writestr("word/fontTable.xml", font_table_xml)
        docx.writestr("word/_rels/document.xml.rels", build_document_rels_xml())


def main() -> int:
    base_dir = Path(__file__).resolve().parents[1]
    source = base_dir / "开题报告.md"
    output = base_dir / "开题报告.docx"
    if len(sys.argv) > 1:
        source = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2:
        output = Path(sys.argv[2]).resolve()

    build_docx(source, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
