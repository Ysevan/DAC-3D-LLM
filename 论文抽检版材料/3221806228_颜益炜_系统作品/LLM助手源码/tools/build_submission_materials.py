from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from textwrap import wrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
MAIN_SYSTEM = REPO_ROOT / "福特科" / "xxp_ui"
OUT = ROOT / ".tmp_submission_materials"
WORKS = OUT / "3221806228_颜益炜_系统作品"


def register_fonts() -> tuple[str, str]:
    candidates = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "simsun.ttc",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc",
    ]
    for font_path in candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont("CNFont", str(font_path)))
            pdfmetrics.registerFont(TTFont("CNFontBold", str(font_path)))
            return "CNFont", "CNFontBold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=20,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=14,
            leading=22,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=12,
            leading=18,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10.5,
            leading=18,
            firstLineIndent=21,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9,
            leading=15,
            spaceAfter=4,
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["Code"],
            fontName=FONT,
            fontSize=7.5,
            leading=10,
        ),
    }


S = styles()


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.drawCentredString(A4[0] / 2, A4[1] - 1.0 * cm, "DAC-3D智能交互助手抽检材料")
    canvas.setStrokeColor(colors.grey)
    canvas.line(2 * cm, A4[1] - 1.15 * cm, A4[0] - 2 * cm, A4[1] - 1.15 * cm)
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, str(doc.page))
    canvas.restoreState()


def p(text: str, style: str = "body"):
    return Paragraph(text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), S[style])


def table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F2F2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BFBFBF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def build_pdf(path: Path, title: str, story):
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2.4 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.0 * cm,
        bottomMargin=1.8 * cm,
    )
    doc.build([p(title, "title"), *story], onFirstPage=header_footer, onLaterPages=header_footer)


def copy_tree_filtered(src: Path, dst: Path, include_exts: set[str] | None = None):
    skip_dirs = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "node_modules",
        ".venv",
        "dist",
        "build",
        ".tmp",
        ".tmp_submission_materials",
        ".tmp_0527_full_job",
        "BypassAIGC",
        "weights",
        "deploy",
        "pre_fusion_images",
        "runtime",
        "datebase",
        "stage",
        "vector_store",
        "dac3d_iim_assistant.egg-info",
    }
    skip_exts = {
        ".pt",
        ".pth",
        ".onnx",
        ".torchscript",
        ".engine",
        ".dll",
        ".zip",
        ".rar",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".docx",
        ".pdf",
        ".log",
        ".pid",
    }
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if any(part in skip_dirs or part.lower().endswith(".zip") for part in rel.parts):
            continue
        if item.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
            continue
        if item.suffix.lower() in skip_exts:
            continue
        if include_exts and item.suffix.lower() not in include_exts:
            continue
        if item.stat().st_size > 1_500_000:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8-sig" if path.suffix.lower() == ".cmd" else "utf-8"
    path.write_text(text, encoding=encoding)


def copy_if_exists(src: Path, dst: Path):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_offline_runtime_assets(dac_dst: Path):
    # Keep only the minimum DAC-3D assets needed for the offline model demo.
    for src in [
        MAIN_SYSTEM / "deploy" / "weights" / "best.pt",
        MAIN_SYSTEM / "deploy" / "inference.py",
        MAIN_SYSTEM / "deploy" / "infer" / "infer_withflow.py",
        MAIN_SYSTEM / "deploy" / "export_model.py",
    ]:
        copy_if_exists(src, dac_dst / src.relative_to(MAIN_SYSTEM))

    sample_src = REPO_ROOT / "福特科" / "pre_fusion_images"
    sample_dst = WORKS / "DAC-3D离线检测示例数据" / "pre_fusion_images"
    if sample_src.exists():
        for image in sorted(sample_src.glob("pos100_surface*_*")):
            if image.is_file():
                copy_if_exists(image, sample_dst / image.name)


def build_works_folder():
    if WORKS.exists():
        shutil.rmtree(WORKS)
    WORKS.mkdir(parents=True)

    assistant_dst = WORKS / "LLM助手源码"
    copy_tree_filtered(ROOT, assistant_dst)

    # Keep frontend source and package metadata, but not node_modules/dist.
    for frontend_name in ["frontend", "ui2"]:
        src = ROOT / frontend_name
        dst = assistant_dst / frontend_name
        if src.exists():
            copy_tree_filtered(src, dst)

    dac_dst = WORKS / "DAC-3D主系统源码_精简版" / "xxp_ui"
    if MAIN_SYSTEM.exists():
        copy_tree_filtered(MAIN_SYSTEM, dac_dst)
        copy_offline_runtime_assets(dac_dst)
        compat_dac_dst = WORKS / "福特科" / "xxp_ui"
        copy_tree_filtered(MAIN_SYSTEM, compat_dac_dst)
        copy_offline_runtime_assets(compat_dac_dst)

    write_text(
        WORKS / "README_抽检复现说明.md",
        """# DAC-3D智能交互助手抽检复现说明

本目录为论文抽检版系统作品，包含 LLM 助手源码、DAC-3D 主系统相关源码精简版、知识库样例、启动脚本和复现说明。

## 目录说明

- `LLM助手源码/`：FastAPI 后端、RAG、Agent Runtime、工具网关、安全策略、知识库文档、React 前端源码与测试用例。
- `DAC-3D主系统源码_精简版/`：DAC-3D 主系统中与离线检测、界面、运行状态、命令桥接和检测结果读取相关的源码。为支持离线模型复现，保留了默认离线检测权重 `deploy/weights/best.pt`。
- `福特科/xxp_ui/`：与原工程和自动化测试兼容的 DAC-3D 主系统路径，内容与精简版一致。
- `DAC-3D离线检测示例数据/`：保留一组 `pre_fusion_images` 三相机示例图，可用于离线检测流程验证。
- `启动助手后端.cmd`：启动助手 FastAPI 后端。
- `启动助手前端.cmd`：启动白色主题 Web 前端。
- `启动DAC-3D离线检测主程序.cmd`：启动 DAC-3D 主系统，进入界面后选择示例数据进行离线检测。
- `运行自动化测试.cmd`：运行助手自动化测试。

## 复现范围

抽检包可复现论文中“LLM 助手原型、知识库问答、命令预览、状态查询、结果解释、工具网关与安全校验”等功能。DAC-3D 主系统侧以“离线模型运行”为复现目标，保留默认离线检测权重和一组示例图；在线相机采集、运动控制卡控制和完整生产数据不属于抽检包复现范围。

## 基本步骤

1. 安装 Python 3.10 及 Node.js。
2. 进入 `LLM助手源码`，执行 `python -m pip install -e .`。
3. 配置 `.env`，或使用 mock 模式运行本地演示。
4. 执行 `python app.py` 启动后端。
5. 进入 `ui2` 后执行 `npm install`、`npm run dev` 启动前端。
6. 在浏览器访问前端地址，输入“当前检测状态是什么？”、“扫描 25mm x 25mm 区域，分辨率 10 微米”等进行验证。
7. 如需验证 DAC-3D 离线模型，进入 `DAC-3D主系统源码_精简版/xxp_ui` 安装 PyQt5、OpenCV、PyTorch、Ultralytics、SAHI 等依赖后执行 `python main.py`，在界面中选择 `DAC-3D离线检测示例数据/pre_fusion_images` 作为离线图片目录。
""",
    )

    write_text(
        WORKS / "启动助手后端.cmd",
        """@echo off
chcp 65001 >nul
cd /d "%~dp0\\LLM助手源码"
python -m pip install -e .
python app.py
pause
""",
    )
    write_text(
        WORKS / "启动助手前端.cmd",
        """@echo off
chcp 65001 >nul
cd /d "%~dp0\\LLM助手源码\\ui2"
set VITE_API_BASE_URL=http://127.0.0.1:7890
npm install
npm run dev
pause
""",
    )
    write_text(
        WORKS / "运行自动化测试.cmd",
        """@echo off
chcp 65001 >nul
cd /d "%~dp0\\LLM助手源码"
python -m pytest tests -q
pause
""",
    )
    write_text(
        WORKS / "启动DAC-3D离线检测主程序.cmd",
        """@echo off
chcp 65001 >nul
cd /d "%~dp0\\DAC-3D主系统源码_精简版\\xxp_ui"
python main.py
pause
""",
    )
    write_text(
        WORKS / "DAC-3D离线模型运行说明.md",
        """# DAC-3D离线模型运行说明

抽检包中的 DAC-3D 主系统只要求复现离线模型运行，不要求连接真实相机、运动控制卡或在线扫描硬件。

## 已保留内容

- 离线检测主流程源码：`image_processor_22.py`
- 默认离线检测权重：`deploy/weights/best.pt`
- 离线推理相关脚本：`deploy/inference.py`、`deploy/infer/infer_withflow.py`
- 一组完整示例图：`DAC-3D离线检测示例数据/pre_fusion_images`

## 运行方式

1. 进入 `DAC-3D主系统源码_精简版/xxp_ui`。
2. 安装 PyQt5、opencv-python、numpy、Pillow、torch、ultralytics、sahi 等依赖。
3. 执行 `python main.py`。
4. 在界面中选择示例图目录，启动离线检测。

如评阅环境缺少 GPU，程序可按代码中的设备选择逻辑回退到 CPU，但推理速度会变慢。
""",
    )


def operation_manual_story():
    return [
        p("一、系统概述", "h1"),
        p("DAC-3D智能交互助手面向三维光学检测软件的辅助交互场景，提供参数问答、自然语言命令预览、运行状态查询、检测结果解释和操作指导等功能。系统不替代 DAC-3D 主检测流程，也不直接控制检测设备；涉及执行的操作需要经过结构化预览、安全校验和用户确认。"),
        p("二、启动方式", "h1"),
        table(
            [
                ["步骤", "操作", "说明"],
                ["1", "配置 Python 与 Node.js 环境", "建议使用 Python 3.10，Node.js 18 及以上。"],
                ["2", "安装助手依赖", "进入 LLM助手源码目录后执行 python -m pip install -e .。"],
                ["3", "配置模型服务", "在 .env 中配置 LLM provider、API 地址、模型名和密钥；也可以使用 mock 模式进行本地演示。"],
                ["4", "启动后端", "执行 python app.py，默认提供 FastAPI 服务。"],
                ["5", "启动前端", "进入 ui2 目录执行 npm install 与 npm run dev。"],
                ["6", "访问系统", "浏览器打开前端地址，进入聊天式交互页面。"],
            ],
            [1.4 * cm, 5.2 * cm, 8.6 * cm],
        ),
        p("三、主要功能操作", "h1"),
        table(
            [
                ["功能", "示例输入", "预期输出"],
                ["参数问答", "曝光时间是什么意思？", "回答参数含义，并给出知识来源。"],
                ["命令预览", "扫描 25mm x 25mm 区域，分辨率 10 微米", "生成结构化命令预览，等待确认。"],
                ["状态查询", "当前检测状态是什么？", "返回 DAC-3D 运行状态或 mock 状态。"],
                ["结果解释", "划痕长度 1.2mm，深度 0.08mm，严重吗？", "结合缺陷规则解释严重度和依据。"],
                ["操作指导", "样品表面反光很强怎么办？", "给出受知识库约束的处理建议。"],
            ],
            [3.0 * cm, 5.8 * cm, 6.4 * cm],
        ),
        p("四、注意事项", "h1"),
        p("助手生成的操作类结果默认为命令预览，不会绕过工具网关直接写入 DAC-3D 命令文件。高风险操作、路径相关操作或字段缺失的请求，需要先完成安全校验和显式确认。外部模型不可用时，应记录错误并切换到 mock 或本地回退流程进行演示。"),
    ]


def deployment_story():
    return [
        p("一、部署环境", "h1"),
        table(
            [
                ["项目", "要求"],
                ["操作系统", "Windows 环境优先；助手后端可在常规 Python 环境运行。"],
                ["Python", "Python 3.10 或兼容版本。"],
                ["Node.js", "Node.js 18 及以上，用于构建和运行 React/Vite 前端。"],
                ["LLM 服务", "支持 OpenAI-compatible 接口；需配置 API 地址、模型名和密钥。"],
                ["DAC-3D 主系统", "真实硬件联调需要相机、运动控制卡、模型权重和检测数据。抽检包保留源码与桥接逻辑。"],
            ],
            [4 * cm, 11.2 * cm],
        ),
        p("二、后端部署", "h1"),
        Preformatted(
            "cd LLM助手源码\npython -m pip install -e .\npython knowledge_base/build_kb.py\npython app.py",
            S["code"],
        ),
        p("后端启动后提供健康检查、状态查询、同步聊天、流式聊天、知识库维护和命令预览等接口。若不连接真实 DAC-3D 主系统，可使用 mock 模式验证核心链路。"),
        p("三、前端部署", "h1"),
        Preformatted(
            "cd LLM助手源码\\ui2\nnpm install\nnpm run dev",
            S["code"],
        ),
        p("前端通过代理访问后端接口，展示回答内容、来源依据、结构化命令、运行状态和安全确认结果。"),
        p("四、DAC-3D 集成说明", "h1"),
        p("系统采用低侵入式集成方式：DAC-3D 主系统负责写出运行状态和检测结果，助手读取状态文件并生成解释；操作请求经工具网关、策略校验、路径白名单和确认流程后，才写入命令文件，由主系统轮询执行。抽检包中 DAC-3D 侧以离线模型运行为复现目标，保留默认离线检测权重和一组示例图，不要求复现在线硬件控制。"),
        p("五、抽检包体积控制", "h1"),
        p("为控制作品文件夹体积，抽检包未包含 node_modules、Python 虚拟环境、冗余模型权重、导出模型全集、离线图片大数据、运行缓存和论文过程文件。专家复现助手原型时可按说明重新安装依赖；DAC-3D 侧保留最小离线运行资源，包括 `deploy/weights/best.pt` 和一组 `pre_fusion_images` 示例图。"),
    ]


def source_code_story():
    story = [
        p("一、源码组成", "h1"),
        table(
            [
                ["目录", "作用"],
                ["LLM助手源码/app.py", "应用入口、服务装配和请求路由。"],
                ["LLM助手源码/agent_runtime.py", "统一 Agent 运行时和工具编排入口。"],
                ["LLM助手源码/rag", "知识库检索、提示词组织和模型服务适配。"],
                ["LLM助手源码/intent", "意图识别、字段抽取和结构化命令预览。"],
                ["LLM助手源码/tool_gateway 与 safety", "工具网关、安全策略、路径白名单和确认流程。"],
                ["LLM助手源码/integration", "DAC-3D 状态、命令和检测结果适配边界。"],
                ["LLM助手源码/ui2", "React/Vite 前端交互界面。"],
                ["DAC-3D主系统源码_精简版/xxp_ui", "主系统界面、运行状态、命令桥接和检测结果相关源码。"],
            ],
            [5.4 * cm, 9.8 * cm],
        ),
        p("二、关键源码摘录", "h1"),
        p("完整源码已随“系统作品”文件夹提供。以下仅摘录与论文工作直接相关的关键文件片段，便于评阅时快速定位。"),
    ]
    snippets = [
        ROOT / "app.py",
        ROOT / "agent_runtime.py",
        ROOT / "rag" / "llm_client.py",
        ROOT / "intent" / "command_generator.py",
        ROOT / "tool_gateway" / "gateway.py",
        ROOT / "safety" / "policy_engine.py",
        ROOT / "integration" / "dac3d_client.py",
        ROOT / "integration" / "result_parser.py",
        MAIN_SYSTEM / "window" / "ui.py",
    ]
    for file in snippets:
        if not file.exists():
            continue
        story.append(p(f"文件：{file.relative_to(REPO_ROOT)}", "h2"))
        text = file.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        selected = "\n".join(lines[:120])
        if len(lines) > 120:
            selected += "\n...（后续源码见系统作品文件夹）"
        wrapped = []
        for line in selected.splitlines():
            if len(line) <= 110:
                wrapped.append(line)
            else:
                wrapped.extend(wrap(line, 110))
        story.append(Preformatted("\n".join(wrapped), S["code"]))
        story.append(Spacer(1, 6))
    return story


def build_pdfs():
    build_pdf(OUT / "3221806228_颜益炜_操作说明书.pdf", "3221806228_颜益炜_操作说明书", operation_manual_story())
    build_pdf(OUT / "3221806228_颜益炜_系统部署说明.pdf", "3221806228_颜益炜_系统部署说明", deployment_story())
    build_pdf(OUT / "3221806228_颜益炜_系统源代码.pdf", "3221806228_颜益炜_系统源代码", source_code_story())


def summarize():
    rows = []
    for item in OUT.rglob("*"):
        if item.is_file():
            rows.append((str(item.relative_to(OUT)), item.stat().st_size))
    total = sum(size for _, size in rows)
    lines = [
        "DAC-3D论文抽检版材料生成记录",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"文件数：{len(rows)}",
        f"总大小：{total / 1024 / 1024:.2f} MB",
        "",
        "主要文件：",
    ]
    for rel, size in sorted(rows)[:500]:
        lines.append(f"- {rel} ({size / 1024:.1f} KB)")
    write_text(OUT / "材料清单.txt", "\n".join(lines))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    build_works_folder()
    build_pdfs()
    summarize()
    print(f"built: {OUT}")


if __name__ == "__main__":
    main()
