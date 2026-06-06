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
            wordWrap="CJK",
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9,
            leading=15,
            alignment=TA_LEFT,
            wordWrap="CJK",
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
    def c(text: str):
        return p(text, "small")

    return [
        p("一、系统概述", "h1"),
        p("本说明书用于指导评阅人员在“3221806228_颜益炜_系统作品”目录内复现 DAC-3D 智能交互助手及 DAC-3D 离线检测演示。系统面向三维光学检测软件的辅助交互场景，提供参数问答、操作指导、运行状态查询、自然语言命令预览、命令安全确认、检测结果解释和知识库信息查看等功能。"),
        p("系统边界与论文一致：DAC-3D 主系统已有缺陷检测模型负责输出划痕、点蚀和飞溅等检测结果；本课题实现的是大语言模型助手、RAG 知识检索、Agent 运行时、工具网关、安全审查、状态文件桥接和结果解释链路，不包含检测模型本身的训练、改造或部署研究。"),
        p("抽检包为控制体积，未包含 Python 虚拟环境、node_modules、运行缓存、大规模离线图片集和冗余模型文件。DAC-3D 侧保留离线模型运行所需的最小资源，包括主系统源码、默认权重 best.pt 和一组示例图像。"),
        p("二、目录与文件说明", "h1"),
        table(
            [
                [c("目录或文件"), c("说明")],
                [c("LLM助手源码"), c("FastAPI 后端、Agent 运行时、RAG、命令生成、安全网关、React/Vite 前端和测试用例。复现论文中的智能助手主要功能时从该目录启动。")],
                [c("DAC-3D主系统源码_精简版"), c("DAC-3D 主系统离线检测、状态写出、命令桥接和结果记录相关源码，用于验证主系统离线模型运行和助手集成边界。")],
                [c("福特科\\xxp_ui"), c("与原项目路径兼容的 DAC-3D 主系统副本，部分自动化测试和旧版入口按该路径查找文件。")],
                [c("DAC-3D离线检测示例数据"), c("少量 pre_fusion_images 示例图，供 DAC-3D 离线检测界面选择和推理使用。")],
                [c("启动助手后端.cmd"), c("启动 FastAPI 服务，默认地址为 http://127.0.0.1:7890。")],
                [c("启动助手前端.cmd"), c("启动 Vite 前端，默认访问地址为 http://127.0.0.1:5173，并指向 7890 后端。")],
                [c("启动DAC-3D离线检测主程序.cmd"), c("启动 DAC-3D 主系统离线演示入口，需要本机具备 PyQt5、OpenCV、PyTorch、Ultralytics 等依赖。")],
                [c("运行自动化测试.cmd"), c("执行助手自动化测试，用于快速验证核心接口、Agent、安全和命令链路。")],
            ],
            [5.0 * cm, 11.0 * cm],
        ),
        p("三、运行环境准备", "h1"),
        table(
            [
                [c("项目"), c("配置与说明")],
                [c("操作系统"), c("建议使用 Windows 10/11。论文系统和抽检包按 Windows 路径与批处理脚本组织。")],
                [c("Python"), c("建议使用 Python 3.10 或兼容版本。后端、RAG、测试脚本和 DAC-3D 离线检测均依赖 Python。")],
                [c("Node.js"), c("建议使用 Node.js 18 及以上，用于运行 React/Vite 前端。")],
                [c("助手依赖"), c("在 LLM助手源码中执行 python -m pip install -e .；如需开发和测试，可安装开发依赖。")],
                [c("前端依赖"), c("在 LLM助手源码\\ui2 中执行 npm install。抽检包不携带 node_modules，需要本地安装。")],
                [c("DAC-3D 离线依赖"), c("需要 PyQt5、opencv-python、numpy、Pillow、torch、ultralytics、sahi。无 GPU 时可按代码逻辑使用 CPU，推理速度会降低。")],
                [c("外部 LLM"), c("可配置 OpenAI-compatible 服务；若不配置真实模型，可使用 mock 模式完成本地功能演示。")],
            ],
            [4.0 * cm, 12.0 * cm],
        ),
        p("四、助手后端启动", "h1"),
        p("推荐直接双击“启动助手后端.cmd”。也可以进入 LLM助手源码目录后手动执行以下命令："),
        Preformatted(
            "python -m pip install -e .\npython knowledge_base/build_kb.py\npython -m uvicorn ui2_server:app --host 127.0.0.1 --port 7890",
            S["code"],
        ),
        p("后端启动后，可通过 http://127.0.0.1:7890/api/health 检查服务状态；通过 http://127.0.0.1:7890/api/runtime 查看模型、知识库、mock 模式和 DAC-3D 集成状态。"),
        p("如果需要接通真实大语言模型，请在本地 .env 中配置 DAC3D_LLM_PROVIDER、DAC3D_LLM_API_BASE_URL、DAC3D_LLM_API_KEY 和 DAC3D_LLM_MODEL_NAME。为保护个人隐私和接口安全，抽检材料未包含个人 token 或 API Key；复查时请使用评阅环境可用的 token 或 API Key。"),
        p("五、助手前端启动与使用", "h1"),
        p("推荐先启动后端，再双击“启动助手前端.cmd”。前端脚本会设置 VITE_API_BASE_URL=http://127.0.0.1:7890，并在本地启动 Vite 服务。浏览器访问 http://127.0.0.1:5173 后即可进入白色主题的聊天式交互页面。"),
        table(
            [
                [c("功能"), c("操作示例与预期结果")],
                [c("参数问答"), c("输入“曝光时间是什么意思？”。系统应基于知识库说明参数含义，并展示来源依据或相关文档信息。")],
                [c("操作指导"), c("输入“样品表面反光很强怎么办？”。系统应给出受知识库约束的处理建议，避免无依据扩展。")],
                [c("状态查询"), c("输入“当前检测状态是什么？”。系统应读取 DAC-3D 状态桥接或 mock 状态，并返回运行状态、进度和当前任务信息。")],
                [c("命令预览"), c("输入“扫描 25mm x 25mm 区域，分辨率 10 微米，离线模式”。系统应抽取扫描区域、分辨率、模式等字段，生成结构化命令预览，不会直接执行。")],
                [c("结果解释"), c("输入“划痕长度 1.2mm，深度 0.08mm，严重吗？”。系统应结合缺陷类型、阈值规则和 DAC-3D 输出字段解释严重度与依据。")],
            ],
            [3.2 * cm, 12.8 * cm],
        ),
        p("六、命令预览与安全确认", "h1"),
        p("操作类请求不会绕过安全链路直接写入 DAC-3D 命令文件。系统会先经过意图识别、字段抽取、结构化校验、路径白名单、风险分类和显式确认。字段缺失时，界面应提示用户补充分辨率、扫描区域、模式或离线目录等信息。"),
        table(
            [
                [c("场景"), c("系统处理方式")],
                [c("字段完整且低风险"), c("生成命令预览，并给出可确认的结构化内容。")],
                [c("扫描区域、分辨率或模式缺失"), c("返回澄清提示，不猜测危险参数。")],
                [c("路径不在白名单"), c("拒绝提交，并说明路径需要位于允许目录。")],
                [c("高风险或执行类请求"), c("必须经过显式确认，不能只凭 LLM 文本直接执行。")],
                [c("提示词要求跳过审核"), c("安全策略优先，系统应拒绝绕过工具网关和确认流程。")],
            ],
            [4.4 * cm, 11.6 * cm],
        ),
        p("通过接口直接测试命令预览时，需要提供会话、操作者和角色信息。下面示例仅用于本地验证："),
        Preformatted(
            'POST http://127.0.0.1:7890/api/commands/preview\n'
            'Headers: X-Session-Id, X-Operator-Id, X-Operator-Roles\n'
            'Body: {"message":"扫描 25mm x 25mm 区域，分辨率 10 微米，离线模式"}',
            S["code"],
        ),
        p("七、DAC-3D 离线检测演示", "h1"),
        p("DAC-3D 主系统的在线硬件控制不作为抽检包复现目标。抽检包重点验证离线模型可以运行，并验证助手能够围绕检测状态和结果进行解释。"),
        table(
            [
                [c("步骤"), c("操作与预期结果")],
                [c("1"), c("双击“启动DAC-3D离线检测主程序.cmd”，应打开 DAC-3D 主系统界面。")],
                [c("2"), c("在界面中选择 DAC-3D离线检测示例数据 下的 pre_fusion_images 图像目录，系统应加载示例图。")],
                [c("3"), c("确认默认权重 deploy\\weights\\best.pt 可用；模型类别应包含 splash、chipping、scratch。")],
                [c("4"), c("启动离线检测，系统应对示例图进行推理，生成缺陷框、检测结果或结果记录。")],
                [c("5"), c("回到助手前端询问当前状态或检测结果，助手应读取状态/结果桥接数据并生成解释。")],
            ],
            [1.6 * cm, 14.4 * cm],
        ),
        p("若评阅环境缺少 GPU，模型仍可使用 CPU 推理，但首轮加载和推理时间会变长。若 PyQt5 或 Ultralytics 未安装，应先按部署说明安装依赖。"),
        p("八、自动化测试与验证", "h1"),
        p("抽检包保留助手测试用例。可双击“运行自动化测试.cmd”，或进入 LLM助手源码目录后执行："),
        Preformatted("python -m pytest tests -q", S["code"]),
        p("测试覆盖配置加载、知识库检索、参数问答、命令预览、安全策略、状态查询、结果解释、Agent 回归和 Prompt Injection 拦截等场景。测试通过只能说明本地原型链路可运行，不等同于真实设备长时间闭环验收。"),
        p("九、常见问题处理", "h1"),
        table(
            [
                [c("问题现象"), c("原因与处理方式")],
                [c("前端提示 502 或无法连接后端"), c("通常是后端未启动，或前端未指向 7890 端口。请先启动后端，并确认启动助手前端.cmd 中 VITE_API_BASE_URL 为 http://127.0.0.1:7890。")],
                [c("接口返回 401"), c("通常是缺少会话标识。接口测试时增加 X-Session-Id；通过前端使用时通常由页面自动处理。")],
                [c("接口返回 403"), c("通常是操作者、角色或路径白名单校验未通过。请补充 X-Operator-Id、X-Operator-Roles，并确认离线目录位于允许范围。")],
                [c("知识库回答缺少依据"), c("通常是知识库未构建或文档未加载。请执行 python knowledge_base/build_kb.py，或在前端查看知识库摘要。")],
                [c("真实 LLM 无响应"), c("通常是 API 地址、Key、模型名或网络不可用。请检查 .env；必要时切换 mock 模式完成本地演示。")],
                [c("DAC-3D 离线界面无法启动"), c("通常是 PyQt5、OpenCV、Torch 或模型依赖缺失。请按部署说明安装依赖，并先运行 Python 语法检查确认源码未损坏。")],
                [c("离线推理很慢"), c("通常是使用 CPU 推理或首次加载模型。请等待模型加载完成；如有 CUDA 环境可配置 GPU。")],
            ],
            [4.8 * cm, 11.2 * cm],
        ),
        p("十、复现检查清单", "h1"),
        table(
            [
                [c("检查项"), c("通过标准")],
                [c("助手后端"), c("访问 /api/health 返回正常；/api/runtime 可看到运行时、知识库和模型状态。")],
                [c("助手前端"), c("浏览器能打开白色主题页面，并能完成参数问答、状态查询和命令预览。")],
                [c("安全链路"), c("不完整命令会澄清；路径或高风险请求不会被直接执行。")],
                [c("知识库"), c("能够显示知识库摘要，并在问答中返回与 DAC-3D 文档相关的依据。")],
                [c("DAC-3D 离线模型"), c("best.pt 可加载，示例图可完成 splash、chipping、scratch 类别推理。")],
                [c("自动化测试"), c("运行 pytest tests -q 后核心测试通过。")],
            ],
            [5.0 * cm, 11.0 * cm],
        ),
    ]


def deployment_story():
    def c(text: str):
        return p(text, "small")

    return [
        p("一、部署目标与边界", "h1"),
        p("本部署说明用于指导评阅人员在本地环境中部署“3221806228_颜益炜_系统作品”目录内的 DAC-3D 智能交互助手和 DAC-3D 离线检测演示环境。部署说明重点说明环境、依赖、配置、启动顺序和验证方式；系统部署完成后的具体功能使用方法，请参见《3221806228_颜益炜_操作说明书》。"),
        p("抽检包支持两类复现目标：一是部署并运行 LLM 助手后端和前端，验证参数问答、状态查询、命令预览、结果解释和安全链路；二是部署 DAC-3D 主系统的离线检测演示，验证默认权重和示例图像可以完成本地推理。真实相机、运动控制卡和在线检测硬件不属于抽检包必须复现范围。"),
        p("二、部署环境要求", "h1"),
        table(
            [
                [c("项目"), c("要求")],
                [c("操作系统"), c("建议使用 Windows 10/11。抽检包中的批处理脚本、路径和 DAC-3D 主系统演示均按 Windows 环境组织。")],
                [c("Python"), c("建议使用 Python 3.10。助手后端、知识库构建、自动化测试和 DAC-3D 离线检测都需要 Python。")],
                [c("Node.js"), c("建议使用 Node.js 18 及以上，用于安装和启动 React/Vite 前端。")],
                [c("LLM 服务"), c("支持 OpenAI-compatible 接口。若没有可用 token/API Key，可使用 mock 模式验证本地链路。")],
                [c("DAC-3D 离线检测"), c("需要 PyQt5、OpenCV、NumPy、Pillow、PyTorch、Ultralytics、SAHI 等依赖。无 GPU 时可使用 CPU 推理，但速度会变慢。")],
            ],
            [4.0 * cm, 12.0 * cm],
        ),
        p("三、目录放置与文件检查", "h1"),
        p("建议将“3221806228_颜益炜_系统作品”整个文件夹放在不含特殊权限限制的本地目录中，例如桌面、D 盘或项目工作目录。路径中可以包含中文，但不建议放在需要管理员权限写入的位置。"),
        table(
            [
                [c("检查项"), c("说明")],
                [c("LLM助手源码"), c("应包含 app.py、ui2_server.py、agent_runtime.py、knowledge_base、rag、intent、tool_gateway、safety、integration、ui2 和 tests 等目录或文件。")],
                [c("DAC-3D主系统源码_精简版"), c("应包含 xxp_ui 主系统目录，以及离线检测相关代码和模型运行依赖入口。")],
                [c("DAC-3D离线检测示例数据"), c("应包含少量 pre_fusion_images 示例图片，用于离线推理演示。")],
                [c("启动脚本"), c("应包含启动助手后端、启动助手前端、启动 DAC-3D 离线检测主程序和运行自动化测试的批处理脚本。")],
            ],
            [5.0 * cm, 11.0 * cm],
        ),
        p("四、助手后端部署", "h1"),
        p("进入系统作品中的 LLM助手源码目录，安装后端依赖并构建知识库。若评阅环境已有依赖，可直接跳过重复安装。"),
        Preformatted(
            "cd LLM助手源码\npython -m pip install -e .\npython knowledge_base/build_kb.py\npython app.py",
            S["code"],
        ),
        p("也可以直接双击系统作品目录下的“启动助手后端.cmd”。后端默认提供 FastAPI 服务，建议使用 127.0.0.1:7890 作为本地复现端口。启动后可访问 /api/health 检查服务是否可用，访问 /api/runtime 查看模型、知识库、mock 模式和 DAC-3D 集成状态。"),
        p("如果需要使用真实大语言模型，请在本地 .env 中配置 provider、API 地址、模型名和 token/API Key。为保护个人隐私和接口安全，抽检材料未包含个人 token 或 API Key；复查时请使用评阅环境可用的 token 或 API Key。"),
        p("五、助手前端部署", "h1"),
        p("进入 LLM助手源码\\ui2 目录安装前端依赖并启动 Vite 服务。抽检包不携带 node_modules，因此首次运行需要执行 npm install。"),
        Preformatted(
            "cd LLM助手源码\\ui2\nnpm install\nnpm run dev",
            S["code"],
        ),
        p("也可以直接双击“启动助手前端.cmd”。该脚本会将 VITE_API_BASE_URL 设置为 http://127.0.0.1:7890，使前端请求指向本地后端。启动完成后，浏览器访问 http://127.0.0.1:5173 即可进入页面。"),
        p("六、DAC-3D 离线检测部署", "h1"),
        p("DAC-3D 侧用于验证离线模型运行能力，不要求复现真实在线硬件控制。部署时应确认默认权重和示例图片存在，并安装 PyQt5、OpenCV、PyTorch、Ultralytics 等依赖。"),
        table(
            [
                [c("项目"), c("部署说明")],
                [c("主系统入口"), c("可双击“启动DAC-3D离线检测主程序.cmd”，或进入 DAC-3D主系统源码_精简版\\xxp_ui 后执行 python main.py。")],
                [c("模型权重"), c("默认离线权重位于 deploy\\weights\\best.pt，类别包含 splash、chipping、scratch。")],
                [c("示例图片"), c("示例图位于 DAC-3D离线检测示例数据 下的 pre_fusion_images 目录，用于本地离线推理演示。")],
                [c("运行方式"), c("打开主系统界面后选择示例图片目录，再启动离线检测。CPU 环境可运行但耗时较长。")],
            ],
            [4.2 * cm, 11.8 * cm],
        ),
        p("七、助手与 DAC-3D 集成配置", "h1"),
        p("系统采用低侵入式集成方式。DAC-3D 主系统负责写出运行状态和检测结果，助手读取状态文件并生成状态回答或结果解释；操作请求先由助手生成结构化命令预览，再经过工具网关、策略校验、路径白名单和显式确认，最后由 DAC-3D 主系统轮询命令文件执行。"),
        p("本地复现时，如果未启动 DAC-3D 主系统，可使用 mock 模式验证助手功能。若需要联调主系统，应确认状态文件目录、命令文件目录和路径白名单配置一致，避免助手写入主系统无法读取的位置。"),
        p("八、部署后验证", "h1"),
        table(
            [
                [c("验证项"), c("检查方式")],
                [c("后端服务"), c("访问 http://127.0.0.1:7890/api/health，应返回服务正常信息。")],
                [c("运行状态"), c("访问 /api/runtime，应能看到模型提供方、知识库状态、mock 模式和 DAC-3D 集成状态。")],
                [c("知识库"), c("执行 knowledge_base/build_kb.py 后，知识库摘要应能显示文档数和分块数。")],
                [c("前端页面"), c("访问 http://127.0.0.1:5173，页面应能连接后端并完成一次问答。")],
                [c("自动化测试"), c("执行 python -m pytest tests -q，核心测试应通过。")],
                [c("离线检测"), c("DAC-3D 主系统应能加载 best.pt，并对示例图片完成一次离线推理。")],
            ],
            [4.2 * cm, 11.8 * cm],
        ),
        p("九、常见部署问题", "h1"),
        table(
            [
                [c("问题"), c("处理建议")],
                [c("前端无法请求后端"), c("确认后端已启动，端口为 7890；确认前端启动脚本中的 VITE_API_BASE_URL 指向 http://127.0.0.1:7890。")],
                [c("pip 或 npm 安装失败"), c("检查网络、镜像源和版本；也可使用学校或评阅环境已有镜像源重新安装。")],
                [c("真实 LLM 调用失败"), c("检查 .env 中 API 地址、模型名和 token/API Key；若暂时不可用，可切换 mock 模式验证本地功能链路。")],
                [c("DAC-3D 界面打不开"), c("优先检查 PyQt5、OpenCV、Torch、Ultralytics 等依赖是否安装；再检查模型权重和示例数据是否存在。")],
                [c("自动化测试失败"), c("先确认安装依赖和知识库构建完成，再查看失败用例是否与外部 LLM、端口占用或路径配置有关。")],
            ],
            [4.4 * cm, 11.6 * cm],
        ),
        p("十、抽检包体积控制说明", "h1"),
        p("为控制作品文件夹体积，抽检包未包含 node_modules、Python 虚拟环境、运行缓存、冗余模型权重、导出模型全集、大规模离线图片和论文过程文件。评阅人员复现时按本说明重新安装依赖即可。DAC-3D 侧仅保留最小离线运行资源，包括 deploy\\weights\\best.pt 和一组 pre_fusion_images 示例图。"),
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
