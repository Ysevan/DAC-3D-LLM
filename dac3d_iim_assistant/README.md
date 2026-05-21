# DAC-3D IIM Assistant

DAC-3D IIM Assistant 是一个面向 DAC-3D 检测流程的中文优先智能交互助手。它不是通用聊天机器人，而是围绕 DAC-3D 手册、运行状态和检测结果构建的任务型原型，用来降低操作员培训成本。

## 产品目标

本项目覆盖四类核心能力：

- 自然语言操作，将 `scan a 10mm x 10mm area` 或 `扫描 10mm x 10mm 区域` 转成结构化扫描预览
- 文档问答，解释参数、流程、状态和手册内容
- 结果解读，结合结构化检测结果和缺陷阈值文档回答缺陷是否严重
- 操作指导，针对反光、扫描不稳定等现场问题给出依据明确的建议

## 当前交付形态

- 中文优先的本地知识库与持久化向量索引
- `mock` 和 `anthropic` 两种 LLM provider 接口
- `mock` DAC-3D 适配层和稳定的 live adapter seam
- `FastAPI` 后端接口和可嵌入 DAC-3D 侧边栏的 `React + Vite` 前端
- `Gradio` 旧版界面保留为兼容回退
- 结构化命令预览、状态展示、结果解析和来源显示

## 仓库结构

- `app.py`: 应用启动、依赖组装、消息路由
- `config.py`: 运行配置、环境变量读取、路径管理
- `knowledge_base/`: 文档、知识库构建与向量持久化
- `rag/`: 检索、提示词、LLM 适配层
- `intent/`: 意图识别和命令生成
- `integration/`: DAC-3D 访问边界和结果解析
- `ui/`: Gradio 兼容界面和 FastAPI 接口层
- `frontend/`: React + Vite 聊天前端
- `tests/`: 冒烟测试与模块测试
- `docs/`: 部署和运行说明

## 本地运行

在仓库根目录执行：

```bash
python -m pip install -e ".[dev]"
python knowledge_base/build_kb.py
cd frontend && npm install && npm run build && cd ..
python app.py --message "这个参数是什么意思？"
python app.py
python -m pytest tests -q
```

默认 `python app.py` 会启动 FastAPI 后端并尝试提供构建后的 React 前端。调试前端时也可以单独执行：

```bash
cd frontend
npm run dev
```

如果当前环境没有构建前端，或者你仍想使用旧版界面，也可以显式执行：

```bash
python app.py --cli
python app.py --gradio
```

## 示例场景

下面的动图展示了 README 示例关键词在助手中的测试结果摘要，便于快速了解每类能力的实际返回效果。

<img src="docs/media/readme-demo.gif" alt="DAC-3D IIM Assistant README 示例场景测试动图" width="900">

- `scan a 10mm x 10mm area`
- `扫描 10mm x 10mm 区域`
- `这个参数是什么意思？`
- `当前检测状态是什么？`
- `这个缺陷严重吗？`
- `样品太反光了应该怎么办？`

## 版本号规范

当前首个版本标签为 `v0.1.0`。

本项目使用语义化版本号（Semantic Versioning）并统一采用 `vMAJOR.MINOR.PATCH` 形式：

- `MAJOR`：出现不兼容变更时递增，例如 API、配置项或运行方式发生破坏性调整
- `MINOR`：向后兼容的新功能递增，例如新增意图类型、Web UI 能力或新的集成接口
- `PATCH`：向后兼容的问题修复递增，例如流式显示修复、检索精度修复、样式或稳定性修补

建议发布规则如下：

- 新的里程碑可用 `v0.2.0`、`v0.3.0` 持续推进原型能力
- 小范围修复使用 `v0.1.1`、`v0.1.2` 这类补丁版本
- 在 `1.0.0` 之前，功能仍可快速演进，但每次对外发布都应打 tag 并记录变更范围

## 主要限制

- 本阶段不接真实 DAC-3D 设备或 SDK，只提供 mock 运行时和清晰的适配接口
- 若未安装 `fastapi`、`uvicorn`、`react` 前端依赖或未构建 `frontend/dist`，默认 Web 入口无法完整工作
- 若未安装 `anthropic`、`chromadb`、`gradio`、`sentence-transformers`，系统会回退到离线可运行路径
- 默认 `DAC3D_EMBEDDING_DOWNLOAD_ALLOWED=false`，若本机未缓存 embedding 模型，会快速回退到 hashing embedding；需要联网下载模型时再显式开启
- 默认知识库只包含仓库内样例文档，真实项目需要补充 DAC-3D 官方手册、FAQ 和检测规则文档
- 本地 embedding 默认兼容中英双语，若语料完全中文，可通过配置切换为中文专用模型
