# DAC-3D-LLM

本仓库包含 DAC-3D 主检测系统、LLM 智能检测助手、知识库文档、模型部署资产和毕业设计相关材料。项目目标是把 DAC-3D 检测流程从传统软件操作升级为“自然语言交互 + 文档问答 + 检测状态读取 + 结果解读”的本地可演示系统。

## 项目定位

DAC-3D-LLM 面向智能检测场景，不是通用聊天机器人。系统应尽量基于以下三类信息回答和执行：

- DAC-3D 官方/项目文档，例如操作手册、参数说明、FAQ、缺陷判定规则。
- DAC-3D 主系统运行状态，例如当前是否检测、检测进度、最新结果、历史样品结果。
- DAC-3D 检测数据和模型结果，例如离线图片检测输出、缺陷类型、置信度、位置和严重程度。

当前演示支持两种形态：

- LLM 智能检测助手原型：可独立启动 Web 页面，用于问答、命令预览、知识库构建和结果解释。
- OpenAI Agents SDK 多 Agent 形态：统一入口 Coordinator 先判断问题，再 handoff 到 DAC-3D QA、DAC-3D Control、DAC-3D Result、Machine、Memory 或 Safety Agent。
- 嵌入式 DAC-3D 演示：从 DAC-3D 主系统按钮跳转到 Web 助手，读取主系统状态和结果数据。
- 工业设备信息管理 AI Agent 原型：在同一 FastAPI/React 技术栈下演示设备状态、历史数据、报警记录、维护文档、异常模式分析和工具调用轨迹。

## 核心能力

### 1. 自然语言操作

用户可以用自然语言描述检测需求，系统将其转换为结构化命令或 YAML 预览。

示例：

```text
我想扫描一个 25mm x 25mm 的楔形滤光片，步长 10 微米
选择 pre_fusion_images 下的图片进行离线检测
停止当前检测
查询当前系统状态
```

### 2. 智能问答

助手基于本地知识库回答 DAC-3D 参数、流程、设计原则和常见问题。

示例：

```text
DAC-3D 的设计原则是什么？
这个参数是什么意思？
离线检测和在线扫描有什么区别？
为什么需要焦前、焦面、焦后三张图？
```

### 3. 检测结果解读

助手可以读取最新检测结果或指定样品结果，并用结构化结果进行解释。

示例：

```text
当前检测结果怎么样？
第三个样品检测结果怎么样？
前面几个样品的缺陷情况如何？
这个缺陷严重吗？
```

### 4. 操作建议

助手可以针对现场问题给出操作建议。

示例：

```text
样品表面反光很强怎么办？
检测速度太慢怎么办？
图像没有缺陷标记怎么办？
离线检测为什么不能停止？
```

### 5. 工业设备信息管理 Agent

新增 `dac3d_iim_assistant/machine_agent/` 原型模块，内置 mock 工业设备数据和文档，可演示：

- 查询当前设备状态、温度、压力、转速、电流、产量和未关闭报警。
- 查询某段时间历史数据和报警记录。
- 总结上个月或最近三个月运行情况。
- 分析报警次数、最高频报警类型、高发时段、正常/异常数据差异。
- 检索设备操作手册、故障代码说明、维护保养规范和报警处理流程。

推荐演示问题：

```text
现在设备状态怎么样？
上个月运行情况怎么样？
最近有哪些异常？
为什么最近温度报警变多了？
E102 错误代码是什么意思？
帮我总结一下这台机器最近三个月的问题。
```

## DAC-3D 主系统改造内容

本项目不是只做了一个独立聊天页面，而是在原 DAC-3D 主系统基础上增加了和 LLM 助手联动的能力。主要修改集中在 `福特科/xxp_ui/`。

### 与 `福特科` 项目原始版本的差异

本说明以 `DAC-3D-LLM\福特科.zip`（2026年5月18日企业发送离线版本） 作为原始 DAC-3D 系统版本，对比当前 `DAC-3D-LLM\福特科\` 目录。

### 代码更新清单

#### `福特科/xxp_ui/window/ui.py`

这是 DAC-3D 主界面的核心改造文件。相比 `福特科.zip` 原始版本，主要增加了智能助手入口、状态桥、命令桥和离线检测控制。

主要更新：

- 新增 `json`、`os`、`subprocess`、`sys`、`webbrowser`、`Path` 等导入，用于写状态文件、启动助手服务、打开浏览器。
- 新增 `_startPos`、`_endPos` 初始化，修复窗口拖动时可能出现的 `_startPos` 属性不存在导致闪退的问题。
- 新增 `assistant_status_file`、`assistant_command_file`、`assistant_command_ack_file`，用于主系统和 LLM 助手之间交换状态与命令。
- 新增 `writeAssistantRuntimeStatus(...)`，把 DAC-3D 当前运行状态写成 JSON，供助手实时读取。
- 新增 `writeAssistantCommandAck(...)`，把主系统对助手命令的接收、拒绝、执行状态写回给助手。
- 新增 `buildAssistantLatestResult(...)`，把最新检测样品结果整理成助手可解释的结构化数据。
- 新增 `pollAssistantCommandFile(...)`，定时读取助手生成的命令文件。
- 支持识别并处理 `start_online_scan`、`start_offline_detection`、`stop_detection`、`query_status`、`get_latest_result`、`validate_offline_folder` 等命令。
- 新增 `describeAssistantCommand(...)`，把结构化命令转成人可读描述。
- 新增 `initAssistantWebButton(...)`，在 DAC-3D 主界面添加智能助手入口按钮。
- 新增 `openAssistantWeb(...)`，自动启动 `dac3d_iim_assistant/app.py` 并打开 `http://127.0.0.1:7890` 前端工作台。
- 新增离线检测控件逻辑，包括 `initOfflineControls(...)`、`chooseOfflineSourceDir(...)`、`setOfflineControlsVisible(...)`。
- 在检测过程中写入 `latest_result` 和 `result_history`，支持助手回答“当前检测结果”“第三个样品结果”“前面几个样品结果”等问题。
- 在启动在线扫描、启动离线检测、停止检测时写入实时状态，支持助手回答“当前系统在做什么”。

#### `福特科/xxp_ui/image_processor_22.py`

这是当前主要检测处理器。相比原始版本，主要增强了离线检测停止、GPU 推理适配、模型路径加载和结果图保存。

主要更新：

- 新增 `offline_stop_requested` 标志位，用于离线检测中断。
- 新增 `request_offline_stop(...)`，收到停止命令后清空待处理状态并向 UI 队列发送结束信号。
- 离线检测提交图片时会检查 `offline_stop_requested`，停止后不再继续提交后续图片。
- 图像处理循环中收到停止请求后会丢弃后续离线图像处理结果，避免停止后仍继续输出结果。
- 支持从 UI 队列接收 `stop_offline_detection` 消息，并调用 `request_offline_stop(...)`。
- 模型加载时根据 `torch.cuda.is_available()` 自动选择 `cuda:0` 或 `cpu`。
- 修正模型路径加载方式，将绝对路径转换为相对项目路径，降低 SAHI/YOLO 在 Windows 路径下加载失败的概率。
- YOLO 模型加载后尝试 `yolo_model.to(device)`，尽量使用 GPU 推理。
- `AutoDetectionModel.from_pretrained(...)` 增加 `device=device`，并保留 `TypeError` 回退逻辑，兼容不同 SAHI 版本。
- 保留并增强缺陷框、缺陷圆心、标注结果图输出逻辑，确保测试结果图片能显示标记。

#### `福特科/xxp_ui/image_processor_2.py`

这是旧版/备用图像处理器。相比原始版本，主要同步了模型加载和 GPU 适配修复。

主要更新：

- 新增 `model_path_for_loader`，避免部分模型加载器无法处理 Windows 绝对路径。
- 根据 CUDA 可用性自动选择 `cuda:0` 或 `cpu`。
- YOLO 模型加载后尝试移动到目标设备。
- SAHI `AutoDetectionModel` 增加 `device` 参数，并保留旧版本兼容回退。

#### `福特科/xxp_ui/Algorithm/Regis_Fusion/Regis_Fusion_three2.py`

这是图像配准/融合相关脚本。相比原始版本，主要修复硬编码路径。

主要更新：

- 将原始固定路径 `F:\福特科\xxp_ui\Algorithm\result` 改为基于当前脚本位置计算的项目相对路径。
- 将原始固定路径 `D:\zycgit\ZDevelop_Confocal\xxp_ui\results\fusion_result` 改为项目内 `results/fusion_result`。
- 这样项目换电脑或换目录后，不需要手动修改代码中的绝对路径。

#### `福特科/xxp_ui/window/login.py`

这是登录窗口逻辑。相比原始版本，主要修复登录后主窗口对象生命周期问题。

主要更新：

- 将局部变量 `window = MyWindow(...)` 改为 `self.main_window = MyWindow(...)`。
- 这样登录窗口持有主窗口引用，避免主窗口对象被 Python 垃圾回收导致登录后闪退或窗口异常关闭。

#### `福特科/xxp_ui/window/assistant_panel.py`

这是当前版本新增文件，原始 `福特科.zip` 中不存在。它是一个嵌入式助手 Dock/桥接层原型，用于后续把助手直接嵌入 DAC-3D 主系统。

主要能力：

- 新增 `DAC3DMainWindowBridge`，把助手命令转成主系统队列动作。
- 支持 `start_online_scan(...)`、`start_offline_detection(...)`、`stop_detection(...)`。
- 支持 `query_current_status(...)`，读取当前运行状态、样品数量、离线模式、离线目录。
- 支持 `validate_offline_folder(...)`，检查离线图片目录是否存在并包含图片。
- 支持 `get_latest_result_summary(...)`，读取最近一次检测结果目录。
- 新增 `AssistantWorker`，避免助手调用阻塞 PyQt UI 线程。
- 新增 `AssistantDock`，提供可嵌入主系统的聊天式助手面板原型。

当前主系统实际演示入口仍以浏览器跳转到 Web 助手为主，`assistant_panel.py` 保留为后续深度嵌入式侧边栏方案。

#### `福特科/xxp_ui/.gitignore`

这是当前版本新增文件，原始 `福特科.zip` 中不存在。它用于控制哪些 DAC-3D 主系统文件进入 Git。

主要规则：

- 忽略 `.idea/`、`.vs/`、`__pycache__/`、`*.pyc` 等本地 IDE 和缓存文件。
- 忽略 `runtime/`、`results/`、`*.log` 等运行时输出。
- 忽略临时图像 `frame.jpg`、`bottom_pos.txt`。
- 对 `deploy/weights/`、`deploy/model/`、`weights/` 等模型目录做特殊放行，并通过根目录 `.gitattributes` 使用 Git LFS 管理。


### 1. 增加智能助手入口

在 DAC-3D 主系统界面中增加了智能助手按钮。用户登录 DAC-3D 后，可以从主系统直接打开 Web 版 LLM 助手。

当前默认跳转地址：

```text
http://127.0.0.1:7890
```

备用 Web UI 仍可访问：

```text
http://127.0.0.1:7860
```

相关文件：

```text
福特科/xxp_ui/window/ui.py
福特科/xxp_ui/window/assistant_panel.py
```

### 2. 增加 DAC-3D 与助手之间的本地状态桥

主系统会把当前运行状态、检测进度、最新检测结果、历史样品结果写入本地状态文件。LLM 助手读取这些文件后，可以回答实时状态类问题。

可回答的问题包括：

```text
当前系统在做什么？
现在检测到第几个样品了？
当前检测结果怎么样？
第三个样品检测结果怎么样？
前面几个样品结果如何？
```

状态桥主要能力：

- 记录当前检测状态，例如 idle、running、stopped、finished、error。
- 记录当前任务类型，例如离线检测、在线扫描。
- 记录当前进度、样品编号、已完成数量。
- 记录最新一次检测结果。
- 记录历史样品结果，支持按第几个样品查询。

### 3. 增加助手命令读取能力

LLM 助手不仅能回答问题，也能生成结构化命令。主系统可以读取这些命令并执行对应动作。

优先支持的命令类型：

```text
start_offline_detection
start_online_scan
stop_detection
query_status
get_latest_result
validate_offline_folder
```

示例自然语言：

```text
扫描 10mm x 10mm 区域
选择 pre_fusion_images 下的图片进行离线检测
停止当前检测
查询当前检测状态
读取最新检测结果
```

### 4. 修复和增强离线检测流程

围绕 DAC-3D 离线检测演示，主系统做了以下增强：

- `main.py` 统一加载当前主要图像处理器。
- `image_processor_22.py` 作为当前主要检测处理逻辑。
- 增加对离线检测停止的支持，避免离线流程无法中断。
- 修复模型加载路径，使部署权重可以按实际路径加载。
- 检测过程中写入状态，便于 LLM 助手实时查询。
- 检测完成后写入最新结果和历史样品结果，便于 LLM 助手解释。

### 5. 保留 GPU / Apple 加速检测能力

系统会检测当前 PyTorch 是否支持 CUDA 或 Apple MPS。如果环境中安装的是 CUDA 版 PyTorch，并且显卡驱动可用，检测模型优先走 CUDA；在 Apple Silicon Mac 上，如果 PyTorch MPS 可用，则走 MPS。如果 macOS/PyTorch 组合导致 MPS 不可用，离线检测会尝试使用现有 ONNX 模型和 ONNX Runtime `CoreMLExecutionProvider`，再不行才回退 CPU。

可以通过环境变量指定推理设备：

```powershell
# 可选值：auto、cuda、mps、coreml、mlx、cpu
set DAC3D_INFERENCE_DEVICE=auto
```

`mlx` 会作为 Apple 加速请求处理。当前检测链路没有原生 MLX 模型后端，因此会优先解析到 PyTorch MPS；如果 MPS 不可用且 ONNX Runtime 暴露 CoreML EP，则使用 ONNX/CoreML。

检查加速设备是否可用：

```powershell
python CUDA.py
```

输出中的 `离线推理设备` 会显示实际使用 `cuda:0`、`mps`、`coreml` 或 `cpu`。

注意：受限沙盒或 CI 环境可能会让 PyTorch MPS 误报不可用。请以普通终端或 PyQt 主程序环境下的 `python CUDA.py` 输出为准。

## 仓库结构

```text
DAC-3D-LLM/
├─ dac3d_iim_assistant/          # LLM 智能检测助手
├─ 福特科/xxp_ui/                # DAC-3D 主检测系统
├─ 福特科/pre_fusion_images/     # 离线演示图片数据
├─ 07.毕业论文初稿.docx          # 毕业论文初稿
├─ .gitattributes                # Git LFS 规则
└─ README.md                     # 本说明文档
```

### 智能助手目录

```text
dac3d_iim_assistant/
├─ app.py                        # 助手入口和消息路由
├─ config.py                     # 配置、环境变量、路径
├─ knowledge_base/               # 文档、向量库、知识库构建
├─ rag/                          # 检索、提示词、LLM Provider
├─ intent/                       # 意图识别和结构化命令生成
├─ integration/                  # DAC-3D 状态/结果适配层
├─ ui/                           # Gradio/FastAPI 界面接口
├─ ui2/                          # Web UI 前端工程
├─ docs/                         # 部署指南和说明文档
└─ tests/                        # 自动化测试
```

### DAC-3D 主系统目录

```text
福特科/xxp_ui/
├─ main.py                       # DAC-3D 主系统入口
├─ window/                       # PyQt5 页面、登录页、主界面
├─ image_processor_22.py         # 当前主要图像检测处理逻辑
├─ deploy/                       # 部署模型、推理脚本和权重
├─ weights/                      # 检测权重
├─ ui/                           # Qt UI 文件和图标资源
├─ Algorithm/                    # 图像配准和融合算法
└─ ultralytics_gcy/              # 本项目使用的 YOLO/Ultralytics 代码
```

## Git LFS 说明

本项目包含模型权重和大量演示图片，普通 Git 不适合直接保存这些大文件。因此以下目录通过 Git LFS 管理：

```text
福特科/pre_fusion_images/
福特科/xxp_ui/deploy/exported_models/
福特科/xxp_ui/deploy/model/
福特科/xxp_ui/deploy/weights/
福特科/xxp_ui/weights/
```

首次克隆仓库后，请安装并拉取 LFS 文件：

```powershell
git lfs install
git lfs pull
```

如果只看到很小的指针文件，而不是实际模型或图片，说明 LFS 文件还没有拉取完成。

查看 LFS 文件：

```powershell
git lfs ls-files
```

上传 LFS 文件：

```powershell
git lfs push origin main --all
```

## 环境要求

推荐环境：

- Windows 10/11
- Python 3.10+
- Git
- Git LFS
- Node.js 18+，仅当前端单独开发时需要
- 可选 NVIDIA GPU 和 CUDA 版 PyTorch，用于加速检测

DAC-3D 主系统依赖 PyQt5、OpenCV、PyTorch、Ultralytics 等库。智能助手依赖 FastAPI/Gradio、Chroma、sentence-transformers、LLM Provider SDK 等库。

## 快速开始：智能助手

进入助手目录：

```powershell
cd C:\Users\xecat\DAC-3D-LLM\dac3d_iim_assistant
```

安装依赖：

```powershell
python -m pip install -e ".[dev]"
```

构建知识库：

```powershell
python knowledge_base\build_kb.py
```

启动 Web 助手：

```powershell
python app.py
```

默认 Web、CLI 和 `/api/chat` 已使用统一 Agent 入口：用户消息先交给 LLM，LLM 再通过 OpenAI Agents SDK 选择 `dac3d_*` 或 `machine_*` 工具。需要配置 `OPENAI_API_KEY`、`DAC3D_AGENT_API_KEY`，或继承 OpenAI-compatible 的 `DAC3D_LLM_API_KEY`：

```powershell
$env:OPENAI_API_KEY="你的 OpenAI API Key"
python app.py --message "当前检测状态是什么？"
python app.py --agent-web  # 兼容旧脚本；Web 默认已是 Agent 入口
```

可选通过环境变量或命令行指定 Agent 模型：

```powershell
$env:DAC3D_AGENT_MODEL_NAME="gpt-5.4-mini"
python app.py --agent --agent-model gpt-5.4-mini
```

第三方 OpenAI-compatible 模型接入方式：

```powershell
$env:DAC3D_AGENT_MODEL_NAME="vendor/dac3d-control-model"
$env:DAC3D_AGENT_API_KEY="第三方模型 API Key"
$env:DAC3D_AGENT_API_BASE_URL="https://llm.example.com/v1"
$env:DAC3D_AGENT_API_TYPE="chat_completions"
python app.py --agent --message "确认立即开始在线扫描"
```

如果 `DAC3D_LLM_PROVIDER=openai_compatible` 已经配置好普通 DAC-3D-LLM 对话模型，Agent 会默认继承 `DAC3D_LLM_MODEL_NAME`、`DAC3D_LLM_API_KEY` 和 `DAC3D_LLM_API_BASE_URL`，并自动使用 `chat_completions`。需要单独给 Agent 换模型时，再设置 `DAC3D_AGENT_*`。

也可以通过命令行覆盖：

```powershell
python app.py --agent --agent-model vendor/dac3d-control-model --agent-api-base-url https://llm.example.com/v1 --agent-api-type chat_completions --message "当前检测状态是什么？"
```

如果已在 `dac3d_iim_assistant` 中执行 `python -m pip install -e ".[dev]"`，还可以使用 Agent-first 命令：

```powershell
dac3d-agent --describe
dac3d-agent --list-tools
dac3d-agent --message "当前检测状态是什么？"
dac3d-agent --preview-command "start online scan"
dac3d-agent --execute-command "start online scan" --confirmed
```

FastAPI/React、Gradio 和 CLI 聊天入口默认使用统一 Agent runtime；`--agent-web` 仅保留给旧脚本。当前 runtime 是多 Agent 系统：`DAC-3D Multi-Agent Coordinator` 是唯一入口，先由 LLM 判断问题类型，再 handoff 到 `DAC-3D QA Agent`、`DAC-3D Control Agent`、`DAC-3D Result Agent`、`Machine Agent`、`Memory Agent`、`Skill Agent` 或 `Safety Agent`。专家 Agent 分别持有自己的工具组，覆盖 DAC-3D 文档问答、状态、结果、命令预览/执行、JSON 历史记忆检索、技能选择、执行前安全审查，以及设备状态、历史、报警、文档、异常归因。`dac3d_preview_command` 只生成结构化命令预览；`dac3d_safety_review` 只审查命令风险和确认要求，不执行；`dac3d_execute_command` 会在用户明确确认、运行时不忙碌、离线目录校验通过后，把控制命令提交到 embedded bridge、file command bridge 或 mock runtime。真实 DAC-3D 主系统运行时，`DAC3D_ENDPOINT` 指向 `dac3d_runtime_status.json`，`DAC3D_COMMAND_PATH` 指向 `dac3d_assistant_command.json`，主系统会轮询该命令文件并触发在线扫描、离线检测或停止检测。

DAC 控制能力现在经过 `DAC3DToolGateway`、`PolicyEngine`、`SafetyGuard` 和命令生命周期状态机。网关声明 `read_dac_status`、`read_latest_result`、`list_allowed_dirs`、`preview_command`、`validate_command`、`submit_command`、`cancel_pending_command`、`read_command_history` 等 MCP-style 工具元数据，并为每个命令预览生成 `preview_id`、确认要求、风险等级、schema/path/runtime 校验结果和命令历史。每次工具调用都会先经过 `PolicyEngine`，未知工具、schema 不匹配、高风险未确认、越权路径、绕过确认文本、直接写 `command.json`、未批准 memory/skill 写入等情况默认 fail-closed。每个可提交 preview 都会进入 `preview_created -> validation_passed -> awaiting_confirmation -> confirmed -> submitted` 生命周期，确认 token 绑定 `preview_hash`，默认 300 秒过期，只能消费一次；修改 preview、过期确认、重复提交或 token 不匹配都会被拒绝并写入命令历史。工具描述现在包含 `readOnlyHint`、`destructiveHint`、`idempotentHint`、`openWorldHint` 和 `annotations`，可供 Web UI 或未来 MCP client 做风险提示；这些字段只是提示，不替代安全 enforcement，真正的 schema 校验、路径白名单、prompt-injection 拦截和确认 token 校验仍由 `PolicyEngine`、`SafetyGuard` 与 `submit_command` 执行。执行类命令不会直接写 bridge 文件；`submit_command` 只会提交当前会话中已生成且通过确认 token 校验的 pending preview。Web UI 遇到可提交的 pending preview 时会显示“批准执行”按钮，点击后调用 `/api/commands/approve` 继续当前会话的 Tool Gateway 提交流程，不需要用户再手写“确认执行”。路径敏感命令会走 `PathPolicy`：输入目录会 canonicalize 并阻止 `..`、symlink escape、UNC/network path、allowlist 外路径和 secret 文件；命令桥写入路径还会单独校验 command output allowlist，避免覆盖配置文件。默认读取目录为 `.tmp/`、`福特科/pre_fusion_images/` 和 `福特科/xxp_ui/runtime/`，可用 `DAC3D_ALLOWED_DIRS` 覆盖；命令输出目录默认来自 file bridge 所在目录和 `.tmp/`，生产环境可用 `DAC3D_COMMAND_OUTPUT_DIR` 收紧。可用 `DAC3D_COMMAND_CONFIRMATION_TTL_SECONDS` 调整确认有效期。

技能系统位于 `dac3d_iim_assistant/skills/`，每个 skill 是一个可版本化目录，包含 `SKILL.md` 以及可选的 schema、examples、模板或参考资料。`SkillRegistry` 会先发现和选择技能，只在当前任务需要时把匹配的 `SKILL.md` 摘要注入上下文，避免把所有流程知识塞进 prompt。当前内置技能包括 `dac-command-preview`、`dac-state-check`、`dac-offline-inspection`、`dac-result-explain`、`dac-fault-recovery`、`dac-memory-maintenance`、`dac-safety-approval` 和 `dac-context-engineering`。`SkillPatchStore` 会把运行经验沉淀成待审核 skill patch，写入 `.tmp/conversation_memory/skill_patches.json`；`dac_skill_propose_patch`、`GET/POST /api/skills/patches` 和 Web 设置面板的“技能补丁审核”只创建、批准或拒绝补丁记录，不会静默修改生产 `SKILL.md`。

统一 Agent 入口现在带有 `ContextBuilder` 上下文工程层。每轮进入 LLM 前，系统会按任务选择并压缩 runtime status、安全策略、相关 skill 和多层记忆；纯指导类问题不会无条件塞入 idle 状态，控制、状态、结果、设备异常或运行态非 idle 时才注入 DAC 运行状态。`parsed_result.context_engineering` 会记录本轮实际使用的 context sections、排除的大块上下文和字符限制，便于调试上下文污染。可用 `DAC3D_CONTEXT_SKILL_LIMIT` 控制按需加载的 skill 数量，用 `DAC3D_CONTEXT_CHAR_LIMIT` 控制单轮上下文包规模。

Context Builder 现在还会给每段上下文打上 `trust_level` 和权限边界：`approved_policy` 可作为系统执行边界，`tool_output` 只能作为 observation，`retrieved_doc` 只能作为证据数据，`approved_memory` 只能用于偏好和背景理解，不能授予工具权限或跳过确认。RAG 文档片段会显式标记 `trust=retrieved_doc`、`can_instruct_agent=false`、`can_influence_tools=false`；如果文档、工具输出或记忆里出现“忽略系统规则”“直接写 command.json”“call submit_command now”等注入文本，会进入 `parsed_result.context_engineering.trust.injection_signals`，同时真正的副作用仍由 `PolicyEngine`、`ToolGateway` 和 confirmation token fail-closed 拦截。

Agent 工作台现在包含本地 Goal Tracker。目标以 JSON 写入 `.tmp/conversation_memory/agent_goals.json`，支持创建目标、追加进度和标记完成；聊天中出现“目标：...”或 `goal:` 这类明确目标信号时，会自动把当前 session 的目标沉淀为可见条目。Web 设置面板的“Agent 目标”会调用 `GET /api/goals`、`POST /api/goals`、`POST /api/goals/{goal_id}/progress` 和 `POST /api/goals/{goal_id}/complete`，用于把长任务目标和每轮 Agent trace、Context Tree、Memory OS 一起纳入工作区视图。

Trace/Eval 闭环位于 `trace_eval/` 和 `evals/cases/`。Agent 每轮会向 `agent_traces.jsonl` 写入 append-only trace，记录 session、意图、工具调用、命令预览 ID、风险决策、批准状态、最终回复和 memory patch ID，并对 token/key/secret/password 类字段做基础脱敏。`POST /api/evals/run` 会运行本地确定性回归集，当前覆盖状态查询、命令预览、批准提交、prompt injection 拦截、记忆写入候选，以及 15 个 security red-team 变体；Web 设置面板也提供“运行评测”按钮，便于把 trace -> eval -> regression report 纳入日常演示。`POST /api/evals/drafts` 可以把最近或指定 trace 转成待审核 eval 草稿，保存到 `evals/drafts/`，不会自动加入 `evals/cases/`；需要人工确认后再把草稿提升为正式回归用例。`POST /api/evals/codex-handoff` 或 `python -m trace_eval.generate_codex_handoff` 会运行本地 eval、读取失败用例和相关 trace，并生成 `docs/generated/codex_handoff_next.md`，用于把失败评测、推荐修改文件和验证命令交给下一轮 Codex 继续处理。

会话历史默认使用本地 JSON + Markdown 记忆，写入 `dac3d_iim_assistant/.tmp/conversation_memory/`。结构参考 Hermes 的分层记忆思路：`MEMORY.md` 保存系统长期核心事实，`USER.md` 保存用户长期偏好，`sessions/*.json` 保存每个 session 的原始对话，`index.json` 支持跨 session 检索，`knowledge_notes/*.md` 与 `knowledge_index.json` 保存主题化经验笔记，`procedures/*.md` 与 `procedure_index.json` 保存已审核流程记忆。Agent 每轮进入 LLM 前会组合当前页面短期历史、核心 Markdown 记忆、当前 session 最近 JSON 对话、session 摘要、主题知识笔记命中、流程记忆命中和长期 JSON 索引检索结果。新增 `LocalMemoryProvider` 把这套能力包装成 Memory OS 接口：每轮会写入 `traces.jsonl`，可把“以后/记住/纠正”等长期偏好生成 `memory_patches.json` 待审核补丁，只有批准后才写入 `MEMORY.md`、`USER.md`、主题笔记或流程记忆。`procedure_memory` patch 批准后会写入带 provenance frontmatter 的 `procedures/*.md`，记录 trace、session、patch 和 human review 信息；`GET/POST /api/memory/procedures` 可读取或提出流程记忆候选。Web 设置面板的“记忆补丁审核”可以读取、批准或拒绝这些候选，后端对应 `GET /api/memory/patches`、`POST /api/memory/patches/{patch_id}/approve` 和 `POST /api/memory/patches/{patch_id}/reject`。记忆只用于理解“刚才那个”“继续上一个”等指代、偏好和项目约定；DAC-3D 实时状态、检测结果和执行命令仍以工具返回为准。可通过 `DAC3D_MEMORY_ENABLED=false` 关闭，或用 `DAC3D_MEMORY_RECENT_TURNS`、`DAC3D_MEMORY_SEARCH_LIMIT`、`DAC3D_MEMORY_CORE_CHAR_LIMIT`、`DAC3D_MEMORY_USER_CHAR_LIMIT`、`DAC3D_MEMORY_NOTE_CHAR_LIMIT` 调整上下文规模。

Memory OS 现在还有独立的污染防护层：`conversation_memory_update`、`conversation_knowledge_write` 和 `conversation_procedure_write` 只生成待审核 patch，不会静默写入长期记忆；每条 patch 都带 `memory_status`、`trust_level`、`provenance`、policy decision 和 conflict 信息。`policy_memory` 需要特权审批，`tool_memory` 不能由 tool output 直接创建，`procedure_memory` 必须有 evidence trace，疑似 secret、跳过确认、自动执行等内容会被拒绝。检索命中会带 `source/trust_level/status`，`rejected`、`deleted`、`superseded` 记忆不会进入上下文。

默认访问：

```text
http://127.0.0.1:7890
```

备用 Web UI 可访问：

```text
http://127.0.0.1:7860
```

单条命令测试：

```powershell
python app.py --message "当前检测状态是什么？"
```

运行测试：

```powershell
python -m pytest tests -q
```

### 助手使用方式

启动后在浏览器打开：

```text
http://127.0.0.1:7860
```

典型使用顺序：

1. 先启动 DAC-3D 主系统。
2. 再启动 `dac3d_iim_assistant` 助手。
3. 在 DAC-3D 主系统里点击智能助手按钮，或直接浏览器访问助手地址。
4. 在助手中输入自然语言问题或操作指令。
5. 如果是查询/解读类问题，助手会读取知识库和 DAC-3D 当前状态。
6. 如果是操作类问题，助手会生成结构化命令预览，必要时等待确认。

推荐演示问题：

```text
当前系统在做什么？
扫描 10mm x 10mm 区域
选择 pre_fusion_images 下的图片进行离线检测
第三个样品检测结果怎么样？
DAC-3D 的设计原则是什么？
样品表面反光很强怎么办？
```

## 快速开始：DAC-3D 主系统

进入主系统目录：

```powershell
cd C:\Users\xecat\DAC-3D-LLM\福特科\xxp_ui
```

启动主系统：

```powershell
python main.py
```

如果提示缺少 PyQt5：

```powershell
python -m pip install PyQt5
```

如果需要 GPU / Apple 加速检测，请安装与你显卡、CUDA 或 macOS/MPS 环境匹配的 PyTorch。安装完成后检查：

```powershell
python CUDA.py
```

输出中的 `离线推理设备` 才是当前 Python 环境实际使用的检测设备。

## 环境配置指南

### 1. Git 和 Git LFS

本项目包含大模型和图片数据，必须安装 Git LFS。

检查 Git：

```powershell
git --version
```

检查 Git LFS：

```powershell
git lfs version
```

首次拉取项目后执行：

```powershell
git lfs install
git lfs pull
```

如果 LFS 上传或下载不稳定，可以降低并发：

```powershell
git config lfs.concurrenttransfers 1
git config lfs.activitytimeout 300
git config lfs.dialtimeout 60
git config lfs.tlstimeout 60
git config http.version HTTP/1.1
```

### 2. Python 环境

推荐使用 Python 3.10 或 3.11。检查版本：

```powershell
python --version
```

建议为项目创建独立虚拟环境：

```powershell
cd C:\Users\xecat\DAC-3D-LLM
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3. DAC-3D 主系统依赖

进入主系统目录：

```powershell
cd C:\Users\xecat\DAC-3D-LLM\福特科\xxp_ui
```

基础依赖示例：

```powershell
python -m pip install PyQt5 opencv-python numpy pillow matplotlib pandas pyyaml
```

安装 CPU 版 PyTorch：

```powershell
python -m pip install torch torchvision torchaudio
```

如果要使用 NVIDIA GPU，请根据本机 CUDA 版本安装对应 CUDA 版 PyTorch；如果要使用 Apple Silicon 加速，请确认 PyTorch MPS 可用。安装后必须确认：

```powershell
python CUDA.py
```

### 4. LLM 助手依赖

进入助手目录：

```powershell
cd C:\Users\xecat\DAC-3D-LLM\dac3d_iim_assistant
```

安装助手依赖：

```powershell
python -m pip install -e ".[dev]"
```

如果缺少某些包，可按需补装：

```powershell
python -m pip install fastapi uvicorn gradio chromadb sentence-transformers python-dotenv httpx
```

### 5. 前端环境

如果需要修改或重新构建 Web UI，需要安装 Node.js 18+。

检查 Node：

```powershell
node -v
npm -v
```

构建前端：

```powershell
cd C:\Users\xecat\DAC-3D-LLM\dac3d_iim_assistant\ui2
npm install
npm run build
```

### 6. LLM API 配置

在 `dac3d_iim_assistant` 目录创建本地 `.env` 文件。在这里我没有把真实 API Key 提交到 GitHub，需要添加你自己的 API Key。

OpenAI-compatible 示例：

```env
DAC3D_LLM_PROVIDER=openai_compatible
DAC3D_LLM_API_BASE_URL=https://your-provider.example/v1
DAC3D_LLM_API_KEY=your_api_key
DAC3D_LLM_MODEL_NAME=your_model_name
```

Anthropic-compatible 示例：

```env
DAC3D_LLM_PROVIDER=anthropic
DAC3D_LLM_API_BASE_URL=https://your-anthropic-compatible-endpoint
DAC3D_LLM_API_KEY=your_api_key
DAC3D_LLM_MODEL_NAME=claude-3-5-haiku-latest
```

离线演示或没有 API 时，可以使用 mock：

```env
DAC3D_LLM_PROVIDER=mock
```

### 7. DAC-3D 状态桥配置

如果要让助手读取 DAC-3D 主系统真实运行状态，需要关闭 mock，并指定本地状态桥文件。

示例：

```env
DAC3D_MOCK_MODE=false
DAC3D_ENDPOINT=file:///C:/Users/xecat/DAC-3D-LLM/dac3d_iim_assistant/.tmp/dac3d_runtime_status.json
```

主系统运行后会写入状态文件，助手即可回答当前状态、最新结果和历史样品结果。

## 主系统与助手联动

当前联动方式以本地文件桥为主：

- DAC-3D 主系统写入运行状态、进度、最新结果和历史样品结果。
- LLM 助手读取这些状态文件，回答“当前系统在做什么”“第三个样品怎么样”等问题。
- 助手生成结构化命令后，主系统可以读取命令文件并执行离线检测、在线扫描、停止检测等动作。

常见联动命令包括：

```text
start_offline_detection
start_online_scan
stop_detection
query_status
get_latest_result
validate_offline_folder
```

## LLM 配置

助手支持 mock 和真实 LLM Provider。开发演示可使用 mock，真实问答可配置 OpenAI-compatible 或 Anthropic-compatible 服务。

不要把真实 API Key 写入 Git。建议写入本地 `.env`：

```env
DAC3D_LLM_PROVIDER=openai_compatible
DAC3D_LLM_API_BASE_URL=https://your-provider.example/v1
DAC3D_LLM_API_KEY=your_api_key
DAC3D_LLM_MODEL_NAME=your_model_name
```

如果使用 Anthropic-compatible 服务：

```env
DAC3D_LLM_PROVIDER=anthropic
DAC3D_LLM_API_BASE_URL=https://your-anthropic-compatible-endpoint
DAC3D_LLM_API_KEY=your_api_key
DAC3D_LLM_MODEL_NAME=claude-3-5-haiku-latest
```

## 知识库维护

知识库源文档位于：

```text
dac3d_iim_assistant/knowledge_base/documents/
```

添加或更新文档后，可以在 Web 页面上传并触发自动重建，也可以手动执行：

```powershell
cd C:\Users\xecat\DAC-3D-LLM\dac3d_iim_assistant
python knowledge_base\build_kb.py
```

向量库输出位于：

```text
dac3d_iim_assistant/knowledge_base/vector_store/
```

## 常用 Git 操作

查看修改：

```powershell
git status
```

提交指定文件：

```powershell
git add "路径/文件名"
git commit -m "Update file"
git push origin main
```

提交全部修改：

```powershell
git add .
git commit -m "Update project"
git push origin main
git lfs push origin main --all
```

如果 LFS 上传超时，可以降低并发：

```powershell
git config lfs.concurrenttransfers 1
git config lfs.activitytimeout 300
git config lfs.dialtimeout 60
git config lfs.tlstimeout 60
git config http.version HTTP/1.1
git lfs push origin main --all
```

## 当前限制

- 真正的 DAC-3D 硬件 SDK 和现场设备控制仍需根据实际设备环境进一步适配。
- 大模型权重和演示图片依赖 Git LFS，克隆后必须执行 `git lfs pull`。
- GPU 加速依赖本机显卡、CUDA、PyTorch 版本三者匹配。
- LLM 真实问答依赖外部 API 或本地模型服务，未配置时只能走 mock 或规则兜底。
- 检测结果解释质量依赖结果文件完整性和知识库中的判定依据。

## 安全说明

- 不要提交真实 API Key。
- 不要提交客户敏感检测数据。
- 生产环境不要使用默认演示账号和弱密码。
- 真实设备控制命令必须保留人工确认或权限控制。

## 推荐演示流程

1. 启动 DAC-3D 主系统。
2. 启动 LLM 智能助手 Web 页面。
3. 在 DAC-3D 主系统中点击智能助手入口跳转。
4. 使用自然语言生成扫描配置，例如“扫描 10mm x 10mm 区域”。
5. 使用离线图片执行检测，例如选择 `福特科/pre_fusion_images/`。
6. 询问“当前系统在做什么”“第三个样品检测结果怎么样”。
7. 询问参数含义、缺陷严重程度和现场操作建议。

## 项目状态

本项目当前定位为本科毕业设计阶段的完整本地演示系统，重点验证 DAC-3D 与 LLM 助手的结合方式，包括知识库问答、意图识别、结构化命令、状态读取和结果解释。后续可继续扩展真实设备控制、权限体系、生产部署和更完整的检测数据闭环。
