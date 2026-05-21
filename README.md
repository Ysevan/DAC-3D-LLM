# DAC-3D-LLM

本仓库包含 DAC-3D 主检测系统、LLM 智能检测助手、知识库文档、模型部署资产和毕业设计相关材料。项目目标是把 DAC-3D 检测流程从传统软件操作升级为“自然语言交互 + 文档问答 + 检测状态读取 + 结果解读”的本地可演示系统。

## 项目定位

DAC-3D-LLM 面向智能检测场景，不是通用聊天机器人。系统应尽量基于以下三类信息回答和执行：

- DAC-3D 官方/项目文档，例如操作手册、参数说明、FAQ、缺陷判定规则。
- DAC-3D 主系统运行状态，例如当前是否检测、检测进度、最新结果、历史样品结果。
- DAC-3D 检测数据和模型结果，例如离线图片检测输出、缺陷类型、置信度、位置和严重程度。

当前演示支持两种形态：

- LLM 智能检测助手原型：可独立启动 Web 页面，用于问答、命令预览、知识库构建和结果解释。
- 嵌入式 DAC-3D 演示：从 DAC-3D 主系统按钮跳转到 Web 助手，读取主系统状态和结果数据。

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
- 新增 `openAssistantWeb(...)`，自动启动 `dac3d_iim_assistant/app.py` 并打开 `http://127.0.0.1:7860`。
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
http://127.0.0.1:7860
```

同时支持切换到第二套 Web UI：

```text
http://127.0.0.1:8000
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

### 5. 保留 GPU 检测能力

系统会检测当前 PyTorch 是否支持 CUDA。如果环境中安装的是 CUDA 版 PyTorch，并且显卡驱动可用，检测模型可以走 GPU；否则自动回退 CPU。

检查 GPU 是否可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

输出 `True` 表示当前 Python 环境可以使用 GPU。

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

默认访问：

```text
http://127.0.0.1:7860
```

如果启用第二套 Web UI，可访问：

```text
http://127.0.0.1:8000
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

如果需要 GPU 检测，请安装与你显卡和 CUDA 版本匹配的 PyTorch。安装完成后检查：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

输出 `True` 才表示当前 Python 环境可使用 GPU。

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

如果要使用 GPU，请根据本机 CUDA 版本安装对应 CUDA 版 PyTorch。安装后必须确认：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
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
