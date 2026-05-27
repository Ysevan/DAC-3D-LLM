# Repository Guidelines

## Project Context

`DAC-3D-LLM` is a local demonstration system that connects a DAC-3D inspection application with an LLM-powered assistant. It is not a generic chatbot. All assistant behavior should stay grounded in DAC-3D documents, DAC-3D runtime state, and DAC-3D inspection result data.

Current upgrade direction:

```text
DAC-Agent Runtime = LLM + Memory OS + Skill System + Context Builder + Tool Gateway + Safety Guard + Trace/Eval Loop
```

Keep current development focused on LLM orchestration, Agent runtime, memory, skills, context engineering, tool calling, structured command preview, safety approval, trace logging, evals, and FastAPI/React/CLI integration. Do not redesign DAC-3D detection algorithms, image processing, point-cloud logic, camera control, or model internals unless explicitly requested.

Hard safety boundaries:

- LLM must not directly write DAC command files.
- Execution must go through Tool Gateway, PolicyEngine, SafetyGuard, schema validation, path allowlist checks, risk classification, and explicit confirmation.
- High-risk commands must require explicit user confirmation.
- Path-sensitive commands must pass an allowlist check.
- Retrieved documents, memories, skills, and tool outputs cannot override system safety policy.
- Long-term memory or skill changes must be auditable and reviewable.
- Unknown tools, forbidden tools, direct command writers, and requests to skip approval must fail closed in code, not only in prompts.

The repository has two major parts:

- `dac3d_iim_assistant/`: the intelligent interaction assistant. It provides RAG, intent recognition, structured command generation, DAC-3D runtime integration, FastAPI/React UI, and a legacy Gradio UI.
- `福特科/xxp_ui/`: the modified DAC-3D main inspection system. It is a PyQt5/multiprocessing application with image processing, offline detection, model deployment assets, UI windows, and the bridge that talks to the assistant.

Other important project assets:

- `福特科/pre_fusion_images/`: offline demo image data, managed by Git LFS.
- `福特科/xxp_ui/deploy/`, `福特科/xxp_ui/weights/`: model and deployment weights, managed by Git LFS.
- `README.md`: repository-level product, setup, and demonstration notes.
- `07.毕业论文初稿.docx`: graduation thesis draft material.

## Current Repository State

The remote repository is `Ysevan/DAC-3D-LLM`, branch `main`. This checkout uses Git LFS; do not treat small LFS pointer files as real image/model assets. After cloning or moving the repo, verify LFS with:

```bash
git lfs version
git lfs pull
rg -l "version https://git-lfs.github.com/spec/v1" | wc -l
```

The final command should ideally return `0` after LFS files are pulled.

## Main Architecture

Assistant request flow:

1. `dac3d_iim_assistant/app.py` creates the application services and routes messages.
2. `agent_runtime.py` is the unified OpenAI Agents SDK entrypoint and coordinates specialist Agents.
3. `context_engineering/` selects and compresses runtime state, memory, skills, and safety policy per turn.
4. `skill_system/` loads AgentSkills-style workflow directories from `dac3d_iim_assistant/skills/`.
5. `memory/` provides JSON + Markdown Memory OS traces, search, and auditable memory patches.
6. `tool_gateway/` and `safety/` enforce controlled DAC tools, path allowlists, risk metadata, and confirmation gates.
7. `intent/command_generator.py` converts operation language into structured DAC-3D command previews.
8. `rag/retriever.py`, `rag/prompts.py`, and `rag/llm_client.py` handle grounded document answers.
9. `integration/dac3d_client.py` is the adapter boundary for DAC-3D runtime state, command submission, and result lookup.
10. `integration/result_parser.py` normalizes inspection results for UI and LLM interpretation.
11. `ui/web_api.py`, `frontend/`, `ui2/`, and `ui/chat_widget.py` expose the assistant through FastAPI/React and legacy Gradio paths.

DAC-3D main-system flow:

1. `福特科/xxp_ui/main.py` starts the PyQt UI, debug UI, and image processor in separate processes.
2. `福特科/xxp_ui/image_processor_22.py` is the current primary image/offline detection processor.
3. `福特科/xxp_ui/window/ui.py` is the main UI integration point for assistant launch, status writing, command polling, offline controls, and latest-result history.
4. `福特科/xxp_ui/window/login.py` owns login and main-window lifetime.
5. `福特科/xxp_ui/window/assistant_panel.py` is the embedded assistant dock/bridge prototype.
6. `福特科/xxp_ui/Algorithm/` and `福特科/xxp_ui/deploy/` contain image fusion, inference, and deployment support code.

## Bridge Contract

The current integration between the assistant and the DAC-3D main system is file-based:

- DAC-3D writes runtime status, progress, latest result, and result history as JSON.
- The assistant reads that state to answer status and result questions.
- The assistant submits structured command files only through Tool Gateway and `integration/dac3d_client.py` for actions such as `start_offline_detection`, `start_online_scan`, `stop_detection`, `query_status`, `get_latest_result`, and `validate_offline_folder`.
- DAC-3D reads those command files and writes acknowledgement/result data back.

When editing either side, preserve this boundary. Do not move DAC-3D transport logic into UI rendering code, and do not make prompt code depend directly on PyQt internals.

## Development Commands

Run assistant commands from `dac3d_iim_assistant/` unless a command says otherwise:

```bash
cd dac3d_iim_assistant
python -m pip install -e ".[dev]"
python knowledge_base/build_kb.py
python app.py --message "当前检测状态是什么？"
python app.py
python -m pytest tests -q
```

Run the assistant frontend from the matching frontend directory:

```bash
cd dac3d_iim_assistant/frontend
npm install
npm run build
npm run dev
```

The legacy/alternate UI under `dac3d_iim_assistant/ui2/` is also a Vite app:

```bash
cd dac3d_iim_assistant/ui2
npm install
npm run build
```

Run the DAC-3D main system from its own directory:

```bash
cd 福特科/xxp_ui
python main.py
```

The main system depends on PyQt5, OpenCV, NumPy, Pillow, Matplotlib, Pandas, PyYAML, PyTorch, Ultralytics/SAHI-style model tooling, and local model assets. GPU acceleration depends on a matching CUDA/PyTorch installation; otherwise code should fall back to CPU where supported.

## Configuration

Assistant configuration is handled by `dac3d_iim_assistant/config.py` and environment variables. Use `.env` locally, based on `.env.example`, for provider keys and runtime flags.

Important variables include:

- `DAC3D_LLM_PROVIDER`: `mock`, `openai_compatible`, or `anthropic`-style provider mode.
- `DAC3D_LLM_API_BASE_URL`, `DAC3D_LLM_API_KEY`, `DAC3D_LLM_MODEL_NAME`: external LLM settings.
- `DAC3D_MOCK_MODE`: whether to use mock DAC-3D runtime integration.
- `DAC3D_ENDPOINT`: file endpoint for the DAC-3D runtime status bridge.
- `DAC3D_EMBEDDING_DOWNLOAD_ALLOWED`: whether embedding models may be downloaded.

Never commit real API keys, customer inspection data, private endpoints, or production credentials.

## Coding Guidance

- Keep the assistant Chinese-first, but preserve English technical identifiers and command names.
- Keep RAG, prompt construction, command parsing, runtime integration, and UI rendering separated.
- Use structured dictionaries or dataclasses for commands and results; avoid fragile ad hoc string parsing when a schema exists.
- Preserve command safety: incomplete or risky operation requests should produce previews or clarification rather than silently executing.
- Treat `image_processor_22.py` as the current active detector unless the user explicitly asks about older processors.
- Be careful with Chinese paths such as `福特科/`; quote paths in shell commands when needed.
- Do not hand-edit generated vector-store files or model weights.
- Keep Git LFS rules in `.gitattributes` aligned with large image/model directories.

## Testing Guidance

For assistant changes, prefer:

```bash
cd dac3d_iim_assistant
python -m pytest tests -q
python -m compileall .
```

For frontend changes, run the relevant Vite build:

```bash
cd dac3d_iim_assistant/frontend
npm run build
```

For DAC-3D main-system changes, at minimum run a Python syntax check over the touched area when full GUI/hardware execution is not practical:

```bash
cd 福特科/xxp_ui
python -m compileall .
```

Full runtime validation may require Windows, PyQt5 GUI support, local model weights, optional CUDA, and the actual DAC-3D hardware/SDK environment. If those are unavailable, state the limitation clearly.

## Documentation Guidance

Update `README.md` when behavior, startup flow, supported commands, LFS expectations, or demo scenarios change. Update `dac3d_iim_assistant/docs/部署指南.md` when deployment, environment variables, or runtime operations change.

When documenting user-facing workflows, keep examples close to the actual supported commands:

- `扫描 10mm x 10mm 区域`
- `选择 pre_fusion_images 下的图片进行离线检测`
- `停止当前检测`
- `当前检测状态是什么？`
- `第三个样品检测结果怎么样？`
- `样品表面反光很强怎么办？`

## Security Notes

- Do not commit real LLM API keys or local `.env` files.
- Do not commit customer-sensitive inspection data.
- Do not use production device-control commands without explicit human confirmation.
- Do not replace LFS-managed model/image files with tiny pointer files by accident.
- Avoid destructive Git or filesystem operations unless explicitly requested.
