# DAC-3D-LLM 生产安全总审查

审查日期：2026-05-27

审查范围：`dac3d_iim_assistant/` 的 Agent、Tool、Command、Path/File、Memory、API/UI、Trace/Logging、Supply Chain/CI 安全面。结论以生产启用 `DAC3D_ALLOW_COMMAND_SUBMIT=true`、真实 DAC-3D 桥接写入、长期 memory 写入为上线假设。

## P0 必须修复

1. Agent/CLI 仍存在确认布尔值执行路径，未绑定 Web API 的 `preview_id`、`preview_hash`、一次性 token、operator/session。`agent_runtime.py` 的 `--execute-command --confirmed` 会直接进入 `runtime.execute_command(... confirmed_by_user=args.confirmed)`；OpenAI Agents SDK 工具 `dac3d_execute_command` 也把 `confirmed_by_user` 暴露给模型工具参数。Web API 已经有 token 链路，但 Agent/CLI 路径仍可绕过 S8 的确认生命周期。上线前必须把所有命令提交统一收口到同一确认服务，或在生产禁用 Agent/CLI 直接提交。参考：`dac3d_iim_assistant/agent_runtime.py:315`、`:560`、`:944`、`:1016`，`dac3d_iim_assistant/agent_core/tools.py:41`。

2. Tool Gateway/PolicyEngine 还不是强制边界。当前工具是 `agent_runtime.py` 内直接注册的一组 `function_tool`，再调用 `DAC3DAgentToolController`；没有一个所有工具调用都必须经过的注册表、schema 校验、policy decision、risk classification 与 fail-closed 网关。高风险命令在 Web API 有确认，但 machine tools、KB rebuild、未来 remote/open-world/file tools 没有统一策略面。上线前必须先引入中心化 `ToolGateway`，所有 Agent 工具只调用网关，未注册工具默认拒绝。

3. Path/File 策略不完整。知识库上传只做了 `Path(file.filename).name` 落到临时目录，离线目录校验和 command bridge 写入直接使用传入或配置路径；目前没有统一 canonicalize、`../` 阻断、symlink escape 阻断、allowlist 外拒绝、secret 文件读取拒绝、任意覆盖拒绝策略。上线前必须新增中心化 PathPolicy，并覆盖 `integration/dac3d_client.py` 的离线目录读取与 command 文件写入。参考：`dac3d_iim_assistant/ui/web_api.py:303`、`dac3d_iim_assistant/integration/dac3d_client.py:251`、`:436`、`:462`。

4. 长期 memory 写入仍未经过 approval 生命周期。`ConversationMemoryStore.append_turn()` 已拒绝 secret-like 内容，但 `agent_runtime.py` 在每轮结束后会直接 `_remember_turn()`，并写入 session JSON 和全局 index；`/api/memory/approve`、`/api/memory/reject` 还是 501 占位。上线前必须把长期写入改成 pending patch，经授权 operator 审批后落库，rejected/deleted memory 不得进入上下文。参考：`dac3d_iim_assistant/agent_runtime.py:917`、`dac3d_iim_assistant/memory/conversation_store.py:142`、`dac3d_iim_assistant/ui/web_api.py:259`。

## P1 建议修复

1. Security eval runner 当前是 fail-closed stub，能覆盖用例清单和阻断期望，但尚未驱动真实 Agent、真实 tool trace、真实 command lifecycle。需要把 `evals/security_cases` 接入 runtime harness，并断言真实工具调用列表。

2. Audit trace 是 JSONL hash chain，可检测内容篡改，但还不是操作系统或外部存储层面的不可变审计。生产建议加只追加权限、轮转、签名或外部日志汇聚。

3. Swagger/OpenAPI 未把 `X-DAC3D-Session-ID`、`X-DAC3D-Operator-ID`、角色头建模为安全依赖。运行时会校验，但接口文档不够清晰，容易误用。

4. 前端高风险确认目前使用 `window.confirm`。安全语义已经明确，但生产 UI 建议换成可访问 modal，并要求 operator 核对 action、scope、preview hash 或短确认语。

5. dependency audit、bandit、ruff、npm audit 已进入 CI，但本地无法在无网络/无 CI token 环境中确认完整执行结果。合并前应以 GitHub Actions 结果为准。

## P2 后续优化

1. Python 依赖缺少锁文件。前端已有 `package-lock.json`，Python 建议补 `requirements.lock`、`uv.lock` 或等价锁定。

2. 静态扫描器是轻量本地守门，后续可接入 semgrep/CodeQL，并把 allowlist 细化到规则级。

3. 把 Chrome 插件的真实 UI/API smoke 固化成可复现 E2E 测试，避免只依赖人工/本地运行记录。

4. 给 FastAPI 增加更完整的安全响应头策略，例如 CSP、`X-Content-Type-Options`、`Referrer-Policy`、`Permissions-Policy`。

5. 给 trace query/export 增加分页、角色权限和按 trace_id 的审计检索 API。

## 已满足项

- Web API 写接口要求 session/operator；确认接口要求 operator、session、preview hash、一次性 token，并拒绝 replay、过期、operator/session/hash mismatch。
- `/api/chat` 与 streaming chat 对执行类文本 fail closed，强制走 preview/confirm 分离链路。
- 生产配置校验禁止生产 wildcard CORS、禁用 debug、要求 redaction、要求 rate limit、禁止 mock writer 下开放命令提交。
- trace 日志已脱敏、带 request_id/trace_id、支持 hash chain 校验、支持 redacted export。
- 前端不使用 `dangerouslySetInnerHTML`、`.innerHTML`、`eval`、`new Function`，并显示高风险确认、preview hash、trace_id、错误 trace_id。
- CI 安全工作流已覆盖 pytest、compileall、ruff、bandit、pip-audit、npm audit、frontend build、static scan、security eval smoke。
- red-team cases 已覆盖 prompt injection、indirect prompt injection、RAG poisoning、memory poisoning、confirmation bypass、path traversal、secret exfiltration、tool misuse、API auth bypass、trace tampering、XSS 输出注入、DoS oversized input 等类别。

## 缺失测试

- Agent/CLI 直接执行命令必须被禁用或强制 token-bound confirmation 的回归测试。
- 所有 function tools 必须经过 `ToolGateway -> PolicyEngine` 的注册、schema、risk、confirmation、fail-closed 测试。
- PathPolicy 的 canonical path、`../`、symlink escape、allowlist、secret-file-read、任意覆盖测试。
- memory approval/reject/delete/patch 生命周期测试，以及 rejected/deleted memory 不进入上下文的测试。
- security eval runner 对真实 runtime/tool trace 的集成测试。
- Chrome 插件 UI smoke 的自动化回归测试。

## 生产上线前阻塞项

- 不能在生产设置 `DAC3D_ALLOW_COMMAND_SUBMIT=true`，直到 P0-1 到 P0-3 完成并有回归测试。
- 不能启用生产长期 memory 写入，直到 P0-4 完成；临时策略应设置 `DAC3D_ENABLE_MEMORY_WRITE=false` 或只保留短期会话上下文。
- remote/open-world/file/skill patch 类能力必须保持关闭，直到统一 ToolGateway/PolicyEngine 与 PathPolicy 完成。
- 生产必须使用明确 CORS origin、明确输入目录 allowlist、明确 command 输出目录、非 mock DAC-3D writer，并保留 rate limit 与 redaction。

## 修复计划

1. S15-1：先修命令执行 P0。把 Agent/CLI 的 `confirmed_by_user` 直通路径替换为 preview_id/token-bound confirmation，或在生产直接拒绝 Agent/CLI submit；补 replay、hash mismatch、session/operator mismatch、CLI bypass 测试。

2. S15-2：引入最小 `ToolGateway`/`PolicyEngine`。所有 Agent tools 注册 schema/risk/context policy；未注册工具 fail closed；high-risk/destructive 工具必须返回 confirmation_required，不能直接 submit。

3. S15-3：引入 `PathPolicy`。统一处理 canonicalize、allowlist、symlink、secret file、bridge output path，接入 uploads、offline folder validation、command bridge。

4. S15-4：改 memory 为 approval patch 生命周期。默认生成 pending memory patch，operator 审批后写入；reject/delete 不再进入 prompt context。

5. S15-5：把 security eval runner 接入真实 runtime，把 P0 修复转为 CI 门禁。
