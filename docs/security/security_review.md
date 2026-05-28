# DAC-3D-LLM 生产安全总审查

审查日期：2026-05-27

审查范围：`dac3d_iim_assistant/` 的 Agent、Tool、Command、Path/File、Memory、API/UI、Trace/Logging、Supply Chain/CI 安全面。结论以生产启用 `DAC3D_ALLOW_COMMAND_SUBMIT=true`、真实 DAC-3D 桥接写入、长期 memory 写入为上线假设。

## S15-1 已修复

- Agent/CLI 的 `confirmed_by_user` 直通提交已被禁用。`dac3d_execute_command` 和 CLI `--execute-command --confirmed` 现在只返回命令预览与 `TOKEN_BOUND_CONFIRMATION_REQUIRED` 安全决策，不再写入 mock runtime 或 command-file bridge。
- 真实命令下发仍只允许走 Web API `/api/commands/preview` + `/api/commands/confirm`，并绑定 `preview_id`、`preview_hash`、一次性 `confirmation_token`、operator 和 session。
- 已补回归测试覆盖英文直通、中文“执行扫描”、pending preview 后“确认执行”、command-file bridge 不写出，以及 Agents SDK 工具调用不改变 runtime。

## S15-2 已修复

- Agent 工具已接入最小 `ToolGateway`/`PolicyEngine`。所有 `AGENT_TOOL_NAMES` 都在注册表中声明风险级别、side effect、参数 schema、确认要求和是否允许 Agent 直用。
- `DAC3DAgentToolController` 的工具入口统一先经过 gateway；未知工具、缺失/额外参数、禁用的写类工具都会 fail closed，且不会调用底层 handler。
- `dac3d_rebuild_knowledge_base` 已禁止 Agent 直接触发，因为它会产生文件系统写入；Web/API 授权路径后续仍可独立治理。
- 运行时摘要现在暴露 `tool_gateway.enforced=true`、注册工具清单和每个工具的风险元数据，工具返回也会带 `tool_gateway` 与 `policy_decision`。

## S15-3 已修复

- 新增中心化 `PathPolicy`，统一处理 canonicalize、`../` 阻断、NUL 字节阻断、symlink escape 阻断、allowlist 外拒绝、secret-like 路径拒绝和 command bridge 任意覆盖拒绝。
- 离线图片目录读取已接入 PathPolicy：`validate_offline_folder` 返回结构化 `path_policy` 拒绝结果；`start_offline_detection` 在真实提交前 fail closed。
- command-file bridge 写入已接入 PathPolicy：只允许 `dac3d_assistant_command.json`，可通过 `DAC3D_ALLOWED_COMMAND_OUTPUT_DIR` 限定输出根目录，并拒绝危险临时文件和 symlink。
- 状态文件读取会拒绝 secret-like 文件和不安全路径；知识库上传文件名会拒绝路径分隔符、`../` 和 secret-like 文件名。
- 已补回归测试覆盖 canonical path、`../`、symlink escape、allowlist、secret-file-read、unsafe upload filename、command bridge 输出目录和任意文件名拒绝。

## S15-4 已修复

- Agent chat runtime 不再把每轮对话直接写入长期 memory；`_remember_turn()` 现在只生成 pending memory patch，并在响应里暴露 `memory_pending_patch` 元数据。
- 新增 memory approval 生命周期：`ConversationMemoryStore.propose_turn()`、`approve_pending_patch()`、`reject_pending_patch()` 和 `delete_turn()`。只有 approved patch 会进入 session JSON 和全局 index。
- `/api/memory/approve`、`/api/memory/reject` 已从 501 占位改为真实 privileged API；新增 `/api/memory/pending` 与 `/api/memory/delete` 供管理员审查和删除。
- rejected patch 不会写入 session/index；deleted turn 会同时从 session JSON 和 index 移除，并写入删除 tombstone，确保不会进入后续 prompt context。
- 已补回归测试覆盖 pending-before-approval、approve commit、reject no-context、delete no-context、API approve/reject/delete 和 Agent adapter 注入前置审批。

## S15-5 已修复

- Security eval runner 已从 fail-closed stub 升级为本地 runtime harness。每个 red-team case 会进入真实 `DAC3DAgentRuntime`、`DAC3DAgentToolController`、`ToolGateway`、`PathPolicy`、memory approval、FastAPI auth 或 audit trace 边界。
- Eval 报告会输出实际 `tool_calls`、被 ToolGateway fail-closed 的 forbidden tool attempts、policy decisions、command submission 状态、memory approval 状态、API status code 和 memory context 可见性。
- CI 中的 `python -m evals.run_security_evals` 现在断言真实命令没有提交、未批准 memory 不进入上下文、未知/危险工具不触发 handler、API auth bypass 被拒绝。
- 已补回归测试确认报告不再只是空 trace：prompt-injection case 记录 `dac3d_execute_command` 和 blocked `write_command`/`submit_command`，memory-poisoning case 只产生 pending patch，API auth bypass 返回 401。

## S15-6 已修复

- Swagger/OpenAPI 已显式建模 DAC-3D API actor 安全头：`X-DAC3D-Session-ID`、`X-DAC3D-Operator-ID` 和 `X-DAC3D-Roles` 都注册为 header apiKey security schemes。
- OpenAPI 会按路径标注安全级别：只读接口要求 session，命令 preview 要求 session/operator，命令确认、知识库构建和 privileged memory/skill API 要求 session/operator/roles。
- 每个受保护 operation 都附带 `x-dac3d-security` 扩展，标出 required headers 和 required roles，降低 Swagger 调试时漏带 operator/session/role 的误用风险。
- 已补回归测试确认 `/openapi.json` 暴露三类安全头、`/api/commands/confirm` 标注 operator/admin/security_admin 写角色、`/api/health` 不被错误标为受保护。

## S15-7 已修复

- `ui2` 高风险命令确认已从浏览器原生确认弹窗改为页面内可访问 dialog。点击“确认执行本次命令”只会打开确认窗口，不会直接下发。
- dialog 会展示本次 action、mode、region、summary、完整 `preview_hash`、trace_id 和风险提示，并声明确认只绑定当前 preview/operator/session。
- operator 必须输入短确认语 `执行 <preview_hash 前 16 位>`，按钮才会启用；Escape/取消只关闭 dialog，不会触发 `/api/commands/confirm`。
- 静态扫描器新增 `browser-window-confirm` 规则，CI 本地扫描会拒绝重新引入浏览器原生确认调用。

## S15-8 已修复

- Audit trace JSONL 仍保留 append-only hash chain，同时新增可选 HMAC-SHA256 事件签名。签名绑定 `event_hash`、`signature_key_id` 和算法，攻击者即使重算 hash chain，也无法在没有独立密钥的情况下伪造合法签名。
- `AuditTraceLogger.verify_hash_chain()` 在配置签名密钥时会同时校验签名缺失、算法不匹配、key id 不匹配和签名不匹配，并在结果中报告 `signature_checked`。
- FastAPI audit logger 会从 `DAC3D_AUDIT_TRACE_SIGNING_KEY` 和 `DAC3D_AUDIT_TRACE_KEY_ID` 读取签名配置；生产配置校验要求 `DAC3D_AUDIT_TRACE_ENABLED=true` 且必须提供签名密钥。
- 已补回归测试覆盖签名 trace 的正常校验，以及“篡改事件并重算 hash chain、但没有签名密钥”会被拒绝。

## P0 必须修复

当前审查范围内的 P0 项已完成。下一步应继续推进 P1/P2 的锁文件、可复现 E2E 和外部审计汇聚。

## P1 建议修复

1. Audit trace 已加入签名 hash chain，可检测篡改和离线重算 hash chain，但还不是操作系统只追加权限或外部日志汇聚。生产建议继续加日志轮转、只追加权限或外部 SIEM/对象存储归档。

2. dependency audit、bandit、ruff、npm audit 已进入 CI，但本地无法在无网络/无 CI token 环境中确认完整执行结果。合并前应以 GitHub Actions 结果为准。

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
- audit trace 支持 HMAC-SHA256 事件签名；生产环境要求开启 trace 并配置签名密钥。
- 前端不使用 `dangerouslySetInnerHTML`、`.innerHTML`、`eval`、`new Function` 或浏览器原生确认调用，并通过可访问 dialog 显示高风险确认、preview hash、trace_id、错误 trace_id。
- CI 安全工作流已覆盖 pytest、compileall、ruff、bandit、pip-audit、npm audit、frontend build、static scan、security eval smoke。
- red-team cases 已覆盖 prompt injection、indirect prompt injection、RAG poisoning、memory poisoning、confirmation bypass、path traversal、secret exfiltration、tool misuse、API auth bypass、trace tampering、XSS 输出注入、DoS oversized input 等类别。
- Agent/CLI 直接执行命令已 fail closed；confirmed flag 只返回 token-bound confirmation 要求，不能直接提交到 mock runtime 或 command-file bridge。
- Agent 工具已通过 `ToolGateway -> PolicyEngine` 强制注册和策略判定；未注册工具默认拒绝，禁用的写类工具不会触发底层 handler。
- Path/File 边界已通过 `PathPolicy` 强制执行；离线目录、状态文件读取、command bridge 写入和知识库上传文件名都进入统一 canonical/allowlist/secret/symlink/traversal 检查。
- 长期 memory 已改为 pending patch 审批生命周期；未批准、已拒绝、已删除的 memory 都不会进入 prompt context。
- Security eval runner 已接入真实本地 Agent runtime/tool harness；报告会记录实际 `tool_calls`、ToolGateway blocked attempts、policy decisions、command submission 状态、memory approval 状态和 API auth 结果。
- Swagger/OpenAPI 已把 DAC-3D session、operator 和 roles header 建模为安全 schemes，并按 operation 标注 required headers/roles。
- `ui2` 高风险命令确认已改为可访问 dialog，要求 operator 核对命令摘要和 preview hash 并输入短确认语后才提交。

## 缺失测试

- Chrome 插件 UI smoke 的自动化回归测试。

## 生产上线前阻塞项

- 生产设置 `DAC3D_ALLOW_COMMAND_SUBMIT=true` 前必须明确配置 `DAC3D_ALLOWED_INPUT_DIRS`、`DAC3D_ALLOWED_COMMAND_OUTPUT_DIR`，并保留 preview/confirm/token-bound confirmation 链路。
- 生产长期 memory 写入必须保留 privileged approval API；如需临时冻结落库，可设置 `DAC3D_ENABLE_MEMORY_WRITE=false`，pending patch 仍不会进入 prompt context。
- remote/open-world/skill patch 类能力必须保持关闭，直到纳入 ToolGateway 风险策略并完成专项 review。
- 生产必须使用明确 CORS origin、明确输入目录 allowlist、明确 command 输出目录、非 mock DAC-3D writer，并保留 rate limit 与 redaction。

## 修复计划

1. S15-1：已完成。Agent/CLI 的 `confirmed_by_user` 直通路径改为 fail-closed，不再直接 submit；真实下发保留 Web API token-bound confirmation。

2. S15-2：已完成。所有 Agent tools 进入最小 `ToolGateway`/`PolicyEngine` 注册表；未注册工具 fail closed；写类 KB rebuild 禁止 Agent 直用。

3. S15-3：已完成。引入 `PathPolicy`，统一处理 canonicalize、allowlist、symlink、secret file、bridge output path，并接入 uploads、offline folder validation、status file read、command bridge。

4. S15-4：已完成。memory 改为 approval patch 生命周期；默认生成 pending memory patch，operator 审批后写入；reject/delete 不再进入 prompt context。

5. S15-5：已完成。security eval runner 已接入真实 runtime/tool trace，把 P0 安全边界转为 CI 门禁。

6. S15-6：已完成。Swagger/OpenAPI 已建模 DAC-3D session/operator/roles 安全头，并为受保护 operation 标注安全级别和所需角色。

7. S15-7：已完成。前端高风险确认已从浏览器原生确认弹窗升级为可访问 dialog，并增加静态扫描规则防止回退。

8. S15-8：已完成。Audit trace 加入 HMAC 签名校验和生产密钥要求，防止仅重算 JSONL hash chain 的离线篡改。
