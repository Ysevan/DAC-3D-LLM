# DAC-3D Production Security Configuration

Production deployments must fail closed before accepting DAC-3D command submission. The FastAPI app validates these settings at startup when `DAC3D_ENV=prod` or `DAC3D_ENV=production`.

## Required Variables

| Variable | Production requirement |
| --- | --- |
| `DAC3D_ENV` | `prod` or `production` enables strict validation. |
| `DAC3D_ALLOW_COMMAND_SUBMIT` | May be true only when the runtime writer is not mock. |
| `DAC3D_REQUIRE_CONFIRMATION` | Must be true. |
| `DAC3D_ALLOWED_INPUT_DIRS` | Must contain at least one allowed input directory. |
| `DAC3D_ALLOWED_COMMAND_OUTPUT_DIR` | Must be set. |
| `DAC3D_ENABLE_MEMORY_WRITE` | Allowed, but memory writes reject secret-like content. |
| `DAC3D_ENABLE_SKILL_PATCH_APPLY` | Must default to false. |
| `DAC3D_ENABLE_REMOTE_TOOLS` | Must be false unless separately reviewed. |
| `DAC3D_ENABLE_OPEN_WORLD_TOOLS` | Must be false unless separately reviewed. |
| `DAC3D_TRACE_REDACTION` | Must be true. |
| `DAC3D_AUDIT_TRACE_ENABLED` | Must be true. |
| `DAC3D_AUDIT_TRACE_SIGNING_KEY` | Must be set from a secret manager; signs every audit event hash. |
| `DAC3D_AUDIT_TRACE_KEY_ID` | Should identify the active signing key for rotation and verification. |
| `DAC3D_CORS_ALLOWED_ORIGINS` | Must be explicit and cannot include `*`. |
| `DAC3D_RATE_LIMIT_ENABLED` | Must be true. |
| `DAC3D_DEBUG_MODE` | Must be false. |
| `DAC3D_MOCK_MODE` | Must not be treated as a real production command writer. |

## Security Response Headers

The FastAPI security middleware attaches browser security headers to API, static UI, Swagger, and structured error responses. The default policy includes CSP, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Permissions-Policy`, frame blocking, and cross-origin isolation baseline headers. HTTPS requests also receive HSTS.

## Audit Trace Query And Export

Audit trace query/export APIs are privileged endpoints. Use `GET /api/audit/traces`, `GET /api/audit/traces/{trace_id}`, or `GET /api/audit/export` only with `X-DAC3D-Session-ID`, `X-DAC3D-Operator-ID`, and an `X-DAC3D-Roles` value containing `admin` or `security_admin`. Responses are redacted, paginated with bounded `limit`/`offset`, and include hash-chain/signature verification metadata.

## Safe Production Example

```bash
DAC3D_ENV=prod
DAC3D_MOCK_MODE=false
DAC3D_REQUIRE_CONFIRMATION=true
DAC3D_TRACE_REDACTION=true
DAC3D_AUDIT_TRACE_ENABLED=true
DAC3D_AUDIT_TRACE_KEY_ID=ops-audit-2026-05
# Set DAC3D_AUDIT_TRACE_SIGNING_KEY from a deployment secret, not from committed files.
DAC3D_RATE_LIMIT_ENABLED=true
DAC3D_ENABLE_REMOTE_TOOLS=false
DAC3D_ENABLE_OPEN_WORLD_TOOLS=false
DAC3D_ENABLE_SKILL_PATCH_APPLY=false
DAC3D_DEBUG_MODE=false
DAC3D_CORS_ALLOWED_ORIGINS=https://dac3d.example.com
DAC3D_ALLOWED_INPUT_DIRS=/opt/dac3d/input
DAC3D_ALLOWED_COMMAND_OUTPUT_DIR=/opt/dac3d/runtime/commands
```

## Local Validation

```bash
cd dac3d_iim_assistant
python -m pytest tests/test_security_config.py -q
```

The production validator lives in `security/production_config.py`. Non-production environments may be permissive, but `warnings()` reports risky choices such as wildcard CORS, disabled confirmation, remote tools, unsigned audit traces, or debug mode.
