---
name: dac-command-preview
description: Convert a DAC-3D operation request into a structured command preview before execution.
risk_level: medium
requires_confirmation: true
triggers:
  - 扫描
  - 命令预览
  - preview
  - command
tools:
  - dac3d_preview_command
  - dac3d_safety_review
memory_layers:
  - procedure_memory
  - tool_memory
---

# DAC Command Preview

Use this skill when the user asks to scan, start offline detection, stop detection, or change a DAC-3D operation.

Required steps:

1. Read current DAC runtime status when execution readiness matters.
2. Generate a structured command preview only.
3. Validate the preview schema and missing fields.
4. Attach risk level and whether confirmation is required.
5. Do not submit the command from this skill.

Output contract:

- `intent`: `operation_preview`
- `command_preview.action`: one registered DAC action.
- `command_preview.payload`: validated command payload.
- `requires_confirmation`: true for all commands that can change runtime state.
- `next_action`: `await_user_confirmation` or `needs_clarification`.

Memory write policy:

- Successful command patterns may become procedure memory only after repeated use or explicit operator approval.
- Failed command arguments should be stored as episodic trace, not permanent memory.
