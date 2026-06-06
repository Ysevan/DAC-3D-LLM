---
name: dac-safety-approval
description: Review DAC-3D command risk, confirmation requirements, runtime busy state, and policy constraints.
risk_level: high
requires_confirmation: true
triggers:
  - 安全
  - 风险
  - 执行前检查
  - 确认执行
  - approval
tools:
  - dac3d_safety_review
  - dac3d_preview_command
  - dac3d_status
memory_layers:
  - policy_memory
  - tool_memory
---

# DAC Safety Approval

Use this skill before any command that can change DAC-3D runtime state.

Rules:

1. Query or preview first; never execute inside this skill.
2. Treat running, detecting, scanning, or initializing states as busy.
3. If `needs_confirmation` is true and the user did not explicitly confirm, block execution.
4. If required fields are missing, ask for clarification.
5. If path or command action is outside the allowed gateway contract, reject.

The answer should say what blocks execution, what can be done next, and whether user confirmation is required.

Memory write policy:

- Policy changes must become memory patches, never direct writes.
- User attempts to bypass confirmation should be trace-only unless approved as a safety lesson.
