---
name: dac-memory-maintenance
description: Maintain DAC-Agent memory patches, corrections, approvals, and forgetting requests.
risk_level: medium
requires_confirmation: true
triggers:
  - 记住
  - 纠正
  - 忘记
  - 记忆
  - memory
tools:
  - conversation_memory_search
  - conversation_memory_profile
  - conversation_memory_patches
  - conversation_memory_approve_patch
  - conversation_memory_reject_patch
memory_layers:
  - operator_memory
  - policy_memory
  - episodic_memory
---

# DAC Memory Maintenance

Use this skill when the user asks to remember, correct, forget, approve, reject, or inspect memory.

Rules:

1. Never silently write long-term memory from a casual statement.
2. Convert durable memory candidates into pending patches.
3. Only approve a patch when the user explicitly says to approve/save it.
4. Reject or correct stale memory when the user says it is wrong.
5. Keep safety policies separate from user preference memory.

Memory write policy:

- User preferences go to `USER.md`.
- Stable project/runtime rules go to `MEMORY.md`.
- Procedure details go to topic knowledge notes.
- Current status and one-off results stay in trace only.
