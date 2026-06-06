---
name: dac-state-check
description: Read and explain current DAC-3D runtime state and progress.
risk_level: low
requires_confirmation: false
triggers:
  - 状态
  - 进度
  - 当前
  - status
tools:
  - dac3d_status
memory_layers:
  - episodic_memory
---

# DAC State Check

Use this skill for status, progress, current task, idle/running/error, and whether a new command can be started.

Required answer fields:

- Runtime state.
- Progress percentage if available.
- Latest status message.
- Whether a new command appears safe to preview.
- Limitation if status is stale or unavailable.

Memory write policy:

- Current state is ephemeral and must not be written to long-term memory.
- Repeated failures may produce an episodic trace or a memory patch for fault recovery.
