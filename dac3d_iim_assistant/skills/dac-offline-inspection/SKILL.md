---
name: dac-offline-inspection
description: Run the safe workflow for DAC-3D offline image inspection.
risk_level: high
requires_confirmation: true
triggers:
  - 离线检测
  - 图片检测
  - offline
  - pre_fusion_images
tools:
  - dac3d_preview_command
  - dac3d_safety_review
  - dac3d_execute_command
memory_layers:
  - procedure_memory
  - project_memory
  - episodic_memory
---

# DAC Offline Inspection

Use this skill when the user wants to inspect a local image folder instead of running live scanning.

Workflow:

1. Identify the image folder.
2. Validate that the folder is allowed and contains expected offline images.
3. Generate a `start_offline_detection` preview.
4. Run safety review.
5. Wait for explicit confirmation before execution.
6. After execution, read status and summarize result.

Failure handling:

- If the folder is missing, ask the user to choose a valid folder.
- If validation fails, do not execute.
- If DAC runtime is busy, explain that the command must wait.

Memory write policy:

- Frequently used allowed folders may become operator or project memory after approval.
- One-off local paths should remain in episodic trace only.
