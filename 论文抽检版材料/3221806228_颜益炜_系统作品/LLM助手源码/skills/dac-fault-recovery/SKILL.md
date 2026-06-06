---
name: dac-fault-recovery
description: Diagnose failed commands, unavailable DAC state, and recurring runtime errors.
risk_level: medium
requires_confirmation: false
triggers:
  - 失败
  - 报错
  - 恢复
  - error
  - recovery
tools:
  - dac3d_status
  - dac3d_answer
  - conversation_memory_search
memory_layers:
  - episodic_memory
  - tool_memory
  - procedure_memory
---

# DAC Fault Recovery

Use this skill when a command fails, status is unavailable, folder validation fails, or the user asks how to recover.

Workflow:

1. Read current status and recent task trace.
2. Identify whether the failure is path, schema, runtime busy, bridge unavailable, result missing, or operator cancellation.
3. Suggest the smallest safe recovery step.
4. If the same failure repeats, propose a memory or skill patch for human review.

Memory write policy:

- Single failures remain episodic.
- Repeated failure patterns can become tool memory or procedure memory after approval.
