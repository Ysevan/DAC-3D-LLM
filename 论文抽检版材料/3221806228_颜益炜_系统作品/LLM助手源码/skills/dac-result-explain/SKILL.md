---
name: dac-result-explain
description: Read and explain latest or indexed DAC-3D inspection result.
risk_level: low
requires_confirmation: false
triggers:
  - 结果
  - 缺陷
  - 样品
  - result
tools:
  - dac3d_latest_result
  - dac3d_answer
memory_layers:
  - document_memory
  - episodic_memory
---

# DAC Result Explain

Use this skill when the user asks about latest result, a specific sample position, defect type, severity, or next inspection action.

Required behavior:

1. Read result data first.
2. Do not invent defects if no result exists.
3. Explain quality label, defect count, defect type, location, confidence, and severity.
4. When useful, cite document guidance with `dac3d_answer`.

Memory write policy:

- Result summaries are episodic, not permanent.
- Repeated defect explanation preferences may become user memory after approval.
