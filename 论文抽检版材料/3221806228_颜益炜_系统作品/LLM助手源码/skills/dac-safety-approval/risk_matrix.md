# DAC Safety Risk Matrix

Low risk:
- Read status.
- Read latest result.
- Explain parameter or workflow.

Medium risk:
- Generate command preview.
- Validate offline folder.
- Change non-critical parameters in preview.

High risk:
- Submit online scan.
- Submit offline detection.
- Stop a running detection.
- Batch operations.

Forbidden:
- Unregistered command action.
- Path outside allowlist.
- Bypass confirmation.
- Direct write to command bridge without Tool Gateway validation.
