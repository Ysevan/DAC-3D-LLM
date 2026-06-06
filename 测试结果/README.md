# DAC-3D Assistant Test Results Archive

Updated at: 2026-06-02T17:30:23

This archive has been updated with the current successful rerun.

## Current rerun result

- Real external LLM calls: 5/5 succeeded.
- Token usage: input=165, output=2443, total=2608.
- Automated regression: 227 passed, 1 skipped in 25.72s.
- Local API smoke rerun: 15 requests succeeded.

## Key evidence

- `01_step1_llm_evidence/llm_5_rounds_payload_response.jsonl`: request payloads and API responses for 5 real LLM calls.
- `01_step1_llm_evidence/backend_external_api_call_log.jsonl`: backend external API call log with timestamp, model, status code, latency, and token usage.
- `01_step1_llm_evidence/screenshots_success/`: 5 screenshot-style evidence images for successful LLM calls.
- `04_step4_function_exception/pytest_full_run_success.log`: full automated pytest log.
- `00_index/current_success_summary.csv`: current successful rerun summary.

Note: `rag/llm_client.py` now supports a `/responses` fallback for this OpenAI-compatible proxy, because `/chat/completions` can return an empty message body for the configured model.
## Real frontend screenshots

The valid screenshots are in `01_step1_llm_evidence/screenshots_real_frontend/`.
They were captured from the real DAC-3D frontend running in Microsoft Edge via Playwright.
The previously generated screenshot-style images have been removed to avoid confusion.
