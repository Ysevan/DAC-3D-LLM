# Repository Guidelines

## Project Goal
This repository builds an I.I.M assistant for DAC-3D. The assistant must reduce operator training cost through four core capabilities:

- natural-language operation, such as converting `scan a 10mm x 10mm area` into a DAC-3D scan configuration
- intelligent Q&A, such as explaining parameters, workflow steps, and manual content
- inspection-result interpretation, such as answering whether a defect is severe
- operator guidance, such as suggesting what to do when a sample is too reflective

The assistant is not a generic chatbot. It must stay grounded in DAC-3D manuals, DAC-3D runtime state, and DAC-3D result data.

## Required Repository Structure
Contributors should work against this target structure:

- `app.py`: main entry point and top-level orchestration
- `config.py`: application settings, provider selection, paths, and runtime flags
- `README.md`: product overview, quick start, and architecture summary
- `knowledge_base/documents/`: source manuals, FAQs, parameter notes, and other raw reference documents
- `knowledge_base/vector_store/`: generated vector index files and retrieval metadata
- `knowledge_base/build_kb.py`: document loading, chunking, embedding, and vector-store build pipeline
- `rag/retriever.py`: retrieval over the vector store
- `rag/llm_client.py`: LLM provider adapter
- `rag/prompts.py`: prompt templates and prompt builders
- `intent/classifier.py`: intent classification
- `intent/command_generator.py`: natural-language to structured-command conversion
- `integration/dac3d_client.py`: DAC-3D API or SDK adapter
- `integration/result_parser.py`: normalization of DAC-3D inspection results
- `ui/chat_widget.py`: chat-style user interface and multi-turn state
- `tests/test_assistant.py`: smoke tests and top-level workflow tests
- `docs/`: contains the deployment and runtime operations guide

## End-to-End Flow
Use this execution path as the default design:

1. `ui/chat_widget.py` receives the user message and manages conversation history.
2. `intent/classifier.py` decides whether the request is a query, operation, interpretation request, or guidance request.
3. Query and guidance flows use `rag/retriever.py`, `rag/prompts.py`, and `rag/llm_client.py`.
4. Operation flows use `intent/command_generator.py`, then optionally call `integration/dac3d_client.py`.
5. Interpretation flows may combine retrieval context with parsed DAC-3D result data from `integration/result_parser.py`.
6. `app.py` coordinates the full pipeline and returns the response to the UI.

Do not move DAC-3D access into UI code or prompt logic into integration modules.

## File-by-File Implementation Requirements

### `README.md`
Document the product goal, the four core capabilities, the repository layout, local run steps, current limitations, and one example for each supported user scenario.

### `app.py`
Implement the application bootstrap and request router. This file should load configuration, create all service objects, expose a `main()` entry point, and define the top-level request flow from user input to final response. It should also handle startup failure, provider errors, and DAC-3D unavailability in a clean way.

Recommended responsibilities:
- initialize config, retriever, LLM client, intent classifier, command generator, DAC-3D client, and UI shell
- route `query`, `operation`, `interpretation`, and `guidance` requests
- return either a text answer, a structured command preview, or an interpreted result summary

### `config.py`
Keep all runtime settings here in a typed configuration object. Add settings for LLM provider, API key, API base URL, model name, timeout, retry count, streaming switch, vector-store type, vector-store path, document path, chunk size, chunk overlap, retrieval top-k, DAC-3D endpoint, and a mock mode for local development. Secrets must come from environment variables or local-only config, not from committed literals.

### `knowledge_base/documents/`
Store raw documents only. Expected content includes DAC-3D user manuals, parameter descriptions, FAQ documents, troubleshooting notes, and sample interpretation guidance. Organize files by source or document type if the corpus grows. Do not place generated embeddings or temporary build artifacts here.

### `knowledge_base/vector_store/`
Store generated vector indexes and metadata only. This directory should contain persistent data produced by `build_kb.py`, such as FAISS indexes, Chroma persistence files, chunk metadata, and source mappings. Do not hand-edit these files.

### `knowledge_base/build_kb.py`
Implement the knowledge-base pipeline. This file should load source documents, normalize text, split content into chunks, create embeddings, and save the vector store plus metadata. It should preserve source document names so retrieved answers can be traced back to manuals or FAQs.

Recommended functions:
- `load_documents()`
- `chunk_documents()`
- `embed_chunks()`
- `persist_vector_store()`
- `build_knowledge_base()`

### `rag/retriever.py`
Implement retrieval over the built vector store. The retriever should return structured results rather than plain strings. Each result should include chunk text, source document, section or title if available, retrieval score, and chunk metadata. Support filtering by document type if needed.

### `rag/llm_client.py`
Implement the LLM abstraction layer. This module should support at least one provider first, but the interface should allow switching among Qwen, ERNIE, and Claude through configuration. Include normal generation, optional streaming generation, timeout handling, retries, and normalized errors.

### `rag/prompts.py`
Store prompt builders, not only one static prompt. Provide separate prompts for:

- manual and parameter Q&A
- defect interpretation
- operator guidance
- command clarification when the operation request is incomplete

Prompts should require grounded answers and discourage unsupported claims.

### `intent/classifier.py`
Classify user input into at least `query`, `operation`, and `interpretation`. A separate `guidance` label is recommended because operational advice is a first-class requirement. The classifier should also return confidence and extracted hints such as scan size, parameter name, or result-analysis keywords.

### `intent/command_generator.py`
Convert operation language into a structured DAC-3D command. The main output should be a validated dictionary and, if required by the integration, a YAML rendering. It should extract scan area, resolution, region, mode, and other operator parameters when present. It should also report missing required fields instead of silently guessing unsafe values.

### `integration/dac3d_client.py`
This should be the only module that talks to DAC-3D APIs, SDKs, or host services. It should support:

- submitting structured scan commands
- querying current inspection status
- retrieving current or recent inspection results
- running in mock mode when DAC-3D is not connected

Keep transport details isolated here so the rest of the assistant does not depend on DAC-3D protocol details.

### `integration/result_parser.py`
Normalize DAC-3D result payloads into a stable assistant-facing schema. Parse defect type, severity, confidence, location, measurement values, and any other fields needed for explanation. This module should prepare data that both the UI and the LLM flow can consume safely.

### `ui/chat_widget.py`
Implement a simple ChatGPT-style chat panel. It should support multi-turn history, clear user and assistant turns, streaming output, and display of structured command previews or interpreted result summaries. The design should make later embedding as a DAC-3D sidebar straightforward.

### `tests/test_assistant.py`
Keep this file as the initial smoke-test entry point. At minimum, test configuration loading, top-level request routing, one retrieval-backed answer path, one command-generation path, and one result-interpretation path. As the codebase grows, split detailed tests into module-specific files and keep this file focused on end-to-end coverage.

### Deployment Guide in `docs/`
Document environment requirements, dependency installation, provider configuration, knowledge-base build steps, local startup steps, DAC-3D integration setup, deployment checks, and common failure recovery steps.

## Minimum Demonstration Scenarios
Any implementation plan should be able to support these examples:

- `scan a 10mm x 10mm area`
- `what does this parameter mean?`
- `is this defect serious?`
- `what should I do if the sample is too reflective?`
- `what is the current inspection status?`

## Delivery Priority
Implement in this order unless blocked:

1. knowledge-base build and vector-store persistence
2. retrieval plus prompt assembly plus LLM client
3. intent classification and command generation
4. chat UI with multi-turn state and streaming
5. DAC-3D live integration and result interpretation

## Build, Test, and Development Commands
Run commands from the repository root.

- `python app.py`: start the scaffold or local demo flow
- `python knowledge_base/build_kb.py`: build the vector store once implemented
- `python -m pytest tests -q`: run automated tests
- `python -m compileall .`: run a syntax check

Update this section whenever new tooling is introduced.

## Coding Style & Naming Conventions
Use 4-space indentation, UTF-8 files, and PEP 8 formatting. Prefer type hints, explicit schemas, and focused classes or functions. Use `snake_case` for files, functions, and methods, `PascalCase` for classes, and `UPPER_CASE` for constants. Keep retrieval, prompt logic, command generation, DAC-3D integration, and UI responsibilities separated.

## Testing Guidelines
Use `pytest` and name files `test_*.py`. Cover document loading, chunking, retrieval quality, prompt assembly, provider failures, intent labels, command validation, DAC-3D mock flows, and result parsing. Avoid live network calls in normal tests.

## Commit & Pull Request Guidelines
This workspace snapshot does not include `.git` history, so there is no existing repository convention to follow. Use short imperative commit subjects such as `Add retriever metadata schema` or `Implement DAC-3D result parser`. Pull requests should describe the user scenario, changed modules, configuration impact, and screenshots when the UI changes.

## Security & Configuration Tips
Do not commit real API keys, DAC-3D endpoints, or customer inspection data. Keep secrets outside source control. If live DAC-3D data is used for development or testing, sanitize logs and keep persisted artifacts under strict control.
