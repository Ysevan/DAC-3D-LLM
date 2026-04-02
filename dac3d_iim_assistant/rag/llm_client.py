"""LLM abstraction for the DAC-3D assistant."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any

from config import AppConfig
from integration.result_parser import ParsedInspectionResult
from rag.retriever import RetrievalItem

SENTENCE_PATTERN = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])\s+")


class LLMProviderError(RuntimeError):
    """Normalized provider error raised by the client."""


class LLMConfigurationError(LLMProviderError):
    """Raised when the requested provider is not correctly configured."""


class ProviderAdapter:
    """Small interface used by the provider registry."""

    def generate(
        self,
        prompt: str,
        *,
        task: str,
        question: str,
        retrieval_items: Sequence[RetrievalItem],
        parsed_result: ParsedInspectionResult | None,
        command: dict[str, Any] | None,
    ) -> str:
        raise NotImplementedError

    def stream_generate(
        self,
        prompt: str,
        *,
        task: str,
        question: str,
        retrieval_items: Sequence[RetrievalItem],
        parsed_result: ParsedInspectionResult | None,
        command: dict[str, Any] | None,
    ) -> Iterator[str]:
        yield self.generate(
            prompt,
            task=task,
            question=question,
            retrieval_items=retrieval_items,
            parsed_result=parsed_result,
            command=command,
        )


class MockProviderAdapter(ProviderAdapter):
    """Deterministic grounded provider for offline demos and tests."""

    def generate(
        self,
        prompt: str,
        *,
        task: str,
        question: str,
        retrieval_items: Sequence[RetrievalItem],
        parsed_result: ParsedInspectionResult | None,
        command: dict[str, Any] | None,
    ) -> str:
        del prompt
        items = list(retrieval_items)
        if task == "greeting":
            provider_name = question.split("||", 1)[0].strip() or "unknown provider"
            model_name = question.split("||", 1)[1].strip() if "||" in question else "unknown model"
            return (
                f"我是由 {provider_name} 提供的 {model_name} 模型。"
                "我目前可用于文档问答、操作指导、检测结果解读、扫描命令预览和运行状态查询。"
            )

        if task == "command_clarification":
            missing_fields = list((command or {}).get("missing_fields", []))
            warnings = list((command or {}).get("warnings", []))
            if not missing_fields:
                return "命令已经具备必需字段，可以继续确认并执行。"
            details = "、".join(missing_fields)
            warning_text = f" 风险提示: {'；'.join(warnings)}" if warnings else ""
            return f"要继续生成可执行扫描命令，还缺少这些字段: {details}.{warning_text}"

        if task == "interpretation" and parsed_result is not None:
            return self._interpret_result(question, self._prioritize_items(items, task), parsed_result)

        prioritized_items = self._prioritize_items(items, task)

        evidence = self._select_evidence(question, prioritized_items)
        if not evidence:
            return (
                "我没有足够的 DAC-3D 文档依据来安全回答这个问题。"
                "请补充相关手册、FAQ 或参数说明后再试。"
            )

        sources = self._format_sources(prioritized_items)
        if task == "guidance":
            guidance_steps = self._build_guidance_steps(question, prioritized_items)
            return f"建议按以下顺序操作: {guidance_steps} 来源: {sources}."
        return f"根据检索到的 DAC-3D 文档，{evidence} 来源: {sources}."

    def stream_generate(
        self,
        prompt: str,
        *,
        task: str,
        question: str,
        retrieval_items: Sequence[RetrievalItem],
        parsed_result: ParsedInspectionResult | None,
        command: dict[str, Any] | None,
    ) -> Iterator[str]:
        answer = self.generate(
            prompt,
            task=task,
            question=question,
            retrieval_items=retrieval_items,
            parsed_result=parsed_result,
            command=command,
        )
        for sentence in SENTENCE_PATTERN.split(answer):
            if sentence:
                yield sentence

    def _select_evidence(
        self,
        question: str,
        retrieval_items: Sequence[RetrievalItem],
        sentence_limit: int = 3,
    ) -> str:
        question_tokens = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", question.lower()))
        scored_sentences: list[tuple[float, str]] = []
        for item_index, item in enumerate(retrieval_items):
            del item_index
            sentences = SENTENCE_PATTERN.split(item.text) if item.text else []
            if not sentences:
                sentences = [item.text]
            for sentence in sentences:
                cleaned = sentence.strip()
                if not cleaned:
                    continue
                sentence_tokens = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", cleaned.lower()))
                overlap = len(question_tokens & sentence_tokens)
                score = (overlap * 3.0) + float(item.score)
                scored_sentences.append((score, cleaned))

        scored_sentences.sort(key=lambda pair: (pair[0], len(pair[1])), reverse=True)
        selected: list[str] = []
        for _, sentence in scored_sentences:
            if sentence not in selected:
                selected.append(sentence)
            if len(selected) >= sentence_limit:
                break
        return " ".join(selected)

    def _build_guidance_steps(
        self,
        question: str,
        retrieval_items: Sequence[RetrievalItem],
        step_limit: int = 3,
        sentences_per_step: int = 2,
    ) -> str:
        question_tokens = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", question.lower()))
        selected_steps: list[str] = []
        seen_sections: set[tuple[str, str]] = set()
        for item in retrieval_items:
            section_key = (item.source, item.section)
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)
            sentences = SENTENCE_PATTERN.split(item.text) if item.text else []
            ranked_sentences: list[tuple[float, str]] = []
            for sentence in sentences or [item.text]:
                cleaned = sentence.strip()
                if not cleaned:
                    continue
                sentence_tokens = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", cleaned.lower()))
                overlap = len(question_tokens & sentence_tokens)
                score = (overlap * 3.0) + float(item.score)
                ranked_sentences.append((score, cleaned))
            ranked_sentences.sort(key=lambda pair: (pair[0], len(pair[1])), reverse=True)
            best_sentences: list[str] = []
            for _, sentence in ranked_sentences:
                if sentence not in best_sentences:
                    best_sentences.append(sentence)
                if len(best_sentences) >= sentences_per_step:
                    break
            combined_step = " ".join(best_sentences).strip()
            if combined_step and combined_step not in selected_steps:
                selected_steps.append(combined_step)
            if len(selected_steps) >= step_limit:
                break

        if not selected_steps:
            return self._select_evidence(question, retrieval_items, sentence_limit=2)
        return " ".join(f"{index}. {step}" for index, step in enumerate(selected_steps, start=1))

    def _interpret_result(
        self,
        question: str,
        retrieval_items: Sequence[RetrievalItem],
        parsed_result: ParsedInspectionResult,
    ) -> str:
        del question
        seriousness = "严重" if parsed_result.severity in {"high", "critical"} else "不属于高风险"
        measurements = ", ".join(
            f"{key}={value}" for key, value in parsed_result.measurements.items()
        )
        evidence = self._select_evidence(parsed_result.rule_reason, retrieval_items, sentence_limit=2)
        sources = self._format_sources(retrieval_items)
        threshold_text = parsed_result.threshold_reference or "未提供明确阈值引用"
        return (
            f"最新检测结果判断为{seriousness}。"
            f"缺陷类型为 {parsed_result.defect_type}，位置在 {parsed_result.location}，"
            f"严重度为 {parsed_result.severity}，置信度 {parsed_result.confidence:.2f}。"
            f"测量值: {measurements or '无'}。"
            f"结构化判定依据: {parsed_result.rule_reason} 阈值说明: {threshold_text}。"
            f"{evidence} 来源: {sources}."
        )

    def _prioritize_items(
        self,
        retrieval_items: Sequence[RetrievalItem],
        task: str,
    ) -> list[RetrievalItem]:
        if not retrieval_items:
            return []
        if task == "guidance":
            preferred_types = {"guidance", "manual"}
        elif task == "interpretation":
            preferred_types = {"defect"}
        else:
            return list(retrieval_items)
        prioritized = [item for item in retrieval_items if item.document_type in preferred_types]
        return prioritized or list(retrieval_items)

    def _format_sources(self, retrieval_items: Sequence[RetrievalItem]) -> str:
        ordered_sources: list[str] = []
        for item in retrieval_items:
            descriptor = f"{item.source} ({item.section})"
            if descriptor not in ordered_sources:
                ordered_sources.append(descriptor)
            if len(ordered_sources) >= 6:
                break
        return ", ".join(ordered_sources) if ordered_sources else "无来源"


class AnthropicProviderAdapter(ProviderAdapter):
    """Anthropic SDK adapter."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _create_client(self) -> Any:
        if not self.config.api_key:
            raise LLMConfigurationError("Anthropic provider requires an API key.")
        try:
            from anthropic import Anthropic
        except Exception as exc:  # pragma: no cover - optional dependency
            raise LLMConfigurationError(
                "Anthropic SDK is not installed. Install the project dependencies first."
            ) from exc

        client_kwargs: dict[str, Any] = {"api_key": self.config.api_key}
        if self.config.api_base_url:
            client_kwargs["base_url"] = self.config.api_base_url
        return Anthropic(**client_kwargs)

    def generate(
        self,
        prompt: str,
        *,
        task: str,
        question: str,
        retrieval_items: Sequence[RetrievalItem],
        parsed_result: ParsedInspectionResult | None,
        command: dict[str, Any] | None,
    ) -> str:
        del task, question, retrieval_items, parsed_result, command
        client = self._create_client()
        try:
            response = client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_generation_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # pragma: no cover - network guarded
            raise LLMProviderError(f"Anthropic request failed: {exc}") from exc

        blocks = getattr(response, "content", [])
        text_parts = [getattr(block, "text", "") for block in blocks]
        answer = "".join(text_parts).strip()
        if not answer:
            raise LLMProviderError("Anthropic provider returned an empty response.")
        return answer

    def stream_generate(
        self,
        prompt: str,
        *,
        task: str,
        question: str,
        retrieval_items: Sequence[RetrievalItem],
        parsed_result: ParsedInspectionResult | None,
        command: dict[str, Any] | None,
    ) -> Iterator[str]:
        del task, question, retrieval_items, parsed_result, command
        client = self._create_client()
        try:
            with client.messages.stream(
                model=self.config.model_name,
                max_tokens=self.config.max_generation_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield str(text)
        except Exception as exc:  # pragma: no cover - network guarded
            raise LLMProviderError(f"Anthropic streaming request failed: {exc}") from exc


class UnsupportedProviderAdapter(ProviderAdapter):
    """Placeholder for providers that are outside this delivery scope."""

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    def generate(
        self,
        prompt: str,
        *,
        task: str,
        question: str,
        retrieval_items: Sequence[RetrievalItem],
        parsed_result: ParsedInspectionResult | None,
        command: dict[str, Any] | None,
    ) -> str:
        del prompt, task, question, retrieval_items, parsed_result, command
        raise LLMConfigurationError(
            f"Provider '{self.provider_name}' is reserved for future work and is not implemented in this workspace."
        )


class LLMClient:
    """Wrap grounded answer generation behind a provider-like interface."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._providers: dict[str, ProviderAdapter] = {
            "mock": MockProviderAdapter(),
            "anthropic": AnthropicProviderAdapter(config),
            "qwen": UnsupportedProviderAdapter("qwen"),
            "ernie": UnsupportedProviderAdapter("ernie"),
        }

    def generate(
        self,
        prompt: str,
        *,
        task: str,
        question: str,
        retrieval_items: Sequence[RetrievalItem] | None = None,
        parsed_result: ParsedInspectionResult | None = None,
        command: dict[str, Any] | None = None,
    ) -> str:
        """Generate a grounded response for the requested task."""
        items = list(retrieval_items or [])
        provider = self._select_provider()
        error: Exception | None = None
        for _ in range(self.config.retry_count + 1):
            try:
                return provider.generate(
                    prompt,
                    task=task,
                    question=question,
                    retrieval_items=items,
                    parsed_result=parsed_result,
                    command=command,
                )
            except LLMProviderError as exc:
                error = exc
            except Exception as exc:  # pragma: no cover - defensive normalization
                error = LLMProviderError(str(exc))
        raise LLMProviderError(str(error or "Unknown LLM provider error."))

    def stream_generate(
        self,
        prompt: str,
        *,
        task: str,
        question: str,
        retrieval_items: Sequence[RetrievalItem] | None = None,
        parsed_result: ParsedInspectionResult | None = None,
        command: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield response chunks for providers that support streaming."""
        provider = self._select_provider()
        try:
            yield from provider.stream_generate(
                prompt,
                task=task,
                question=question,
                retrieval_items=list(retrieval_items or []),
                parsed_result=parsed_result,
                command=command,
            )
        except LLMProviderError:
            raise
        except Exception as exc:  # pragma: no cover - defensive normalization
            raise LLMProviderError(str(exc)) from exc

    def _select_provider(self) -> ProviderAdapter:
        provider_name = self.config.provider.strip().lower()
        provider = self._providers.get(provider_name)
        if provider is None:
            raise LLMConfigurationError(f"Unknown provider: {self.config.provider}")
        return provider
