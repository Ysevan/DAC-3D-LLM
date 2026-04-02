"""Intent classification compatibility layer for the DAC-3D assistant."""

from __future__ import annotations

from dataclasses import dataclass, field

from intent.parser import IntentParser
from intent.schemas import ParsedIntent


@dataclass(slots=True)
class IntentResult:
    """Legacy intent label plus confidence and extracted hints."""

    label: str
    confidence: float
    hints: dict[str, object] = field(default_factory=dict)
    parsed_intent: ParsedIntent | None = None


class IntentClassifier:
    """Classify incoming user requests into DAC-3D flows."""

    def __init__(self, parser: IntentParser | None = None) -> None:
        self.parser = parser or IntentParser()

    def classify(self, text: str) -> IntentResult:
        """Return the predicted intent for the input."""
        parsed = self.parser.parse(text)
        return IntentResult(
            label=parsed.intent,
            confidence=parsed.confidence,
            hints=dict(parsed.hints),
            parsed_intent=parsed,
        )
