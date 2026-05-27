"""CLI entrypoint for generating the next Codex handoff document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local DAC-Agent evals and generate docs/generated/codex_handoff_next.md.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Eval category to include. Can be supplied more than once.",
    )
    parser.add_argument(
        "--recent-trace-limit",
        type=int,
        default=8,
        help="Number of recent traces to include after failed-eval traces.",
    )
    parser.add_argument(
        "--output",
        default="docs/generated/codex_handoff_next.md",
        help="Handoff markdown output path. Relative paths are resolved under dac3d_iim_assistant/.",
    )
    parser.add_argument(
        "--rebuild-kb",
        action="store_true",
        help="Rebuild the local knowledge base before running evals.",
    )
    parser.add_argument(
        "--fail-on-eval-failure",
        action="store_true",
        help="Exit with status 1 when the eval run has failures.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from agent_runtime import DAC3DAgentChatAdapter, DAC3DAgentRuntime, LOCAL_VALIDATION_MODEL_NAME
    from app import DAC3DAssistant
    from config import AppConfig

    config = AppConfig.from_env()
    config.agent_model_name = LOCAL_VALIDATION_MODEL_NAME
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = config.base_dir / output_path

    assistant = DAC3DAssistant.create(config=config, rebuild_kb=bool(args.rebuild_kb))
    adapter = DAC3DAgentChatAdapter(DAC3DAgentRuntime(assistant=assistant, config=config))
    result = adapter.generate_codex_handoff(
        categories=[str(item) for item in args.category] or None,
        recent_trace_limit=args.recent_trace_limit,
        output_path=output_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.fail_on_eval_failure and int(result.get("eval_summary", {}).get("failed") or 0) > 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI usage.
    raise SystemExit(main())
