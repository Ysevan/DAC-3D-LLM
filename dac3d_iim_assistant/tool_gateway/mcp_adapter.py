"""Minimal MCP-compatible descriptor adapter for DAC Tool Gateway."""

from __future__ import annotations

from typing import Any

from tool_gateway.gateway import ToolDescriptor


def to_mcp_tool_descriptor(descriptor: ToolDescriptor | dict[str, Any]) -> dict[str, Any]:
    """Map one internal tool descriptor to a lightweight MCP-style descriptor."""
    payload = descriptor.to_dict() if isinstance(descriptor, ToolDescriptor) else dict(descriptor)
    annotations = dict(payload.get("annotations") or {})
    return {
        "name": str(payload.get("name") or ""),
        "description": str(payload.get("description") or ""),
        "inputSchema": payload.get("input_schema") or {},
        "outputSchema": payload.get("output_schema") or {},
        "annotations": {
            "readOnlyHint": annotations.get("readOnlyHint", payload.get("readOnlyHint") is True),
            "destructiveHint": annotations.get("destructiveHint", payload.get("destructiveHint") is True),
            "idempotentHint": annotations.get("idempotentHint", payload.get("idempotentHint") is True),
            "openWorldHint": annotations.get("openWorldHint", payload.get("openWorldHint") is True),
        },
        "x-dac3d": {
            "risk_level": payload.get("risk_level"),
            "requires_confirmation": payload.get("requires_confirmation") is True,
            "allowed_scopes": list(payload.get("allowed_scopes") or []),
            "enforcement": "PolicyEngine+SafetyGuard",
        },
    }


def to_mcp_tool_descriptors(descriptors: list[ToolDescriptor] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map internal DAC Tool Gateway descriptors to MCP-style descriptors."""
    return [to_mcp_tool_descriptor(descriptor) for descriptor in descriptors]


def default_mcp_resource_descriptors() -> list[dict[str, Any]]:
    """Return read-only DAC-Agent resource descriptors for future MCP clients."""
    return [
        {
            "uri": "dac3d://runtime/status",
            "name": "Current DAC-3D runtime status",
            "description": "Latest normalized DAC-3D state, progress, and runtime message.",
            "mimeType": "application/json",
            "annotations": {"readOnlyHint": True},
            "x-dac3d": {"source": "read_dac_status", "layer": "runtime_state"},
        },
        {
            "uri": "dac3d://results/latest",
            "name": "Latest DAC-3D inspection result",
            "description": "Most recent inspection result summary exposed through the controlled adapter.",
            "mimeType": "application/json",
            "annotations": {"readOnlyHint": True},
            "x-dac3d": {"source": "read_latest_result", "layer": "inspection_result"},
        },
        {
            "uri": "dac3d://memory/profile",
            "name": "DAC-Agent memory profile",
            "description": "Conversation memory profile, curated Markdown memory, and searchable indexes.",
            "mimeType": "application/json",
            "annotations": {"readOnlyHint": True},
            "x-dac3d": {"source": "conversation_memory_profile", "layer": "memory_os"},
        },
        {
            "uri": "dac3d://skills/catalog",
            "name": "DAC-Agent skill catalog",
            "description": "Discoverable local DAC-Agent skills and progressive-disclosure metadata.",
            "mimeType": "application/json",
            "annotations": {"readOnlyHint": True},
            "x-dac3d": {"source": "dac_skill_list", "layer": "skill_system"},
        },
    ]


def default_mcp_prompt_descriptors() -> list[dict[str, Any]]:
    """Return reusable prompt descriptors for future MCP/Agents SDK orchestration."""
    return [
        {
            "name": "dac3d_status_question",
            "description": "Ask for the current DAC-3D runtime state and progress.",
            "arguments": [
                {
                    "name": "question",
                    "description": "Operator's natural-language status question.",
                    "required": True,
                }
            ],
        },
        {
            "name": "dac3d_command_preview",
            "description": "Turn an operation request into a structured DAC-3D command preview.",
            "arguments": [
                {
                    "name": "instruction",
                    "description": "Natural-language operation instruction.",
                    "required": True,
                }
            ],
        },
        {
            "name": "dac3d_result_explanation",
            "description": "Explain the latest or selected DAC-3D inspection result.",
            "arguments": [
                {
                    "name": "sample_position",
                    "description": "Optional sample position to inspect.",
                    "required": False,
                }
            ],
        },
        {
            "name": "dac_agent_memory_or_skill_update",
            "description": "Propose a memory note or skill patch from repeated operator workflow experience.",
            "arguments": [
                {
                    "name": "observation",
                    "description": "Workflow observation that may become a memory or skill patch.",
                    "required": True,
                }
            ],
        },
    ]


def to_mcp_capability_manifest(
    descriptors: list[ToolDescriptor] | list[dict[str, Any]],
    *,
    gateway_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a lightweight MCP-compatible capability manifest.

    This is intentionally an adapter payload, not a network MCP server. It lets
    FastAPI, CLI, and future orchestration layers discover the same tool,
    resource, prompt, and deployment metadata without replacing Tool Gateway.
    """
    gateway_manifest = gateway_manifest or {}
    allowed_dirs = list(gateway_manifest.get("allowed_dirs") or [])
    tools = to_mcp_tool_descriptors(descriptors)
    resources = default_mcp_resource_descriptors()
    prompts = default_mcp_prompt_descriptors()
    return {
        "name": "dac-agent-runtime",
        "description": "MCP-compatible capability manifest for DAC-3D Agent Runtime.",
        "protocol": {
            "style": "mcp-compatible",
            "server": "adapter_manifest_only",
            "transport": "none",
        },
        "capabilities": {
            "tools": {"count": len(tools), "listChanged": False},
            "resources": {"count": len(resources), "listChanged": False},
            "prompts": {"count": len(prompts), "listChanged": False},
        },
        "tools": tools,
        "resources": resources,
        "prompts": prompts,
        "roots": [
            {
                "uri": root,
                "name": f"DAC allowed root {index + 1}",
                "description": "Configured DAC-Agent local root visible to path-sensitive tools.",
            }
            for index, root in enumerate(allowed_dirs)
        ],
        "deployment_modes": [
            "local_fastapi",
            "cli",
            "future_mcp_server",
            "future_agents_sdk_orchestration",
        ],
        "x-dac3d": {
            "backend": gateway_manifest.get("backend", "dac_tool_gateway"),
            "tool_gateway_workflow": gateway_manifest.get("workflow"),
            "adapter_scope": "discovery_and_orchestration_metadata",
            "internal_gateway_remains_authoritative": True,
        },
    }
