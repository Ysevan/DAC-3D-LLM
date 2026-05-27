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
