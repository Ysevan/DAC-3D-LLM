"""Controlled DAC tool gateway."""

from tool_gateway.gateway import DAC3DToolGateway, ToolDescriptor, ToolInvocationResult
from tool_gateway.mcp_adapter import (
    default_mcp_prompt_descriptors,
    default_mcp_resource_descriptors,
    to_mcp_capability_manifest,
    to_mcp_tool_descriptor,
    to_mcp_tool_descriptors,
)

__all__ = [
    "DAC3DToolGateway",
    "ToolDescriptor",
    "ToolInvocationResult",
    "default_mcp_prompt_descriptors",
    "default_mcp_resource_descriptors",
    "to_mcp_capability_manifest",
    "to_mcp_tool_descriptor",
    "to_mcp_tool_descriptors",
]
