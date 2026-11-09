"""MCP access to the deterministic tools: a stdio server, a client, and the gateway the agents call.

The gateway's transport is set by ONEVIEW_TOOL_TRANSPORT: "direct" (default, in-process calls) or "mcp" (the same
functions behind an MCP stdio server). The tool logic lives in src/tools/ and is not changed by either transport."""
