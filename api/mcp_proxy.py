"""Lightweight stdio proxy for Claude Desktop.

This script is launched by Claude Desktop via stdio. It proxies all MCP
requests to the deployed HTTP server, so Claude Desktop gets the full
tool set without needing a local DB connection.

Configure Claude Desktop (~/.config/claude/claude_desktop_config.json or
~/Library/Application Support/Claude/claude_desktop_config.json):

    {
      "mcpServers": {
        "pt-caselaw-dgsi": {
          "command": "/Users/franciscocosta/repos/pt-caselaw-dgsi/.venv312/bin/python3.12",
          "args": ["/Users/franciscocosta/repos/pt-caselaw-dgsi/api/mcp_proxy.py"],
          "env": {
            "MCP_REMOTE_URL": "https://YOUR-SERVER.onrender.com/mcp/"
          }
        }
      }
    }

Set MCP_REMOTE_URL to the deployed server's streamable-HTTP endpoint.
"""
from __future__ import annotations

import os
import sys

from fastmcp import FastMCP
from fastmcp.server import create_proxy

# The deployed server URL — override via env var or edit the constant below.
# Streamable-HTTP endpoint:  https://your-server/mcp/
# SSE endpoint (fallback):   https://your-server/sse
REMOTE_URL: str = os.environ.get(
    "MCP_REMOTE_URL",
    "https://REPLACE-WITH-YOUR-SERVER.onrender.com/mcp/",
)

if "REPLACE-WITH-YOUR-SERVER" in REMOTE_URL:
    print(
        "ERROR: MCP_REMOTE_URL is not set. "
        "Set the MCP_REMOTE_URL env var in claude_desktop_config.json.",
        file=sys.stderr,
    )
    sys.exit(1)

proxy: FastMCP = create_proxy(REMOTE_URL, name="PT Caselaw DGSI")

if __name__ == "__main__":
    proxy.run(show_banner=False)
