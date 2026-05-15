"""Lightweight stdio proxy for Claude Desktop.

This script is launched by Claude Desktop via stdio. It proxies all MCP
requests to the deployed HTTP server, so Claude Desktop gets the full
tool set without needing a local DB connection.

Configure Claude Desktop (~/Library/Application Support/Claude/claude_desktop_config.json):

    {
      "mcpServers": {
        "pt-caselaw-dgsi": {
          "command": "/Users/franciscocosta/repos/pt-caselaw-dgsi/.venv312/bin/python3.12",
          "args": ["/Users/franciscocosta/repos/pt-caselaw-dgsi/api/mcp_proxy.py"],
          "env": {
            "PYTHONPATH": "/Users/franciscocosta/repos/pt-caselaw-dgsi",
            "MCP_REMOTE_URL": "https://pt-caselaw-dgsi.onrender.com/mcp/"
          }
        }
      }
    }

auth=None on the transport skips the MCP 2025 OAuth discovery loop
(GET /.well-known/oauth-protected-resource → 404 → fail) for public servers.
"""
from __future__ import annotations

import os
import sys

from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server import create_proxy

# Trailing slash is required — without it FastAPI issues a 307 redirect
# which the MCP client does not follow for POST requests.
REMOTE_URL: str = os.environ.get(
    "MCP_REMOTE_URL",
    "https://pt-caselaw-dgsi.onrender.com/mcp/",
)

# auth=None bypasses the OAuth discovery requests that FastMCP sends by
# default when connecting to a streamable-HTTP server.  Our server is
# public; there is no OAuth provider to discover.
_transport = StreamableHttpTransport(REMOTE_URL, auth=None)
_client = Client(_transport)

proxy: FastMCP = create_proxy(_client, name="PT Caselaw DGSI")

if __name__ == "__main__":
    proxy.run(show_banner=False)
