"""Combined entrypoint: FastAPI + MCP over streamable HTTP at /mcp and SSE at /sse.

Run with:
    uvicorn api.app:app

REST API:            http://localhost:8000/docs
MCP streamable HTTP: http://localhost:8000/mcp   (Claude Code, Anthropic API connector)
MCP SSE:             http://localhost:8000/sse   (Claude Desktop)
"""
from fastmcp.server.http import create_sse_app
from fastmcp.utilities.lifespan import combine_lifespans

from api.main import app, lifespan as _api_lifespan
from api.mcp_server import mcp

# Streamable HTTP transport (Claude Code, Anthropic API MCP connector)
_mcp_http_app = mcp.http_app(path="/")

# SSE transport (Claude Desktop)
_mcp_sse_app = create_sse_app(
    server=mcp,
    message_path="/sse/messages",
    sse_path="/sse",
)

app.router.lifespan_context = combine_lifespans(
    _api_lifespan,
    _mcp_http_app.lifespan,
    _mcp_sse_app.lifespan,
)

app.mount("/mcp", _mcp_http_app)

# Add SSE routes directly so they are reachable at /sse and /sse/messages
for route in _mcp_sse_app.routes:
    app.routes.append(route)
