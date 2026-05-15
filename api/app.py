"""Combined entrypoint: FastAPI + MCP over streamable HTTP at /mcp and SSE at /sse.

Run with:
    uvicorn api.app:app

REST API:            http://localhost:8000/docs
MCP streamable HTTP: http://localhost:8000/mcp   (Claude Code, Anthropic API connector)
MCP SSE:             http://localhost:8000/sse   (Claude Desktop)
"""
import json

from fastmcp.server.http import create_sse_app
from fastmcp.utilities.lifespan import combine_lifespans
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.main import app, lifespan as _api_lifespan
from api.mcp_server import mcp

# ---------------------------------------------------------------------------
# MCP probe middleware
# Claude and MCP clients send a plain GET to /mcp before opening a session.
# The FastMCP streamable-HTTP handler returns 406 for GETs without the SSE
# Accept header.  Using a @app.get("/mcp") route to fix this causes FastAPI
# to return 405 for POST /mcp (path matches but method not allowed).
# A middleware intercepts only GET at /mcp paths and returns 200 {"auth":"none"}.
# All other methods (POST, DELETE) pass straight through to the mounted app.
# /.well-known/ endpoints intentionally return 404 — that is the RFC 9728
# signal for "no OAuth required".  Returning 200 from those would start the
# full OAuth registration flow in any compliant client.
# ---------------------------------------------------------------------------
_MCP_PROBE_PATHS = {"/mcp", "/mcp/"}
_MCP_PROBE_BODY = json.dumps({"server": "PT Caselaw DGSI", "protocol": "MCP", "auth": "none"}).encode()


class _MCPProbeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "GET" and request.url.path in _MCP_PROBE_PATHS:
            return Response(
                content=_MCP_PROBE_BODY,
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        return await call_next(request)


app.add_middleware(_MCPProbeMiddleware)

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
