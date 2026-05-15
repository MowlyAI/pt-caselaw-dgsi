"""Combined entrypoint: FastAPI + MCP over streamable HTTP at /mcp and SSE at /sse.

Run with:
    uvicorn api.app:app

REST API:            http://localhost:8000/docs
MCP streamable HTTP: http://localhost:8000/mcp   (Claude Code, Anthropic API connector)
MCP SSE:             http://localhost:8000/sse   (Claude Desktop)
"""
import json
import logging

from fastmcp.server.http import create_sse_app
from fastmcp.utilities.lifespan import combine_lifespans
from starlette.types import ASGIApp, Receive, Scope, Send

from api.main import app, lifespan as _api_lifespan
from api.mcp_server import mcp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure-ASGI MCP middleware  (replaces BaseHTTPMiddleware to avoid SSE buffering)
#
# Responsibility 1 — GET /mcp or /mcp/ → 200 JSON probe reply
#   MCP clients probe the endpoint with a plain GET before opening a session.
#   FastMCP's streamable-HTTP handler returns 406 for plain GETs, so we
#   answer here before the request ever reaches the mounted sub-app.
#
# Responsibility 2 — GET /.well-known/oauth-protected-resource[/*] → 200 PRM
#   RFC 9728 "Protected Resource Metadata" with no authorization_servers tells
#   MCP clients this is a PUBLIC resource — no OAuth required.
#   Returning 404 causes MCP 2025-03-26 clients to fall through to the legacy
#   path: /.well-known/oauth-authorization-server → /register, all of which
#   also return 404, making the client report an auth error and abort.
#   A 200 PRM with an empty (or absent) authorization_servers list stops
#   compliant clients at discovery step 1.
#
# Responsibility 3 — pass through everything else, logging the status code.
# ---------------------------------------------------------------------------

_PRM_PREFIX = "/.well-known/oauth-protected-resource"
_MCP_PROBE_BODY = json.dumps({
    "server": "PT Caselaw DGSI",
    "protocol": "MCP",
    "auth": "none",
}).encode()


async def _send_json(send: Send, status: int, body: bytes) -> None:
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"cache-control", b"no-store"),
            (b"access-control-allow-origin", b"*"),
        ],
    })
    await send({"type": "http.response.body", "body": body, "more_body": False})


class _MCPMiddleware:
    """Pure ASGI middleware — no BaseHTTPMiddleware buffering of SSE streams."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        method: str = scope.get("method", "")

        # ── 1. MCP probe (GET) or path normalisation (POST) ─────────────────
        if path == "/mcp" or path == "/mcp/":
            if method == "GET":
                await _send_json(send, 200, _MCP_PROBE_BODY)
                return
            # For POST/DELETE rewrite /mcp → /mcp/ to avoid Starlette's 307
            if path == "/mcp":
                scope = dict(scope)
                scope["path"] = "/mcp/"
                scope["raw_path"] = b"/mcp/"

        # ── 2. OAuth PRM — list ourselves as the authorization server ─────────
        if method == "GET" and (
            path == _PRM_PREFIX or path.startswith(_PRM_PREFIX + "/")
        ):
            headers_dict = dict(scope.get("headers", []))
            host = headers_dict.get(b"host", b"localhost").decode()
            scheme = scope.get("scheme", "https")
            base = f"{scheme}://{host}"
            body = json.dumps({
                "resource": f"{base}/mcp/",
                "authorization_servers": [base],
            }).encode()
            await _send_json(send, 200, body)
            return

        # ── 3. Pass through with response-code logging ──────────────────────
        async def _logged_send(message: dict) -> None:
            if message.get("type") == "http.response.start":
                logger.info("response %s %s → %d", method, path, message["status"])
            await send(message)

        await self.app(scope, receive, _logged_send)


app.add_middleware(_MCPMiddleware)

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
