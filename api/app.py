"""Combined entrypoint: FastAPI + MCP over streamable HTTP at /mcp.

Run with:
    uvicorn api.app:app

REST API:     http://localhost:8000/docs
MCP endpoint: http://localhost:8000/mcp
"""
from fastmcp.utilities.lifespan import combine_lifespans

from api.main import app, lifespan as _api_lifespan
from api.mcp_server import mcp

_mcp_app = mcp.http_app(path="/")

app.router.lifespan_context = combine_lifespans(_api_lifespan, _mcp_app.lifespan)
app.mount("/mcp", _mcp_app)
