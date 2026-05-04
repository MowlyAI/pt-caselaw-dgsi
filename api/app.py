"""Combined entrypoint: FastAPI + MCP SSE mounted at /mcp.

Run with:
    uvicorn api.app:app

The REST API is at http://localhost:8000/ (docs: /docs).
The MCP SSE endpoint is at http://localhost:8000/mcp/sse.
"""
from api.main import app
from api.mcp_server import mcp

app.mount("/mcp", mcp.sse_app())
