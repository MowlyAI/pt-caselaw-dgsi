"""Combined entrypoint: FastAPI + MCP over streamable HTTP at /mcp.

Run with:
    uvicorn api.app:app

REST API:     http://localhost:8000/docs
MCP endpoint: http://localhost:8000/mcp/
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.main import app, lifespan as _api_lifespan
from api.mcp_server import mcp

_mcp_app = mcp.http_app(path="/")


@asynccontextmanager
async def _combined_lifespan(fastapi_app: FastAPI):
    async with _api_lifespan(fastapi_app):
        async with _mcp_app.lifespan(_mcp_app):
            yield


app.router.lifespan_context = _combined_lifespan
app.mount("/mcp", _mcp_app)
