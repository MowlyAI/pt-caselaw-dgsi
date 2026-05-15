"""Quick smoke-test of the five critical HTTP behaviours for MCP connectivity."""
import httpx

BASE = "http://localhost:8001"
INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
               "clientInfo": {"name": "t", "version": "1"}},
}
MCP_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

TESTS = [
    ("GET",  "/mcp",                                    None,   200),
    ("GET",  "/mcp/",                                   None,   200),
    ("POST", "/mcp/",                                   INIT,   200),
    ("POST", "/mcp",                                    INIT,   200),
    ("GET",  "/.well-known/oauth-protected-resource",   None,   404),
]

with httpx.Client(follow_redirects=True, timeout=10) as c:
    ok = True
    for method, path, body, expected in TESTS:
        r = c.request(method, BASE + path, json=body, headers=MCP_HEADERS if body else {})
        status = "✓" if r.status_code == expected else "✗"
        print(f"{status} {method} {path} -> {r.status_code} (expected {expected})")
        if r.status_code != expected:
            ok = False
    print()
    print("ALL PASS" if ok else "FAILURES ABOVE")
