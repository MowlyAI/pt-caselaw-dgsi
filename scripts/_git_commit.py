"""Helper: stage and commit pending changes."""
import subprocess, sys

REPO = "/Users/franciscocosta/repos/pt-caselaw-dgsi"

def run(args, **kw):
    r = subprocess.run(args, cwd=REPO, capture_output=True, text=True, **kw)
    print(" ".join(args))
    if r.stdout: print(r.stdout.rstrip())
    if r.stderr: print(r.stderr.rstrip(), file=sys.stderr)
    return r

status = run(["git", "status", "--short"])
if not status.stdout.strip():
    print("Nothing to commit.")
    sys.exit(0)

run(["git", "add", "api/app.py", "api/main.py", ".env.local"])
run(["git", "commit", "-m",
     "fix: middleware for MCP GET probe; fix .env.local duplicate key\n\n"
     "- Replace @app.get('/mcp') routes (which returned 405 for POST)\n"
     "  with _MCPProbeMiddleware in app.py. The middleware intercepts\n"
     "  only GET /mcp and GET /mcp/ and returns 200 {auth:none}.\n"
     "  POST /mcp and POST /mcp/ pass straight through to the FastMCP\n"
     "  mounted app unchanged. This eliminates both the 406 (plain GET)\n"
     "  and the 405 (POST blocked by a GET-only FastAPI route).\n"
     "- Remove duplicate OPENROUTER_API_KEY placeholder from .env.local.\n"
     "  python-dotenv uses first-seen value, so the placeholder\n"
     "  'your_openrouter_api_key_here' was shadowing the real key,\n"
     "  causing embed_query to receive a 200 response with no 'data'\n"
     "  key and raising KeyError on every search call."
])
run(["git", "push", "origin", "main"])
run(["git", "log", "--oneline", "-4"])
