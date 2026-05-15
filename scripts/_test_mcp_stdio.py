"""Test the MCP server stdio handshake end-to-end."""
import json
import subprocess
import time

proc = subprocess.Popen(
    [".venv312/bin/python3.12", "-m", "api.mcp_server"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd="/Users/franciscocosta/repos/pt-caselaw-dgsi",
)

messages = [
    {"jsonrpc": "2.0", "method": "initialize", "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1"},
    }, "id": 1},
    {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
    {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2},
]

for msg in messages:
    proc.stdin.write((json.dumps(msg) + "\n").encode())
proc.stdin.flush()

time.sleep(10)
proc.terminate()
out, err = proc.communicate(timeout=5)

print("=== stdout ===")
for line in out.decode().splitlines():
    if not line.strip():
        continue
    try:
        d = json.loads(line)
        if "result" in d:
            result = d["result"]
            if "serverInfo" in result:
                print(f"[id={d['id']}] initialize OK — server: {result['serverInfo']}")
            elif "tools" in result:
                names = [t["name"] for t in result["tools"]]
                print(f"[id={d['id']}] tools/list OK — {names}")
            else:
                print(f"[id={d.get('id')}] result: {result}")
        elif "error" in d:
            print(f"[id={d.get('id')}] ERROR: {d['error']}")
    except json.JSONDecodeError:
        print("raw:", line[:120])

print("\n=== stderr (last 5 lines) ===")
for line in err.decode().splitlines()[-5:]:
    print(line)
