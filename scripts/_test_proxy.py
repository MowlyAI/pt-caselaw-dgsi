#!/usr/bin/env python3
"""Test the MCP proxy by launching it against the local server and doing the MCP handshake."""
import json
import os
import subprocess
import sys
import threading
import time

REPO = "/Users/franciscocosta/repos/pt-caselaw-dgsi"
PYTHON = f"{REPO}/.venv312/bin/python3.12"
PROXY = f"{REPO}/api/mcp_proxy.py"
REMOTE_URL = os.environ.get("MCP_REMOTE_URL", "http://localhost:8000/mcp/")

env = dict(os.environ)
env["MCP_REMOTE_URL"] = REMOTE_URL
env["PYTHONPATH"] = REPO

proc = subprocess.Popen(
    [PYTHON, PROXY],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=os.path.expanduser("~"),
    env=env,
)

lines_out: list[str] = []
lines_err: list[str] = []


def read_out() -> None:
    for line in proc.stdout:
        lines_out.append(line.decode(errors="replace").rstrip())


def read_err() -> None:
    for line in proc.stderr:
        lines_err.append(line.decode(errors="replace").rstrip())


t1 = threading.Thread(target=read_out, daemon=True)
t2 = threading.Thread(target=read_err, daemon=True)
t1.start()
t2.start()

req = (
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "Claude", "version": "1.0"},
            },
        }
    )
    + "\n"
)

proc.stdin.write(req.encode())
proc.stdin.flush()
print(f"[+] Sent initialize to proxy → {REMOTE_URL}", flush=True)

time.sleep(8)
proc.kill()
proc.wait()

print("\n--- STDOUT ---")
for line in lines_out:
    print(line)

print("\n--- STDERR ---")
for line in lines_err:
    print(line)

print(f"\n[rc={proc.returncode}]")
