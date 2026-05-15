#!/usr/bin/env python3
"""Simulate how Claude Desktop launches the MCP server and test the stdio handshake."""
import json
import os
import subprocess
import sys
import threading
import time

REPO = "/Users/franciscocosta/repos/pt-caselaw-dgsi"
PYTHON = f"{REPO}/.venv312/bin/python3.12"

env = dict(os.environ)
env["PYTHONPATH"] = REPO

proc = subprocess.Popen(
    [PYTHON, "-m", "api.mcp_server"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=os.path.expanduser("~"),  # simulate Claude Desktop launching from home dir
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

# MCP initialize request
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

proc.stdin.write(req.encode())  # send IMMEDIATELY — no sleep, just like Claude Desktop does
proc.stdin.flush()
print(f"[+] Sent initialize immediately, waiting for response...", flush=True)

time.sleep(10)
proc.kill()
proc.wait()

print("\n--- STDOUT ---")
for line in lines_out:
    print(line)

print("\n--- STDERR ---")
for line in lines_err:
    print(line)

print(f"\n[rc={proc.returncode}]")
