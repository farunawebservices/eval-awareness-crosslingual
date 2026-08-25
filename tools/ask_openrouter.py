#!/usr/bin/env python3
import json
import os
import sys
import urllib.request

prompt = sys.stdin.read()
if not prompt.strip():
    raise SystemExit("Pipe a prompt into this script.")

key = os.environ["OPENROUTER_API_KEY"]
payload = {
    "model": "anthropic/claude-opus-5",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a careful research coding assistant. Do not claim to have "
                "run commands or edited files. Return only analysis, exact commands, "
                "or unified diffs that the user can inspect and apply."
            ),
        },
        {"role": "user", "content": prompt},
    ],
    "temperature": 0.2,
}

request = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    },
)

with urllib.request.urlopen(request, timeout=180) as response:
    data = json.load(response)

print(data["choices"][0]["message"]["content"])
