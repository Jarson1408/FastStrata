#!/usr/bin/env python3
"""A tiny terminal chat for a running FastStrata server (start it with run-<model>.bat / run-<model>.sh first, or
let `python polystrata.py --model <name>` start it).

    python chat.py [--port 8080] [--think none|low|medium|high]

Type a message and press Enter.  /image <path> attaches a picture to your next message (when the server was set up
with images), /think <none|low|medium|high> sets how long the model thinks first, /reset starts a new conversation,
/quit leaves.  /model shows the model that answers, /model list the installed ones, /model switch <name> changes to
another (one is in memory at a time), /model info [<name>] says what a model is, /model unload gives its memory
back until the next message.  Standard library only.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request


def headers(key: str = "") -> dict:
    return {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}


def api(base: str, path: str, body=None, key: str = "", timeout: float = 900):
    """A JSON request to the server: (status, answer).  Status 0: the server did not answer."""
    req = urllib.request.Request(base + path, None if body is None else json.dumps(body).encode(), headers(key))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except ValueError:
            return e.code, {"error": {"message": e.reason}}
    except (OSError, ValueError) as e:
        return 0, {"error": {"message": str(e)}}


def stream(url, messages, think, max_tokens, key=""):
    body = {"model": "strata", "messages": messages, "stream": True, "max_tokens": max_tokens,
            "reasoning_effort": think}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers(key))
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])["choices"][0]["delta"]
            yield d.get("reasoning_content") or "", d.get("content") or ""


def describe(e: dict) -> str:
    """A model of GET /api/models in a few lines."""
    size = "-".join(x for x in (e.get("params_total"), "A" + e["params_active"] if e.get("params_active") else None) if x)
    vram = e.get("vram_gb") or {}
    rows = [("name", e["id"]), ("model", e.get("display_name")), ("size", size), ("quantization", e.get("quantization")),
            ("engine", e.get("backend")), ("context", e.get("context")),
            ("images", "yes" if e.get("multimodal") else "no"), ("guesses ahead (MTP)", "yes" if e.get("mtp") else "no"),
            ("VRAM", f"{vram['least']:g} GB at least, {vram['all']:g} GB holds all of it" if vram else None),
            ("file", f"{e['ram_gb']:g} GB, read in place" if e.get("ram_gb") else None),
            ("status", e.get("status") + (f" ({e['problem']})" if e.get("problem") else "")),
            ("config", e.get("config")), ("notes", e.get("notes"))]
    return "\n".join(f"  {k:20s} {v}" for k, v in rows if v not in (None, ""))


def model_command(base: str, words: list[str], key: str = "") -> bool:
    """/model ...  True when another model answers from now on (the conversation starts again)."""
    what = words[0].lower() if words else ""
    if what in ("switch", "load") and len(words) > 1:
        print(f"(switching to {words[1]}: the model in memory is stopped first, then this one is loaded ...)", flush=True)
        code, out = api(base, "/api/models/load", {"model": " ".join(words[1:])}, key)
        if code == 200:
            print(f"(the model is now {out['model']['id']})")
            return True
        print(f"({out.get('error', {}).get('message', 'the switch failed')})")
        if code == 404 and out.get("error", {}).get("models"):
            print("(installed: " + ", ".join(out["error"]["models"]) + ")")
        return False
    if what == "unload":
        code, out = api(base, "/api/models/unload", {}, key)
        print(f"({out.get('status') or out.get('error', {}).get('message')})")
        return False
    code, out = api(base, "/api/models", None, key)
    if code != 200:
        print(f"(the server does not list its models: {out.get('error', {}).get('message', code)})")
        return False
    models = out["models"]
    if what == "list":
        for e in models:
            vram = e.get("vram_gb")
            print(f"  {'*' if e.get('current') else ' '} {e['id']:28s} {e.get('backend') or '':10s} "
                  f"{e.get('quantization') or '':12s} {e['status']}"
                  + (f", VRAM {vram['least']:g}-{vram['all']:g} GB" if vram else "")
                  + (f"  [{e['problem']}]" if e.get("problem") else ""))
        return False
    if what in ("", "info"):
        name = " ".join(words[1:]).lower()
        e = next((e for e in models if (e["id"].lower() == name if name else e.get("current"))), None)
        print(describe(e) if e else "(no such model; /model list shows the names)" if name else "(no model is in use)")
        return False
    print("(usage: /model | /model list | /model switch <name> | /model info [<name>] | /model unload)")
    return False


def chat(base: str, think: str = "medium", max_tokens: int = 4096, key: str = "") -> int:
    url = base + "/v1/chat/completions"
    gray, reset = ("\033[90m", "\033[0m") if sys.stdout.isatty() else ("", "")
    messages, pending = [], []
    print(f"FastStrata chat ({url}).  /image <path> = attach a picture, /think none|low|medium|high (now: {think}), "
          f"/model = the model and how to change it, /reset = new conversation, /quit = leave.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if user in ("/quit", "/exit"):
            return 0
        if user == "/reset":
            messages, pending = [], []
            print("(new conversation)")
            continue
        if user == "/model" or user.startswith("/model "):
            if model_command(base, user.split()[1:], key):
                messages, pending = [], []
                print("(new conversation)")
            continue
        if user.startswith("/think"):
            level = user[6:].strip().lower()
            if level in ("none", "low", "medium", "high"):
                think = level
                print(f"(thinking: {think})")
            else:
                print("(usage: /think none|low|medium|high)")
            continue
        if user.startswith("/image"):
            path = user[6:].strip().strip('"').strip("'")
            if not os.path.isfile(path):
                print(f"(no such file: {path or '?'} - usage: /image <path to a picture>)")
                continue
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            data = base64.b64encode(open(path, "rb").read()).decode()
            pending.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            print(f"(picture attached: {os.path.basename(path)} - now type your question)")
            continue
        if not user:
            continue
        if pending:
            messages.append({"role": "user", "content": [*pending, {"type": "text", "text": user}]})
            pending = []
        else:
            messages.append({"role": "user", "content": user})
        answer, n, t0 = [], 0, time.time()
        print("model> ", end="", flush=True)
        try:
            in_think = False
            for reasoning, content in stream(url, messages, think, max_tokens, key):
                if reasoning:
                    if not in_think:
                        print(gray, end="")
                        in_think = True
                    print(reasoning, end="", flush=True)
                if content:
                    if in_think:
                        print(reset + "\n", end="")
                        in_think = False
                    print(content, end="", flush=True)
                    answer.append(content)
                n += 1
            if in_think:
                print(reset, end="")
        except OSError as e:
            print(f"\n(could not reach the server at {url}: {e})")
            messages.pop()
            continue
        dt = time.time() - t0
        print(f"\n{gray}[{n} chunks in {dt:.1f} s]{reset}")
        messages.append({"role": "assistant", "content": "".join(answer)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--think", default="medium", choices=["none", "low", "medium", "high"],
                    help="how long the model thinks before answering (none = answer directly)")
    ap.add_argument("--no-think", action="store_true", help="same as --think none")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""),
                    help="the server's API key, when it has one (also $STRATA_API_KEY)")
    a = ap.parse_args()
    return chat(f"http://{a.host}:{a.port}", "none" if a.no_think else a.think, a.max_tokens, a.api_key)


if __name__ == "__main__":
    sys.exit(main())
