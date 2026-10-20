#!/usr/bin/env python3
"""FastStrata from a terminal: which models are installed, a chat with one of them, and a benchmark.

    python polystrata.py --list-models
    python polystrata.py --model <name>                chat with it; /model in the chat changes it
    python polystrata.py benchmark --model <name>      speeds, memory and cache figures, as a table and as JSON

A model is one that setup installed (its run config, strata-<name>.json, is beside this file).  One model is in
memory at a time.  With a server already running on the port, --model switches that server to the model; without
one, a server is started here and stopped when the chat or the benchmark ends.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import chat as terminal  # noqa: E402
from serve import models as registry  # noqa: E402

SHORT = "Explain in three sentences why the sky is blue."
DOC = ("docs/GLM.md", "README.md")   # the long prompt is a question about this repository's own text


class Server:
    """The server a command talks to: the one running on the port, or one started here for the model."""

    def __init__(self, entry: dict, port: int, key: str, quiet: bool = False):
        self.base, self.key, self.child = f"http://127.0.0.1:{port}", key, None
        code, health = terminal.api(self.base, "/health", None, key, timeout=5)
        if code == 200 and health.get("service") == "strata":
            code, out = terminal.api(self.base, "/api/models/load", {"model": entry["id"]}, key)
            if code != 200:
                raise SystemExit(f"polystrata: {out.get('error', {}).get('message', 'the server did not load the model')}")
            return
        if code:
            raise SystemExit(f"polystrata: port {port} answers, but not as FastStrata; use --port")
        print(f"starting {entry['id']} on port {port} (a first start takes a minute or two) ...", flush=True)
        out = subprocess.DEVNULL if quiet else None
        self.child = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                                       "--config", entry["config"], "--port", str(port)], stdout=out, cwd=str(ROOT))
        while True:
            if self.child.poll() is not None:
                raise SystemExit(f"polystrata: the server ended before it was ready (exit code {self.child.returncode})")
            code, health = terminal.api(self.base, "/health", None, key, timeout=5)
            if code == 200 and health.get("loaded"):
                return
            time.sleep(1)

    def close(self):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()               # the server ends its engine on the way out
            try:
                self.child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                self.child.kill()


def entry_of(name: str) -> dict:
    models = registry.discover(ROOT)
    e = registry.find(models, name)
    if e is None:
        raise SystemExit(f"polystrata: no model named {name!r}.  Installed: " +
                         (", ".join(m["id"] for m in models) or "none (run setup first)"))
    if e["status"] == "unavailable":
        raise SystemExit(f"polystrata: {e['id']} cannot run: {e['problem']}")
    return e


def port_of(entry: dict, asked: int | None) -> int:
    if asked:
        return asked
    try:
        return int(json.loads(Path(entry["config"]).read_text(encoding="utf-8-sig")).get("port") or 8080)
    except (OSError, ValueError):
        return 8080


def list_models(port: int | None, key: str) -> int:
    models = registry.discover(ROOT)
    if not models:
        print("no model is installed: run setup (START-HERE.bat, or ./setup.sh)")
        return 1
    code, out = terminal.api(f"http://127.0.0.1:{port or 8080}", "/api/models/current", None, key, timeout=3)
    current = (out.get("model") or {}).get("id") if code == 200 and out.get("loaded") else None
    for e in models:
        print(registry.line(e, current))
    if current:
        print(f"\n* in memory now (the server on port {port or 8080})")
    return 0


class Watch(threading.Thread):
    """The server's hardware readings while a request runs: the most VRAM and RAM in use, the mean loads."""

    def __init__(self, base, key):
        super().__init__(daemon=True)
        self.base, self.key, self.stop, self.rows = base, key, threading.Event(), []

    def run(self):
        while not self.stop.wait(0.5):
            code, out = terminal.api(self.base, "/metrics", None, self.key, timeout=5)
            if code == 200 and out.get("hardware"):
                self.rows.append(out["hardware"])

    def result(self) -> dict:
        self.stop.set()
        self.join()

        def of(key, f):
            vals = [r[key] for r in self.rows if isinstance(r.get(key), (int, float))]
            return f(vals) if vals else None
        gib = lambda v: None if v is None else round(v / 2**30, 2)   # noqa: E731
        mean = lambda v: round(sum(v) / len(v), 1)                    # noqa: E731
        return {"vram_peak_gib": gib(of("gpu_mem_used", max)), "ram_peak_gib": gib(of("ram_used", max)),
                "gpu_util_mean": of("gpu_util", mean), "cpu_util_mean": of("cpu", mean)}


def long_prompt(tokens: int, part: float = 1.0) -> str:
    text = "\n\n".join((ROOT / p).read_text(encoding="utf-8") for p in DOC if (ROOT / p).exists())
    text = (text * (1 + tokens * 4 // max(len(text), 1)))[:int(tokens * 4 * part)]   # about four characters a token
    return "Here is part of a manual:\n\n" + text + "\n\nIn three sentences, what does this software do?"


def run_once(srv: Server, name: str, prompt: str, max_tokens: int) -> dict:
    """One request, timed here and by the engine."""
    import urllib.request
    watch = Watch(srv.base, srv.key)
    watch.start()
    body = {"model": "strata", "messages": [{"role": "user", "content": prompt}], "stream": True,
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(srv.base + "/v1/chat/completions", json.dumps(body).encode(), terminal.headers(srv.key))
    t0, first = time.time(), None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: ") and line != "data: [DONE]" and first is None:
                d = json.loads(line[6:]).get("choices", [{}])[0].get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    first = time.time() - t0
    hw = watch.result()
    code, out = terminal.api(srv.base, "/metrics", None, srv.key)
    rec = (out.get("requests") or [{}])[0] if code == 200 else {}
    read = (rec.get("prompt_read") if rec.get("prompt_read") is not None else rec.get("prompt_tokens")) or 0
    return {"prompt": name, "prompt_tokens": rec.get("prompt_total"), "output_tokens": rec.get("output_tokens"),
            "ttft_s": round(first, 3) if first is not None else None,
            "prefill_tok_s": round(read / (rec["prompt_ms"] / 1000), 1) if rec.get("prompt_ms") and read else None,
            "decode_tok_s": rec.get("decode_tok_s"),
            "cache_hit_rate": rec.get("hit_rate"),
            "mtp_offered": rec.get("drafts_offered"), "mtp_accepted": rec.get("drafts_accepted"),
            "mtp_acceptance": round(rec["drafts_accepted"] / rec["drafts_offered"], 3)
            if rec.get("drafts_offered") else None,
            **hw}


def benchmark(a) -> int:
    entry = entry_of(a.model)
    srv = Server(entry, port_of(entry, a.port), a.api_key, quiet=True)
    try:
        runs = [run_once(srv, "short", SHORT, a.max_tokens)]
        for n in a.prompt_tokens:
            if entry["context"] and n + a.max_tokens + 512 > entry["context"]:
                print(f"(skipped the {n}-token prompt: the context is {entry['context']})")
                continue
            # Four characters a token is a guess, and a text with tables comes to more: 24000 is 31,600 tokens
            # for one model and more than the 32,768 of the context for another.  A prompt that is refused is cut.
            for part in (1.0, 0.95, 0.9, 0.85):
                try:
                    runs.append(run_once(srv, f"long-{n}", long_prompt(n, part), a.max_tokens))
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 400 or part == 0.85:
                        raise
                    print(f"(the {n}-token prompt does not fit the context: trying {part - 0.05:.0%} of it)")
    finally:
        srv.close()
    report = {"model": entry["id"], "backend": entry["backend"], "quantization": entry["quantization"],
              "context": entry["context"], "max_tokens": a.max_tokens, "runs": runs}
    cols = [("prompt", "prompt"), ("prompt_tokens", "tokens in"), ("output_tokens", "out"), ("ttft_s", "TTFT s"),
            ("prefill_tok_s", "prefill tok/s"), ("decode_tok_s", "decode tok/s"), ("cache_hit_rate", "cache hits"),
            ("mtp_acceptance", "MTP right"), ("vram_peak_gib", "VRAM GiB"), ("ram_peak_gib", "RAM GiB"),
            ("gpu_util_mean", "GPU %"), ("cpu_util_mean", "CPU %")]
    print(f"\n{entry['id']} ({entry['backend']}, {entry['quantization']}, context {entry['context']})")
    print("  ".join(f"{title:>13s}" for _, title in cols))
    for r in runs:
        print("  ".join(f"{'-' if r.get(k) is None else r[k]!s:>13s}" for k, _ in cols))
    print("\nVRAM and RAM are the machine's use at its highest while the request ran, not the engine's alone.")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"written: {a.json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["chat", "benchmark"], default="chat")
    ap.add_argument("--list-models", action="store_true", help="the installed models, and which is in memory")
    ap.add_argument("--model", help="a model's name, as --list-models shows it (a part of it is enough when only "
                                    "one name has it)")
    ap.add_argument("--port", type=int, help="the server's port (default: the model's own, from its config)")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""))
    ap.add_argument("--think", default="medium", choices=["none", "low", "medium", "high"])
    ap.add_argument("--max-tokens", type=int, help="a reply's length (chat: 4096, benchmark: 256)")
    ap.add_argument("--prompt-tokens", type=lambda s: [int(x) for x in s.split(",")], default=[2000, 8000],
                    help="benchmark: the long prompts' sizes in tokens, roughly (default 2000,8000)")
    ap.add_argument("--json", help="benchmark: also write the figures to this file")
    a = ap.parse_args()
    if a.list_models:
        return list_models(a.port, a.api_key)
    if not a.model:
        ap.error("give --model <name> (see --list-models)")
    if a.command == "benchmark":
        a.max_tokens = a.max_tokens or 256
        return benchmark(a)
    entry = entry_of(a.model)
    srv = Server(entry, port_of(entry, a.port), a.api_key)
    try:
        return terminal.chat(srv.base, a.think, a.max_tokens or 4096, a.api_key)
    finally:
        srv.close()


if __name__ == "__main__":
    sys.exit(main())
