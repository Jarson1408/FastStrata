"""serve/test_gateway_endpoints.py - llama.cpp's routes the FastStrata Gateway sends (POST /completion, /v1/completions,
/tokenize, /v1/tokenize), against the mock engine (no GPU, no pack): the answer's shape, the stream, the context
check, the refusal of other sites' pages, an engine error after the stream began, the API monitor; and that the
three copies of this code (serve/server.py, tools/apply_gateway_endpoints.py, patches/gateway_completion_endpoints.patch)
agree, since the tool and the patch put it back after an upstream merge.

    python -m unittest serve.test_gateway_endpoints -v
"""
from __future__ import annotations

import ast
import contextlib
import http.client
import importlib.util
import inspect
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import CTX_SLACK, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
ANSWER = "Hello from the mock engine."
TOKENS = len(ANSWER) + 1                         # one token a byte, and the end-of-turn token
EVIL = "http://evil.example.com"


def load_tool():
    spec = importlib.util.spec_from_file_location("apply_gateway_endpoints", ROOT / "tools/apply_gateway_endpoints.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def server_text() -> str:
    return (ROOT / "serve/server.py").read_bytes().decode("utf-8").replace("\r\n", "\n")


class CountingEngine(MockEngine):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class ErrEngine(MockEngine):
    """The engine's ERR line (or #879's non-finite logits) after a few tokens: StrataEngine raises a ValueError."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if i == 3:
                raise ValueError("verify: layer 31 never rang (an illegal memory access was encountered)")
            yield t


class Endpoints(unittest.TestCase):
    engine_class = CountingEngine

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = self.engine_class(self.tok, ANSWER, max_context=CTX)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body, headers=None):
        """(status, the body as bytes, what the server printed)"""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                c.request("POST", path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
                r = c.getresponse()
                raw = r.read()
            return r.status, raw, out.getvalue()
        finally:
            c.close()

    def events(self, raw: bytes) -> list:
        """The `data:` payloads of an SSE body, [DONE] as the string."""
        out = []
        for line in raw.decode().split("\n"):
            if line.startswith("data: "):
                payload = line[6:]
                out.append(payload if payload == "[DONE]" else json.loads(payload))
        return out


class Answers(Endpoints):
    def test_not_streamed(self):
        for path in ("/completion", "/v1/completions"):
            code, raw, _ = self.post(path, {"prompt": "hi", "n_predict": 100})
            self.assertEqual(code, 200, path)
            body = json.loads(raw)
            self.assertEqual(body["content"], ANSWER, path)
            self.assertIs(body["stop"], True)
            self.assertIs(body["stopped_limit"], False)
            self.assertEqual(body["tokens_predicted"], TOKENS)
            self.assertEqual(body["tokens_evaluated"], len(self.tok.encode("hi", parse_special=True)))

    def test_limit(self):
        code, raw, _ = self.post("/completion", {"prompt": "hi", "n_predict": 5})
        body = json.loads(raw)
        self.assertEqual((code, body["content"], body["stopped_limit"], body["tokens_predicted"]),
                         (200, ANSWER[:5], True, 5))

    def test_streamed(self):
        code, raw, _ = self.post("/completion", {"prompt": "hi", "n_predict": 100, "stream": True})
        self.assertEqual(code, 200)
        self.assertNotIn(b"HTTP/1", raw)
        events = self.events(raw)
        self.assertEqual(events[-1], "[DONE]")
        last = events[-2]
        self.assertIs(last["stop"], True)
        self.assertEqual((last["content"], last["stopped_limit"], last["tokens_predicted"]),
                         ("", False, TOKENS))
        pieces = events[:-2]
        self.assertTrue(pieces and all(e["stop"] is False for e in pieces))
        self.assertEqual("".join(e["content"] for e in pieces), ANSWER)

    def test_prompt_as_token_ids(self):
        ids = self.tok.encode("hello there", parse_special=True)
        code, raw, _ = self.post("/completion", {"prompt": ids, "n_predict": 10})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)["tokens_evaluated"], len(ids))
        self.assertEqual(self.engine.last_prompt[:len(ids)], ids)

    def test_bad_prompt_is_a_400(self):
        code, raw, _ = self.post("/completion", {"prompt": {"text": "hi"}, "n_predict": 10})
        self.assertEqual(code, 400)
        self.assertIn("prompt must be a string or list", json.loads(raw)["error"]["message"])

    def test_more_than_the_context_is_refused(self):
        prompt = "x" * (CTX - CTX_SLACK - 10)
        code, raw, _ = self.post("/completion", {"prompt": prompt, "n_predict": 100})
        self.assertEqual(code, 400)
        self.assertIn("exceeds context window", json.loads(raw)["error"]["message"])
        self.assertEqual(self.engine.calls, 0)
        code, raw, _ = self.post("/completion", {"prompt": prompt})     # no n_predict: what the context leaves
        self.assertEqual(code, 200)
        self.assertLessEqual(json.loads(raw)["tokens_predicted"], 10)

    def test_tokenize(self):
        for path in ("/tokenize", "/v1/tokenize"):
            code, raw, _ = self.post(path, {"content": "abc <|im_end|>"})
            self.assertEqual(code, 200, path)
            self.assertEqual(json.loads(raw)["tokens"], self.tok.encode("abc <|im_end|>", parse_special=True))
        code, raw, _ = self.post("/tokenize", {"content": 5})
        self.assertEqual(code, 400)

    def test_api_monitor_has_output_tokens_and_speed(self):
        self.svc.api_monitor = True
        for stream in (False, True):
            code, _, _ = self.post("/completion", {"prompt": "hi", "n_predict": 100, "stream": stream})
            self.assertEqual(code, 200)
            record = self.svc.request_records(self.svc.request_records()[0]["id"])
            self.assertEqual(record["path"], "/completion")
            self.assertEqual(record["output"], ANSWER, stream)
            self.assertEqual(record["usage"]["completion_tokens"], TOKENS)
            self.assertEqual(record["usage"]["prompt_tokens"], len(self.tok.encode("hi", parse_special=True)))
            self.assertEqual(record["state"], "completed")


class OtherSites(Endpoints):
    """A web page of another site may not run the model through these routes without an API key (B-2026-01)."""

    def test_foreign_page_is_refused(self):
        for path in ("/completion", "/v1/completions", "/tokenize", "/v1/tokenize"):
            for ctype in ("text/plain", "application/json"):
                body = {"prompt": "hi", "n_predict": 100000, "content": "hi"}
                code, raw, log = self.post(path, body, {"Content-Type": ctype, "Origin": EVIL})
                self.assertEqual(code, 403, (path, ctype))
                self.assertIn("api_key", json.loads(raw)["error"]["message"])
                self.assertIn("evil.example.com", log)
        self.assertEqual(self.engine.calls, 0)

    def test_the_gateway_and_own_pages_pass(self):
        # the Gateway: application/json, no Origin; FastStrata's own page and a local app: JSON with their Origin
        for headers in ({}, {"Origin": f"http://127.0.0.1:{self.port}"}, {"Origin": "http://localhost:3000"}):
            self.assertEqual(self.post("/completion", {"prompt": "hi", "n_predict": 4}, headers)[0], 200, headers)
            self.assertEqual(self.post("/tokenize", {"content": "hi"}, headers)[0], 200, headers)
        self.assertEqual(self.post("/completion", {"prompt": "hi", "n_predict": 4},
                                   {"Content-Type": "text/plain"})[0], 200)       # curl -d without -H

    def test_a_key_decides_instead(self):
        self.svc.api_key = "s3cret"
        headers = {"Content-Type": "text/plain", "Origin": EVIL}
        self.assertEqual(self.post("/completion", {"prompt": "hi"}, headers)[0], 401)
        self.assertEqual(self.post("/completion", {"prompt": "hi", "n_predict": 4},
                                   {**headers, "Authorization": "Bearer s3cret"})[0], 200)

    def test_every_post_route_refuses_a_foreign_page(self):
        """Every path _post names, not a list kept by hand: a new route cannot skip the check unnoticed."""
        tree = ast.parse(inspect.getsource(server.make_handler).replace("\r\n", "\n"))
        post = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_post")
        paths = sorted({n.value + ("x" if n.value.endswith("/") else "") for n in ast.walk(post)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("/")
                        and len(n.value) > 1 and " " not in n.value})
        self.assertIn("/completion", paths)
        self.assertIn("/tokenize", paths)
        self.assertGreaterEqual(len(paths), 15)
        for path in paths:
            for ctype, codes in (("application/json", {403}), ("text/plain", {403, 415})):
                code, _, _ = self.post(path, {"prompt": "hi", "content": "hi", "model": "m",
                                              "messages": [{"role": "user", "content": "hi"}]},
                                       {"Content-Type": ctype, "Origin": EVIL})
                self.assertIn(code, codes, (path, ctype))
        self.assertEqual(self.engine.calls, 0)


class EngineErrorInTheStream(Endpoints):
    """D-2026-01: the engine's error after the stream began is an error event and [DONE], not a second HTTP response
    written into the open stream (which the Gateway read as an answer cut off for no reason)."""
    engine_class = ErrEngine

    def test_streamed(self):
        code, raw, log = self.post("/completion", {"prompt": "hi", "n_predict": 100, "stream": True})
        self.assertEqual(code, 200)
        self.assertNotIn(b"HTTP/1", raw)
        events = self.events(raw)
        self.assertEqual(events[-1], "[DONE]")
        self.assertIn("illegal memory access", events[-2]["error"]["message"])
        self.assertEqual(events[-2]["error"]["type"], "server_error")
        # no closing "stop": true: the client must not take the cut answer for a whole one
        self.assertFalse(any(isinstance(e, dict) and e.get("stop") for e in events))
        self.assertEqual("".join(e["content"] for e in events[:-2]), ANSWER[:3])
        self.assertIn("illegal memory access", log)
        self.assertEqual(self.svc.metrics()["requests"][0]["finish"], "error")

    def test_not_streamed_is_a_plain_error(self):
        code, raw, _ = self.post("/completion", {"prompt": "hi", "n_predict": 100})
        self.assertEqual(code, 400)
        self.assertIn("illegal memory access", json.loads(raw)["error"]["message"])


@unittest.skipUnless((ROOT / "tools/apply_gateway_endpoints.py").is_file()
                     and (ROOT / "patches/gateway_completion_endpoints.patch").is_file(),
                     "the tool and the patch are the working repository's (the public tree has neither)")
class ThreeCopies(unittest.TestCase):
    """serve/server.py, tools/apply_gateway_endpoints.py and the patch carry the same code: a fix in one of them only
    would be undone by the next merge from upstream that runs the tool or the patch."""

    def setUp(self):
        self.tool = load_tool()
        self.text = server_text()

    def test_the_tool_has_the_servers_methods(self):
        start, end = self.text.index(self.tool.METHODS_START), self.text.index(self.tool.METHODS_END)
        self.assertEqual(self.text[start:end], self.tool.COMPLETION_METHODS + "\n")
        self.assertEqual(self.tool.apply(self.text), self.text)                  # nothing left to put in

    def test_the_tool_replaces_an_older_copy(self):
        # the copy from before 2026-10-23: no cross-site check, no error event in the stream
        _, upstream_gate, gate = self.tool.PIECES[0]
        old = self.text.replace(gate, upstream_gate, 1)
        start, end = old.index(self.tool.METHODS_START), old.index(self.tool.METHODS_END)
        stale = old[start:end].split("            except (ValueError, GpuBusy")[0] + "\n"
        old = old[:start] + stale + old[end:]
        self.assertNotEqual(old, self.text)
        self.assertEqual(self.tool.apply(old), self.text)

    def test_the_patch_and_the_tool_turn_upstreams_file_into_this_one(self):
        git = shutil.which("git")
        if git is None:
            self.skipTest("git is not installed")
        patch = (ROOT / "patches/gateway_completion_endpoints.patch").read_bytes().replace(b"\r\n", b"\n")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "serve").mkdir()
            target = Path(tmp) / "serve/server.py"
            target.write_bytes(self.text.encode("utf-8"))
            (Path(tmp) / "p.patch").write_bytes(patch)
            # outside a repository git apply works as patch(1): the patch taken out gives upstream's file
            r = subprocess.run([git, "-c", "core.autocrlf=false", "apply", "-R", "p.patch"], cwd=tmp,
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, "the patch no longer matches serve/server.py: " + r.stderr)
            upstream = target.read_bytes().decode("utf-8")
        self.assertNotIn("def _completion(", upstream)
        self.assertIn('            if path.startswith("/v1/") and self._foreign_page():\n', upstream)
        self.assertEqual(self.tool.apply(upstream), self.text)


if __name__ == "__main__":
    unittest.main()
