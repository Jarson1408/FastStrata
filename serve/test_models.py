"""The model registry (serve/models.py) and the switch between models, with engines that are not processes."""
import contextlib
import io
import json
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import models as registry
from serve import server as S
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, ModelBusy, Service, SwitchFailed, serve


def gguf(path: Path, **meta) -> Path:
    """A GGUF with metadata and no tensors: what the registry reads of a model file."""
    def text(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta))
    for key, v in meta.items():
        key = key.replace("__", ".")
        if isinstance(v, bool):
            out += text(key) + struct.pack("<I?", 7, v)
        elif isinstance(v, int):
            out += text(key) + struct.pack("<II", 4, v)
        else:
            out += text(key) + struct.pack("<I", 8) + text(v)
    path.write_bytes(out)
    return path


def install(root: Path, name: str, arch: str, file: str, **more) -> Path:
    """A model as setup leaves it: its file, its tokenizer directory and its run config."""
    model = gguf(root / file, general__architecture=arch, general__name=name.title(),
                 general__size_label="35B-A3B", **{arch + "__context_length": 262144, **more})
    tok = root / (name + "-tok")
    tok.mkdir()
    (tok / "vocab.json").write_text("{}")
    exe = root / "engine"
    exe.write_text("")
    cfg = root / f"strata-{name}.json"
    cfg.write_text(json.dumps({"exe": str(exe), "args": ["--model", str(model), "--max-context", "8192"],
                               "cwd": str(root), "tokenizer": str(tok), "model_name": name}))
    return cfg


class Stopped(MockEngine):
    """An engine that can be stopped and started again, and counts both."""
    def __init__(self, tok, say="Hello."):
        super().__init__(tok, say, max_context=4096)
        self.running, self.stops, self.starts = True, 0, 0

    def alive(self):
        return self.running

    def unload(self):
        self.running = False
        self.stops += 1

    def restart(self):
        self.running = True
        self.starts += 1

    def close(self):
        self.running = False


class Registry(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        install(self.root, "qwen36", "qwen35moe", "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf", qwen35moe__nextn_predict_layers=1)
        install(self.root, "gemma", "gemma4", "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf")
        registry._seen.clear()

    def tearDown(self):
        self.dir.cleanup()

    def test_a_model_is_what_its_file_says(self):
        q = registry.find(registry.discover(self.root), "qwen36")
        self.assertEqual((q["backend"], q["mtp"], q["context"], q["status"]), ("qwen35moe", True, 8192, "available"))
        self.assertEqual((q["params_total"], q["params_active"], q["quantization"]), ("35B", "3B", "UD-Q3_K_M"))
        self.assertEqual(q["display_name"], "Qwen36")
        g = registry.find(registry.discover(self.root), "gemma")
        self.assertEqual((g["backend"], g["mtp"]), ("gemma4", False))

    def test_the_name_setup_gave_comes_before_the_files(self):
        cfg = self.root / "strata-gemma.json"
        cfg.write_text(json.dumps({**json.loads(cfg.read_text()), "display_name": "Gemma 4 26B-A4B UD-Q3_K_M"}))
        g = registry.find(registry.discover(self.root), "gemma")
        self.assertEqual(g["display_name"], "Gemma 4 26B-A4B UD-Q3_K_M")

    def test_what_cannot_run_says_why(self):
        install(self.root, "other", "llama", "other.gguf")
        cfg = install(self.root, "gone", "gemma4", "gone.gguf")
        (self.root / "gone.gguf").unlink()
        by = {e["id"]: e for e in registry.discover(self.root)}
        self.assertEqual(by["other"]["status"], "unavailable")
        self.assertIn("'llama'", by["other"]["problem"])
        self.assertIn("missing", by["gone"]["problem"])
        self.assertEqual(Path(by["gone"]["config"]), cfg)

    def test_models_json_adds_and_corrects(self):
        (self.root / "models.json").write_text(json.dumps({"gemma": {"display_name": "Gemma 4", "status": "x"}}))
        g = registry.find(registry.discover(self.root), "gemma")
        self.assertEqual((g["display_name"], g["status"]), ("Gemma 4", "available"))

    def test_find_takes_a_name_a_file_or_the_one_match(self):
        models = registry.discover(self.root)
        self.assertEqual(registry.find(models, "strata-gemma.json")["id"], "gemma")
        self.assertEqual(registry.find(models, "QWEN")["id"], "qwen36")
        self.assertIsNone(registry.find(models, "e"))       # in neither name alone
        self.assertIsNone(registry.find(models, "glm"))


class Switch(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.a = install(self.root, "alpha", "qwen35moe", "alpha.gguf")
        self.b = install(self.root, "beta", "gemma4", "beta.gguf")
        registry._seen.clear()
        self.tok = ByteTokenizer()
        self.old = Stopped(self.tok)
        self.svc = Service(self.old, self.tok, ChatTemplate(S.ROOT / "serve/chat_template.jinja"), model_name="alpha")
        self.svc.config_path, self.svc.models_root = str(self.a), self.root
        self.new = Stopped(self.tok, "Hi.")
        patches = [mock.patch.object(S, "load_tokenizer", lambda p: self.tok),
                   mock.patch.object(S, "start_model", lambda cfg, tok, lazy=False: (self.new, None, {"top_k": 5}, None, []))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.dir.cleanup()

    def test_the_old_model_stops_before_the_new_one_answers(self):
        e = self.svc.switch_model("beta")
        self.assertEqual((e["id"], e["status"], e["current"]), ("beta", "loaded", True))
        self.assertEqual((self.old.stops, self.old.running), (1, False))
        self.assertIs(self.svc.engine, self.new)
        self.assertEqual((self.svc.model, self.svc.sampling_defaults), ("beta", {"top_k": 5}))
        self.assertEqual([m["id"] for m in self.svc.models() if m["current"]], ["beta"])

    def test_a_model_that_does_not_start_leaves_the_old_one_loaded(self):
        def fails(cfg, tok, lazy=False):
            raise RuntimeError("no VRAM")
        with mock.patch.object(S, "start_model", fails):
            with self.assertRaisesRegex(SwitchFailed, "no VRAM.*alpha is loaded again"):
                self.svc.switch_model("beta")
        self.assertIs(self.svc.engine, self.old)
        self.assertEqual((self.old.stops, self.old.starts, self.old.running, self.svc.model), (1, 1, True, "alpha"))

    def test_no_switch_under_a_request_and_none_to_a_model_that_cannot_run(self):
        with self.svc.status_lock:
            self.svc.status["busy"] = True
        with self.assertRaises(ModelBusy):
            self.svc.switch_model("beta")
        self.assertEqual(self.old.stops, 0)
        with self.svc.status_lock:
            self.svc.status["busy"] = False
        (self.root / "beta.gguf").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.svc.switch_model("beta")
        with self.assertRaises(KeyError):
            self.svc.switch_model("gamma")
        self.assertEqual(self.old.stops, 0)

    def test_the_routes(self):
        httpd = serve(self.svc, port=0)
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def call(path, body=None):
            req = urllib.request.Request(base + path, None if body is None else json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.status, json.load(r)
            except urllib.error.HTTPError as e:
                return e.code, json.load(e)

        code, listing = call("/api/models")
        self.assertEqual((code, listing["current"], [m["id"] for m in listing["models"]]), (200, "alpha", ["alpha", "beta"]))
        self.assertEqual(call("/api/models/load", {"model": "gamma"})[0], 404)
        code, out = call("/api/models/load", {"model": "beta"})
        self.assertEqual((code, out["model"]["id"]), (200, "beta"))
        code, cur = call("/api/models/current")
        self.assertEqual((cur["model"]["id"], cur["loaded"], cur["name"]), ("beta", True, "beta"))
        self.assertEqual(call("/api/models/unload", {}), (200, {"status": "unloaded"}))
        self.assertEqual(call("/api/models/current")[1]["loaded"], False)


class Start(unittest.TestCase):
    """main() with --engine strata as far as "ready": 0.1.42's line for the replicas' cards read `groups`, a name of
    start_model()'s, and every start of a real engine ended in a NameError."""

    def test_main_reaches_ready(self):
        real_sleep = time.sleep

        def sleep(s):                       # Ctrl+C in main()'s wait, real sleeps in the server's threads
            if threading.current_thread() is threading.main_thread():
                raise KeyboardInterrupt
            real_sleep(s)
        with tempfile.TemporaryDirectory() as d:
            cfg = install(Path(d), "alpha", "qwen35moe", "a.gguf")
            tok = ByteTokenizer()
            out = io.StringIO()
            with mock.patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(cfg), "--port", "0"]), \
                    mock.patch.object(S, "load_tokenizer", lambda p: tok), \
                    mock.patch.object(S, "start_model", lambda cfg, tok, lazy=False: (Stopped(tok), None, {}, None, None)), \
                    mock.patch.object(S.time, "sleep", sleep), mock.patch.object(S.signal, "signal"), \
                    contextlib.redirect_stdout(out):
                self.assertEqual(S.main(), 0)
        self.assertIn("ready: http://127.0.0.1:0/v1", out.getvalue())


if __name__ == "__main__":
    unittest.main()
