"""chat.py against a server with engines that are not processes: the chat loop, and /model in it.

    python -m unittest tools.test_chat
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chat  # noqa: E402
from serve import models as registry  # noqa: E402
from serve import server as S  # noqa: E402
from serve import test_models as models_tests  # noqa: E402   (the module: its helpers, not its tests)
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402


class Mine(io.StringIO):
    """What the thread that made it prints.  The server's threads are in this process too, and their lines go on
    to the stream that was there."""

    def __init__(self, beyond):
        super().__init__()
        self.beyond, self.thread = beyond, threading.get_ident()

    def write(self, s):
        return super().write(s) if threading.get_ident() == self.thread else self.beyond.write(s)


class Chat(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        root = Path(self.dir.name)
        first = models_tests.install(root, "alpha", "qwen35moe", "alpha.gguf")
        models_tests.install(root, "beta", "gemma4", "beta.gguf")
        registry._seen.clear()
        self.tok = ByteTokenizer()
        self.old = models_tests.Stopped(self.tok, "Hello.")
        self.new = models_tests.Stopped(self.tok, "Hi.")
        svc = Service(self.old, self.tok, ChatTemplate(S.ROOT / "serve/chat_template.jinja"), model_name="alpha")
        svc.config_path, svc.models_root = str(first), root
        for p in (mock.patch.object(S, "load_tokenizer", lambda p: self.tok),
                  mock.patch.object(S, "start_model", lambda cfg, tok, lazy=False: (self.new, None, {}, None, None))):
            p.start()
            self.addCleanup(p.stop)
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def say(self, *lines) -> str:
        """What the chat prints when these lines are typed and the input then ends."""
        typed = iter(lines)

        def fake_input(prompt=""):
            print(prompt, end="")
            return next(typed)                          # StopIteration is not EOFError: end the input as a terminal does

        def ended(prompt=""):
            try:
                return fake_input(prompt)
            except StopIteration:
                raise EOFError from None

        out = Mine(sys.stdout)
        with mock.patch("builtins.input", ended), contextlib.redirect_stdout(out):
            self.assertEqual(chat.chat(self.base, "none", 64), 0)
        return out.getvalue()

    def asked(self, engine) -> str:
        return self.tok.decode(engine.last_prompt)

    def test_a_question_is_answered_and_remembered(self):
        out = self.say("first question", "second question")
        self.assertEqual(out.count("model> Hello."), 2)
        self.assertIn("first question", self.asked(self.old))    # the second request carries the first turn
        self.assertIn("second question", self.asked(self.old))

    def test_reset_and_think(self):
        out = self.say("first question", "/reset", "/think low", "/think much", "", "second question", "/quit", "x")
        self.assertIn("(new conversation)", out)
        self.assertIn("(thinking: low)", out)
        self.assertIn("(usage: /think none|low|medium|high)", out)
        self.assertNotIn("first question", self.asked(self.old))
        self.assertEqual(out.count("model> "), 2)                 # the empty line and what follows /quit ask nothing

    def test_model_list_and_info(self):
        out = self.say("/model list", "/model", "/model info beta", "/model info gamma", "/model what")
        self.assertRegex(out, r"\* alpha +qwen35moe")
        self.assertRegex(out, r"  beta +gemma4")
        self.assertRegex(out, r"name +alpha\n")
        self.assertRegex(out, r"name +beta\n")
        self.assertIn("(no such model; /model list shows the names)", out)
        self.assertIn("(usage: /model | /model list | /model switch <name> | /model info [<name>] | /model unload)", out)
        self.assertNotIn("model> ", out)

    def test_switch_starts_a_new_conversation_with_the_other_model(self):
        out = self.say("first question", "/model switch beta", "second question", "/model list")
        self.assertIn("(the model is now beta)", out)
        self.assertIn("(new conversation)", out)
        self.assertIn("model> Hello.", out)
        self.assertIn("model> Hi.", out)
        self.assertEqual((self.old.stops, self.old.running), (1, False))
        self.assertIn("second question", self.asked(self.new))
        self.assertNotIn("first question", self.asked(self.new))
        self.assertRegex(out, r"\* beta +gemma4")

    def test_a_switch_that_fails_keeps_the_conversation(self):
        out = self.say("first question", "/model switch gamma", "second question", "/model unload")
        self.assertIn("(installed: alpha, beta)", out)
        self.assertNotIn("(new conversation)", out)
        self.assertIn("first question", self.asked(self.old))
        self.assertIn("(unloaded)", out)

    def test_a_server_that_is_not_there(self):
        with mock.patch.object(self, "base", "http://127.0.0.1:9"):
            out = self.say("a question", "/model list")
        self.assertIn("(could not reach the server at http://127.0.0.1:9/v1/chat/completions", out)
        self.assertIn("(the server does not list its models", out)


if __name__ == "__main__":
    unittest.main()
