"""The benchmark's long prompt is cut when the model's context refuses it: four characters a token is a guess, and
the same text is more tokens for one tokenizer than for another.  No server and no model are used.

    python -m unittest tools.test_benchmark_prompt
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import polystrata  # noqa: E402


def refused(code):
    return urllib.error.HTTPError("http://x/v1/chat/completions", code, "refused", None, None)


class LongPrompt(unittest.TestCase):
    def run_benchmark(self, answers):
        """benchmark() with a server that answers the long prompt's tries with `answers` in turn: an HTTP status
        that refuses it, or None for a run.  Returns (the prompts' lengths, what was printed)."""
        sent, left = [], list(answers)

        def run_once(srv, name, prompt, max_tokens):
            if name != "short":
                sent.append(len(prompt))
                code = left.pop(0)
                if code:
                    raise refused(code)
            return {"prompt": name}

        entry = {"id": "m", "backend": "b", "quantization": "q", "context": 32768}
        a = argparse.Namespace(model="m", port=None, api_key="", max_tokens=256, prompt_tokens=[24000], json=None)
        out = io.StringIO()
        with mock.patch.object(polystrata, "entry_of", lambda m: entry), \
                mock.patch.object(polystrata, "port_of", lambda e, p: 1), \
                mock.patch.object(polystrata, "Server", mock.Mock()), \
                mock.patch.object(polystrata, "run_once", run_once), contextlib.redirect_stdout(out):
            polystrata.benchmark(a)
        return sent, out.getvalue()

    def test_a_prompt_that_fits_is_sent_once(self):
        sent, out = self.run_benchmark([None])
        self.assertEqual(len(sent), 1)
        self.assertNotIn("does not fit", out)

    def test_a_refused_prompt_is_cut(self):
        sent, out = self.run_benchmark([400, 400, None])
        self.assertEqual(len(sent), 3)
        self.assertGreater(sent[0], sent[1])
        self.assertGreater(sent[1], sent[2])
        self.assertAlmostEqual(sent[2] / sent[0], 0.9, places=2)
        self.assertIn("the 24000-token prompt does not fit the context: trying 95% of it", out)

    def test_another_error_is_not_tried_again(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.run_benchmark([500])

    def test_it_gives_up_at_85_percent(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.run_benchmark([400, 400, 400, 400])


if __name__ == "__main__":
    unittest.main()
