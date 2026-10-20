"""Tests for setup.py's log (FASTSTRATA LOG): what a user sends when setup stops must carry what setup printed and
what went wrong, and must not carry their account name, their PC's name or a key. No network, no GPU; the log goes to
a temporary folder and the console is restored after each test.

    python -m unittest tools.test_setup_log
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402


class TestWhatTheLogMustNotCarry(unittest.TestCase):
    def r(self, text):
        return setup.redact(text, home=r"C:\Users\anna", user="anna", pc="ANNA-PC")

    def test_the_home_folder_in_both_spellings(self):
        self.assertNotIn("anna", self.r(r"C:\Users\anna\FastStrata and C:/Users/anna/FastStrata").lower())
        self.assertNotIn("anna", self.r(r'"C:\\Users\\anna\\x.gguf"').lower())

    def test_the_account_and_the_pc_where_they_stand_alone(self):
        out = self.r("user anna on ANNA-PC; annabelle stays")
        self.assertIn("<user>", out)
        self.assertIn("<pc>", out)
        self.assertIn("annabelle", out, "a longer word that contains the name is not the name")

    def test_keys_and_tokens(self):
        out = self.r('--api-key s3cret "api_key": "s3cret" hf_' + "a" * 30 + " sk-" + "b" * 30)
        self.assertNotIn("s3cret", out)
        self.assertNotIn("a" * 30, out)
        self.assertNotIn("b" * 30, out)

    def test_short_names_are_left_alone(self):
        """A two-letter account would take every 'jo' out of a log."""
        self.assertEqual(setup.redact("jo jo", home="", user="jo", pc="x"), "jo jo")


class _LogCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.saved = (sys.stdout, sys.stderr, setup.SETUP_LOG)

    def tearDown(self):
        if setup.SETUP_LOG is not None:
            setup.SETUP_LOG[1].close()
        sys.stdout, sys.stderr, setup.SETUP_LOG = self.saved
        self.tmp.cleanup()

    def start(self):
        console = io.StringIO()
        sys.stdout = console
        path = setup.start_log(["--yes", "--api-key", "s3cret"], folder=self.folder)
        return path, console

    def text(self, path):
        setup.SETUP_LOG[1].flush()
        return path.read_text(encoding="utf-8")


class TestTheLogCarriesTheRun(_LogCase):
    def test_what_the_console_shows_goes_in_redacted(self):
        path, console = self.start()
        with mock.patch.dict(setup.os.environ, {"USERNAME": "anna", "USER": "anna"}):
            setup.say("  [ok] the model is in C:/x for anna")
        self.assertIn("the model is in", console.getvalue())
        log = self.text(path)
        self.assertIn("the model is in", log)
        self.assertNotIn("for anna", log)
        self.assertNotIn("s3cret", log, "the command line's key is not in the header")

    def test_a_stop_says_where_the_log_is(self):
        path, console = self.start()
        with self.assertRaises(SystemExit):
            setup.fail("no NVIDIA GPU found", "install the driver")
        self.assertIn(str(path), console.getvalue())
        self.assertIn(setup.REPORT_URL, console.getvalue())
        self.assertIn("no NVIDIA GPU found", self.text(path))

    def test_an_unexpected_error_goes_in_whole(self):
        path, _ = self.start()
        try:
            raise ValueError("bad value")
        except ValueError:
            with redirect_stdout(io.StringIO()):
                setup.log_crash(*sys.exc_info())
        log = self.text(path)
        self.assertIn("UNEXPECTED ERROR", log)
        self.assertIn("ValueError: bad value", log)
        self.assertIn("Traceback", log)

    def test_a_commands_exit_code_goes_in(self):
        path, _ = self.start()
        setup.run([sys.executable, "-c", "pass"], quiet=True)
        self.assertIn("(exit 0)", self.text(path))

    def test_a_progress_line_goes_in_once_as_it_ended(self):
        """A download's progress is one line rewritten in place (\\r); the log had every step of it on one line."""
        path, _ = self.start()
        sys.stdout.write("\r  model:  1 / 9 GB")
        sys.stdout.write("\r  model:  9 / 9 GB")
        sys.stdout.write("\n  [ok] model downloaded\n")
        log = self.text(path)
        self.assertIn("  model:  9 / 9 GB\n", log)
        self.assertNotIn("1 / 9 GB", log)

    def test_an_answer_goes_in(self):
        path, _ = self.start()
        with mock.patch("builtins.input", return_value="n"):
            self.assertEqual(setup.ask("Do you want images?", ["y", "n"], "y", False), "n")
        self.assertEqual(setup.ask("Keep it?", ["y", "n"], "y", True), "y")
        log = self.text(path)
        self.assertIn("Do you want images? [Y/n]: n", log)
        self.assertIn("Keep it? [Y/n]: y (--yes)", log)

    def test_a_detail_is_in_the_log_and_not_on_the_screen(self):
        path, console = self.start()
        setup.detail("  PCIe: 3.0 x8")
        self.assertNotIn("PCIe", console.getvalue())
        self.assertIn("PCIe: 3.0 x8", self.text(path))
        with mock.patch.object(setup, "DETAILS", True):
            setup.detail("  shown with --details")
        self.assertIn("shown with --details", console.getvalue())

    def test_the_ten_newest_are_kept(self):
        for i in range(12):
            (self.folder / f"setup-20260101-0000{i:02d}.log").write_text("", encoding="utf-8")
        path, _ = self.start()
        logs = sorted(self.folder.glob("setup-*.log"))
        self.assertEqual(len(logs), setup.SETUP_LOGS_KEPT)
        self.assertIn(path, logs)

    def test_a_folder_that_cannot_be_written_is_no_reason_to_stop(self):
        sys.stdout = io.StringIO()
        with mock.patch.object(Path, "mkdir", side_effect=PermissionError("read-only")):
            self.assertIsNone(setup.start_log([], folder=self.folder / "no"))
        self.assertIsNone(setup.SETUP_LOG)


if __name__ == "__main__":
    unittest.main()
