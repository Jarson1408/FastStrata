"""The files tools/faststrata_names.py reads say FastStrata where they mean this project.  A text that means the
project this one is a fork of names it by a link, as "the PolyStrata project" or as "upstream PolyStrata": the tool
leaves those alone, and the bare word it would turn into FastStrata the next time it runs.

    python -m unittest tools.test_faststrata_names
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import faststrata_names as N  # noqa: E402


class Names(unittest.TestCase):
    def test_examples(self):
        N.demo()

    def test_nothing_is_left_to_rename(self):
        files = io.StringIO()
        try:
            with mock.patch.object(sys, "argv", ["faststrata_names.py", "--check"]), \
                    contextlib.redirect_stdout(files), contextlib.redirect_stderr(io.StringIO()):
                left = N.main()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("the tool lists the files with git: this is no checkout")
        self.assertEqual(left, 0, "python tools/faststrata_names.py --check lists:\n" + files.getvalue())


if __name__ == "__main__":
    unittest.main()
