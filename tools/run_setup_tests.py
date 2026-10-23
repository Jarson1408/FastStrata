"""Run setup's own tests: every tools/test_setup_*.py and tools/test_faststrata_names.py, each file in a Python of its
own, the way the CI (.github/workflows/setup-tests.yml) and release/make_faststrata_release.py run them.  They need no
GPU, no download and nothing installed beyond Python.

    python tools/run_setup_tests.py [-v] [--server]

-v prints each file's output too; without it only a failing file's last lines are printed.  --server runs the
server's tests instead (every serve/test_*.py, against the mock engine): they need the packages of requirements.txt
(jinja2), so the CI runs them after setup's, once those are installed.  Each file's line says how many of its tests
ran and how many it skipped, so "ok" on a file that skipped its tests shows.  Exit code 1 when a file fails.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_files(root: Path = ROOT) -> list:
    return sorted((root / "tools").glob("test_setup_*.py")) + [root / "tools" / "test_faststrata_names.py"]


def server_test_files(root: Path = ROOT) -> list:
    return sorted((root / "serve").glob("test_*.py"))


def counts(output: str) -> tuple[int, int]:
    """(tests run, tests skipped) out of unittest's summary ("Ran 12 tests", "OK (skipped=2)"); 0 where it has none."""
    ran = sum(int(n) for n in re.findall(r"^Ran (\d+) tests?", output, re.M))
    skipped = sum(int(n) for n in re.findall(r"skipped=(\d+)", output))
    return ran, skipped


def command(f: Path) -> list:
    # the server's tests import serve.* and most have no __main__: run as modules, from the root
    return ([sys.executable, "-m", "unittest", f"serve.{f.stem}"] if f.parent.name == "serve"
            else [sys.executable, str(f)])


def main() -> int:
    verbose = "-v" in sys.argv[1:]
    files, failed = server_test_files() if "--server" in sys.argv[1:] else test_files(), []
    ran = skipped = 0
    start = time.monotonic()
    for f in files:
        t = time.monotonic()
        r = subprocess.run(command(f), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        n, k = counts(r.stdout + r.stderr)
        ran, skipped = ran + n, skipped + k
        print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {f.relative_to(ROOT).as_posix()}: {n} tests"
              f"{f', {k} skipped' if k else ''} ({time.monotonic() - t:.1f} s)", flush=True)
        if verbose or r.returncode != 0:
            print((r.stdout + r.stderr)[-4000 if not verbose else None:], flush=True)
        if r.returncode != 0:
            failed.append(f.name)
    print(f"\n{len(files) - len(failed)} of {len(files)} test files pass ({ran} tests, {skipped} skipped) in "
          f"{time.monotonic() - start:.0f} s" + (f"; failing: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
