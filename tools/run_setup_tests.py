"""Run setup's own tests: every tools/test_setup_*.py and tools/test_faststrata_names.py, each file in a Python of its
own, the way the CI (.github/workflows/setup-tests.yml) and release/make_faststrata_release.py run them.  They need no
GPU, no download and nothing installed beyond Python.

    python tools/run_setup_tests.py [-v]

-v prints each file's output too; without it only a failing file's last lines are printed.  Exit code 1 when a file
fails.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_files(root: Path = ROOT) -> list:
    return sorted((root / "tools").glob("test_setup_*.py")) + [root / "tools" / "test_faststrata_names.py"]


def main() -> int:
    verbose = "-v" in sys.argv[1:]
    files, failed = test_files(), []
    start = time.monotonic()
    for f in files:
        t = time.monotonic()
        r = subprocess.run([sys.executable, str(f)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {f.name} ({time.monotonic() - t:.1f} s)", flush=True)
        if verbose or r.returncode != 0:
            print((r.stdout + r.stderr)[-4000 if not verbose else None:], flush=True)
        if r.returncode != 0:
            failed.append(f.name)
    print(f"\n{len(files) - len(failed)} of {len(files)} test files pass in {time.monotonic() - start:.0f} s"
          + (f"; failing: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
