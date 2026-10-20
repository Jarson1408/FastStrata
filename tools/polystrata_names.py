#!/usr/bin/env python3
"""Write this project's name, PolyStrata, where the text of a Strata tree says Strata.

PolyStrata is built on Strata (https://github.com/Niko1221/Strata) and takes its changes, and Strata's text names
Strata. Run this after taking changes from it:

    python tools/polystrata_names.py            # rewrite the files
    python tools/polystrata_names.py --check    # list what it would change, change nothing

It replaces the word Strata in the files a user reads or runs (documents, setup, the server, the web app, the
tools and their tests) and nothing else; a second run changes nothing. It leaves alone the engines' sources and
build files, the benchmark records, the paper, docs/FAQ.md (it is about both projects), and the places where
Strata means the project this one is built on (KEEP). To name that project in new text, link it,
[Strata](https://github.com/Niko1221/Strata), or write "the Strata project".
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = ("src/", "include/", "tests/", "cmake/", "third_party/", "bench/", "sycl/src/", "sycl/tools/", "tools/vision/",
        "docs/paper/", "docs/COMMUNITY_BENCHMARKS.md", "docs/FAQ.md", "tools/polystrata_names.py")
KINDS = (".md", ".py", ".html", ".js", ".css", ".bat", ".sh", ".yml", ".yaml")
NAMES = ("Dockerfile",)
KEEP = "|".join((
    r"\[Strata\]\(https://github\.com/Niko1221/Strata", # a Markdown link to it
    r"Niko1221/Strata\">Strata",                        # an HTML link to it
    r"(?:Niko1221|eddoursul)/Strata",                   # their repositories, in an address or by name
    r"[Tt]he\s+Strata project|Strata 项目|Strata maintainers",
    r"Strata-Paper",                                    # the paper's file
    r"Strata\.Example",                                 # a host name a test spells in capitals
    r"class Strata\b|: Strata(?=[,)])|\bStrata\(",      # the MCP server's class
    r"Desktop[\\/]Strata",                              # a path on a developer's own PC
    r"\\\(Strata SYCL engine",                          # a label the Intel page strips; another program writes it
    r'replace\("Strata ",',                             # the engine's own build_info, which the benchmark strips
    r"NVFP4(?:-GPTQ)?-Strata",                          # the NVFP4 packs' Hugging Face repositories
))
WORD = re.compile(f"{KEEP}|(?<![A-Za-z0-9_])(Strata)(?![A-Za-z0-9_])")


def rename(text: str) -> str:
    return WORD.sub(lambda m: "PolyStrata" if m.group(1) else m.group(0), text)


def files() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines()
            if not f.startswith(SKIP) and (f.endswith(KINDS) or f in NAMES)]


def main() -> int:
    check = "--check" in sys.argv[1:]
    changed = 0
    for f in files():
        path = ROOT / f
        try:
            with open(path, encoding="utf-8", newline="") as fh:
                old = fh.read()
        except (OSError, UnicodeDecodeError) as e:
            print(f"skipped {f}: {e}", file=sys.stderr)
            continue
        new = rename(old)
        if new == old:
            continue
        changed += 1
        print(f)
        if not check:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(new)
    print(f"{changed} files " + ("would change" if check else "changed"), file=sys.stderr)
    return 1 if check and changed else 0


def demo() -> None:
    same = ("https://github.com/Niko1221/Strata/releases", "[Strata](https://github.com/Niko1221/Strata) by",
            "StrataEngine", "PolyStrata", "paper/Strata-Paper.pdf", "class Strata:", "M.Strata(root)",
            "strata: Strata)", "The Strata project's author", "the\nStrata project's 0.1.41",
            '<a href="https://github.com/Niko1221/Strata">Strata</a>')
    for s in same:
        assert rename(s) == s, s
    assert rename("Strata's folder, Strata-data and Strata.") == "PolyStrata's folder, PolyStrata-data and PolyStrata."
    assert rename("Strata は Strata（IQ3_S）") == "PolyStrata は PolyStrata（IQ3_S）"
    assert rename(rename("run Strata")) == "run PolyStrata"


if __name__ == "__main__":
    demo()
    sys.exit(main())
