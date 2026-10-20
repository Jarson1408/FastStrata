#!/usr/bin/env python3
"""Write this project's name, FastStrata, where the text of a PolyStrata tree says PolyStrata.

FastStrata is a fork of PolyStrata (https://github.com/VecSzn/PolyStrata) and takes its changes, and PolyStrata's
text names PolyStrata (PolyStrata takes Strata's changes and writes its own name over them with
tools/polystrata_names.py). Run this after taking changes from it:

    python tools/faststrata_names.py            # rewrite the files
    python tools/faststrata_names.py --check    # list what it would change, change nothing

It replaces the word PolyStrata in the files a user reads or runs (setup, the start and update scripts, the server,
the web app, the tools and their tests, the pages that describe this project) and nothing else; a second run changes
nothing. It leaves alone:

- what programs and users find things by: folder names (`PolyStrata-data`, a `PolyStrata...` or `PolyStrata*` folder,
  the settings folder `PolyStrata`), paths (`C:\\tools\\PolyStrata`), and the word in lower case (`polystrata.py`,
  the data folders' rule, the Linux settings folder); binaries, config files, options and environment variables
  have other names;
- the engines' sources and build files, the benchmark records and the measurement scripts (`bench/`,
  `docs/benchmarks/`, `test_scripts/`), the paper, the Intel, Mac and AMD-packaging setups FastStrata does not use,
  the tools of the Strata engine's packs and its research data (`data/`), and SECURITY.md, whose reporting address
  is the PolyStrata project's;
- the pages of docs/ that describe Strata or PolyStrata as such, whose numbers were measured on those projects: only
  the pages in DOCS are this project's; and the files that describe upstream's models (the MCP server, the GGUF
  inspector, the Docker image), which FastStrata has not adapted;
- README.md, README.pl.md and AGENTS.md, which are FastStrata's own and name PolyStrata where they mean it;
- the places where PolyStrata means the project this one is a fork of (KEEP). To name it in new text, link it,
  [PolyStrata](https://github.com/VecSzn/PolyStrata), or write "the PolyStrata project" or "upstream PolyStrata".

Do not run tools/polystrata_names.py on this tree: FastStrata's text names the Strata project's engine ("the Strata
engine"), which that tool would rename.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = ("src/", "include/", "tests/", "cmake/", "third_party/", "bench/", "sycl/", "metal/", "patches/", "ref/",
        "data/", "test_scripts/", "SECURITY.md", "tools/vision/", "tools/hip/", "tools/qwen35/", "tools/pack_layer.py", "tools/strata_pack.py",
        "tools/mtp_pack.py", "tools/nvfp4_convert.py", "tools/polystrata_names.py", "tools/test_polystrata_names.py",
        "tools/faststrata_names.py", "AGENTS.md", "README.md", "README.pl.md",
        # upstream's, and about its models (Qwen3.8-Flash-Next's families): FastStrata has not adapted them
        "Dockerfile", "docker-entrypoint.sh", "tools/strata_mcp.py", "tools/test_strata_mcp.py", "tools/strata_inspect.py")
DOCS = ("docs/AI_SETUP.md", "docs/FASTSTRATA_OPTIONS.md", "docs/media/")
KINDS = (".md", ".py", ".html", ".js", ".css", ".bat", ".sh", ".yml", ".yaml", ".svg")
NAMES = ("Dockerfile",)
KEEP = "|".join((
    r"\[PolyStrata\]\(https://github\.com/VecSzn/PolyStrata",  # a Markdown link to it
    r"VecSzn/PolyStrata\">PolyStrata",                        # an HTML link to it
    r"VecSzn/PolyStrata",                                     # its repository, in an address or by name
    r"[Tt]he\s+PolyStrata\s+project|[Uu]pstream\s+PolyStrata",
    r"PolyStrata-[A-Za-z0-9]",                                # a folder: PolyStrata-data, PolyStrata-main, PolyStrata-q8
    r"PolyStrata(?:\*|\.\.\.)",                               # the data folders' names: PolyStrata*, PolyStrata...
    r"named\s+`?PolyStrata",                                  # the same rule in words
    r"starts\s+with\s+`PolyStrata`",
    r"/\s*[\"']PolyStrata[\"']",                              # the settings folder, a folder in a test: / "PolyStrata"
    r"[\\/]PolyStrata(?![A-Za-z0-9_])",                       # a path: C:\tools\PolyStrata, /home/you/PolyStrata
    r"PolyStrata[\\/]",                                       # a path that starts with it
    r"PolyStrata\s+with\s+the\s+Q8\s+cache",                  # the engine FastStrata started from (the log)
))
WORD = re.compile(f"{KEEP}|(?<![A-Za-z0-9_])(PolyStrata)(?![A-Za-z0-9_])")


def rename(text: str) -> str:
    return WORD.sub(lambda m: "FastStrata" if m.group(1) else m.group(0), text)


def wanted(f: str) -> bool:
    if f.startswith(SKIP) or not (f.endswith(KINDS) or f in NAMES):
        return False
    return not f.startswith("docs/") or f.startswith(DOCS)


def files() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if wanted(f)]


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
    same = ("https://github.com/VecSzn/PolyStrata/issues", "[PolyStrata](https://github.com/VecSzn/PolyStrata) by",
            '<a href="https://github.com/VecSzn/PolyStrata">PolyStrata</a>', "FastStrata", "polystrata.py",
            "PolyStrataEngine", "the PolyStrata project's release", "upstream PolyStrata's setup",
            "PolyStrata-data", "D:\\PolyStrata-data", "Downloads\\PolyStrata-main\\PolyStrata-main", "PolyStrata* folders",
            "must be named PolyStrata... (e.g.", 'Path.home() / "AppData" / "Roaming") / "PolyStrata" / "settings.json"',
            "C:\\tools\\PolyStrata", "/home/you/PolyStrata/tools/strata_mcp.py", "C:\\Users\\you\\PolyStrata\\tools",
            "PolyStrata\\engine-poly", "whose name starts with `PolyStrata` (e.g.", "PolyStrata with the Q8 cache added",
            "github.com/Jarson1408/PolyStrata-q8")
    for s in same:
        assert rename(s) == s, s
    assert rename("PolyStrata's setup, PolyStrata. (PolyStrata)") == "FastStrata's setup, FastStrata. (FastStrata)"
    assert rename('title PolyStrata\r\nsay("PolyStrata is updated")') == 'title FastStrata\r\nsay("FastStrata is updated")'
    assert rename('props["build_info"] = "PolyStrata " + v') == 'props["build_info"] = "FastStrata " + v'
    assert rename(rename("run PolyStrata")) == "run FastStrata"
    assert wanted("setup.py") and wanted("serve/web/index.html") and wanted("docs/AI_SETUP.md")
    assert wanted("docs/media/how-it-works-qwen36.svg") and wanted("tools/strata_bench.py")
    assert not wanted("tools/strata_mcp.py") and not wanted("Dockerfile") and not wanted("docs/MCP_SERVER.md")
    assert not wanted("docs/DETAILS.md") and not wanted("docs/OPTIMIZATION_LOG.md") and not wanted("src/poly/main.cpp")
    assert not wanted("test_scripts/agent-session.json") and not wanted("README.md")


if __name__ == "__main__":
    demo()
    sys.exit(main())
