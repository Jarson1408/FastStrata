# AGENTS.md

FastStrata runs Qwen3.6-35B-A3B and Ornith 1.5, mixture-of-experts models of 35B weights with 3B active, on a gaming
PC: one NVIDIA graphics card plus system RAM, on Windows or Linux, measured on a laptop with a 6 GB card and on 4 GB
simulated on it. It is a fork of [PolyStrata](https://github.com/VecSzn/PolyStrata), which is built on
[Strata](https://github.com/Niko1221/Strata), and keeps their parts: the PolyStrata project's GGUF engine `strata-poly`
(`src/poly/`), which FastStrata speeds up; the Strata project's Qwen3.8-Flash-Next engine (`src/`, `include/`), the
Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and the one-click installer
(`setup.py`, started by `START-HERE.bat` / `setup.sh`). Setup offers FastStrata's two models alone
(`FASTSTRATA_FAMILIES`, four presets); upstream PolyStrata's other models still install with `--family`, without
FastStrata's options. What FastStrata changed and every number: [README.md](README.md) (README.pl.md is its Polish
version: keep the two in step) and [docs/OPTIMIZATION_LOG.md](docs/OPTIMIZATION_LOG.md).

The repository is github.com/Jarson1408/FastStrata (it was github.com/Jarson1408/PolyStrata-q8). Its address
is in README.md, README.pl.md, docs/AI_SETUP.md, UPDATE.bat, update.sh, serve/server.py, serve/web/index.html
and .github/ISSUE_TEMPLATE/ (`git grep Jarson1408/FastStrata` lists them). The names programs and the user's Gateway
find things by stay as they are: the binaries (`strata-poly`, `strata-poly-kvq8.exe`), the run configs
(`strata-*.json`), the folders (`PolyStrata-data`, the data folders' `PolyStrata...` rule, the settings folder
`PolyStrata`), the options and the environment variables.

## Installing FastStrata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the preset by its card and RAM, run setup
non-interactively, start and verify the server, and connect the user's apps. Never expose the server beyond
`127.0.0.1` without `--api-key`.

## Working on the code

- `strata-poly` and its models: [docs/BACKENDS.md](docs/BACKENDS.md). Every FastStrata change, how it was measured and
  what it gave: [docs/OPTIMIZATION_LOG.md](docs/OPTIMIZATION_LOG.md); a new change gets a step there, measured with
  `test_scripts/run_serve.py`, the old and the new build in turns. Each option in detail:
  [docs/FASTSTRATA_OPTIONS.md](docs/FASTSTRATA_OPTIONS.md).
- The Strata engine, the server's API and all its settings: [docs/DETAILS.md](docs/DETAILS.md) and the
  [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Text taken from upstream PolyStrata names that project. After merging from it, `python tools/faststrata_names.py`
  writes FastStrata again; its first lines say what it leaves alone (README.md, README.pl.md and this file among
  them). Upstream PolyStrata merges Strata and renames its text with `tools/polystrata_names.py`; do not run that tool
  here, it would rename "the Strata engine" in FastStrata's text.
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_speedups.py`).
- FastStrata's speedups (`docs/OPTIMIZATION_LOG.md`) are written into a new Qwen3.6 or Ornith install's config by
  `poly_speedups()` in `setup.py`, each where the PC's card, CPU and RAM and the preset take it (`--speedups on|off`;
  the presets are `FAMILIES[f]["models"]` of `FASTSTRATA_FAMILIES`, picked by `poly_pick()`, each with "measured",
  "cpu_multi" and, where `--graph-experts` cannot read its experts, "graph_experts": False); a new one gets its
  condition there and its fake machines in `tools/test_setup_speedups.py`. The attention cache's format (fp16,
  `--kv-q8`, then the 4-bit `--kv-q4` and `--kv-q4-keys q4` only where `--kv-q8` does not fit) is chosen in
  `poly_kv_plan()` (marked `FASTSTRATA KV-Q4`).
- The FastStrata Gateway (github.com/Jarson1408/FastStrata-Gateway, the clean tree that the user's Local-LLM-Gateway
  builds from its `faststrata` branch with `install/release/make_release.py`) is set up by `faststrata_gateway()` in
  `setup.py` (marked `FASTSTRATA GATEWAY`): pinned to `GATEWAY_COMMIT` and checked by `GATEWAY_FILES_SHA256`
  (`gateway_files_sha256()` of the archive's files), so a new gateway release changes both. Every setup run writes
  `logs/setup-<date>-<time>.log` (marked `FASTSTRATA LOG`: what setup printed, account and PC names and keys taken
  out, ten kept). Tests: `tools/test_setup_gateway.py`, `tools/test_setup_log.py`.
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
