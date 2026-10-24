# Changelog

FastStrata's releases. The version is `FASTSTRATA_VERSION` in `setup.py`, which setup prints at its start and writes
into its log; each release is tagged `v<version>`. The engine inside keeps the version of upstream Strata's source
(`CMakeLists.txt`), which is another number. Every measurement named here: [docs/RESULTS.md](docs/RESULTS.md) and
[docs/OPTIMIZATION_LOG.md](docs/OPTIMIZATION_LOG.md).

## v0.3.0 - 2026-10-24

A new engine: with LODESTAR on, a turn's prompt is read in 7.7 s where it took 19.3 (the user's own OpenCode session
replayed on the 6 GB test laptop, the ready-made engines of v0.2.0 and of this release). The ready-made engine of this
release is a new file, `strata-poly-windows-x64-59a3ba3a4a47ac90.zip` (RTX 20 to RTX 50, CUDA 13.0).

- `--prompt-cache-message N` (step 11 of docs/OPTIMIZATION_LOG.md): the engine also keeps a checkpoint where the
  prompt's last message begins. The gateway's LODESTAR frame ends every prompt and the next prompt leaves it out; the
  Qwen3.6 model's recurrent layers could only go back to an older checkpoint, so every turn read 11,541 tokens again
  on that session, and now 2,640. Without LODESTAR the time is the same (5.0 s a turn). Off unless given; setup writes
  `--prompt-cache-message 2048` on NVIDIA.
- Without the option the engine takes every prompt up where v0.2.0's does and reads the same tokens.

## v0.2.0 - 2026-10-23

The fixes from two independent code audits of v0.1.1 (the server's routes for the gateway, the check of Python's
installer, the tests the CI runs), and the context's cache named apart from the model's Q4 / Q3. The engine is the
same as in v0.1.0 and v0.1.1: this release carries the same ready-made
`strata-poly-windows-x64-3ba1bf3a4bf413eb.zip`.

- The context's memory (the attention cache) is named apart from the model's Q4 / Q3: setup's menu says "up to 144K
  context with an 8-bit cache", the context question says what the cache is, and the README explains the two.
- The server's `/completion` and `/tokenize` (what the gateway sends) refuse a web page of another site without an
  API key, as `/v1/*` does: before, any site open in the browser could have the model write on the PC's card.
- An engine error in the middle of a streamed `/completion` reaches the gateway as an error event and `[DONE]`, no
  longer as a second HTTP response written into the stream; the API monitor shows `/completion`'s answer and speed.
- The web app's pages cannot be shown in another site's frame (`X-Frame-Options`, `frame-ancestors`).
- Where setup downloads python.org's Python 3.12 installer (no Python and no winget), it runs it only with the
  pinned SHA-256; START-HERE.bat checks the same.
- The CI runs the server's tests too (every `serve/test_*.py` against the mock engine), and Dependabot looks at the
  pinned packages once a month.

## v0.1.1 - 2026-10-22

Setup and the gateway, from the first installs on other PCs. The engine is the same as in v0.1.0: this release
carries the same ready-made `strata-poly-windows-x64-3ba1bf3a4bf413eb.zip`.

- The gateway, release `47f4cd8`: a read of a file the gateway spilled reaches the model whole. Before, an 8 KB file
  was replaced by a preview, the spill file the preview pointed to was replaced by the same preview, and the model
  read it again until the loop breaker stopped it, never having seen the file.
- Setup asks whether the NVIDIA card shows the desktop, with what it found as the answer (Windows: the screens
  Windows lists for each graphics; Linux: /sys/class/drm; else nvidia-smi). Where it does, the safe context is 10%
  below the longest that fits (128K instead of 144K on a 6 GB card); the VRAM in use is no longer taken off the card.
- Step 2's first choice is called the safe one.
- The gateway needs Python 3.12 or newer: where the PC has none, setup now installs Python 3.12 for the user's
  account on Windows (asked; winget, else python.org's installer, no administrator) and tells Linux the command,
  where it used to leave the gateway out with a warning.
- An NVIDIA driver older than 580 that still runs the CUDA 12 engine is told that setup compiles the engine because
  of it, and how to update the driver.
- The answer length: asked with the context where the models are chosen by hand (8K, 16K or 32K; the safe one is the
  preset's own, Qwen3.6 16K and Ornith 8K), or `--max-output N` / `MODEL=N`; at most a quarter of the context. Setup
  keeps it in the model's run config, and the gateway and OpenCode's `limit.output` take it from there.
- Setup's tests pass on Linux too (the CI's Ubuntu job): three upstream tests took Windows for granted.
- README: the gateway for OpenCode up front; a section on AMD cards (FastStrata's speedups are for NVIDIA RTX;
  with an AMD card, upstream PolyStrata).

## v0.1.0 - 2026-10-21

The first release: Qwen3.6-35B-A3B and Ornith 1.5 on one NVIDIA card and system RAM, four presets.

**Speed**, on the test laptop (RTX 3060 Laptop 6 GB, Ryzen 7 5800H, 64 GB DDR4), Qwen3.6-35B-A3B UD-Q4_K_M, against
the engine FastStrata started from (upstream PolyStrata's `strata-poly` with the 8-bit cache, 2026-10-17): writing
after a 2K prompt 56.4 -> 71.8 tok/s, after 37K 41.9 -> 66.9, after 139K 24.7 -> 48.8; a coding agent's session of 35
answers 62.4 -> 82.8; reading a 37K prompt 47.2 -> 36.4 s.

**The engine** (each opt-in, written into the config where the PC takes it):

- `--kv-q8`: the attention cache in 8 bits, so 144K fits a 6 GB card (72K in 16 bits).
- `--threads 8`: one CPU thread a core.
- `--attn-split`: an answer's attention in one pass on the tensor cores.
- `--graph-experts`: each layer of an answer as one CUDA graph.
- `--elastic-prefill 1024`: prompts read in calls of 2,048 tokens on VRAM the expert cache lends.
- `--suffix-drafts 256`: guesses from every conversation the engine has seen (SuffixDecoding).
- `--cpu-multi`: the CPU's K-quant experts read once for all the tokens of a window.
- `--kv-q4`, `--kv-q4-keys q4`: a 4-bit cache, only where `--kv-q8` does not fit the context.
- A conversation cache of 12 GiB in 8 slots.

**Setup:**

- A ready-made engine for Windows in this release (`strata-poly-windows-x64-3ba1bf3a4bf413eb.zip`, 155 MB:
  `strata-poly.exe` and the image encoder, RTX 20 to RTX 50, CUDA 13.0, built by `tools/make_poly_release.py` in about
  40 minutes on the test laptop): setup downloads it where the PC takes it (an NVIDIA card from RTX 20 on, driver 580 or
  newer, a CPU with AVX2), checks its SHA-256, and compiles the engine only elsewhere.
- One menu of the four presets (Qwen3.6 UD-Q4_K_M and UD-Q3_K_M, Ornith 1.5 APEX-I-QUALITY and APEX-I-COMPACT),
  judged from what was measured on the 6 GB laptop and a simulated 4 GB card; other cards get setup's estimate less
  10%, marked "estimated".
- The FastStrata Gateway for OpenCode, pinned to one commit and checked by its files' SHA-256; OpenCode found or
  installed, its config written with a `.bak`; a desktop shortcut **FastStrata**. LODESTAR is off at the start.
- A log of every run in `logs/`, with account and PC names and keys taken out, for a bug report.
- `UNINSTALL.bat` (`setup.py --uninstall`): the gateway's settings and the shortcut to the Recycle Bin, the provider
  `faststrata` out of OpenCode's config.
- Continuous tests: setup's tests on Windows and Linux on every push (`.github/workflows/setup-tests.yml`).

**Not in this release:** a ready-made engine (setup compiles it, about five minutes, once), shipped expert profiles.
Not measured: cards of 8 GB and more, a PC with 32 GB of RAM, Linux, AMD cards, a task benchmark.
