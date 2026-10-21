<h1 align="center">FastStrata</h1>

<p align="center"><b>A 35B mixture-of-experts coding model on a gaming PC with a small graphics card</b><br>
Qwen3.6-35B-A3B and Ornith 1.5 · Windows and Linux · NVIDIA · MIT</p>

<p align="center">
<a href="https://github.com/Jarson1408/FastStrata/actions/workflows/setup-tests.yml"><img src="https://github.com/Jarson1408/FastStrata/actions/workflows/setup-tests.yml/badge.svg" alt="setup tests"></a>
<a href="https://github.com/Jarson1408/FastStrata/releases"><img src="https://img.shields.io/github/v/release/Jarson1408/FastStrata?label=release" alt="release"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

<p align="center"><b>English</b> · <a href="README.pl.md">Polski</a></p>

<!-- The demo: OpenCode writing code on the 6 GB laptop, the speed counter beside it. When
docs/media/faststrata-opencode-6gb.gif is in the repository, put this line here, and in README.pl.md:
<p align="center"><img src="docs/media/faststrata-opencode-6gb.gif" width="900" alt="OpenCode writing code with Qwen3.6-35B-A3B on a laptop with a 6 GB RTX 3060, the speed counter beside it"></p>
-->

FastStrata runs Qwen3.6-35B-A3B, a coding model of 35 billion weights of which 3 billion work on each token, on a
gaming laptop with a 6 GB graphics card. On an RTX 3060 Laptop with 64 GB of RAM it writes **72 tok/s after a short
prompt, 67 after a 37K-token one and 49 after 139K**, with a 144K context. The card holds the attention and the
experts an answer uses most; the processor computes the rest from RAM.

It is a fork of [PolyStrata](https://github.com/VecSzn/PolyStrata) that made the same laptop up to twice as fast:
x1.27 after a short prompt, x1.60 after 37K, x1.98 after 139K, and x1.33 for a coding agent's session. Every number
here was measured on that one laptop, the old and the new engine in turns:
[the results](docs/RESULTS.md), [how](docs/OPTIMIZATION_LOG.md).

**Contents:** [Will my PC run it?](#will-my-pc-run-it) · [Install in three steps](#install-in-three-steps) ·
[Speed](#speed) · [Use it](#use-it) · [OpenCode and the gateway](#opencode-and-the-faststrata-gateway) ·
[LODESTAR](#lodestar-project-memory-for-a-small-model) · [If setup stops](#if-setup-stops) ·
[Uninstall](#uninstall) · [Configuration](#configuration) · [Where it comes from](#where-it-comes-from) ·
[Credits and license](#credits-and-license)

## Will my PC run it?

One NVIDIA card, GeForce RTX 20 series or newer:

| VRAM | What it gives | How it is known |
| --- | --- | --- |
| **4 GB: the minimum** | A short context, 16K to 48K, writing at 42 to 56 tok/s. 32K is the least a coding agent like OpenCode works with comfortably. | Measured on a simulated 4 GB card: the 6 GB card with 2 GiB held by another program ([how](docs/RESULTS.md#on-a-4-gb-gaming-card)) |
| **6 GB: sensible** | **A 144K context** with the Q4 model, writing at 49 to 72 tok/s. | Measured: [the test laptop](docs/RESULTS.md#the-test-pc) |
| **8 GB: optimal** | About 248K with Q4, and more of the experts in VRAM. | Setup's estimate less 10%; not measured with FastStrata |

| | |
| --- | --- |
| **RAM** | While it runs the engine holds 24-29 GB with the 23 GB model files and 19-22 GB with the 17 GB ones (measured on the 6 GB laptop, which has 64 GB). Setup checks what your PC has free. A PC with 32 GB was not measured. |
| **Disk** | 17 to 24 GB for the model, on an SSD: the experts are read from the file. |
| **Processor** | x86-64, with AVX2 for `--cpu-multi`; 8 cores were measured. |
| **System** | Windows 10 / 11 or Linux, a current NVIDIA driver. An AMD card on Linux gets the CPU options only, not measured. No Mac. |

Setup checks all of this itself and says what fits before it downloads anything: `START-HERE.bat --check` only
reports. The full table by card and preset: [docs/RESULTS.md](docs/RESULTS.md#what-you-need).

## Install in three steps

1. **Download** [the zip][zip] and unzip it anywhere, or `git clone` [the repository][repo].
2. **Run setup:** on Windows double-click **`START-HERE.bat`**; on Linux run **`./setup.sh`** in the folder. Press
   Enter to take what it recommends.
3. **Use it:** open OpenCode and pick a model of the provider **FastStrata**, or chat in the browser at
   `http://127.0.0.1:8080`.

In step 2 setup checks the PC and recommends a preset and a context for it: Enter installs that one with no more
questions. Or choose yourself from the four presets judged for this PC, with the speed measured on the test laptop and
what each costs the answers: one or several (`1,3`), each with its own context. They run one at a time (the gateway
swaps them when OpenCode asks for another), so each is judged on its own; their downloads add up. Then setup downloads
the ready-made engine from FastStrata's latest release (155 MB: Windows, an NVIDIA card from RTX 20 to RTX 50 with
driver 580 or newer, a CPU with AVX2; checked against the SHA-256 GitHub publishes) or, on any other PC, compiles it
(about five minutes on the test laptop, once; on Windows it offers to install Visual Studio 2022's compiler and the
CUDA toolkit), downloads the models (17 to 24 GB each, resumable), writes
each one's config with the options this PC takes, and offers the
[gateway for OpenCode](#opencode-and-the-faststrata-gateway). The screen shows short sentences; `--details` shows every
measurement and command too, and [the log](#if-setup-stops) always has them.

**Next time**, open the desktop shortcut **FastStrata** (with the gateway) or run `START-HERE.bat` again: it starts
the installed model and downloads nothing twice. The first start after a reboot reads the model's file from the SSD:
on the test laptop the engine loaded in 9.9 s and the first answer came at 27 tok/s. A newer FastStrata unzipped next
to the installed one sets itself up the same way on its first run: its models with their contexts, and the gateway
(taken over, or offered); the model files are reused.

For a script or an AI assistant, `--yes` takes every recommended answer (the recommended preset and its context);
the test laptop's setup named in full is

```bat
START-HERE.bat --family qwen36 --model UD-Q4_K_M --context 147456 --yes
```

Several presets at once: `--models UD-Q4_K_M,APEX-I-QUALITY` (or `--model` given twice), with `--context N` for all of
them or `--context UD-Q4_K_M=147456,APEX-I-QUALITY=65536` for each (`./setup.sh` takes the same options on Linux).
Or paste this into Claude Code, Cursor, Codex or Copilot:

```text
Set up FastStrata on this PC for me: https://github.com/Jarson1408/FastStrata - follow docs/AI_SETUP.md there.
```

## Speed

Writing, tokens a second, on the test laptop (RTX 3060 Laptop 6 GB, Ryzen 7 5800H, 64 GB DDR4), 2026-10-20, greedy
answers, the engine FastStrata started from against FastStrata, each the mean of two runs (139K: one):

| Preset | After 2K | After 37K | After 139K | Agent benchmark |
| --- | ---: | ---: | ---: | ---: |
| **Qwen3.6 UD-Q4_K_M** (22.7 GB, recommended) | 56.4 -> **71.8** | 41.9 -> **66.9** | 24.7 -> **48.8** | 62.4 -> **82.8** |
| Qwen3.6 UD-Q3_K_M (17.1 GB) | 58.8 -> **66.2** | 37.0 -> **58.6** | 26.4 -> **45.6** | 68.4 -> **81.2** |
| Ornith 1.5 APEX-I-QUALITY (23.7 GB) | 45.8 -> **51.0** | 28.9 -> **39.6** | 21.5 -> **36.8** | 47.0 -> **61.8** |
| Ornith 1.5 APEX-I-COMPACT (17.4 GB) | 60.8 -> **63.1** | 41.8 -> **56.2** | 24.6 -> **43.6** | 60.4 -> **76.1** |

- The agent benchmark is an OpenCode-like session of 35 answers with prompts up to 14,268 tokens.
- UD-Q3_K_M is not faster than UD-Q4_K_M on FastStrata: it is for a PC with about 5 GB less RAM.
- APEX-I-COMPACT is not recommended for coding: it changes names in code that APEX-I-QUALITY gets right.
- Not measured: a task benchmark, thinking on, sampled answers, other cards and CPUs, Linux.

What each change gave, every preset in full, prompt reading times and what the smaller files cost the answers:
[docs/RESULTS.md](docs/RESULTS.md).

## Use it

- **In the browser:** `http://127.0.0.1:8080`, with Chat, a live Monitor of the model and the PC (the decode speed
  among it), and About.
- **Apps and coding agents:** an OpenAI-compatible provider at `http://127.0.0.1:8080/v1` (any key, any model name);
  Anthropic's API at `/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` and `"fit_max_tokens":
  true` in the config, [why](docs/FAQ.md#can-i-use-it-with-claude-code-codex-or-other-apps)); the Responses API at
  `/v1/responses`; llama.cpp's `/completion` and `/tokenize`.
- **From another device:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Never without a key.

### OpenCode and the FastStrata Gateway

After the model, setup offers the [FastStrata Gateway](https://github.com/Jarson1408/FastStrata-Gateway), the app
between [OpenCode](https://opencode.ai) and the model. It turns the model's tool calls into the ones OpenCode reads,
stops a turn that loops, starts the model when OpenCode asks for it and swaps it for another installed preset, and runs
the [LODESTAR](#lodestar-project-memory-for-a-small-model) project memory (off at the start, with a switch). Say
yes, and setup:

- puts the gateway in `gateway\` in this folder, with its own Python environment (about 300 MB);
- writes the installed presets into its settings, each with the context setup chose for this PC;
- finds OpenCode where it is installed, or offers to install it (through npm or scoop, whichever the PC has);
- writes the presets into OpenCode's config (`~/.config/opencode/opencode.json`, the provider `faststrata`), keeping a
  `.bak` of the file and leaving the rest of it as it was; the folder is made where there is none;
- makes a desktop shortcut **FastStrata**, which opens the gateway's app; the app starts the model when OpenCode asks,
  and closing it frees the card and the RAM. The model's start takes about a minute on the 6 GB test laptop: the
  first answer after the app opens, or after a swap to another model, waits for it, and OpenCode waits too.

In OpenCode, pick a model of the provider FastStrata. Later, or on an install made without it:
`START-HERE.bat --gateway on` (`./setup.sh --gateway on`). With `--yes` setup leaves it out unless `--gateway on` is
given; `--opencode install|skip` and `--no-shortcut` answer the rest.

### LODESTAR: project memory for a small model

A model with 3B active parameters reads a project's instruction files (`README.md`, `AGENTS.md`) but does not keep to
them over a long task: it skips steps and forgets what was decided. LODESTAR is the gateway's protocol for that. It is
off after install; the app's box **Enforce LODESTAR Memory Protocol** (or *LODESTAR Protocol* in the tray menu)
turns it on or off at any time.

- **Deploy it into a project.** The app's **Deploy Template** button (or the tray's *Deploy LODESTAR Template...*)
  copies `AGENTS.md`, `CLAUDE.md` and the `.context/` folder into the project folder you pick. The protocol works in a
  folder it was deployed to.
- **Memory in the session.** With LODESTAR on, the gateway holds the model to one cycle a task: before it changes a
  file it reads the project's rules and memory and writes the goal and its steps into
  `.context/memory/active-context.md`; at the end it logs what was done and closes the goal. A session that was cut
  off, or a new one, starts where the last one stopped.
- **Semantic memory for long projects.** The lesson of each finished task is saved in `.context/semantic-memory/` and
  searched at the start of the next one. It runs `python .context/semantic-memory/embed.py`, which needs
  `sentence-transformers` in that Python (`pip install sentence-transformers`; it brings PyTorch). The gateway's own
  environment does not install it.

## If setup stops

Every run writes what it printed into `logs\setup-<date>-<time>.log` (the ten newest are kept), with your account and
PC names and any API key taken out. Where setup stops it names that file: attach it to an [issue][issues], and say
what you did. More: [docs/INSTALL.md](docs/INSTALL.md), [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Uninstall

Close the FastStrata app first (its tray icon > Quit). Then double-click **`UNINSTALL.bat`** in the FastStrata folder
(Linux: `python3 setup.py --uninstall`). It lists what it will do and asks once:

- the gateway's settings, `%APPDATA%\faststrata-gateway` (Linux: `~/.config/faststrata-gateway`), go to the Recycle
  Bin (the trash);
- the desktop shortcut **FastStrata** goes to the Recycle Bin;
- OpenCode's config loses the provider `faststrata` and nothing else; the file before is kept beside it as `.bak` (or
  `.bak-<date>-<time>` where a `.bak` is there already).

What another FastStrata folder on the PC set up is left unless you say so. Then it names what is yours to delete: the
FastStrata folder itself (the engine, the gateway, their Python environments) and the model files, with their size;
other FastStrata folders on the PC may share the model files. OpenCode, Python and the build tools setup may have
installed stay too.

## Configuration

Setup writes the engine's options into the model's run config, `strata-<family>-<preset>.json`, under `"args"`. On
the test laptop, for UD-Q4_K_M at 144K:

```text
--max-context 147456 --kv-q8 --prompt-batch 512 --tail-skip 6 --threads 8 --attn-split --graph-experts
--elastic-prefill 1024 --suffix-drafts 256 --cpu-multi --conversation-cache-mib 12288 --conversation-cache-slots 8
```

Each where the PC and the preset take it ([which PC gets which](docs/RESULTS.md#configuration)), and setup prints each
with what it gave. All of them off: `START-HERE.bat --setup --speedups off`, as PolyStrata's setup wrote the config;
`--speedups on` turns them on again. Every option in detail: [docs/FASTSTRATA_OPTIONS.md](docs/FASTSTRATA_OPTIONS.md).

## Where it comes from

FastStrata is the third project in a line. [Strata](https://github.com/Niko1221/Strata) (Niko1221 and contributors)
runs Qwen3.8-Flash-Next with its experts split between the card and the CPU, and brought the installer, the server and
the web app. [PolyStrata](https://github.com/VecSzn/PolyStrata) (VecSzn) added `strata-poly`, an engine that reads a
model's GGUF file in place, for Qwen3.6-35B-A3B, Ornith 1.5 and others. FastStrata narrows it to the two models that
fit a gaming PC, four presets in all, and makes `strata-poly` fast on a small card: ten measured steps, nine changes
to the engine and its settings ([each one](docs/RESULTS.md#what-faststrata-changed)).

For Qwen3.8-Flash-Next use [Strata](https://github.com/Niko1221/Strata). Plans: expert profiles shipped with each
preset; a mode for PCs with 16 to 24 GB of RAM.
What changed in each release: [CHANGELOG.md](CHANGELOG.md).

## Credits and license

- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors: the Qwen engine, the installer, the
  server and the web app.
- [PolyStrata](https://github.com/VecSzn/PolyStrata) by VecSzn: the `strata-poly` engine FastStrata speeds up.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT): the GGUF reader, the kernels `strata-poly` builds
  on, and the CPU arithmetic `--cpu-multi` keeps to the bit.
- Qwen3.6-35B-A3B by the [Qwen team](https://huggingface.co/Qwen), quantized by
  [Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF); Ornith 1.5 by ornith-ai, in
  [mudler](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF)'s APEX quantization. Each model's own
  license applies to its files.
- Ideas: SuffixDecoding (Oliaro et al., NeurIPS 2025) for `--suffix-drafts`; Expert Deferral (KTransformers, SOSP
  2025), measured in step 6; KIVI and KVQuant for the 4-bit cache's layouts.

FastStrata is open source under the [MIT License](LICENSE), as Strata and PolyStrata are; the license keeps Strata's
copyright notice. A few parts have their own licenses ([which](docs/HOW_IT_WORKS.md#license)). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).

Found a problem? Open an [issue][issues] and attach the setup log or `strata-<family>-<preset>.log` from the
FastStrata folder. A result from your own PC is welcome: `BENCH.bat` (`./bench.sh`),
[how](docs/COMMUNITY_BENCHMARKS.md). How the server is secured: [SECURITY.md](SECURITY.md), upstream PolyStrata's.

[repo]: https://github.com/Jarson1408/FastStrata
[zip]: https://github.com/Jarson1408/FastStrata/archive/refs/heads/main.zip
[issues]: https://github.com/Jarson1408/FastStrata/issues
