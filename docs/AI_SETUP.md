# Installing FastStrata for a user (instructions for AI agents)

This page is written for an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...) that a user asked to
set up FastStrata on their PC. FastStrata runs Qwen3.6-35B-A3B or Ornith 1.5, mixture-of-experts models of 35B weights
with 3B active, on one NVIDIA graphics card plus system RAM, and serves an OpenAI- and Anthropic-compatible API on
`http://127.0.0.1:8080`. What it is: the [README](../README.md); its presets and every measured number: [RESULTS.md](RESULTS.md).

Work through the steps in order. Tell the user what you are doing in plain words; they may not be technical.

## 0. Ground rules

- **Never expose the server without an API key.** Keep the default `--host 127.0.0.1`. Only if the user asks for
  access from other devices, use `--host 0.0.0.0` **together with** `--api-key <a long random secret>`, and give the
  user the key. Never put a tunnel or port forward in front of a server without a key.
- Do not change the user's system beyond what setup does (setup installs Python for the user account if needed, and
  everything else inside the FastStrata folder and its `PolyStrata-data` folder). Ask before installing drivers.
- The model download is 17 to 24 GB a preset, 1 GB more to read pictures. Confirm the user is fine with that before
  you start, especially on a metered connection.
- No ready-made engine of FastStrata exists yet: setup compiles it. On Windows it offers to install Visual Studio
  2022's Build Tools and the CUDA toolkit first. Setup is long-running: run it in the background or with a long
  timeout and poll its output; do not kill it because it is quiet for a while. It is resumable: running the same
  command again continues where it stopped.

## 1. Check the PC

Setup's own check prints the GPU(s), driver, RAM, CPU, free disk and what this PC has for each preset, and installs
nothing except Python and the `.venv` it needs to run (step 2 gets the repository first):

```
Windows:  START-HERE.bat --check
Linux:    ./setup.sh --check
```

To check by hand:

| What | Windows (PowerShell) | Linux |
| --- | --- | --- |
| OS | `[Environment]::OSVersion` | `cat /etc/os-release` |
| NVIDIA GPU, VRAM, driver | `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv` | the same |
| RAM | `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB` | `free -g` |
| Free disk | `Get-PSDrive -PSProvider FileSystem` | `df -h .` |

Requirements:

- **GPU:** one NVIDIA GeForce RTX 20 series card or newer. FastStrata was measured on a 6 GB laptop card and on 4 GB
  simulated on it. GTX 10 series and older, and integrated GPUs, are not supported. An AMD card on Linux runs as a
  beta, with the options that run on the CPU only.
- **Driver:** a current NVIDIA driver (580 or newer). If it is missing or too old, tell the user to update it (NVIDIA
  App or nvidia.com/drivers) and restart; do not install drivers yourself unless they ask.
- **RAM:** while it runs the engine holds 24-29 GB with a 23 GB preset and 19-22 GB with a 17 GB one (measured beside
  a 6 GB card). **CPU:** x86-64 with AVX2. **Disk:** 17 to 25 GB free, on an SSD.
- **OS:** Windows 10/11 or Linux.

If the PC does not meet them, say which part is missing and stop.

## 2. Get FastStrata

```
git clone https://github.com/Jarson1408/FastStrata.git
```

Without git: download https://github.com/Jarson1408/FastStrata/archive/refs/heads/main.zip and unzip it. Pick a
drive with enough free space: the model files go to a `PolyStrata-data` folder **next to** the FastStrata folder (or
`--data-dir <path>` on another drive).

## 3. Pick the model and preset

Ask the user which model, if they have no preference: **Qwen3.6-35B-A3B** (`--family qwen36`) for coding; Ornith 1.5
(`--family ornith`) has the same architecture and other weights.

| Preset | `--family` / `--model` | File | RAM in use while it runs |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B UD-Q4_K_M (recommended) | `--family qwen36 --model UD-Q4_K_M` | 22.7 GB | 24-27 GB |
| Qwen3.6-35B-A3B UD-Q3_K_M | `--family qwen36 --model UD-Q3_K_M` | 17.1 GB | 19-22 GB |
| Ornith 1.5 APEX-I-QUALITY | `--family ornith --model APEX-I-QUALITY` | 23.7 GB | 26-29 GB |
| Ornith 1.5 APEX-I-COMPACT | `--family ornith --model APEX-I-COMPACT` | 17.4 GB | 19-22 GB |

- Leave `--model` out and setup takes the preset it recommends for the PC: the first of the four that runs 32K or
  more there, judged from what was measured (the 6 GB laptop: UD-Q4_K_M at 144K with the 8-bit cache; a simulated
  4 GB card: UD-Q3_K_M up to 48K, UD-Q4_K_M and APEX-I-COMPACT at 32K) and setup's estimate for other cards. The
  sensible setting is UD-Q4_K_M with the 8-bit cache on a 6 GB card or larger; 8 GB is the optimum (the whole 256K by
  the estimate); 4 GB the minimum, a short context.
- UD-Q3_K_M is not faster than UD-Q4_K_M; it needs about 5 GB less RAM. **APEX-I-COMPACT is not recommended for
  coding**: it changes names in code that APEX-I-QUALITY gets right ([RESULTS.md](RESULTS.md#models-and-presets)).
- **Several presets at once**, if the user wants to switch between them (Qwen3.6 for code, Ornith for something
  else): `--models UD-Q4_K_M,APEX-I-QUALITY`, or `--model` given more than once. They run one at a time (the gateway
  swaps them when OpenCode asks for another), so each needs the RAM of the table above on its own; their downloads and
  disk space add up. Each gets its own context: `--context N` for all of them, `--context MODEL=N` for one
  (`--context UD-Q4_K_M=147456,APEX-I-QUALITY=65536`). The first named is the one the gateway starts first. If one of
  them fails (a download), the others are kept and setup says how to add it: `--setup --model <it>`.

## 4. Run setup without questions

From the FastStrata folder (Windows: in `cmd`, or `cmd /c START-HERE.bat ...` from PowerShell):

```
Windows:  START-HERE.bat --yes --family qwen36 --no-start
Linux:    ./setup.sh --yes --family qwen36 --no-start
```

The flags (all of them: `START-HERE.bat --help`):

| Flag | Meaning |
| --- | --- |
| `--yes` | take the recommended answer to every question (no prompts) |
| `--family qwen36\|ornith` | the model |
| `--model UD-Q4_K_M\|UD-Q3_K_M\|APEX-I-QUALITY\|APEX-I-COMPACT` | the preset (default: the one setup recommends for this PC, with its context); with `--yes` it is also the consent to a preset setup's estimate finds too big; given more than once, several presets |
| `--models A,B` | several presets at once, e.g. `UD-Q4_K_M,APEX-I-QUALITY`; the first is started first |
| `--max-output N` or `MODEL=N` | the longest answer in tokens, its thinking too (default: Qwen3.6 16384, Ornith 8192; at most a quarter of the context); the gateway's and OpenCode's limit |
| `--context N` or `MODEL=N` | context in tokens, `144K` = 147456 (default: the longest that keeps the cache in 16 or 8 bits on this card, up to 147456); `N` for every preset, `MODEL=N` for one (`UD-Q4_K_M=147456,APEX-I-QUALITY=65536`) |
| `--vision yes\|no\|gpu\|cpu` | read pictures; `--yes` leaves them off |
| `--gpu N` | the card (default: the one with the most VRAM) |
| `--speedups on\|off` | FastStrata's options (default on); `off` writes the config upstream PolyStrata's setup wrote |
| `--data-dir PATH` | where the model files go |
| `--port N` | the server port (default 8080) |
| `--host H --api-key K` | listen beyond this PC; **only together with a key** |
| `--no-start` | install only, do not start the server |
| `--gateway on` | also the FastStrata Gateway for OpenCode, its OpenCode config and a desktop shortcut (with `--yes` only when given; on an install with a model, `--gateway on` alone sets up just that) |
| `--opencode install\|skip` | OpenCode where it is missing (`--yes` skips it unless `install`) |
| `--no-shortcut` | no desktop shortcut |
| `--setup` | install another model or change settings of an installed one |
| `--check` | only check the PC |

`--no-start` is recommended for agents: the server runs in the foreground until its window is closed, which would
block your shell. Start it separately in step 6.

Notes:

- **The attention cache.** Where the context does not fit the card in 16 bits, setup takes 8 bits (`--kv-q8`, the
  6 GB laptop's measured setting), and where 8 bits do not fit either it asks about 4 (`--kv-q4`, `--kv-q4-keys q4`),
  which costs the answers more. On a 4 GB card it keeps 8 bits up to the contexts measured there.
- **Linux:** setup uses `sudo apt` (or dnf/pacman) to install Python with venv and the build tools if they are
  missing. You cannot type the user's password: if a `sudo` step is needed, ask the user to run it (e.g.
  `sudo apt install python3-venv build-essential git`) and then rerun setup.
- **The compile:** setup installs the compiler and the CUDA toolkit (it asks first; `--yes` accepts) and compiles the
  engine for the card, some minutes, once.

## 5. While it works, tell the user

- It downloads 17 to 24 GB from Hugging Face; how long depends on their connection (at 100 Mbit/s about half an
  hour). It can be stopped and continues where it left off.
- It compiles the engine once (some minutes).
- **The first start after the PC starts reads the model's file from the SSD:** on the 6 GB test laptop the engine
  loaded in 9.9 s and the first answer came at 27 tok/s; later answers come faster.
- Nothing leaves their PC: the model runs locally.

## 6. Start the server

Setup prints the start script it wrote, one a preset: `run-<family>-<preset>.bat` (`.sh` on Linux), lower case,
e.g. `run-qwen36-ud-q4_k_m.bat`.

```
Windows (PowerShell):  Start-Process -FilePath ".\run-qwen36-ud-q4_k_m.bat"            (opens its own window)
Linux:                 nohup ./run-qwen36-ud-q4_k_m.sh > strata-server.out 2>&1 &
```

Or `START-HERE.bat` / `./setup.sh` without flags, which starts the installed model (it asks which one when several are
installed; add `--yes` to take the first). The server opens the user's browser on `http://127.0.0.1:8080` when it is
ready. Closing its window (or stopping the process) stops the model.

## 7. Verify

The HTTP server answers once the model is loaded. Poll:

```
curl http://127.0.0.1:8080/health
```

It returns `{"status": "ok", "model": ..., "max_context": ..., "images": ..., "api_key": ..., "loaded": true, ...}`.
Then:

```
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"strata\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 64, \"reasoning_effort\": \"none\"}"
```

With an API key, add `-H "Authorization: Bearer <key>"`. The server window prints a `ready: http://127.0.0.1:8080/v1`
line when it is ready; the engine log is `strata-<family>-<preset>.log` in the FastStrata folder.

## 8. Connect the user's apps

- **Browser:** `http://127.0.0.1:8080` - Chat, a live Monitor, and About (settings and addresses).
- **Any OpenAI-compatible app or agent:** base URL `http://127.0.0.1:8080/v1`, any API key (or the configured one),
  any model name.
- **Anthropic-compatible apps:** `http://127.0.0.1:8080/v1/messages`.
- **Codex CLI and other Responses API apps:** `http://127.0.0.1:8080/v1/responses` (stateless; Codex's
  `config.toml`: [DETAILS.md](DETAILS.md#the-responses-api-and-codex-cli)).
- **llama.cpp-style clients:** `/completion` and `/tokenize`.
- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, `ANTHROPIC_MODEL` set to a Claude model name it knows
  (FastStrata ignores the name), and any `ANTHROPIC_AUTH_TOKEN` (or the configured key). Add
  `"fit_max_tokens": true` to the model's config: Claude Code asks for 32,000 tokens of answer in every request, and
  the server refuses a request whose prompt leaves less room than that in the context.
- **OpenCode, through the FastStrata Gateway** (the recommended way for a coding agent: it parses the model's tool
  calls, stops loops, starts and swaps the models and keeps the project memory): ask the user, then
  `START-HERE.bat --gateway on` (`./setup.sh --gateway on`). It writes the provider `faststrata` into
  `~/.config/opencode/opencode.json` (base URL `http://localhost:18000/v1`, a `.bak` of the file kept) and makes a
  desktop shortcut **FastStrata** for the gateway's app, which starts the model when OpenCode asks. Do not start the
  server yourself as well: the gateway starts it on port 8080.
- **Thinking level:** `"reasoning_effort": "none" | "low" | "medium" | "high"` (default high).
- FastStrata answers one request at a time; several conversations are parked in RAM and come back without being read
  again. API details: [DETAILS.md](DETAILS.md#using-it).

## 9. When something fails

| Symptom | What to do |
| --- | --- |
| `none of ...'s presets fits this PC` / `too big for this PC's RAM` | Too little RAM or VRAM for every preset (step 3); naming one with `--model ... --yes` installs it anyway, slowly. |
| `the NVIDIA driver is too old` | The user updates the driver and restarts; rerun setup. |
| Download, install or compile stopped | Rerun the same setup command; it continues. |
| `port 8080 is already in use` | FastStrata is already running (check `/health`), or use `--port 8081`. |
| Very slow, disk busy, or "the engine stopped unexpectedly" | Not enough free RAM: close programs, or set up the smaller preset (`--setup --model UD-Q3_K_M`). |
| `prompt ... exceeds the context` | Rerun setup with `--setup --context <bigger>`. |
| Python or build tools could not be installed | Install what the message names (links are printed), then rerun. |
| `the gateway needs Python 3.12 or newer` | Install Python 3.12 from python.org ("Add python.exe to PATH"), then `--gateway on` again. |
| Anything else | Setup names its log (`logs/setup-<date>-<time>.log`, account and PC names taken out): give it to the user to attach to an issue at https://github.com/Jarson1408/FastStrata/issues. |

More: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and the [full table](DETAILS.md#troubleshooting) (both upstream
PolyStrata's pages). If it still fails, collect `strata-<family>-<preset>.log` and the setup output, and suggest an
issue at https://github.com/Jarson1408/FastStrata/issues.

## 10. If the user wants it removed

Ask the user to close the FastStrata app (tray icon > Quit), then from the FastStrata folder:
`cmd /c UNINSTALL.bat --yes` (Linux: `python3 setup.py --uninstall --yes`). It moves the gateway's settings
(`%APPDATA%\faststrata-gateway`) and the desktop shortcut to the Recycle Bin and takes the provider `faststrata` out of
OpenCode's config (a `.bak` first); with `--yes` it leaves what another FastStrata folder set up. It prints the
FastStrata folder and the model files' folder with their size: tell the user to delete them, if they want, once you
have finished (the model files may be shared with another FastStrata folder).

`tools/strata_mcp.py`, the MCP server of upstream PolyStrata ([MCP_SERVER.md](MCP_SERVER.md)), still describes the
PolyStrata project's model set; use the steps above for FastStrata.
