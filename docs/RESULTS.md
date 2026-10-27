# FastStrata: what changed, the results, the settings

The details behind the [README](../README.md): where FastStrata comes from, each change and what it gave, the
four presets, the test PC, every result, the 4 GB card, what setup offers by card, and the engine's options. How
each number was measured, run by run: [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md).

**Contents:** [Where it comes from](#where-it-comes-from) · [What FastStrata changed](#what-faststrata-changed) ·
[Models and presets](#models-and-presets) · [The test PC](#the-test-pc) · [Results](#results) ·
[4 GB cards](#on-a-4-gb-gaming-card) · [What you need](#what-you-need) · [Configuration](#configuration) ·
[How it was measured](#how-it-was-measured)

## Where it comes from

FastStrata is the third project in a line:

| Project | By | What it is |
| --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata) | Niko1221 and contributors | The original. The `strata` engine runs Qwen3.8-Flash-Next, a model of 24,576 experts of which each word needs 10: the graphics card keeps the experts asked for most and learns which those are as you use it, every expert sits in RAM and the processor computes the ones the card lacks at the same time, and a draft layer guesses the next words for the model to check. Around it: the one-click installer, a server with an OpenAI- and Anthropic-compatible API, and a web app with a chat and a live monitor. |
| [PolyStrata](https://github.com/VecSzn/PolyStrata) | VecSzn | Adds a second engine, `strata-poly`, which reads a model's GGUF file where it lies, with no second copy: GLM 5.3 Flash first, then Qwen3.6-35B-A3B, Ornith 1.5, Gemma 4, Tiel Coder and DeepSeek V4 Flash. The dense weights live in VRAM; the routed experts are split between a VRAM cache, filled by how much each expert is used and following what an answer uses, and the CPU, which computes the others from RAM while the card does its share. The model's own draft (MTP) block guesses up to three tokens ahead, and the model checks them in one pass. |
| **FastStrata** | this fork | Narrows PolyStrata to the two models above that fit a gaming PC, Qwen3.6-35B-A3B and Ornith 1.5, four presets in all, and makes `strata-poly` fast on a small card: fourteen measured steps, twelve changes to the engine and its settings ([below](#what-faststrata-changed)). |

PolyStrata kept the attention cache in 16 bits, 22 KB a token for Qwen3.6, which holds 72K on a 6 GB card. FastStrata
added an 8-bit cache, `--kv-q8` (2026-10-17, 11.7 KB a token: 144K on the 6 GB card), and a 4-bit one, `--kv-q4`
(step 9: 184K, and the model's whole 256K with the keys in 4 bits too).

For Qwen3.8-Flash-Next, use [Strata](https://github.com/Niko1221/Strata). PolyStrata's other models are not part of
FastStrata: setup still installs them when named with `--family`, as PolyStrata set them up, and says FastStrata does
not support them.

## What FastStrata changed

Each change was measured on [the test PC](#the-test-pc) with Qwen3.6-35B-A3B UD-Q4_K_M, the build before it and the
build with it in turns. "Lossless" means the model computes the same sums, so the answers move no more than between
two runs of one build.

| Step | Option | What it does | What it gave on the test PC | Answers |
| ---: | --- | --- | --- | --- |
| | `--kv-q8` (2026-10-17) | The attention cache in 8 bits: 11.7 KB a token where 22 KB. | A 144K context on the 6 GB card, where 16 bits fit 72K. | Lossy: a mean KL of 0.0025-0.0038 against the 16-bit cache, one token it was 60% or more sure of changed in 4,500 steps. |
| 1 | `--threads 8` | One CPU thread a core. The engine took all 16 threads, which left its own thread, the one that hands the card its work, no core. | After a 37K prompt 46.3 and 46.7 tok/s where 41.7 and 42.6. | Lossless. |
| 2 | conversation cache of 12 GiB, 8 slots | More of a coding agent's conversations come back without being read again (was 8 GiB, 4). | Not measured for speed. | Lossless. |
| 3 | `--attn-split` | An answer's attention in one pass over the cache, on the tensor cores. | After 37K 60.2 and 61.7 tok/s where 46.1 and 48.5; one call of the attention at 37K 167 µs where 649. | The products take 16-bit halves: the top token changed at 4 of 256 steps, each a near tie. |
| 4 | `--graph-experts` | Each layer of an answer as one CUDA graph, its experts in VRAM included; the card and the CPU signal each other through pinned memory. | 10% less time a window of tokens after 2K, 7% after 37K. | Lossless: the same sums, bit for bit. |
| 5 | `--elastic-prefill 1024` | A prompt read in calls of 2,048 tokens where 512, on VRAM the expert cache lends for the read. | Reading 2K x1.49, 37K x1.30 (36.0 s where 47.0), 139K x1.07; writing unchanged; 144K still fits. | The sums of `--prompt-batch 2048`. |
| 6 | (measured, turned down) | A yardstick for what a setting costs the answers, and two new lossy knobs. | See [below](#measured-and-turned-down). | |
| 7 | `--suffix-drafts 256` | Guesses from every conversation the engine has seen since it started, in 160 MiB of RAM (SuffixDecoding). | An agent's task given again in a new conversation 99.3 tok/s where 88.9; a plain chat unchanged. | Lossless: every guess is checked by the model. |
| 8 | `--cpu-multi` | The CPU's Q4_K, Q5_K and Q6_K experts: each row read and unpacked once for all the tokens of a window. | The CPU's experts at 40-43 GB/s where 30-40, of the 42-43 a plain read of the RAM gives; after 2K 78.2 tok/s where 74.6 (1-2% in a later check). | Lossless: ggml's bits. |
| 9 | `--kv-q4`, `--kv-q4-keys q4` | The attention cache's values in 4 bits (8.9 KB a token), then its keys too (6.2 KB), for a card too small for the context. | 184K and the whole 256K on the 6 GB card, where `--kv-q8` holds 144K. | Lossy: a mean KL of 0.0040-0.0079 and 0.0090-0.0127; 8 and 10 sure tokens changed in 4,500 steps. Off unless the context needs it. |
| 10 | the four presets | Old against new for every preset, what the smaller files cost the answers, and a simulated 4 GB card. | [Results](#results). | |
| 11 | `--prompt-cache-message 2048` | A checkpoint where the prompt's last message begins, so the gateway's LODESTAR frame at the end of every prompt is not read again with the turn. | With LODESTAR, a turn's prompt 7.7 s where 19.3 (11,541 tokens read a turn -> 2,640); without it the same 5.0 s. | Lossless; without LODESTAR it costs 0.16 s a prompt and saves as much. |
| 12 | (measured) | Which experts the cache should hold: a routing trace (`STRATA_ROUTE_TRACE`) and a simulator of cache rules, checked against the engine's own counts. | No rule that does not know the future was more than 3% better than the engine's; tables by kind of work (code, documents) did not help. | |
| 13 | `--lend-staging` | The prompt batches' staging buffer (500 MiB) holds cached experts while an answer is written, and goes back before the next batch. | The agent benchmark 85.0 and 83.9 tok/s where 82.1 and 82.0; a simulated 4 GB card a third of the experts from VRAM where none, 11-19% faster without guesses. | Lossless: as between two runs without it. |
| 15 | pieces of 64 MiB; `--stage-next` | A layer's experts copied to the card in pieces, the host's copy under the card's (always on); with `--stage-next` a second staging buffer the expert cache lends a prompt's read, the next layer's experts copied while a layer runs. | An agent's 1,501-token turn after 38K read in 5.0 s with the pieces and 3.9 s with `--stage-next`, where 6.4; the agent benchmark's prompts 90 and 79.6 s where 108. | The pieces: the same tokens. `--stage-next`: as between two runs without it. |
| 17 | `--conversation-disk` | The parked conversations on disk too, written in the background after every answer, so that an engine restart (the Gateway's Stop or close, a model swap) does not read them again. | The next turn of a 40K conversation after a restart 5.5 s where 40.0, of a 70K one 7.2 s where 83.0; the agent benchmark as fast as without it. | Lossless: the tokens after a restart are those without one. |

`--tail-skip 6`, which leaves out an expert outside VRAM that every token of a window ranks seventh or eighth, is
PolyStrata's setting for Qwen3.6 and stays: x1.15 on the same text for a mean KL of 0.0060 (step 6). Ornith 1.5 runs
without it, as in PolyStrata: what it costs Ornith was not measured.

### Measured and turned down

- **`--weight-cut P`** (step 6), which keeps experts until they hold P of a token's routing weight: slower than the
  tail skip at the same cost (x1.09 against x1.15 at a KL of 0.006).
- **`--defer-experts K`** (step 6, Expert Deferral from KTransformers), the CPU's last experts computed while the
  card runs on: with `--tail-skip 6 --defer-experts 5` 5% faster on the same text in the check, but it changed 3
  tokens the reference was 60% or more sure of in 3,000 steps, where today's setting changed none. A changed sure
  token in code is a wrong name or bracket, so it stays off, in the engine, until a task benchmark says it is
  harmless. `--tail-skip 5` (x1.20, a KL of 0.0102, two sure tokens changed) was turned down the same way.
- **2 MB pages and locked RAM** (step 8): the CPU's experts copied into 2 MB pages read no faster than the model's
  file in 4 KB pages (best rounds 0.8-1.6% apart), and would lock 18.7 GiB of RAM that Windows cannot page out.
- **A copy of the experts repacked for the CPU** (step 8): `--cpu-multi` already reads them at the RAM's speed, and a
  copy takes another 18.7 GiB.
- **The next layer's experts read ahead** while the card runs a layer (step 8): inside the spread from run to run.

## Models and presets

| Preset | File | RAM in use while it runs | VRAM | Experts | Cost in answers |
| --- | ---: | ---: | --- | --- | --- |
| **Qwen3.6-35B-A3B UD-Q4_K_M** ([Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) | 22.7 GB | 24.1-27.2 GB | 4.7 GB before any expert; 962 experts (1.7 GiB) fit the 6 GB card at a start | Q4_K, Q5_K, Q6_K: every option reaches them | the reference |
| **Qwen3.6-35B-A3B UD-Q3_K_M** | 17.1 GB | 18.5-21.7 GB | 4.8 GB; 1,341 experts (1.7 GiB) | IQ3_XXS, IQ4_XS: neither `--cpu-multi` nor `--graph-experts` | against UD-Q4_K_M: a mean KL of 0.020, 5 sure tokens changed in 1,500 steps, all in prose or punctuation |
| **Ornith 1.5 APEX-I-QUALITY** ([mudler](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF)) | 23.7 GB | 25.5-28.6 GB | 4.4 GB; 1,080 experts (1.9 GiB) | Q6_K, Q5_K, IQ4_XS: `--cpu-multi` on 20 of 41 layers | the reference |
| **Ornith 1.5 APEX-I-COMPACT** | 17.4 GB | 19.2-22.4 GB | 4.2 GB; 1,823 experts (2.4 GiB) | Q3_K, Q4_K: `--cpu-multi` on 10 of 41 layers | against APEX-I-QUALITY: a mean KL of 0.088, 26-27 sure tokens changed in 1,320 steps, names in code among them |

- **RAM in use** is the engine's working set measured on the test PC (step 10): one conversation up to four parked
  ones, nearly all of the model's file among it, since the engine reads every expert outside VRAM from it while it
  writes. GB are 10⁹ bytes.
- **VRAM** is setup's estimate of what a preset takes before any expert at a 32K context (measured on an RTX 5090),
  then the experts that fit the 6 GB test card at a start.
- **UD-Q3_K_M is not faster than UD-Q4_K_M on FastStrata**: 66.2 against 71.8 tok/s after 2K, 58.6 against 66.9
  after 37K, 45.6 against 48.8 after 139K. Its IQ3_XXS experts get neither `--cpu-multi` nor `--graph-experts`, and
  the CPU takes as long or longer for each. It reads prompts faster. It is for a PC with about 5 GB less RAM.
- **APEX-I-COMPACT is not recommended for coding.** It costs 4.3 times what UD-Q3_K_M costs (once in 50 tokens a token
  the larger file was sure of changes), and among those changes are names in code (`fake_ctypes.c` -> `fake_ctypes.Win`,
  `.Struct` -> `.w`). It writes x1.22 faster than APEX-I-QUALITY on the same text.
- Ornith 1.5 has Qwen3.6-35B-A3B's architecture with other weights, and its draft block guesses right less often, so
  it gains less from guesses. Both models read pictures (about 1 GB more to download).
- **For Qwen3.8-Flash-Next, use upstream [Strata](https://github.com/Niko1221/Strata).**

## The test PC

| | |
| --- | --- |
| Laptop | AMD Ryzen 7 5800H (Zen 3, 8 cores, 16 threads, AVX2), 64 GB DDR4-3200 in two channels (42-43 GB/s read by 8 threads) |
| Graphics card | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB GDDR6 (336 GB/s), on PCIe 3.0 x8 |
| Disks | NVMe SSDs; every model's file was read into the file cache before its runs |
| System | Windows 11 Pro, CUDA 13.0, the engine built for sm_86 |
| Engine | old: the 2026-10-17 build with `--max-context 147456 --kv-q8 --prompt-batch 512 --tail-skip 6 --conversation-cache-mib 8192`, 16 threads; FastStrata: kv-q8 at 150beb0b with setup's flags for this laptop ([Configuration](#configuration)) |

## Results

Old against FastStrata on the test PC, 2026-10-20, greedy, the engine on its own as the server drives it. Each figure
is the mean of two runs, the 139K prompt one run each way. *The same text*: each engine writes the same reference
answers to five agent prompts of 1.3K to 3.7K tokens, so how often a guess comes out right no longer moves the speed;
it is the steadiest figure. *Agent benchmark*: an OpenCode-like session of 35 answers (an agent edits three files, a
subagent reviews them, the task again in a new conversation, and a plain chat), prompts up to 14,268 tokens.

| Writing, tok/s, old -> FastStrata | After 2K | The same text | After 37K | After 139K | Agent benchmark |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 UD-Q4_K_M | 56.4 -> **71.8** (x1.27) | 62.3 -> **72.9** (x1.17) | 41.9 -> **66.9** (x1.60) | 24.7 -> **48.8** (x1.98) | 62.4 -> **82.8** (x1.33) |
| Qwen3.6 UD-Q3_K_M | 58.8 -> **66.2** (x1.13) | 67.3 -> **69.4** (x1.03) | 37.0 -> **58.6** (x1.58) | 26.4 -> **45.6** (x1.73) | 68.4 -> **81.2** (x1.19) |
| Ornith 1.5 APEX-I-QUALITY | 45.8 -> **51.0** (x1.11) | 46.4 -> **52.4** (x1.13) | 28.9 -> **39.6** (x1.37) | 21.5 -> **36.8** (x1.71) | 47.0 -> **61.8** (x1.32) |
| Ornith 1.5 APEX-I-COMPACT | 60.8 -> **63.1** (x1.04) | 57.9 -> **63.6** (x1.10) | 41.8 -> **56.2** (x1.34) | 24.6 -> **43.6** (x1.77) | 60.4 -> **76.1** (x1.26) |

The longer the prompt, the more FastStrata gives. After short prompts UD-Q4_K_M gains the most, the one file every
step reaches. The files are measured each on its own text and are not compared with each other here.

### Qwen3.6-35B-A3B UD-Q4_K_M (22.7 GB)

| | Old | FastStrata | |
| --- | ---: | ---: | ---: |
| Writing after a 2K prompt | 56.4 tok/s | 71.8 tok/s | x1.27 |
| Writing the same text | 62.3 tok/s | 72.9 tok/s | x1.17 |
| Writing after a 37K prompt | 41.9 tok/s | 66.9 tok/s | x1.60 |
| Writing after a 139K prompt | 24.7 tok/s | 48.8 tok/s | x1.98 |
| Reading the 37K prompt | 47.2 s | 36.4 s | x1.30 |
| Reading the 139K prompt | 265.1 s | 248.2 s | x1.07 |
| Agent benchmark, all 35 answers | 62.4 tok/s | 82.8 tok/s | x1.33 |
| The same task again in a new conversation | 65.3 tok/s | 93.6 tok/s | x1.43 |
| 4 GB card, 32K context, `--kv-q8` | not run | 55.0 tok/s (one of two runs stopped) | |
| RAM in use while it runs | 24.1-27.2 GB | 24.1-27.2 GB | |

### Qwen3.6-35B-A3B UD-Q3_K_M (17.1 GB)

| | Old | FastStrata | |
| --- | ---: | ---: | ---: |
| Writing after a 2K prompt | 58.8 tok/s | 66.2 tok/s | x1.13 |
| Writing the same text | 67.3 tok/s | 69.4 tok/s | x1.03 |
| Writing after a 37K prompt | 37.0 tok/s | 58.6 tok/s | x1.58 |
| Writing after a 139K prompt | 26.4 tok/s | 45.6 tok/s | x1.73 |
| Reading the 37K prompt | 45.7 s | 34.8 s | x1.31 |
| Reading the 139K prompt | 258.8 s | 241.2 s | x1.07 |
| Agent benchmark, all 35 answers | 68.4 tok/s | 81.2 tok/s | x1.19 |
| The same task again in a new conversation | 72.8 tok/s | 91.5 tok/s | x1.26 |
| 4 GB card, 32K, `--kv-q8` | 35.3 tok/s | 44.7 tok/s | x1.27 |
| 4 GB card, 16K / 48K, `--kv-q8` | not run | 54.9 / 42.0 tok/s | |
| Answers against UD-Q4_K_M | | a mean KL of 0.020, top token changed at 4.1% of steps, 5 sure ones in 1,500 (UD-Q4_K_M's own tail skip: 0.006, 1.4%, none) | |
| RAM in use while it runs | 18.5-21.7 GB | 18.5-21.7 GB | |

### Ornith 1.5 35B-A3B APEX-I-QUALITY (23.7 GB), no tail skip

| | Old | FastStrata | |
| --- | ---: | ---: | ---: |
| Writing after a 2K prompt | 45.8 tok/s | 51.0 tok/s | x1.11 |
| Writing the same text | 46.4 tok/s | 52.4 tok/s | x1.13 |
| Writing after a 37K prompt | 28.9 tok/s | 39.6 tok/s | x1.37 |
| Writing after a 139K prompt | 21.5 tok/s | 36.8 tok/s | x1.71 |
| Reading the 37K prompt | 47.7 s | 36.7 s | x1.30 |
| Reading the 139K prompt | 269.2 s | 236.9 s | x1.14 |
| Agent benchmark, all 35 answers | 47.0 tok/s | 61.8 tok/s | x1.32 |
| The same task again in a new conversation | 47.9 tok/s | 73.6 tok/s | x1.54 |
| 4 GB card | not run | not run | |
| RAM in use while it runs | 25.5-28.6 GB | 25.5-28.6 GB | |

### Ornith 1.5 35B-A3B APEX-I-COMPACT (17.4 GB), no tail skip

| | Old | FastStrata | |
| --- | ---: | ---: | ---: |
| Writing after a 2K prompt | 60.8 tok/s | 63.1 tok/s | x1.04 |
| Writing the same text | 57.9 tok/s | 63.6 tok/s | x1.10 |
| Writing after a 37K prompt | 41.8 tok/s | 56.2 tok/s | x1.34 |
| Writing after a 139K prompt | 24.6 tok/s | 43.6 tok/s | x1.77 |
| Reading the 37K prompt | 46.5 s | 35.2 s | x1.32 |
| Reading the 139K prompt | 268.6 s | 224.6 s | x1.20 |
| Agent benchmark, all 35 answers | 60.4 tok/s | 76.1 tok/s | x1.26 |
| The same task again in a new conversation | 63.5 tok/s | 89.4 tok/s | x1.41 |
| 4 GB card, 32K, `--kv-q8` | not run | 47.8 tok/s | |
| Answers against APEX-I-QUALITY | | a mean KL of 0.088, top token changed at 7.7% of steps, 26-27 sure ones in 1,320 (APEX-I-QUALITY run again: 0.010, 2.0%, 1-4) | |
| RAM in use while it runs | 19.2-22.4 GB | 19.2-22.4 GB | |

What these do not cover: a task benchmark (HumanEval, a tool call's success rate), thinking on, sampled answers, a
second run at 139K (a second run of one build moved by several percent there), the server and an agent app in front
of the engine, other cards and CPUs, Linux.

## On a 4 GB gaming card

No 4 GB card was at hand. The 4 GB card here is **the test PC's 6 GB card with 2 GiB of its VRAM held by another
process** (`test_scripts/vram_hold.py --card 4096`), which left the engine 3,082 MiB free, about what a 4 GB card
leaves once the desktop runs on it. It shows what fits in that VRAM. It does not show a real 4 GB card's memory
bandwidth, its PCIe link (this one is 3.0 x8) or how much of it a given desktop takes.

FastStrata, a prompt that fills the context, then 256 tokens written:

| Preset | Context | `--kv-q8` | `--kv-q4` | Prompt read in |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 UD-Q3_K_M | 16K | 54.9 tok/s | 56.0 tok/s | 17.8 s |
| | 32K | 39.6, 49.8 tok/s (two runs) | 48.0, 48.4 tok/s | 37.3-37.5 s |
| | 48K | 42.0 tok/s | 47.9 tok/s | 60.2-60.5 s |
| Qwen3.6 UD-Q4_K_M | 32K | 55.0 tok/s; 53.5 without `--graph-experts` | | 39.2-39.3 s |
| Ornith 1.5 APEX-I-COMPACT | 32K | 47.8 tok/s | | 39.7 s |

- **Every run fitted, and nothing was paged to system RAM.** Neither Qwen3.6 file leaves room for one expert in VRAM
  beside its dense weights and caches, so the CPU computes them all; APEX-I-COMPACT's smaller dense weights leave it
  398 experts at the start.
- **Old -> new at 32K with UD-Q3_K_M: 35.3 -> 44.7 tok/s, x1.27**; the prompt in 39.1 -> 37.4 s.
- **UD-Q4_K_M writes faster than UD-Q3_K_M there**, 55.0 against 44.7 tok/s at 32K: with every expert on the CPU,
  Q4_K experts take `--cpu-multi` and IQ3_XXS ones do not. It needs 24-27 GB of RAM against 19-22.
- **`--kv-q8` fits 48K**, so the 4-bit cache is not needed up to there. Past 48K was not run on the simulated card;
  at 64K (step 9, UD-Q4_K_M) the card's memory went to system RAM and writing fell to 19.1 tok/s with `--kv-q8` and
  31.0 with `--kv-q4`.
- **One run of UD-Q4_K_M stopped** with "graph experts: the card gave up waiting for the host"; the second wrote 55.0
  tok/s and one without `--graph-experts` 53.5. The cause was not found, so setup gives no `--graph-experts` to a card
  under 5 GB.
- **RAM on a 32 GB PC:** an engine with a 17 GB file (UD-Q3_K_M, APEX-I-COMPACT) holds 19-22 GB of RAM while it runs,
  which leaves a 32 GB PC (34.4 GB) 12-15 GB for Windows and other programs; the test PC's Windows and desktop used
  15.6-20.5 GB that day, so a lighter desktop is needed. The 23 GB files hold 24-29 GB and leave a 32 GB PC too
  little. Not measured on a PC with 32 GB.
- **Setup on a 4 GB card** offers these presets with the contexts measured here and the 8-bit cache: UD-Q3_K_M up
  to 48K, UD-Q4_K_M and APEX-I-COMPACT at 32K. Its own estimate of the cache, made for larger cards, calls a 4 GB card
  too small for any context, so below 5.5 GB setup goes by these runs.

## What you need

### Which card

| VRAM | What it gives | How it is known |
| --- | --- | --- |
| **4 GB: the minimum** | A short context. UD-Q3_K_M runs 16K to 48K, UD-Q4_K_M and APEX-I-COMPACT 32K, all with the 8-bit cache, writing at 42 to 56 tok/s. 16K is the smallest context setup offers; 32K is the least a coding agent like OpenCode works with comfortably. | Measured on a simulated 4 GB card (3,082 MiB free), [above](#on-a-4-gb-gaming-card) |
| **6 GB: sensible** | **Q4 (UD-Q4_K_M) with the 8-bit cache at 144K**: every result on this page. | Measured: [the test PC](#the-test-pc) |
| **8 GB: optimal** | Q4 with the 8-bit cache at about 248K (Ornith the whole 256K window), or 144K with the 16-bit cache, and more of the experts in VRAM. | Setup's estimate less 10%; not measured with FastStrata (PolyStrata ran these presets on an 8 GB laptop card) |

The setting to take: **Q4 and the 8-bit cache** (`--kv-q8`). Setup recommends it on a 6 or 8 GB card and a PC with
about 25 GB of RAM free for it; on a 4 GB card it recommends the smaller Q3.

| | |
| --- | --- |
| **Graphics card** | One NVIDIA GeForce RTX 20 series card or newer. Measured on a 6 GB laptop card and on 4 GB simulated on it. PolyStrata ran these presets on an 8 GB laptop card and a 32 GB card without FastStrata's options ([docs/BACKENDS.md](BACKENDS.md#benchmark)). |
| **RAM** | 24-29 GB in use for the 23 GB files and 19-22 GB for the 17 GB files on the 6 GB card (above). A card with more VRAM needs less. |
| **Processor** | x86-64 with AVX2 for `--cpu-multi`; 8 cores were measured. |
| **Disk** | 17 to 24 GB for the model's file, 1 GB more to read pictures. An SSD: the experts are read from the file. |
| **System** | Windows 10 / 11 or Linux, a current NVIDIA driver. |

What setup's menu says of each preset, by card: the longest context it offers with the cache in 16 or 8 bits, and
the RAM it wants (in GiB as the system reports it; a "32 GB" PC reports a little under 32). The 6 GB and 4 GB rows
were measured with the whole card free; the others, and a card whose desktop takes some of its VRAM, get setup's
estimate less 10%, and the menu says "estimated". Reports from other PCs will replace the estimates:

| Card | Qwen3.6-35B-A3B | Ornith 1.5 |
| --- | --- | --- |
| 4 GB | UD-Q4_K_M 32K from 26.1 GB of RAM, UD-Q3_K_M 48K from 20.6 | APEX-I-QUALITY 24K (estimated) from 27.6, APEX-I-COMPACT 32K from 21.8 |
| 6 GB | UD-Q4_K_M 144K from 24.7, UD-Q3_K_M 144K from 19.3 | APEX-I-QUALITY 144K from 25.9, APEX-I-COMPACT 144K from 19.9 |
| 8 GB | UD-Q4_K_M 248K from 22.6, UD-Q3_K_M 240K from 17.2 | APEX-I-QUALITY 256K from 23.7, APEX-I-COMPACT 256K from 17.7 |
| 12 GB | UD-Q4_K_M from 18.3, UD-Q3_K_M from 12.9 | APEX-I-QUALITY from 19.4, APEX-I-COMPACT from 13.4 |
| 16 GB | UD-Q4_K_M from 14.0 | APEX-I-QUALITY from 15.1 |

Setup recommends the first preset of the menu that runs 32K or more on the PC, and the longest context up to 144K: a
longer one reads a full prompt for minutes (139K in about four on the test PC).

An AMD card on Linux gets the options that run on the CPU (`--threads`, `--suffix-drafts`, `--cpu-multi`, the
conversation cache); the others are CUDA code, not built for AMD. Not measured with FastStrata. Not on a Mac.

## Configuration

Setup writes the engine's options into the model's run config, `strata-<family>-<preset>.json` (for example
`strata-qwen36-ud-q4_k_m.json`), under `"args"`. On the test PC, for UD-Q4_K_M at 144K:

```text
--max-context 147456 --kv-q8 --prompt-batch 512 --tail-skip 6 --threads 8 --attn-split --graph-experts
--elastic-prefill 1024 --lend-staging --stage-next --suffix-drafts 256 --cpu-multi
--conversation-cache-mib 12288 --conversation-cache-slots 8 --prompt-cache-message 2048
--conversation-disk <the data folder>\conversations --conversation-disk-mib 4096
```

Each where the PC and the preset take it, and setup prints each with what it gave:

| Option | Setup gives it to |
| --- | --- |
| `--threads N` | a CPU whose cores are alike and run two threads each: one thread a core (a hybrid Intel CPU keeps its own rule) |
| `--attn-split` | NVIDIA from Turing (RTX 20) on |
| `--graph-experts` | one NVIDIA card from Volta on with 5 GB of VRAM or more; not UD-Q3_K_M, whose experts it does not read |
| `--elastic-prefill 1024` | one NVIDIA card |
| `--lend-staging` | one NVIDIA card |
| `--stage-next` | one NVIDIA card with 5 GB of VRAM or more |
| `--suffix-drafts 256` | a PC with the RAM setup estimates the preset wants |
| `--cpu-multi` | a CPU with AVX2 and a preset with K-quant experts (not UD-Q3_K_M) |
| `--kv-q8`, `--prompt-batch 512` | NVIDIA, where the context asked for does not fit the card in 16 bits; setup asks |
| `--kv-q4`, then `--kv-q4-keys q4` | NVIDIA, only where even `--kv-q8` with batches of 512 does not fit; setup says what it costs and asks |
| conversation cache 12 GiB, 8 slots | a PC with 24 GB of RAM beyond what the preset wants |
| `--prompt-cache-message 2048` | NVIDIA |
| `--conversation-disk`, `--conversation-disk-mib 4096` | NVIDIA, with the conversation cache on, where the data folder's disk has 20 GB free |

- **All speedups off:** `START-HERE.bat --setup --speedups off` (`./setup.sh --setup --speedups off`) writes the
  config as PolyStrata's setup did, and later setup runs keep that choice; `--speedups on` turns them on again. One
  option alone: delete it from `"args"`; a later `--setup` run writes it back.
- **A long context on a small card:** where `--kv-q8` does not fit, replace `"--kv-q8"` with `"--kv-q4"` (or
  `"--kv-q4-keys", "q4"`) and raise `--max-context`. It costs answers ([step 9](OPTIMIZATION_LOG.md)), so take
  it only for a context that does not fit otherwise.
- **Never turned on by setup:** `--weight-cut`, `--defer-experts`, a deeper tail skip ([above](#measured-and-turned-down)).
- Every option in detail, with its own measurements: [docs/FASTSTRATA_OPTIONS.md](FASTSTRATA_OPTIONS.md). The
  rule is `poly_speedups()` in [setup.py](../setup.py), its tests `tools/test_setup_speedups.py`.

## How it was measured

- [docs/OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md): every step, what was measured, how, against what, and what
  was not measured.
- The scripts the log names (`test_scripts/`: `run_serve.py`, the engine on its own; `agent_bench.py`, the agent
  session; `quality.py`, what a setting costs the answers; `fit_run.py`, how long a context fits; `vram_hold.py`, a
  smaller card simulated; `proc_mem.py`, the engine's RAM; `faststrata_runs.py`, step 10's runs) are the working
  repository's and are not published here; ask in an issue for one you want to run.
- Old and new engines in turns, a fresh copy of the expert profile each run, nothing else on the card, greedy answers.
- A report from your own PC is welcome: `BENCH.bat` (`./bench.sh`) measures a chat, long prompts and an agent-style
  session ([how](COMMUNITY_BENCHMARKS.md#run-the-benchmark)). Please say what you did not test.
