# FastStrata's options in detail

What each of FastStrata's options does, when setup gives it, and what it gave on the test PC (an RTX 3060
Laptop 6 GB, a Ryzen 7 5800H, 64 GB of DDR4-3200, Windows 11, Qwen3.6-35B-A3B UD-Q4_K_M). The summary and
the four presets' results are in the [README](../README.md); every run is in
[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md).

## All of them together

Eight changes to the `strata-poly` engine and its settings, each measured on one laptop with Qwen3.6-35B-A3B, the
old and the new build in turns. Every run, and what was tried and left out, is in
[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md). All of them together, as measured after step 8 (step 10 measured them
again against the 2026-10-17 engine for all four presets: [RESULTS.md](RESULTS.md#results)):

| | Before: the engine's defaults with `--kv-q8` | Now: setup's config for this laptop |
| --- | ---: | ---: |
| Writing after a 2K prompt | 46.7 to 61.4 tok/s | 71.3 to 73.3 tok/s |
| Writing after a 37K prompt | 41.7, 42.6 tok/s | 62.1, 64.9 tok/s |
| Writing after a 100K prompt | 34.4 tok/s | 63.9 tok/s |
| Writing after a 139K prompt | 23 tok/s (at 150K, 2026-10-17) | 49.4 tok/s |
| Reading a 2K prompt | 5.4 s | 3.6 s |
| Reading a 37K prompt | 47 s | 36 s |
| Reading a 139K prompt | about 270 s (150K, 2026-10-17) | 245.7 s |

Measured on: a laptop with an NVIDIA RTX 3060 Laptop GPU (6 GB, PCIe 3.0 x8), an AMD Ryzen 7 5800H (8 cores, 16
threads), 64 GB of DDR4-3200 (60 GB to Windows), Windows 11, CUDA 13.0; Qwen3.6-35B-A3B UD-Q4_K_M (22.7 GB);
`--max-context 147456 --kv-q8 --prompt-batch 512 --tail-skip 6`; 2026-10-18 and 19 unless marked; greedy; the engine
on its own as the server drives it (`test_scripts/run_serve.py`), 300 tokens written after 2K, 250 after 37K, 400
after 139K. The 100K figures are from the session that wrote `--attn-split` (128-token answers). On top of that, a
coding agent's task given again in a new conversation was written at 97.9 and 106.7 tok/s where 90.1 (step 7).

A ninth, `--kv-q4`, is not a speedup: it fits a longer context into a smaller card, at a cost to the answers
([below](#--kv-q4-the-context-cache-in-4-bits-for-a-card-too-small-for-the-context)).

## What setup turns on, and when

FastStrata's two models come in two presets each:

| Preset | File | Download | RAM beside an 8 GB card | VRAM setup asks for | Speed measured with FastStrata's options | `--cpu-multi`, `--graph-experts` |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen3.6-35B-A3B UD-Q4_K_M | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22.7 GB | ~27 GB | 6.7 GB | yes: every number on this page, and step 10 | both: Q4_K, Q5_K and Q6_K experts |
| Qwen3.6-35B-A3B UD-Q3_K_M | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | 17.1 GB | ~21 GB | 6.8 GB | step 10 | neither: its experts are IQ3_XXS and IQ4_XS, which neither reads |
| Ornith 1.5 APEX-I-QUALITY | `Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf` | 23.7 GB | ~28 GB | 6.4 GB | step 10 | `--cpu-multi` on its 20 K-quant layers of 41; `--graph-experts` |
| Ornith 1.5 APEX-I-COMPACT | `Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf` | 17.4 GB | ~22 GB | 6.2 GB | step 10 | `--cpu-multi` on its 10 Q4_K layers of 41 (Q3_K takes the old path); `--graph-experts` |

The RAM and VRAM are setup's estimates, at a 32K context: the VRAM a preset takes before any expert (measured on an
RTX 5090) and 2 GB of experts, and in RAM the experts the card does not hold, read through the OS file cache, and
10 GB for the rest. A larger card needs less RAM: about 10 GB once it holds every expert. A longer context takes more
VRAM: 26 KB a token past 32K, 16 KB with `--kv-q8`, 13 KB with `--kv-q4`, 10 KB with `--kv-q4-keys q4`.

Setup takes the first preset of the model that this PC has the RAM and VRAM for: UD-Q4_K_M (APEX-I-QUALITY), else
UD-Q3_K_M (APEX-I-COMPACT), else it says the PC is too small. `--model NAME --yes` installs a preset anyway: the 6 GB
laptop of the numbers above has less VRAM than setup asks for and runs UD-Q4_K_M at a 144K context, installed so. The
options are the architecture's; `--cpu-multi` and `--graph-experts` go only to a preset whose experts they read.

Setup (`START-HERE.bat`, `setup.sh`) then writes these into the model's config (`strata-<model>.json`) where the PC's
card, CPU and RAM take them, and prints each one with what it gave:

| Option | What it does | What it gave on the laptop | Setup gives it to |
| --- | --- | --- | --- |
| `--threads N` | one CPU thread a core for the experts; the engine takes every logical CPU | after 37K: 46.3, 46.7 tok/s with 8 threads, 41.7, 42.6 with all 16 (step 1) | a CPU whose cores are alike and run two threads each. A hybrid CPU keeps setup's earlier rule: with more efficiency cores than performance cores, the performance cores but one and half the efficiency cores |
| `--attn-split` | an answer's attention in one pass over the cache, on the tensor cores | after 37K: 60.2, 61.7 tok/s where 46.1, 48.5 (step 3) | NVIDIA cards from Turing (RTX 20) on |
| `--graph-experts` | each layer of an answer as one CUDA graph | 10% less time a window after 2K, 7% after 37K (step 4) | one NVIDIA card from Volta on with 5 GB of VRAM or more, not a layer split, not UD-Q3_K_M |
| `--elastic-prefill 1024` | a long prompt read in wider batches, on VRAM the expert cache lends it | 37K read in 36.0 s where 47.0, 139K in 245.7 s where 264.1 (step 5) | one NVIDIA card |
| `--suffix-drafts 256` | guesses from every conversation the engine has seen, in 160 MiB of RAM | a task given again in a new conversation 97.9, 106.7 tok/s where 90.1; a plain chat as before (step 7) | a PC with the RAM setup estimates the preset wants |
| `--cpu-multi` | the CPU's Q4_K, Q5_K and Q6_K experts read once for all the tokens of a window, the same bits | 78.2 tok/s after 2K where 74.6; 1-2% in a later check (step 8) | a preset with K-quant experts (not UD-Q3_K_M), on a CPU with AVX2 |
| `--kv-q8` | the attention cache in 8 bits: 11.7 KB a token where 22 KB | a 144K context on the 6 GB card | NVIDIA, when the context asked for does not fit the card in 16 bits by setup's estimate. Setup asks (`--yes` takes it) |
| `--prompt-batch 512` | prompt batches of 512 tokens where the engine takes 1,024: 0.7 GB more VRAM for the context | the 144K context (with `--kv-q8`) | as `--kv-q8`, where 8 bits alone do not make the context fit |
| `--kv-q4`, then `--kv-q4-keys q4` | the attention cache's values in 4 bits and its keys in 8 (8.9 KB a token), then the keys in 4 bits too (6.2 KB); lossy, more than `--kv-q8` ([below](#--kv-q4-the-context-cache-in-4-bits-for-a-card-too-small-for-the-context)) | 184K and 256K on the 6 GB card, where `--kv-q8` holds 144K | NVIDIA, only where even `--kv-q8` with batches of 512 does not fit the context asked for, by setup's estimate. Setup says what it costs and asks (`--yes` takes it) |
| `--conversation-cache-mib 12288`, `--conversation-cache-slots 8` | 12 GiB for 8 parked conversations where 8 GiB for 4 | not measured for speed (step 2) | a PC with 24 GB of RAM more than setup estimates the preset wants (the laptop: 60 against 29) |

- **AMD cards** get `--threads`, `--suffix-drafts`, `--cpu-multi` and the conversation cache: the others are CUDA
  code that was neither built nor run for HIP.
- **Ornith 1.5** gets them all; step 10 measured them with it. It gets no `--tail-skip`: what that costs its answers
  was not measured. The models FastStrata does not support get none of these options.
- **What setup never turns on:** what changes the answers to buy speed. `--tail-skip 6` stays as setup wrote it for
  Qwen3.6 before, and `--weight-cut` and `--defer-experts` stay off (step 6: `--defer-experts 5` was 5% faster and
  changed tokens the reference answers were sure of; `--weight-cut` was slower than the tail skip at the same cost).
  `--kv-q8` rounds the cache to 8 bits, and `--kv-q4` to 4: against the 16-bit cache the first changed the most
  likely token at 1.2 to 2.3% of an answer's steps and once in 4,500 where that cache was 60% or more sure, the second
  at 1.9 to 3.3% and 8 times (step 9). Setup gives them only where the context does not fit otherwise, the one of the
  most bits that fits, says what it costs, and asks.
- **The engine.** No ready-made engine of this fork exists. The release's engines are named after the hash of the
  source they were built from, so setup finds none for this source and compiles it (it installs the C++ compiler and
  the CUDA toolkit first, asking once; some minutes). Before it writes an option, setup looks for it in the engine's
  file: an engine without it (the PolyStrata project's release, which has none of them) does not get it, and setup says so.
- **Turning them off.** `START-HERE.bat --setup --speedups off` (`./setup.sh --setup --speedups off` on Linux) writes
  the config as setup did before them, and setup run again keeps that choice; `--speedups on` turns them on again.
  One option alone: delete it from `args` in `strata-<model>.json`; a later `--setup` run writes it back. The rule is
  `poly_speedups()` in [setup.py](../setup.py), its tests `tools/test_setup_speedups.py`.
- **A card under 5 GB** gets no `--graph-experts`: on the simulated 4 GB card one run of two stopped with "graph
  experts: the card gave up waiting for the host" (cause not found), and it gave about 3% there (55.0 against 53.5
  tok/s, step 10).
- **Not measured:** other cards and CPUs, Linux, AMD cards, a real 4 GB card, a layer split. Not built: the engine on Linux (where GCC compiles `--cpu-multi`'s AVX2 file), HIP and SYCL.

## `--kv-q8`: the context cache in 8 bits

This fork adds the option **`--kv-q8`** to the `strata-poly` engine: the attention caches of Qwen3.5 / 3.6
(GGUF architecture `qwen35moe`) are kept in Q8_0, int8 values with an fp16 scale for each 32, instead of fp16. For
Qwen3.6-35B-A3B that is 11.7 KB a token instead of 22 KB. The cache takes its VRAM from the expert cache as the
conversation grows, so the same card holds about twice the context: on a 6 GB laptop card 144K tokens, where fp16
runs out at about 75K. Without the option the engine is unchanged: its fp16 attention kernels give the old kernels'
results bit for bit (a test of `attend()` and `attend_batch()` on fixed inputs).

Setup gives it where the context asked for needs it ([above](#what-setup-turns-on-and-when)). By hand: add
`"--kv-q8"` to the `args` of the model's config (`strata-qwen36-<size>.json`) and raise `--max-context`. There is no
ready-made engine with the option: setup compiles it from this repo (by hand: CUDA 13.0, Visual Studio 2022 Build
Tools; the steps are in [AGENTS.md](../AGENTS.md#local-modifications--gateway-integration-do-not-overwrite), item 5). It
was measured with Qwen3.6-35B-A3B only; Ornith 1.5 has the same architecture and was not tested.

**Speed by how full the context is**, the PC and settings below:

| Context in use | Writing the answer | Reading the prompt the first time |
| --- | ---: | ---: |
| a short chat (under 1K tokens) | 40-55 tok/s | at once |
| 3K | ~55 tok/s | ~7 s |
| 32K | 44-47 tok/s | 40 s (~800 tok/s) |
| 62K | 39-41 tok/s * | ~55-60 s * |
| 120K | 27-28 tok/s * | ~160 s * |
| 130K | ~27 tok/s * | ~4 min * |
| 146K (the window almost full) | 21.5 tok/s | 261 s (~560 tok/s) |

\* Measured with the default `--prompt-batch 1024`. The batch did not change the writing speed (47 against 49 tok/s at
32K), and with 512 a prompt reads about 17% slower, which the reading times of these rows include.

A long prompt is read once: the same conversation continues from the server's conversation cache, and the next answer
starts after about 0.2 s. The writing speed depends on the text too, since the model's MTP block guesses tokens ahead:
code and repetitive text go faster than free prose. With fp16 and a 64K window the same PC wrote 49 tok/s at 32K and
37 at 62K (the same prompts with `--kv-q8`: 54 and 41).

| Measured on | |
| --- | --- |
| Computer | laptop, AMD Ryzen 7 5800H (8 cores, 16 threads), 60 GB RAM, Windows 11 Pro |
| Graphics card | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB, driver 617.42 |
| Model | Qwen3.6-35B-A3B UD-Q4_K_M by Unsloth (22.7 GB, with the MTP block) |
| Engine | `strata-poly` built with CUDA 13.0 for sm_86; `--kv-q8 --max-context 147456 --prompt-batch 512 --tail-skip 6 --conversation-cache-mib 8192` |
| Test | 2026-10-17, greedy, reasoning off, answers of about 190 tokens, the prompt cut from this repo's `docs/*.md` |

**Why 144K and not more:** at 159K tokens (`--prompt-batch 512`) no expert was left in VRAM and none of it was free.
Past that, Windows moves the card's memory to system RAM, which slowed decoding several times in measurements by
the engine's authors. At 144K about 0.17 GB of experts stay as a margin that other programs can take. More experts in VRAM
did not speed up the writing here (891 against 1,207 experts: 49 tok/s at 32K both times), so the VRAM goes to the
context.

## `--attn-split`: an answer's attention in one pass over the cache

While an answer is written, the engine runs 1 to 8 tokens at a time (the one it knows and its guesses). For each
attention layer the old kernels read the keys, write every score to memory, read the scores back for the softmax and
then read the values in only 16 launch blocks, at about 46 GB/s of the card's 336 GB/s. At a 37K context that took
a quarter of the time an answer was written (24.5% in the profile below). With **`--attn-split`** one kernel reads each 512 rows of the cache once for
all the tokens of the call, on the tensor cores, and keeps the scores, the softmax and the weighted values as it goes;
a small second kernel adds the blocks up. It is for Qwen3.5 / 3.6 (with or without `--kv-q8`) on NVIDIA cards from
Turing (RTX 20) on; anywhere else, and without the option, the old kernels run unchanged. To use it, add
`"--attn-split"` to the `args` of the model's config.

| Context in use | Writing the answer, old kernels | with `--attn-split` | Time a window of tokens takes |
| --- | ---: | ---: | ---: |
| 2K | 72.7 tok/s | 75.4 tok/s | 37.7 -> 37.4 ms |
| 37K | 55.2 tok/s | 72.9 tok/s | 55.8 -> 43.5 ms |
| 100K | 34.4 tok/s | 60.4 tok/s | 77.3 -> 50.1 ms |

The same PC as above, `--kv-q8 --max-context 147456 --prompt-batch 512 --tail-skip 6`, greedy, the prompt cut from
this repo's source code, the old and the new kernels in turn twice; each value is the mean of four answers of 128
tokens that went on from the cached prompt. Code is easy to guess, so 84-94% of the guesses were right: these
speeds are higher than the table above for any text, the ratio between the two columns is the point. A window's
time does not depend on how many guesses were right. The first answer after a prompt, which also records the
layers' CUDA graphs, went from 50.0 to 74.3 tok/s at 37K and from 32.3 to 52.3 at 100K.

The attention of one call alone (`poly_attend_split_test --time`, random cache in Qwen3.6's shape, Q8_0):

| Tokens in the call | at 37K: old -> new | at 100K | at 147K |
| --- | ---: | ---: | ---: |
| 1 | 649 -> 167 us | 1,681 -> 401 us | 2,505 -> 573 us |
| 2 | 1,121 -> 195 us | 2,636 -> 439 us | 3,917 -> 622 us |
| 4 | 2,607 -> 254 us | 6,077 -> 573 us | 9,011 -> 853 us |
| 8 | 4,312 -> 421 us | 10,657 -> 1,104 us | 15,755 -> 1,655 us |

With one token at 147K the new kernel reads the cache at 279 GB/s. In an Nsight Systems profile of an answer at 37K
the attention kernels went from 24.5% of the time to 6.6% (841 to 170 us a call).

The products take the queries, the cache's values and the softmax's weights as halves and add floats, as the engine
already does when it reads a prompt, so the answers can move a little. On random caches the results differ from the
same attention in doubles by up to 7e-4 of the largest value (the old kernels: under 1e-6). Following the same answer
token by token, the most likely token changed at 4 of 256 steps (4 prompts, `--window 1`) and at 3 of 256 with
guesses on, each time where two tokens were within 0.12 of each other in probability; for comparison, the old kernels
with `--window 1` against `--window 8` changed it at 8 of 256 steps. A token still comes out the same, bit for bit,
whatever tokens are beside it in a call. Not tested: AMD cards, Turing, other models than Qwen3.6-35B-A3B.

## `--graph-experts`: a layer as one launch, the card and the CPU taking turns

While an answer is written, every layer goes from the card to the CPU and back. The card runs the layer up to its
router; the engine's thread waits for it, routes, hands the card the experts that are in VRAM (five launches) and
computes the others itself, then hands the card the rest of the layer. In an Nsight Systems profile of an answer of
400 tokens after a short prompt, the engine's thread spent 3.9 of the 5.8 seconds inside CUDA calls: 85,000 kernel
launches (1.2 s), 9,200 waits for the router's event (1.1 s) and 24,900 graph launches (0.9 s).

With **`--graph-experts`** each layer of an answer is one recorded CUDA graph: the end of the layer before it, the
layer up to its router, the shared expert, and the experts in VRAM. Those experts' kernels read the list the host
makes of them (which slot, which token, what weight) from VRAM, so their launches are the same for every token.
Inside the graph the card waits on a word in pinned memory for that list and for the CPU's experts, and writes one
when the router's scores are out; the host spins on that word instead of waiting for an event. The engine's thread
launches each layer's graph while the card is on the one before, and otherwise only routes and computes the CPU's
experts. It is for Qwen3.5 / 3.6 on one NVIDIA card from Volta (compute capability 7.0) on; anywhere else the
layers run as without it. To use it, add `"--graph-experts"` to the `args` of the model's config.

| Context in use | Writing the answer, without | with `--graph-experts` | Time a window of tokens takes |
| --- | ---: | ---: | ---: |
| 2K | 79.3 tok/s | 81.8 tok/s | 34.3 -> 30.7 ms |
| 37K | 73.4 tok/s | 75.2 tok/s | 41.4 -> 39.4 ms |
| 100K | 62.1 tok/s | 63.9 tok/s | 45.1 -> 44.8 ms |

The same PC and settings as the `--attn-split` table, with `--attn-split --threads 8` as this PC's config has them;
the build without the option and the one with it in turn twice, each value the mean of four answers of 128 tokens
that went on from the cached prompt. How many guesses were right moves the tokens/s from answer to answer (69 to
82 tok/s at 37K without the option), so the time a window takes is the steadier figure. An earlier run of the same
code gave 33.1 -> 29.2 ms at 2K, 40.5 -> 38.9 at 37K and 43.0 -> 42.8 at 100K. At 100K the card's attention is
most of a layer, and that is the same either way.

In the profile above the answer took 5.1 s where 5.8, and the engine's thread spent 1.5 s in CUDA calls: 7,700 graph
launches (0.58 s), 14,700 kernel launches (0.17 s: the guessing block's, the logits' and those of the few calls
that are not replayed, none of which are in graphs) and 0.70 s waiting for the card at the end of a window or a
guess; 5 ms went to events.

What was tried and left out: a first version took a window's tokens through the layers together only, where the
engine can also take them in two halves, one half's layer on the card while the CPU computes the other's experts.
It was no faster (33.6 -> 32.9 ms at 2K, 43.0 -> 43.9 at 100K), so the halves are in, chosen by timing as
before. A thread of its own for the launches was slower than the engine's thread (31.4 against 29.2 ms at 2K, 44.5
against 42.8 at 100K, in the same run): it took a core from the CPU's experts. The experts' kernel was first laid out
as the kernel it replaces, an entry of the list a row of blocks; sized for the longest list, the blocks past a
shorter one's end took 45 microseconds at 32 entries, longer than the products. A block now takes every second
entry, which is within 15% of the old kernel from 1 to 32 entries (`poly_expert_list_test --time`).

The host routes as before and every kernel gives the bits of the launches it replaces, so the sums are the old
path's: with the tail skip off (`STRATA_ROUTE_TAIL_SKIP=0`) and the expert cache held as it was loaded
(`--expert-swaps 0 --vram-reserve-mib 1024`), the build without the option and the one with it wrote the same 712
greedy tokens on five prompts. With `--tail-skip` an answer also depends on whether a window went through a layer
whole or in halves, which the engine chooses by timing: the build without the option, run twice, parted from itself
on four of the five prompts, and the one with it parted from it at the same tokens. `poly_expert_list_test` (needs
a card) checks the new kernel against the old one, bit for bit. Not tested: other cards, other models than
Qwen3.6-35B-A3B; on AMD (HIP) it is not built and the option leaves the layers as they were.

## `--elastic-prefill`: a long prompt in a wider batch, on VRAM the expert cache lends it

A prompt is read in calls of `--prompt-batch` tokens, and a wider batch reads faster: on this PC 793 tok/s in calls of
512 and 1,040 in calls of 2,048 (37K tokens). But the batch's buffers take about 0.4 MiB of VRAM a token for as long
as the engine runs, out of the expert cache: 962 experts in VRAM with 512, 601 with 2,048. That is why the 144K
config reads in calls of 512. While a prompt is read the expert cache helps little, since every expert outside it
passes through the staging buffer anyway.

With **`--elastic-prefill N`** a prompt of N tokens or more borrows that VRAM for the read: the expert cache gives up
the experts it values least (those it would give up for a longer context), the wider batch's buffers are mapped in
their place, and when the read is done the buffers go and the same experts are uploaded again, before the answer's
first token. The batch is the widest that fits in the VRAM free and the expert cache, of `--elastic-batch` (default
4096) and its halves, and never wider than the longest stretch of the prompt between two of the server's checkpoints,
which no call crosses (one every 2,048 tokens by default, `--prompt-cache-every`). Near a full context nothing fits,
and the prompt is read as before. It is for Qwen3.5 / 3.6 on one card that maps VRAM a piece at a time; anywhere
else, and without the option, nothing changes. To use it, add `"--elastic-prefill", "1024"` to the `args` of the
model's config.

| Prompt read fresh | without | with `--elastic-prefill 1024` | Calls of | The expert cache lent / had back in |
| --- | ---: | ---: | ---: | ---: |
| 2K (1,972 tokens) | 5.49 s (359 tok/s) | 3.66 s (539 tok/s) | 2,048 | 860 MiB / 0.24 s |
| 37K | 46.7 s (801 tok/s) | 36.0 s (1,037 tok/s) | 2,048 | 832 MiB / 0.16 s |
| 100K | 165.7 s (603 tok/s) | 139.5 s (717 tok/s) | 2,048, then 1,024 | 832 and 384 MiB / 0.16 and 0.08 s |

The same PC as above with this PC's Qwen3.6 config as it is (`--kv-q8 --max-context 147456 --prompt-batch 512
--tail-skip 6 --threads 8 --attn-split --graph-experts --conversation-cache-mib 12288 --conversation-cache-slots 8`,
the prompt cache's checkpoints on), the build without the option (this PC's `engine-poly` build) and the one with it
in turn twice, a new engine for each prompt so that nothing is resumed, then 128 tokens written. The engine reads at
most 65,536 tokens at a time, and before each read the attention caches take the VRAM for its rows: the 100K prompt
is two reads, and the second finds less of the expert cache to lend. The time to lend the cache (20 to 60 ms) and to
have it back is in the prompt's time.

Without checkpoints (`--prompt-cache 0`) a call can be longer than 2,048 tokens: the 37K prompt as an engine's first
request went in calls of 4,096 in 33.7 s (1,110 tok/s, against 46.5 s without the option), the cache lending 1.4 GiB
and having it back in 0.29 s. With `--prompt-cache 0` the 100K and 37K prompts back to back (137,344 tokens, the 144K
window nearly full) read in 243.9 s against 262.5 s: the first of the three reads went in calls of 2,048, the other
two fell back to 512 (192 to 256 MiB left in the cache), and the run ended with 100 MiB of VRAM free, as without the
option. Otherwise an engine with the option ended its runs with 14 to 16 MiB less VRAM free (58 against 74 MiB after
2K); where that leaves less than half of what the engine keeps free, the next request takes it from the expert cache.

The answer after the prompt is written as fast as before: after 37K with no guesses (`--window 1`), 53.7 tok/s
without the option and 53.9 with it, 18.6 ms a token either way. With guesses a window took 39.7 -> 34.1 ms after 2K,
42.8 -> 41.6 after 37K and 43.0 -> 43.2 after 100K, and the tokens/s moved by up to 7% either way: the two builds
wrote different text (a wider batch splits the sums otherwise, as `--prompt-batch 2048` does), and some text is
guessed more easily.

The read in calls of b tokens is the one `--prompt-batch b` makes: with no expert cache in either run (another program
held VRAM while the engine loaded), `--elastic-batch 2048` and `--prompt-batch 2048` wrote the same 128 greedy tokens
on two 6K prompts, and so did 4096; `--prompt-batch 512` parted from them after 112 and after 11 tokens. With a
cache the two differ from the first token's probabilities on (the same 41 of 48 greedy tokens, where 512 and 2048
share 24): an expert in the cache and one that is copied for the read are multiplied in different launches, and the
answer runs with 601 experts in VRAM in one case and about 915 in the other. After a read every cached expert was
checked against the file, byte for byte. Without the option the build gives the same tokens and the same top-3
probabilities at every step as the build before it (two 6K prompts, 64 steps).

What was measured and left out: the staging buffer (a layer's experts, about 0.5 GB here) also lies idle while an
answer is written, and could be lent to the expert cache then. With 64 MiB instead, the cache had 1,200 experts in
place of 962, and an answer after 2K with no guesses came at 60.7 tok/s against 58.7 (57.2 to 62.7 from run to run),
while every prompt would have to give the 0.5 GB back first. Not tested: other cards, other models than
Qwen3.6-35B-A3B, Linux; on AMD (HIP) and Intel (SYCL) it is not built.

## `--weight-cut` and `--defer-experts`: less time on the CPU's experts, at a cost to the answers

On a 6 GB card most of an answer's time goes to the experts outside VRAM, which the CPU computes. `--tail-skip N`
(setup gives Qwen3.6 6) leaves out an expert outside VRAM that every token of a window ranks at place N+1 or later.
Two more ways, both off unless asked for:

- **`--weight-cut P`**, in place of the tail skip: a token keeps its first two experts and those in VRAM, then those
  outside VRAM in the router's order until what it keeps holds P of its experts' weight, and does without the rest;
  its weights are scaled up to their sum again.
- **`--defer-experts K`** (Expert Deferral, from KTransformers), with `--graph-experts`: a layer's experts outside
  VRAM that every token of a window ranks at place K or later are computed while the card runs the end of the layer
  and the next one's start, and their sum is added at the end of the next layer. None of the last layer's, and
  prompt batches are not touched. It goes with the tail skip: `--tail-skip 6 --defer-experts 5` leaves out places 7
  and 8 and defers 5 and 6.

What they cost was measured with `test_scripts/quality.py`: every setting writes the same greedy reference answers
(five agent-style prompts of 1.3k to 3.7k tokens, 300 tokens each, the reference with no lossy knob), and the KL of
its 20 most likely tokens against the reference's is taken at every step. Two runs of the reference's own flags
differ by a KL of 0.0013 (which experts the cache holds moves with timing). RTX 3060 Laptop 6 GB, this PC's Qwen3.6
config's flags; speed on the same text, against the reference's flags run before and after:

| Setting | Top-1 changed | Mean KL | Speed |
| --- | ---: | ---: | ---: |
| `--tail-skip 6` (setup's) | 1.6% | 0.0060 | x1.15 |
| `--tail-skip 5` | 2.6% | 0.0102 | x1.20 |
| `--weight-cut 0.9` | 2.3% | 0.0061 | x1.09 |
| `--defer-experts 7` | 1.9% | 0.0033 | x1.11 |
| `--tail-skip 6 --defer-experts 5` | 2.4% | 0.0095 | x1.24 |

The cut by the router's weight is slower than the tail skip at the same KL. Deferral gives more speed than the tail
skip at the same KL (`--defer-experts 7` x1.11 against `--tail-skip 7` x1.05, both about 0.0034) as long as it
defers little: what the CPU computes late must be done before the next layer routes. The changes of the top token
under the tail skip and the deferrals are at near ties (the reference gave its token less than 60%), as between two
runs of one build. Every setting, the 2k and 37k speeds and the recommendation:
[docs/OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md), step 6. Without the options the build gives the same top-20
probabilities as the build before it, byte for byte. Not tested: other cards, other models than Qwen3.6-35B-A3B, a
task benchmark; on AMD (HIP) and Intel (SYCL) it is not built, and the deferral needs `--graph-experts` (CUDA).

## `--suffix-drafts`: guesses from every conversation the engine has seen

While an answer is written, strata-poly checks guesses a window at a time: the model's draft block guesses up to 3
tokens, and `--lookup` (on by default) guesses what followed an earlier repeat of the last tokens inside the
conversation it holds. A coding agent repeats itself across conversations too: a subagent quotes the code the main
agent just wrote, a second session gets the same task, tool calls look alike from one conversation to the next.

With **`--suffix-drafts MIB`** the guesses from repeats come from every prompt and every answer since the engine
started (SuffixDecoding, Oliaro et al., NeurIPS 2025), kept in an index of about MIB MiB of RAM that drops its oldest
text first, in place of `--lookup`. Where the last tokens were seen in several places, it guesses what most of them
went on with, and how sure it is (the share of the places that agree) goes into the choice between its guess and the
draft block's. A conversation's next turn adds only its new tokens. **`--suffix-corpus FILE`** adds files to guess
from (token ids, a file a line: `tools/suffix_corpus.py` makes one). Every guess is checked by the model, as before,
so the answers are the ones the engine writes without it. To use it, add `"--suffix-drafts", "256"` to the `args` of
the model's config: 160 MiB of RAM, and about 1 us of the engine's thread a guess.

| Conversation, replayed in one engine | Answers | Without | `--suffix-drafts 256` | Guessed tokens right a window |
| --- | ---: | ---: | ---: | ---: |
| an agent edits three files (read, edit, read again) | 15 | 89.0 tok/s | 90.2 tok/s | 2.84 -> 2.97 |
| a subagent reads them and quotes the new code | 2 | 86.0 tok/s | 87.0 tok/s | 2.36 -> 2.49 |
| a new conversation is given the first one's task again | 14 | 88.9 tok/s | **99.3 tok/s** | 2.79 -> 3.78 |
| a plain chat about bicycles (no tools, no files) | 4 | 74.0 tok/s | 74.9 tok/s | 1.00 -> 0.97 |

RTX 3060 Laptop 6 GB, Qwen3.6-35B-A3B UD-Q4_K_M, this PC's Qwen3.6 config (`--kv-q8 --max-context 147456
--prompt-batch 512 --tail-skip 6 --threads 8 --attn-split --graph-experts --elastic-prefill 1024
--conversation-cache-mib 12288 --conversation-cache-slots 8`), greedy, thinking off, `test_scripts/agent_bench.py`:
the four conversations one after the other in one engine, the same prompts for both settings, three runs each in
turns. Within one conversation the default lookup already sees everything the agent read and wrote, so the gain there
is inside the spread from run to run (87 to 92 tok/s); it comes where a conversation repeats another. The plain chat
is not slowed. With the option, the top-20 probabilities of `test_scripts/quality.py`'s answers moved as little as
between two runs without it, and without the option they are the build before's, byte for byte. Every number:
[docs/OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md), step 7. Not tested: thinking on, conversations past 15K
tokens, through the server, other cards and models; AMD (HIP) and Intel (SYCL) builds (the index is host code).

## `--cpu-multi`: the CPU's experts at the speed of the RAM

On a 6 GB card most of the experts an answer meets are computed on the CPU (89% after a 139K prompt here), and the
card waits for them about a third of the time. strata-poly took their dot products from ggml-cpu a row and a token at
a time: an expert two tokens of a window use had each row unpacked twice, and the down products went in short pieces
of many experts. While an answer was written they read 32 to 35 GB/s of the 42 to 43 that a plain read of this
laptop's RAM gives. With **`--cpu-multi`** the Q4_K, Q5_K and Q6_K experts go through the engine's own AVX2 rows: ggml's
arithmetic with the loop over the tokens inside, so a row is read and unpacked once for every token that uses it and
each token gets ggml's result to the bit, and the down products in wider tasks. The answers are the ones without it:
with the expert cache fixed, the top-20 probabilities are the build before's byte for byte. To use it, add
`"--cpu-multi"` to the `args` of the model's config. Other expert types and CPUs without AVX2 run as before.

| | Without | `--cpu-multi` |
| --- | ---: | ---: |
| The CPU's experts alone (`poly_cpu_experts_bench`), windows of 1 / 4 / 8 tokens | 38-40 / 34 / 30 GB/s | **41-42 / 42-43 / 40-41 GB/s** |
| Writing after a 2K prompt (three runs each) | 74.6 tok/s, 29.4 ms a window | **78.2 tok/s, 28.2 ms a window** |
| Writing after a 139K prompt (four runs each) | 52.1 tok/s, 44.3 ms a window | 50.7 tok/s, 43.9 ms a window |
| Writing the same answers (`test_scripts/quality.py`, 1.3K to 3.7K prompts, six runs each) | 63.3 tok/s | **66.2 tok/s** |
| `test_scripts/agent_bench.py`, 35 answers of an agent session and a chat (two runs each) | 84.2 tok/s | **87.3 tok/s** |

RTX 3060 Laptop 6 GB, Ryzen 7 5800H with DDR4-3200, Qwen3.6-35B-A3B UD-Q4_K_M, this PC's Qwen3.6 config (`--kv-q8
--max-context 147456 --prompt-batch 512 --tail-skip 6 --threads 8 --attn-split --graph-experts --elastic-prefill 1024
--suffix-drafts 256`), greedy, the two builds in turns. After 139K the CPU's experts read faster (35.5 to 41.8 GB/s
against 34.1 and 35.2) but the answers were not measurably faster: many windows there go through the layers in two
halves, whose calls share few experts, and the runs' texts differed. With the cache's swaps on, which experts the card
holds follows the timing of its uploads, so a faster CPU moves the answers as much as a second run does, except at one
step of one of the five prompts, where all three runs changed a token the reference held at 67%. Every number,
the bench and what was tried and left out (a guess of the next layer's experts read ahead, a repacked copy in 2 MB
pages): [docs/OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md), step 8. Not tested: other CPUs, through the server,
Linux and GCC, AMD (HIP) and Intel (SYCL) builds (host code).

## `--kv-q4`: the context cache in 4 bits, for a card too small for the context

For a card that cannot hold the context its user needs with `--kv-q8` (a 4 GB gaming card, or the 6 GB laptop past
144K), **`--kv-q4`** keeps the attention cache's values in Q4_0 (4-bit codes, an fp16 scale for each 32, as ggml's
`q4_0`) and its keys in Q8_0, and **`--kv-q4-keys q4`** the keys in Q4_0 too. Qwen3.5 / 3.6 only, as `--kv-q8`. It is
not for speed, and it changes answers more than `--kv-q8` does; setup gives it only where `--kv-q8` would not fit the
context asked for, and says what it costs ([above](#what-setup-turns-on-and-when)). By hand: replace `"--kv-q8"` in
the model's config (`strata-qwen36-<size>.json`) with `"--kv-q4"`, or with `"--kv-q4-keys", "q4"`, and raise
`--max-context`.

| Cache | A token of Qwen3.6 | Context on the 6 GB laptop (UD-Q4_K_M) | Top token changed against fp16 | Of them where fp16 was 60% or more sure | Mean KL |
| --- | ---: | --- | ---: | ---: | ---: |
| fp16 | 22 KB | 72K (141 experts left in VRAM) | (floor 0.6-0.7%) | 0 | (floor 0.0006-0.0007) |
| `--kv-q8` | 11.7 KB | 144K (35 left) | 1.2-2.3% | 1 in 4,500 steps | 0.0025-0.0038 |
| `--kv-q4` | 8.9 KB | 184K (105 left; about 200K by its bytes) | 1.9-3.3% | 8 in 4,500 | 0.0040-0.0079 |
| `--kv-q4-keys q4` | 6.2 KB | 256K, the model's whole context (140 left) | 2.7-4.1% | 10 in 4,500 | 0.0090-0.0127 |

The cost was measured with `test_scripts/quality.py`: every cache writes the fp16 cache's greedy answers and prints its
20 most likely tokens at each step; prompts of 1,976, 37,433 and 100,056 tokens (500 steps each, two runs) and the five
agent prompts of 1.3K to 3.7K (300 steps each, one run). It did not grow with the context. The confident changes were
tokens fp16 held at 60 to 68%, but they include an operator in a formula (`*` for `/`, which `--kv-q4-keys q4` was 81%
sure of) and a name in code (`out` for `in`). Writing after 37K: no change (61.1 to 66.4 tok/s with each); after 139K
46.6 and 46.5 tok/s with `--kv-q8`, 51.4 and 53.2 with `--kv-q4`, 55.1 and 56.8 with `--kv-q4-keys q4`, because the
smaller cache leaves more experts in VRAM (12%, 30% and 42% of those an answer met came from it): speed bought with the
answers, not a reason to use it. The 4 GB card, simulated with `test_scripts/vram_hold.py` (a second process holds 2 GB
of the 6 GB card, leaving the engine 3.0 GB): at 32K the model ran with `--kv-q4` (55.4 tok/s), `--kv-q4-keys q4` (53.5)
and `--kv-q8` (50.0) alike, with no expert in VRAM, and not with fp16; at 64K neither 4 nor 8 bits fitted without
Windows paging the card's memory (31.0 and 19.1 tok/s). So at 32K a 4 GB card does not need it; what it buys between 32K
and 64K is for a later measurement.

Measured on the laptop of the numbers above (RTX 3060 Laptop 6 GB, Ryzen 7 5800H, Qwen3.6-35B-A3B UD-Q4_K_M),
2026-10-20; every number, the kernel test (`poly_kv_q4_test`) and the 4 GB simulation:
[docs/OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md), step 9. Without the flags the engine is the build before's, byte
for byte. Not tested: a task benchmark (HumanEval, tool calls), thinking on, a real 4 GB card, Ornith 1.5; AMD (HIP)
and Intel (SYCL) builds.
