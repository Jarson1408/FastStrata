# Optimization log: strata-poly on this PC

Every change made to speed up `strata-poly` on this laptop, with what it gave, measured here. Newest step last.
Each step says what was measured, how, and against what. Numbers from a session that made a change are marked as
that session's; numbers measured in this log's own runs are not marked. The scripts it names (`test_scripts/`) are the
working repository's and are not part of the published source.

## The machine and the model

- AMD Ryzen 7 5800H (Zen 3, 8 cores / 16 threads, AVX2, 16 MB L3), 64 GB DDR4-3200 dual channel.
- NVIDIA RTX 3060 Laptop GPU, 6 GB GDDR6 (336 GB/s), PCIe 3.0 **x8** (`nvidia-smi`: width 8 of 16), Windows 11
  Pro (WDDM), CUDA 13.0.
- Qwen3.6-35B-A3B UD-Q4_K_M (22.7 GB), the daily config `strata-qwen36-ud-q4_k_m.json`: `--max-context 147456
  --kv-q8 --prompt-batch 512 --tail-skip 6` and the expert profile of the pack. 10,496 routed experts of about
  1.8 MiB; 962 of them fit in VRAM at a start.

## How it is measured

- `test_scripts/run_serve.py LABEL PROMPT MAX_NEW [flags]`: the engine on its own in `--serve` mode, one prompt,
  greedy, a fresh copy of the expert profile every run. `ENGINE=<path>` runs another build with the same flags.
- Prompts: `test_scripts/prompt-code-2k.ids` (1,976 tokens: a question about `cpu_experts.cpp`, 300 tokens written),
  `test_scripts/prompt-engine-37k.ids` (37,433 tokens: `engine.cpp` and `BACKENDS.md`, 250 written),
  `test_scripts/prompt-short-46.ids` (46 tokens, for profiles).
- Old and new builds run in turns (A, B, A, B), nothing else on the GPU.
- Two figures for decoding: tokens/s, and **ms a window**, the decode time divided by the windows of guessed tokens
  (generated - 1 - guesses right). Tokens/s moves with how many guesses come out right, so it differs from answer
  to answer. Ms a window is steadier, but it also grows when the windows are longer, so the two are read together.
- Answers are not the same from run to run even with one build: with `--tail-skip 6` which experts are skipped
  depends on what the cache holds, and the cache moves with the timing of its uploads.
- Profiles: Nsight Systems 2025.3.2 (`nsys profile -t cuda --cuda-graph-trace=node`).
- What a setting costs in answers: `test_scripts/quality.py`, every setting writing the same reference answers
  (step 6).

## 0. Where the time went before any change (2026-10-18)

Daily config with 8 CPU threads (step 1; the 16-thread runs are there), 1,976-token prompt, 300 tokens:

| What was changed for the run | tok/s | ms a window | Experts from VRAM |
| --- | ---: | ---: | ---: |
| nothing (8 threads, see step 1) | 63.7 to 68.8 | 30.7 to 35.2 | 57% |
| no guesses (`--no-draft`) | 55.9 | 18.4 | 61% |
| cache frozen as loaded (`--expert-swaps 0`) | 63.5 | 35.7 | 31% |
| every expert on the CPU (`--no-gpu-experts`) | 55.2 | 36.1 | 0% |
| no tail skip (`--tail-skip 0`) | 57.4 | 37.7 | 50% |

- The expert cache in VRAM is worth little here: 0% from VRAM wrote 55 tok/s, 57% wrote 68.
- Profile, 46-token prompt and 400 tokens, 8 threads: the card was busy 57% of the decode. The engine's own thread
  spent 4.26 s of the 6.28 s inside CUDA calls: 100,352 kernel launches at 13 us, 27,745 graph launches at 39 us,
  0.9 s waiting on events, 0.7 s on streams, 1,440 graphs recorded. 82k kernels went outside graphs, nearly all
  of them the experts held in VRAM.
- Profile after the 37k prompt: the attention's three kernels took 27% of the decode (value sum 440 us a call,
  scores 215, softmax 179). The value sum read the cache at about 46 GB/s of the card's 336.
- Reading the 37k prompt: 793 tok/s with `--prompt-batch 512` (962 experts in VRAM at the start), 935 with 1024
  (840) and 1,040 with 2048 (601). The number of CPU threads does not change it.
- The expert profile: the 841 most used experts serve 25.6% of the uses, 2,000 serve 45%.

## 1. `--threads 8` (2026-10-18, all three configs)

The engine's default is every logical CPU, 16 here.

| Prompt | 16 threads | 8 threads | Other counts |
| --- | --- | --- | --- |
| 1,976 tokens, tok/s | 46.7 (file cache cold), 49.8, 60.0, 61.4 | 63.7, 66.3, 67.8, 68.0, 68.2, 68.8 | 12: 67.5; 15: 66.0, 67.2; 6: 61.2 |
| 1,976 tokens, ms a window | 51.3, 44.5, 36.7, 33.8 | 30.7 to 35.2 | |
| 37,433 tokens, tok/s | 42.6, 41.7 | 46.3, 46.7 | |

- Only all 16 is slow; 8, 12 and 15 are alike. The engine's own thread, which hands the card its work, then has no
  CPU to itself.
- No change to the arithmetic.
- One round of runs during another session's compile was slower for every count (8 threads 59.1, 15 threads 35.6)
  and is not in the table.

## 2. Conversation cache: 8 slots and 12 GiB (2026-10-18, Qwen3.6)

Was 4 slots and 8 GiB. Not measured for speed. The reason: the log showed a conversation pushed out with 4 parked,
a 120k-token conversation parks in 1.2-1.7 GB, and reading 150k tokens again takes about 270 s.

## 3. `--attn-split` (2026-10-18, commits 4945ab29 and 3999332c)

An answer's attention as one pass over the caches on the tensor cores, blocks of 512 rows, then a join.

| Prompt | Old: tok/s, ms a window | `--attn-split`: tok/s, ms a window |
| --- | --- | --- |
| 37,433 tokens | 46.1, 48.5; 43.6, 42.1 | **60.2, 61.7; 34.8, 34.5** |
| 1,976 tokens | 69.6, 68.1; 28.5, 29.3 | 69.2, 71.5; 32.2, 30.7 (more guesses right, longer windows) |

- 8 threads in both builds. The session that wrote it measured, with 128-token answers: 2k 72.7 -> 75.4 tok/s, 37k
  55.2 -> 72.9, 100k 34.4 -> 60.4; one call of the attention at 37k 649 -> 167 us; its share of the decode at 37k
  24.5% -> 6.6%.
- Answers: the products take halves, so the most likely token can change where two tokens are nearly tied. The
  37k answer was the same for its first 62 tokens. Forced to one answer, the session found the top token changed at
  4 of 256 steps, each a near tie; the old kernels with windows of 1 against windows of 8 change it at 8 of 256.
  `poly_attend_split_test` passes.
- Through the server (`serve/server.py --config`, as the Gateway starts it), a short chat: 72.2 and 74.7 tok/s,
  where AGENTS.md had 40-55 for 2026-10-17.

## 4. `--graph-experts` (2026-10-18, commits d91a0cd2 and e36ccc21)

Each layer of an answer is one CUDA graph, the experts in VRAM included: their kernels read the list of experts from
VRAM, the card waits on a word in pinned memory for the host's list and for the CPU's experts, and the host spins on
a word the card writes instead of waiting on an event. The host routes as before, so the sums are the old ones bit
for bit.

| Prompt | Old (`--attn-split --threads 8`): tok/s; ms a window | `--graph-experts` added |
| --- | --- | --- |
| 1,976 tokens | 72.1, 70.0; 32.1, 31.2 | 74.9, 70.9; **28.1, 28.7** |
| 37,433 tokens | 58.3, 62.7; 35.9, 34.5 | 64.6, 64.5; **32.9, 32.7** |

- About 10% less time a window at 2k and 7% at 37k. The session that wrote it measured 34.3 -> 30.7 ms at 2k,
  41.4 -> 39.4 at 37k and 45.1 -> 44.8 at 100k.
- Its profile (400 tokens after a short prompt): kernel launches 85k -> 14.7k, the engine's thread inside CUDA
  calls 3.9 s -> 1.5 s, event waits 1.1 s -> 5 ms, the answer 5.8 s -> 5.1 s.
- **What it does not change:** the card's own work. The same kernels compute the same values, so the GPU is not
  less loaded; it waits less between kernels, and its utilization in `nvidia-smi` should go up rather than down
  (not measured). The CPU is not freed either: the engine's thread spins on the card's word where it used to wait
  in the driver.
- `poly_expert_list_test` passes. Through the server, a short chat: 69.0 and 79.5 tok/s.

## 5. `--elastic-prefill 1024` (2026-10-19, merge c43c5714 of commits c3619a55, 065abee8, 337610cb, cb1019b6)

A prompt read of 1,024 tokens or more goes in wider calls: the expert cache lends its least valued experts for the
read, the wider batch's buffers are mapped in VMM, and after the read the same experts are uploaded again. The
batch is no wider than the stretch between two checkpoints of the conversation cache, 2,048 with its default, so
the daily config reads in calls of 2,048 where it read in 512. Near the end of the context the VRAM free is too
little and it falls back to 512 by itself.

Old: `--threads 8 --attn-split --graph-experts` (step 4); new: the same with `--elastic-prefill 1024`.

| Prompt | Reading, old | Reading, new | Writing after it, ms a window: old -> new |
| --- | --- | --- | --- |
| 1,976 tokens | 5.4, 5.4 s (363 tok/s) | **3.6, 3.6 s (542 tok/s), x1.49** | 29.2, 29.2 -> 28.5, 28.3 |
| 37,433 tokens | 46.8, 47.1 s (800, 795 tok/s) | **36.0, 36.1 s (1,040, 1,038 tok/s), x1.30** | 34.9, 32.6 -> 33.2, 32.5 |
| 139,056 tokens (`test_scripts/make_prompt.py 139000`), one run each | 264.1 s (526 tok/s) | **245.7 s (566 tok/s), x1.07** | 42.2 -> 44.6 (48.8 -> 49.4 tok/s) |

- Writing is unchanged. At 139k the first 65,536 tokens went in calls of 2,048 (the cache lent 832 MiB, 529 -> 70
  experts, in 48 ms and had them back in 166 ms), the rest in calls of 512. 148 MiB of VRAM were free at the end
  in both builds, so the 144k context still fits.
- The session that wrote it measured 2k 5.49 -> 3.66 s, 37k 46.7 -> 36.0 s, 100k 165.7 -> 139.5 s. Reads in calls of
  4,096 need `--prompt-cache 0` (no checkpoints, so a conversation is read again from its start), which is not
  worth it for agents.
- Answers: a read in calls of b gives what `--prompt-batch b` gives; with an expert cache the sums differ as between
  batch sizes. `poly_attend_split_test` and `poly_expert_list_test` still pass after the merge.
- Through the server: a 10.5k-token prompt read in calls of 2,048, first token after 11.0 s, written at 55 tok/s; a
  short chat at 73 tok/s.
- Side effect fixed: the session's test runs had added 10.1M uses (1.2%) to the pack's expert profile; 13 of the
  962 experts the cache starts with had changed. The profile was put back to a copy from 2026-10-18 22:58, before
  the session; the file with the test runs' counts is `expert-profile.txt.bak-20261019-elastic-tests`. A check
  through the server with the daily config writes to that profile too (a few hundred tokens' worth here).

## 6. What the lossy knobs cost, and two new ones (2026-10-19, branch `poly/lossy-knobs`; measured by the session that wrote them)

Steps 1 to 5 leave every expert in. The daily config's `--tail-skip 6` leaves some out, and what that costs had not
been measured. This step builds a yardstick, measures the tail skip
with it, and adds two knobs to hold against it: a cut by the router's weight (`--weight-cut P`) and Expert Deferral
(`--defer-experts K`). No default and no config was changed.

**The new knobs** (commit e0051bf0; both opt-in, `Engine::moe()` and `Engine::turn_run()` in `src/poly/engine.cpp`):

- `--weight-cut P`, in place of the tail skip: a token keeps its first two experts and those in VRAM, then those
  outside VRAM in the router's order until what it keeps holds P of its eight experts' weight, and does without the
  rest; its weights are scaled to their sum again. An expert one token of a window keeps is computed for all that
  chose it (the CPU reads it once for all of them). On the `--graph-experts` path the cut runs before the card reads
  the cached experts' weights.
- `--defer-experts K` (Expert Deferral, KTransformers, SOSP 2025), with `--graph-experts`: a layer's experts outside
  VRAM that every token of the window ranks at place K or later are computed after the host has told the card that
  the layer's other CPU experts are done, while the card runs the end of the layer and the next layer up to its
  router, and the next layer adds their sum to the residual at its end. Not the last layer's. Prompt batches are not
  touched (they go through `moe_rows()`); the prompt's last token, which goes in alone, is. Without `--graph-experts`
  the engine says so and defers nothing. With `--tail-skip N` the skip comes first: `--tail-skip 6 --defer-experts 5`
  leaves out places 7 and 8 and defers places 5 and 6.

Without either flag the build gives the base build's (kv-q8 at 96ae9168) top-20 probabilities byte for byte
(`--tail-skip 0`, the cache pinned with `--expert-swaps 0 --vram-reserve-mib 1024`, the five prompts below, two runs
of each build), and with `--tail-skip 6 --window 1` those of this branch's first build, whose tail skip is the base
build's. In each check one run of the 20 differed from its own build's other run (in the first check a run of the
base build): the same happens without any change.

**The yardstick** (`test_scripts/quality.py`, commit 9a1421e7). Five agent-style requests that
`test_scripts/quality_prompts.py` makes from this repository's files, in Qwen3.6's chat template with thinking off:
a change to `cpu_experts.cpp` answered with code (3,749 tokens), an OpenCode-like agent turn answered with an `edit`
tool call in Qwen3.6's format (2,460), an explanation of `guesses.hpp` (1,985), a short article from
`HOW_IT_WORKS.md` (1,716) and a pytest file for `serve/winjob.py` (1,342). The reference writes a greedy answer of
300 tokens to each with the daily config's flags and no lossy knob (`--tail-skip 0`). A setting then writes the same
answers (`--follow`) and prints its 20 most likely tokens at every step (`--top-probs 20`): 1,500 steps a run.

- Top-1 changed: the steps at which the setting's most likely token is not the reference's; in brackets, how many of
  them the reference gave 60% or more. Every other change is at a near tie.
- KL: KL(reference || setting) over the reference's 20 tokens of a step, both renormalized over them; the mean over
  the steps and the 99th percentile.
- PPL: the perplexity of the reference answers under the setting (the reference itself: 1.1887).
- Same text: the tokens/s of writing those answers, the setting run between two runs of the reference's flags (r, s1,
  r, s2, ..., r). Every setting writes the same tokens and gets about as many guesses right, so this is the steadiest
  speed figure here: 25 runs of the reference's flags wrote 55.8 to 63.0 tok/s, 60.1 on average, 2.7% apart (one
  standard deviation). One run a setting, two for `--tail-skip 6` (with `--defer-experts 7`, which then has nothing
  left to defer: x1.14) and for `--tail-skip 6 --defer-experts 6` and `5` (x1.19 and x1.19, x1.24 and x1.24).
- The floor: the reference's flags again. Two runs of one build differ because what the expert cache holds, and so
  which experts the card computes and which the CPU, moves with the timing of its uploads.

Speed of the answers each setting writes itself: `test_scripts/speed_rounds.py`, `test_scripts/run_serve.py` with
the 1,976-token prompt (300 tokens written) and the 37,433-token one (250), each setting next to a run of the
reference, two rounds in alternating order. Every run: RTX 3060 Laptop 6 GB, Ryzen 7 5800H, Qwen3.6-35B-A3B UD-Q4_K_M,
`--kv-q8 --max-context 147456 --prompt-batch 512 --threads 8 --attn-split --graph-experts --elastic-prefill 1024`
and the setting's flags, a fresh copy of the expert profile each run.

| Setting | Top-1 changed | Mean KL | KL p99 | PPL | Same text | 2k: tok/s (ref) | 37k: tok/s (ref) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none, again (the floor, 25 runs) | 1.10% (0) | 0.0013 | 0.018 | 1.1903 | x1 | | |
| `--tail-skip 7` | 1.67% (0) | 0.0035 | 0.042 | 1.1887 | x1.05 | 66.8 (65.5) | 62.6 (54.7) |
| **`--tail-skip 6`** (today) | 1.60% (1) | 0.0060 | 0.067 | 1.1933 | **x1.15** | 75.0 (68.3) | 66.4 (52.7) |
| `--tail-skip 5` | 2.63% (2) | 0.0102 | 0.121 | 1.1991 | x1.20 | 79.9 (66.4) | 69.2 (54.5) |
| `--weight-cut 0.95` | 1.23% (0) | 0.0016 | 0.020 | 1.1898 | x1.02 | 67.2 (66.5) | 51.5 (55.4) |
| `--weight-cut 0.9` | 2.27% (1) | 0.0061 | 0.066 | 1.1958 | x1.09 | 72.0 (65.2) | 58.0 (52.5) |
| `--weight-cut 0.85` | 3.07% (4) | 0.0105 | 0.130 | 1.2051 | x1.14 | 72.0 (66.8) | 64.5 (55.6) |
| `--defer-experts 7` | 1.87% (0) | 0.0033 | 0.039 | 1.1924 | x1.11 | 70.6 (65.5) | 60.5 (56.2) |
| `--defer-experts 5` | 2.87% (0) | 0.0074 | 0.089 | 1.1970 | x1.07 | 73.7 (67.9) | 66.3 (55.1) |
| `--tail-skip 7 --defer-experts 6` (1 run) | 2.27% (0) | 0.0063 | 0.076 | 1.1945 | x1.15 | 73.6 (65.4) | 64.2 (56.4) |
| `--tail-skip 6 --defer-experts 6` | 2.20% (0) | 0.0079 | 0.083 | 1.1956 | x1.19 | 78.2 (64.7) | 66.2 (56.0) |
| `--tail-skip 7 --defer-experts 5` (1 run) | 2.40% (0) | 0.0083 | 0.105 | 1.1983 | x1.17 | 73.3 (65.8) | 66.4 (56.3) |
| **`--tail-skip 6 --defer-experts 5`** | 2.43% (0) | 0.0095 | 0.093 | 1.1987 | **x1.24** | **80.9 (66.2)** | **69.9 (54.8)** |

Two runs (3,000 steps) a setting unless marked. The same runs as time a window (ref next to it), which also shrinks
when fewer guesses come out right: at 2k 28.3 (33.4), 29.5 (33.5), 27.9 (32.8), 32.0 (33.3), 29.2 (33.0), 24.2
(32.5), 27.5 (32.8), 28.1 (33.8), 27.9 (33.8), 28.6 (33.1), 24.0 (33.5), 27.6 (34.1) ms in the table's order; at 37k
35.1 (36.3), 31.8 (39.1), 29.9 (37.0), 39.5 (37.4), 35.4 (38.5), 33.6 (37.2), 33.2 (37.2), 33.2 (36.8), 30.8 (37.1),
30.1 (36.8), 29.6 (37.0), 30.2 (36.7). The reference itself wrote 64.5 to 69.1 tok/s at 2k and 52.5 to 57.9 at 37k.

What the numbers say:

- **Every change of the top token in the floor, the tail skips and the deferrals is at a near tie.** Of 37,500 floor
  steps 1.1% change, all where the reference gave its token less than 60%. `--tail-skip 6` changes one token the
  reference held at 60% or more in 3,000 steps, `--tail-skip 5` two, `--weight-cut 0.85` four; the deferrals none.
- **Today's `--tail-skip 6` costs a mean KL of 0.0060**, 4.6 times the floor, for x1.15 on the same text. The other
  engine's tail skip measured 0.0058 on an RTX 3060 12 GB and did not move HumanEval (76 and 78 of 82 against 78;
  `docs/DETAILS.md`, route tail skip).
- **The cut by the router's weight is slower than the tail skip at the same cost:** x1.09 against x1.15 at a KL of
  0.006, x1.14 against x1.20 at 0.010, and it changes more tokens the reference was sure of. For the same KL it leaves
  out fewer experts (about 0.80 against 1.22 a token and layer at 0.006; the counts include the draft block's), so
  an expert it leaves out costs the answers more than one the tail skip leaves out. Without the renormalization (a
  test build, one run) `--weight-cut 0.9` had a KL of 0.0052 and x1.10, and changed two tokens the reference held at
  60% or more: no better.
- **Deferral is faster than leaving out at the same cost when it defers little:** `--defer-experts 7` (places 7 and
  8, 1.1 experts a token and layer) x1.11 at a KL of 0.0033, where `--tail-skip 7` gives x1.05 at 0.0035. Deferring
  places 5 to 8 (2.3 a token and layer) gives less (x1.07) at twice the KL: what the CPU computes late must be done
  before the host routes the next layer, and the card's part between the two (the end of a layer and the next one up
  to its router) is short; more deferred work than fits there only moves the wait.
- **On top of the tail skip it adds speed:** `--tail-skip 6 --defer-experts 5` x1.24 on the same text, the fastest
  of all there, and the most tokens/s after 2k and after 37k (`--tail-skip 5` close behind, 79.9 and 69.2), at a KL
  of 0.0095 with no changed token the reference held at 60% or more.

**Recommendation.** If the bar is the cost the other engine measured as not moving HumanEval (a KL of about 0.006),
today's `--tail-skip 6` is already the fastest setting that meets it (`--tail-skip 7 --defer-experts 6` is as fast
and costs the same). The most speed for answers whose top token changes only at near ties is **`--tail-skip 6
--defer-experts 5`**: on the same text 8% faster than today (x1.24 against x1.15), 80.9 against 75.0 tok/s after
2k and 69.9 against 66.4 after 37k, for a KL of 0.0095 against 0.0060, a top-1 change at 2.4% of the steps against
1.6%, all at near ties, and a perplexity of the reference answers of 1.1987 against 1.1933. It is less than
`--tail-skip 5` costs (0.0102, two confident changes) and faster. To try it, add `"--defer-experts", "5"` to the
args of `strata-qwen36-ud-q4_k_m.json` and run a build of this branch. `--weight-cut` is not worth using here.

Found and fixed on the way: the first build's deferral read the tail skip's experts in the wrong order when both
were on (KL 0.0227 for `--tail-skip 6 --defer-experts 5`); `turn_run()`'s tail skip now keeps the experts' numbers
and places in step with them (the same experts in the same order, so the same sums), and the combinations were
measured again, quality and speed. Not measured: a task benchmark (HumanEval, a tool-call success rate) on this
engine, sampled answers (temperature above 0), thinking on, contexts past 37k, answers longer than 300 tokens.
Not built: HIP and SYCL (no toolchain here); the deferral is CUDA only, as `--graph-experts` is.

**Checked again after the merge (2026-10-19, kv-q8 at 2942d948 built in `build-poly`), and the decision.**
`poly_attend_split_test` and `poly_expert_list_test` pass. The same yardstick, a reference written with this build
(`test_scripts/quality.py ref`), then one or two runs (1,500 steps each) of each setting:

| Setting | Top-1 changed | Of them where the reference had 60% or more | Mean KL | KL p99 | PPL | tok/s writing the same text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none, again (the floor) | 0.40% | 0 | 0.00087 | 0.015 | 1.1747 | 60.4 |
| `--tail-skip 6` (today), two runs | 1.53%, 1.60% | 0, 0 | 0.00616, 0.00615 | 0.078, 0.077 | 1.1794, 1.1795 | 69.3, 69.5 |
| `--tail-skip 6 --defer-experts 6` | 1.87% | 0 | 0.00776 | 0.096 | 1.1807 | 71.4 |
| `--tail-skip 6 --defer-experts 5`, two runs | 2.07%, 2.07% | **2, 1** | 0.00850, 0.00838 | 0.107, 0.117 | 1.1792, 1.1804 | 72.8, 73.0 |

- `--defer-experts 5` was 5% faster than today on the same text here (the session measured 8%), and in 3,000 steps
  it changed 3 tokens the reference held at 60% or more, where today's setting changed none (the session saw none
  for it either).
- **Not deployed.** In an answer of code a changed confident token is a wrong name or bracket, and 5% does not pay
  for it without a task benchmark that says it is harmless. Today's `--tail-skip 6` stays: as the session found, it
  is the fastest setting at the cost the other engine measured as not moving HumanEval. `--defer-experts 6` (+3%,
  no confident change in 1,500 steps, KL 0.0078) is not worth it either.
- `engine-poly\strata-poly-kvq8.exe` is still the build of step 5: the knobs are off unless named, and the session
  found this build's top-20 byte for byte the step-5 build's without them, so there is nothing to install for now.

## 7. Drafts from every conversation the engine has seen: `--suffix-drafts` (2026-10-19, branch `poly/suffix-drafts`; measured by the session that wrote it)

Steps 1 to 5 made a window of guessed tokens cheaper. This step is about the guesses. The engine has two sources: the
model's draft block (up to 3 tokens, each from the one before) and `--lookup 3` (on by default), which guesses up to 7
tokens from an earlier repeat of the last 3 or more tokens inside the conversation it holds, among the four latest
places a trigram was seen. An agent repeats itself across conversations too: a subagent quotes what the main agent
wrote, a second session gets the same task, tool calls look alike. `--suffix-drafts MIB` (SuffixDecoding, Oliaro et
al., NeurIPS 2025, arXiv 2411.04975) guesses from every prompt and answer since the engine started instead. Opt-in;
no default and no config was changed.

**What it does** (commit on `poly/suffix-drafts`: `src/poly/suffix_index.hpp`, `src/poly/main.cpp`, an overload of
`DraftPolicy::choose()` in `src/spec/draft_policy.cpp`):

- The text is a ring of tokens, sequences one after another with a mark between them; when it is full the oldest
  text goes. For K = 24, 8 and 3, every place is chained to the place before it that ended with the same K tokens
  (hash chains, as an LZ77 matcher keeps them). A draft starts at the longest K whose last K tokens were seen before;
  up to 64 places of its chain, the newest first, are compared backwards token by token, up to 64 tokens, and those
  that match furthest are kept: the occurrences of the deepest suffix-tree node the sequence ends in.
- What followed them is voted on a token at a time: the token most of them have next is the draft's, its share of
  them is the chance it is right, and the places with another token drop out. The draft ends where the product of
  the shares falls under 0.3, where the places run out, or after as many tokens as the match is long (the paper's
  limit with its factor 1). The engine's window is a single chain, so the best branch is taken, not a tree.
- A prompt's tokens are indexed only from where it leaves the longest of the last 16 sequences seen that starts like
  it, after a lead of 64 tokens a match may run back into: a conversation's next turn adds its new tokens only. The
  answer's tokens go in as they are written. `--suffix-corpus FILE` adds files (token ids, a sequence a line;
  `tools/suffix_corpus.py` makes one from text files).
- The draft takes the place of the lookup's in the round as it was: the window is 8 tokens for a match of 8 or more,
  else 4, and the policy picks the lookup's window or the draft block's by the expected tokens a millisecond with
  each window size's measured time. The new overload of `choose()` takes the draft's shares as well: the rate the
  policy has seen for the match length is multiplied by them, so a draft whose places disagree is cut short or left
  for the block's window. The old `choose()` is the new one with no shares, the same arithmetic.
- Memory: 40 bytes a token of the ring, so `--suffix-drafts 256` makes one of 4,194,304 tokens in 160 MiB of RAM,
  plus the 16 sequences remembered (4 bytes a token, under 1 MiB in the benchmark). Time on the engine's thread:
  1.1, 1.0 and 1.3 us a lookup on average in the three runs below, 3.6 us the most for one answer's lookups.
  `poly_suffix_index_test` (26 checks: the vote, the deepest match over ten more frequent shorter ones, no draft
  across a sequence's end, a prompt's earlier turns not indexed again, a full ring forgetting its oldest text, a
  30-token match found behind 10,000 newer places of its trigram) times 4.0M tokens of code-like text: 0.17 us to
  index a token, 0.71 us for a draft and a token.

**The benchmark** (`test_scripts/agent_bench.py`). Four conversations in Qwen3.6's chat template with thinking off,
one after the other in one engine, as a server sends them:

- *edit*: an OpenCode-like agent (the tools of `test_scripts/quality_prompts.py`) asked to move the code that builds
  the engine's environment into a function `cuda_env()` in `test_scripts/run_serve.py` and `speed_rounds.py`, a file
  at a time: read, edit, read again; then a two-line summary; then the same in `quality.py`. 15 answers, the prompt
  growing from 823 to 14,268 tokens. The agent's first edit of `run_serve.py` failed (two matches) and it read the
  file again and retried, as agents do.
- *review*: another conversation, a subagent asked to read the three files and quote each `cuda_env()`. 2 answers.
- *redo*: the edit conversation's task again in a new conversation, on the files as they were (a retry after a reset,
  or a second session with the same task). 14 answers; the model took another path from its 8th answer on.
- *chat*: four questions about bicycles, no system prompt, no tools, no files: the control. 4 answers.

The files are the repository's at d59f1ae0. The tools are simulated (read gives the file's numbered lines, edit
replaces `oldString` when it is in the file once, grep searches the commit, bash answers "(no output)"). `make` let
the engine write every answer greedily with the daily flags and ran the tool calls it made; the answers were parsed
into text and tool calls by `serve/frontend.py` and the messages rendered with `frontend.ChatTemplate`, as the server
does; it keeps them in `test_scripts/agent-session.json`. `run` replays the 35 assistant turns' prompts in order with
GEN lines (`turn_at` as `serve/server.py` sends it), so every setting reads the same prompts and writes its own
answers. Old: the daily flags (`--max-context 147456 --kv-q8 --prompt-batch 512 --tail-skip 6 --attn-split --threads 8
--graph-experts --elastic-prefill 1024 --conversation-cache-mib 12288 --conversation-cache-slots 8`); new: the same
and `--suffix-drafts 256`; one build (this branch's; without the flag it is the old build, see below), three runs
each in turns (old, new; new, old; old, new), a fresh engine and a fresh copy of the expert profile each run.

| Conversation | Tokens written a run, old / new | Old: tok/s (each run) | New: tok/s (each run) | ms a window, old -> new | Guesses right a window, old -> new | Suffix drafts' windows, right / guessed (a run) |
| --- | --- | --- | --- | --- | --- | --- |
| edit | 2,195 / 2,119 | 89.0 (87.2, 88.1, 91.8) | 90.2 (89.8, 91.2, 89.6) | 43.1 -> 43.9 | 2.84 -> 2.97 | 202, 741 / 898 |
| review | 637 / 648 | 86.0 (86.2, 85.1, 86.6) | 87.0 (86.3, 90.1, 84.7) | 39.0 -> 40.0 | 2.36 -> 2.49 | 47, 178 / 209 |
| redo | 1,934 / 1,688 | 88.9 (88.9, 88.5, 89.4) | **99.3 (98.1, 102.2, 97.8)** | 42.6 -> 48.0 | 2.79 -> 3.78 | 286, 1,136 / 1,202 |
| chat | 988 / 1,003 | 74.0 (74.4, 72.6, 74.9) | 74.9 (74.2, 74.5, 76.0) | 27.1 -> 26.3 | 1.00 -> 0.97 | 5, 1 / 15 |
| all 35 answers | 5,753 / 5,458 | 85.6 (85.1, 84.7, 87.1) | **89.0 (87.9, 90.2, 88.7)** | 38.0 -> 38.4 | 2.26 -> 2.42 | 541, 2,056 / 2,323 |

- The runs write different text where a near tie falls the other way (with `--tail-skip 6` which experts are left
  out moves with the cache's timing): of 35 answers, 18.7 (old) and 20.3 (new) a run were the recorded ones, and the
  new runs wrote fewer tokens in *redo*. On the 13 answers that all six runs wrote as recorded (1,293 tokens):
  92.7 tok/s (91.5, 91.6, 94.9) -> 96.2 (95.7, 97.9, 95.0); of them *redo*'s 5 (275 tokens) 91.6 -> 106.1 and
  *edit*'s 7 (903 tokens) 93.4 -> 94.1.
- An earlier recording of the same session without *redo* (20 answers, three runs each in turns): 83.3 tok/s (83.7,
  82.5, 83.8) -> 83.8 (83.2, 84.0, 84.2); *edit* 88.6 -> 89.2, *review* 85.0 -> 88.0, *chat* 73.6 -> 73.2.
- Sampled answers (`--sample`: temperature 0.7, top-p 0.8, top-k 20, a seed a turn, the same in every run), two runs
  each in turns: all 79.4 tok/s (78.4, 80.4) -> 81.3 (83.2, 79.3); *edit* 82.9 -> 81.9, *review* 77.2 -> 79.8, *redo*
  83.0 -> 89.1 (93.2, 85.0), *chat* 67.2 -> 68.5: the same picture as greedy, the gain smaller and less steady.
- Where the guesses come from, without the model: `src/poly/suffix_replay.cpp` gives each source the prompts of a
  run and the answers it wrote (`agent_bench.py export`) and counts the answer tokens a draft of the window's length
  had right. The first old run's answers: *edit* 68% of the tokens with the lookup and 69% with the index, *review*
  47% and 59%, *redo* 67% and 84%, *chat* 4% and 4%.

**Lossless in practice.** `test_scripts/quality.py` with this build: a reference (no flag, `--tail-skip 0`), then
each setting following its answers, 1,500 steps a run:

| Setting | Top-1 changed | Of them where the reference had 60% or more | Mean KL | KL p99 | PPL | tok/s writing the same text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none, again (the floor), two runs | 1.07%, 0.80% | 0, 0 | 0.00134, 0.00109 | 0.023, 0.014 | 1.1906, 1.1911 | 57.8, 59.2 |
| `--suffix-drafts 256`, two runs | 0.60%, 0.53% | 0, 0 | 0.00129, 0.00126 | 0.019, 0.019 | 1.1898, 1.1905 | 58.9, 58.9 |

The flag moves the probabilities no more than a second run without it does. On these single prompts there is no
earlier conversation to guess from, and the speed is the floor's. **The default path:** without the flag this build gives the top-20 probabilities of the base build (kv-q8 at
d59f1ae0, built in the same folder) byte for byte, at all 1,500 steps of two runs of each build (`--tail-skip 0
--expert-swaps 0 --vram-reserve-mib 1024`, as in step 6).

What the numbers say:

- **Within one conversation the index adds little:** the lookup already sees everything the agent read and wrote in
  it, the file it copies from and its own earlier calls. *edit* gains 1%, inside the spread from run to run.
- **Across conversations it pays:** the conversation given the same task again was written 12% faster (16% on the
  same text), the subagent's review 1 to 3%. *redo*'s windows are fuller (3.78 guesses right a window against 2.79)
  and take longer (48.0 against 42.6 ms).
- **Why not more: a window's tokens cost about as much as they give here.** In one run of a build that printed the
  policy's every round (lines not committed), the rounds whose guesses were all right took, by the window's tokens:
  2 tokens 27.7 ms (72 tok/s), 3 34.5 ms (87), 4 44.5 ms (90), 5 48.4 ms (103), 8 about 80 ms (100, three rounds);
  the policy's own estimates were about 46 ms for 4 and 85 ms for 8. Each token of a window brings its own experts,
  most of them on the CPU (step 0). So drafts that are always right give at most about 100 tok/s on this PC, against
  90 for the draft block's windows of 4 when its 3 guesses are right, which on *redo*'s text they mostly were: of the
  74 rounds of *redo* in that run where the policy took the block's window over the index's draft, 69 had all their
  guesses right. *redo* came at 99.
- **No loss where there is nothing to repeat:** the chat came at 74.9 against 74.0 tok/s; the index's drafts went into
  5 windows a run, 1 of their 15 guesses right. A lookup costs about 1 us against a window's 26 to 48 ms.

**Recommendation.** It is lossless in practice and costs 160 MiB of RAM, so it can stay on: it gains where an agent
repeats an earlier conversation (a retry, a second session on the same task, subagents) and loses nothing measurable
elsewhere. To use it, add `"--suffix-drafts", "256"` to the args of `strata-qwen36-ud-q4_k_m.json` and run a build of
this branch. Not deployed by this session.

Not measured: thinking on, conversations longer than 15k tokens, a session long enough to fill the ring (the eviction
is only unit-tested), through `serve/server.py` and the Gateway (the engine on its own, as the server drives it),
`--suffix-corpus` for speed (a smoke test: three source files, 6,531 tokens, loaded and guessed from), windows longer
than 8 tokens, other models. Not built: HIP and SYCL (the index and the flag are host code in `main.cpp`).

**Checked again after the merge, and switched on (2026-10-19, kv-q8 fast-forwarded to 21e092e1, built in
`build-poly`).**

- Tests: `poly_suffix_index_test` (26 checks), `draft_policy_test` and `poly_guesses_test` (7 checks) pass, built in
  `build-poly-tests` with `STRATA_BUILD_TESTS` and `STRATA_BUILD_CONVERSATION_TESTS` on; `poly_attend_split_test`
  and `poly_expert_list_test` pass.
- The agent benchmark, one run of the installed step-5 engine between two of this build with `--suffix-drafts 256`
  (tok/s over the run's answers):

  | Conversation | Old | New, two runs |
  | --- | ---: | --- |
  | all 35 answers | 85.9 | 87.2, 90.6 |
  | edit | 88.7 | 86.3, 89.9 |
  | review | 84.8 | 84.8, 87.0 |
  | redo (the task again in a new conversation) | 90.1 | **97.9, 106.7** |
  | chat | 74.9 | 74.4, 73.4 |

- Lossless: `test_scripts/quality.py` with the flag, 1,500 steps: top-1 changed 0.13% (the floor 0.40%), none the
  reference held at 60% or more, mean KL 0.00090 (the floor 0.00087), PPL 1.1757 (1.1747).
- Installed over `engine-poly\strata-poly-kvq8.exe` (the step-5 build kept as `*.bak-20261019-suffixdrafts`, as is
  the config) and `--suffix-drafts 256` added to `strata-qwen36-ud-q4_k_m.json`.
- Through the server (`serve/server.py` with a copy of the config whose expert profile is a copy, so the pack's file
  was not written: its time stayed 08:45): the same request to rewrite `test_scripts/run_serve.py` with a new
  function, sent twice in two conversations, greedy, 600 tokens: 99.0 tok/s the first time, 107.3 the second; the
  index's drafts went into 62 and 69 windows with 429 of 431 and 452 of 469 guesses right.

## 8. The CPU's experts at the speed of the RAM: `--cpu-multi` (2026-10-19, branch `claude/cpu-experts-repack`; measured by the session that wrote it)

After step 7, 89% of the experts an answer meets after a 139k prompt are computed on the CPU, and the card spends 36%
of its time waiting for them (the profile under "Where it stands"). This step measures how fast the CPU reads those
experts against what the RAM gives, and makes the reading faster without changing a result. Opt-in; no default and
no config was changed.

**How it was measured.**

- `poly_cpu_experts_bench MODEL.gguf` (new, `src/poly/cpu_experts_bench.cpp`, no card): a plain read of memory by 8
  threads, each its own part, with 32-byte loads, of a 2 GiB buffer of the process's own and of the 18.7 GiB of
  routed experts in the mapped GGUF; then `cpu_experts()` over windows of 1 to 8 tokens, every MoE layer in turn as an
  answer goes, eight experts a token drawn by the pack's expert profile: the distinct experts' bytes over the time.
- In the engine, `STRATA_CPU_STATS=1` (new) prints at exit, for the calls of up to 16 tokens (an answer's): the
  calls, the distinct experts and their bytes (gate, up and down: 1.81 MiB an expert) and the time in `cpu_experts()`
  from the engine thread's call to its return, also by phase. An expert two tokens of a call use counts once.
- Everything else as in the steps before: `test_scripts/run_serve.py` with the daily flags `--threads 8 --attn-split
  --graph-experts --elastic-prefill 1024 --suffix-drafts 256`, a fresh copy of the expert profile each run, old and
  new in turns. Old: kv-q8 at db6035dc built in this branch's folder (the engine code of the installed step-7
  build).

**What the RAM gives, and what the experts got.**

| What was read, 8 threads | GB/s |
| --- | ---: |
| Plain read, own buffer, 4 KB pages | 42.2 (37.1 in a first run right after a build) |
| The same, prefetching 512 / 2,048 / 8,192 bytes ahead | 43.2 / 41.8 / 41.1 |
| Plain read of the routed experts in the mapped GGUF | 42.1 (first run 36.4); 42.9 prefetching 2 KB ahead |
| 16 threads: own buffer; the experts in the file | 40.6; 40.1 |
| 2 MB pages (after a restart, a slower day of the machine: 4 KB pages in the same runs 38.3-38.6) | 35.9, 37.0, 38.3; see the end of this step |
| `cpu_experts()` in the bench, windows of 1 / 2 / 4 / 8 tokens | 37.6-39.9 / 38.1-39.2 / 33.6-33.8 / 29.9-30.4 |
| In the engine, writing after the 1,976-token prompt (6 runs) | 32.3, 34.6, 34.8, 35.1, 35.2, 35.3 |
| In the engine, writing after the 139,056-token prompt (2 runs) | 34.1, 35.2 |

- The engine's figures are this build's without the flag: the old code with the counters (the base build has none).
- The ceiling here is 42 to 43 GB/s (DDR4-3200, dual channel: 51.2 in theory). While an answer was written the
  experts got 32 to 35 GB/s, 75 to 84% of it. After 139k they took 4.98 s of the 8.11 s of writing 400 tokens: gate
  and up 2.78 s (about 38 GB/s, at the ceiling already), the down products 2.09 s (31 GB/s), the activations' and
  hidden values' quantizers 0.11 s; 12,811 calls of 6.9 distinct experts on average, 1.15 uses each.
- In the bench the old path loses most where a window's tokens share experts (30 GB/s for 8 tokens): ggml's
  `vec_dot` is called a row and a token at a time, so an expert two tokens use has each of its rows unpacked twice;
  and a down task took 16 rows of every expert of the call, short pieces of many places.

**What `--cpu-multi` does** (commit 3ba85edf: `src/poly/cpu_kernels_avx2.cpp`, `multi_products()` in
`src/poly/cpu_experts.cpp`, `src/poly/main.cpp`):

- ggml-cpu's AVX2 dot products of Q4_K, Q5_K and Q6_K against Q8_K, with the loop over the tokens inside the loop
  over the blocks: an expert's row is read and its nibbles, high bits and 6-bit scales unpacked once for every token
  of the call that uses the expert (8 at a time), and each token then runs ggml's integer chain and float operations
  in ggml's order, so its result is ggml's to the bit. One call goes over all of a task's rows.
- The down products in tasks of 64 rows (was 16). The SwiGLU and each token's sum are the old path's expressions in
  its order. The CPU is checked for AVX2, FMA and F16C at the start; other types and older CPUs take the old path.
- No extra RAM: buffers of a few hundred KB a thread.

| `cpu_experts()` in the bench, three rounds in turns | 1 token | 2 tokens | 4 tokens | 8 tokens |
| --- | ---: | ---: | ---: | ---: |
| old | 37.6-39.9 GB/s | 38.1-39.2 | 33.6-33.8 | 29.9-30.4 |
| `--cpu-multi` | **41.4-42.4 GB/s** | **42.3-43.0** | **42.1-42.9** | **39.9-40.6** |

Down tasks of 16, 32, 48, 64, 96, 128 and 256 rows gave 41.9-42.2, 42.0-42.7, 42.6-43.0, 42.2-42.4, 41.9-42.9, about
40 and 38.4-38.7 GB/s at 1 to 4 tokens (fewer tasks than several a thread leave threads idle at the end); gate/up
tasks of 16 or 64 rows instead of 32 were no better; software prefetch 1 to 8 KB ahead of the rows changed nothing
(39.3 to 40.8 GB/s with none and with 1, 2, 4 and 8 KB, down tasks of 128 rows) and was left out.

**Answers: the same bits.**

- `poly_cpu_kernels_test` (new, no model): `kq::rows()` against ggml's `vec_dot` on random Q4_K, Q5_K and Q6_K rows for
  1 to 8 tokens (3,822 products), and `cpu_experts()` with and without the flag on random experts of the model's
  shapes, 1 to 13 tokens sharing experts, the SwiGLU's limit reached and not, and GELU (524,288 values): every value
  the same bits. `poly_cpu_experts_bench --check` on the model's own experts: 0 of 12,595,200 values differ.
- In the engine, `test_scripts/quality.py`, with the expert cache fixed (`--tail-skip 0 --expert-swaps 0
  --vram-reserve-mib 1024`, as in step 6): the old build, this build without the flag and this build with
  `--cpu-multi`, two runs each, give the same top-20 probabilities byte for byte at all 1,500 steps (one run of the
  old build differed from its own other run on one prompt). With swaps off at the default reserve (`--expert-swaps
  0`), without and with the flag: byte for byte the same too.
- With swaps on (the daily case) the expert cache's contents follow the timing of its uploads, and the card and the CPU
  compute an expert with different arithmetic (Q8_1 against Q8_K inputs), so runs differ (the floor). A faster CPU
  moves that timing:

| Setting, reference written without the flag | Top-1 changed | Of them where the reference had 60% or more | Mean KL | KL p99 | PPL | tok/s writing the same text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none, again (the floor), three runs | 0.80%, 0.93%, 1.00% | 0, 0, 0 | 0.00130, 0.00114, 0.00123 | 0.018, 0.015, 0.021 | 1.1973, 1.1971, 1.1977 | 62.7, 62.8, 63.8 |
| `--threads 6`, two runs; `--threads 12` (other timing, the same bits) | 0.67%, 1.00%; 0.80% | 0, 0; 0 | 0.00107, 0.00109; 0.00105 | 0.019, 0.015; 0.014 | 1.1970, 1.1975; 1.1965 | 60.8, 61.2; 63.9 |
| **`--cpu-multi`**, three runs | 0.87%, 1.27%, 1.00% | **2, 1, 2** | 0.00138, 0.00111, 0.00130 | 0.016, 0.014, 0.017 | 1.1955, 1.1969, 1.1982 | **66.2, 65.3, 65.6** |

| Setting, reference written with `--cpu-multi` | Top-1 changed | Of them where the reference had 60% or more | Mean KL | KL p99 | PPL | tok/s writing the same text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `--cpu-multi`, again (its floor), three runs | 0.13%, 0.40%, 0.53% | 0, 0, 0 | 0.00046, 0.00049, 0.00106 | 0.007, 0.008, 0.017 | 1.1863, 1.1857, 1.1862 | 66.9, 67.2, 66.2 |
| without the flag, three runs | 0.60%, 0.73%, 0.60% | 0, 0, 0 | 0.00111, 0.00112, 0.00119 | 0.018, 0.017, 0.016 | 1.1859, 1.1860, 1.1857 | 63.9, 62.5, 64.1 |

- All five changes the first table counts as confident are at the same two steps of one answer (the explanation of
  `guesses.hpp`, steps 202 and 203: the reference 67.6% and 66.3%, the runs 43.5 to 47.3% and 38.6 to 41.2%). In
  the second yardstick, written with the flag, step 202 is a near tie (its reference took the other token at about
  50%) and nothing changes a confident token either way. With the cache fixed or the swaps off the two paths give the
  same bits. So the change comes from which experts the card holds at that step, which the faster CPU moves the same
  way each run, not from the arithmetic; by the yardstick's letter it is not at the floor, and it is on one prompt.
- On the same text (the yardstick's forced answers, 1,342 to 3,749-token prompts, `--tail-skip 0`) `--cpu-multi`
  wrote 65.3 to 67.2 tok/s (66.2 on average, 6 runs) where the old path wrote 62.5 to 64.1 (63.3): x1.046.

**Speed.** Old (kv-q8 at db6035dc) and new (this branch) in turns, the daily flags, `STRATA_CPU_STATS=1` for the new
build:

| Prompt | Old: tok/s; ms a window | This build without the flag | `--cpu-multi` | CPU experts: GB/s, s a run (without -> with the flag) |
| --- | --- | --- | --- | --- |
| 1,976 tokens, 300 written, three rounds | 74.4, 73.0, 76.3 (74.6); 30.2, 28.2, 29.9 (29.4) | 74.4, 76.5, 76.0; 28.1, 29.8, 30.5 | **78.0, 78.9, 77.8 (78.2); 28.4, 28.1, 28.2 (28.2)** | 35.2, 35.3, 34.6 -> 36.8, 36.6, 36.9; 1.92 -> 1.71 s |
| 139,056 tokens, 400 written, four rounds | 52.2, 52.7, 51.4, 52.0 (52.1); 43.2, 42.8, 44.7, 46.5 (44.3) | 50.6; 45.3 (one run) | 54.3, 47.6, 49.7, 51.3 (50.7); 41.7, 42.8, 46.2, 44.9 (43.9) | 35.2 -> 41.8, 35.5, 36.9, 37.5; 4.72 -> 3.93, 4.76, 4.74, 4.51 s |

The agent benchmark (`test_scripts/agent_bench.py rounds 2`, step 7's four conversations, prompts up to 14,268
tokens), this build with `--suffix-drafts 256` without and with `--cpu-multi`, two runs each in turns:

| Conversation | tok/s without (each run) | with `--cpu-multi` | ms a window | Guesses right a window |
| --- | --- | --- | ---: | ---: |
| edit (15 answers) | 84.6 (83.1, 86.1) | 87.1 (87.2, 87.0) | 43.2 -> 46.3 | 2.65 -> 3.04 |
| review (2) | 81.8 (78.5, 85.2) | 83.9 (85.8, 82.1) | 39.7 -> 42.8 | 2.25 -> 2.59 |
| redo (14) | 97.0 (95.9, 98.0) | 103.1 (103.8, 102.4) | 61.8 -> 59.9 | 4.98 -> 5.17 |
| chat (4) | 68.6 (67.7, 69.5) | **71.9 (72.1, 71.7)** | **28.7 -> 26.8** | 0.97 -> 0.93 |
| all 35 answers | 84.2 (82.3, 86.2) | **87.3 (87.8, 86.8)** | 41.4 -> 41.5 | 2.49 -> 2.63 |

The chat is the cleanest of the four: its guesses are as often right either way, and its windows took 7% less time.
In the agent's conversations more guesses came out right with the flag (longer windows, so more time a window), which
is the answers' text moving, not the flag.

- **After 2k: x1.05** (78.2 against 74.6 tok/s, 4% less time a window), steady over three rounds; the CPU's experts
  read 5% faster (36.6 to 36.9 against 34.6 to 35.3 GB/s).
- **After 139k: no gain that shows above the spread** (50.7 against 52.1 tok/s, 43.9 against 44.3 ms a window; one
  run each way wrote other text with more or fewer guesses right). The CPU's experts read faster (35.5 to 41.8 GB/s;
  40.6 in a run of an earlier build of the branch; against 34.1 and 35.2) and took 22.3 to 27.3 ms a window against
  27.1 and 28.6, 45.7 to 53.8 us a distinct expert against 54.3 and 56.1. Many windows there go through the layers in
  two halves (the engine times both ways and takes the faster: 60 to 72 CPU calls a window for 41 layers, the fastest
  run having the fewest), and a call has 1.2 uses an expert on average: little to share, and the wider down tasks are
  what is left.

**Tried and not kept.**

- *The next layer's experts read into the caches while the card runs that layer's head* (the CPU is idle about 40% of
  an answer's time). A layer's experts are known only from its router, so they were guessed from the same layer's
  last turns (a weight of 0.7 a turn). From a log of the routing: with 8 guesses a layer the guess held 24% of the
  CPU's experts after 2k (5.85 a layer and window) and 27% after 139k (11.2), 38% of the guesses right; the last
  turn's set alone held 22% and 34%. The workers read the guesses in 64 KB pieces and stopped at the card's flag for
  the next router. At 37k, two rounds: ms a window 30.2, 30.4 without, 29.8, 30.3 with 6 guesses, 31.0, 30.8 with 10;
  the CPU's time an expert 50.9, 55.6 / 49.4, 52.8 / 50.6, 53.7 us: inside the spread. The bound, in the bench: with
  all of a call's experts in the caches beforehand a window of 1 token took 11.2 ms against 15.5 (3 of its 8 experts:
  12.7). A guess from the next layer's router on this layer's input might do better; not tried. Not committed (the
  patch is in the session's scratchpad).
- *A repacked copy of the experts for the CPU, rows interleaved for AVX2 as ik_llama.cpp's `-rtr`, in 2 MB pages:*
  not built into the engine. `--cpu-multi` already reads at the plain read's speed with 4 KB pages in the bench (42-43
  GB/s at 1 to 4 tokens), so a layout made for the arithmetic has nothing left to win; only larger pages could raise
  the ceiling, and they did not (below). A plain copy of all the experts (18.68 GiB, made in 1.2 s) in the bench's own 4 KB pages read as fast
  as the mapped file (`--copy normal`, two rounds in turns, windows of 1, 2 and 4 tokens: 35.1 to 38.6 against 34.8 to
  38.4 GB/s, a slower and noisier hour of the machine than the runs above; the same bits by `--check`). A copy would
  take another 18.7 GiB of the 59.9 GB Windows sees (44.3 GB free by Windows' count with no engine running; the
  engine's own RAM and up to 12 GiB of parked conversations come out of that, and the GGUF's 22.7 GB in the file
  cache would be pushed out), and 2 MB pages cannot be paged out.

**2 MB pages (2026-10-19, after a restart).** The user gave their account the "Lock pages in memory"
right, in effect after signing in again. `poly_cpu_experts_bench` asks for the privilege (AdjustTokenPrivileges) and
for 2 MB pages (VirtualAlloc with MEM_LARGE_PAGES), falls back to 4 KB pages when either fails and says which pages
it got and why (before the restart it said "in 4 KB pages, as the large ones were not had: the account does not hold
SeLockMemoryPrivilege"). After the restart, nothing else running, 47.6 GB free, the card idle; every large copy said
"in 2048 KB pages" (18.68 GiB, allocated and copied in 1.1 to 1.2 s):

| Run, 8 threads | 4 KB pages | 2 MB pages |
| --- | --- | --- |
| Plain read of a 2 GiB buffer, the two in the same run, three runs | 38.6, 38.5, 38.3 GB/s | 35.9, 37.0, 38.3 GB/s |
| `--multi --copy normal` / `--copy large`, three rounds in turns, windows of 1 / 2 / 4 / 8 tokens | 31.9-38.4 / 37.3-38.2 / 33.5-38.4 / 36.7-37.6 GB/s | 30.1-38.9 / 31.1-38.8 / 33.7-38.7 / 31.5-38.0 GB/s |
| The best of the three rounds | 38.4 / 38.2 / 38.4 / 37.6 | 38.9 / 38.8 / 38.7 / 38.0 |

- `--copy large --check`: 0 of 12,595,200 values differ from the plain path's.
- **2 MB pages do not read measurably faster here.** The best rounds are 0.8 to 1.6% apart, and each kind had rounds
  that fell to 30-33 GB/s (the hour after a restart, Windows busy beside it); the plain read was no faster either, so
  the 4 KB pages' address translation is not what holds these reads back. So no engine option for
  the CPU's experts in 2 MB pages was built: it would lock 18.7 GiB that cannot be paged out, for nothing measurable.
  The whole machine read 38-39 GB/s on this day, against 42-43 in the runs above.

**Not measured or not covered:** the server and the Gateway (the engine on its own, as before); thinking on;
`--cpu-share`'s CPU experts in a prompt batch (they go through the same code; these runs' prompts went through the
card); other CPUs (Intel, AVX-512, other cache sizes) and other models' K-quant experts; Linux and GCC (the files
take `-mavx2 -mfma -mf16c` there; not built here), HIP and SYCL (host code only, not built). The I-quant, Q4_0, Q8_0
and MXFP4 experts of other packs take the old path.

**Recommendation.** Its arithmetic is the old path's to the bit and it takes no RAM, so it can be switched on: 4 to
5% more after short and medium prompts (2k, the yardstick's same text, the agent benchmark); after a long one the
CPU's part is faster but the answer is not measurably so. What it moves in the answers is what a cache filled at
another pace moves, with the one place in the yardstick noted above. The remaining gap in the engine is a call's
fixed costs (four hand-offs to the workers a layer and window; after 139k a call took 344 to 378 us on average for
12 to 15 MB, which a plain read moves in 288 to 364 us) and the halves' calls with one use an expert. To use it, add
`"--cpu-multi"` to the args of `strata-qwen36-ud-q4_k_m.json` and run a build of this branch. Not deployed by this
session.

**Checked again after the merge, and switched on (2026-10-19, kv-q8 fast-forwarded to 7d9924fd, built in
`build-poly`).**

- Tests: `poly_cpu_kernels_test` ("ok"), `poly_workers_test` (4,000 checks), `poly_attend_split_test` and
  `poly_expert_list_test` pass.
- `test_scripts/run_serve.py` after the 1,976-token prompt, the installed step-7 engine and this build with
  `--cpu-multi` in turns, three rounds, the daily flags: 75.8, 74.1, 67.2 against 75.9, 75.5, 69.7 tok/s (72.4 against
  73.7 on average); ms a window 30.6, 30.6, 31.1 against 30.5, 29.1, 32.0.
- The agent benchmark, new, old, new: all 35 answers 92.0 and 87.5 against 88.8 tok/s; edit 92.8 and 91.9 against
  90.0, review 88.2 and 85.9 against 85.5, redo 112.9 and 94.1 against 100.1 (other text in each run), chat 70.7 and
  71.3 against 72.9.
- `test_scripts/quality.py` with the flag, 1,500 steps: top-1 changed 0.60% (the floor 0.40%), none the reference
  held at 60% or more, mean KL 0.00113 (0.00087), PPL 1.1756 (1.1747).
- **Here the gain is 1 to 2% and inside the spread** (the session measured 4 to 5%). Switched on all the same: the
  arithmetic is the old path's to the bit, it takes no RAM, and no measurement here put it below the old path on
  average. Installed over `engine-poly\strata-poly-kvq8.exe` (the step-7 build and config kept as
  `*.bak-20261019-cpumulti`) and `--cpu-multi` added to `strata-qwen36-ud-q4_k_m.json`.
- Through the server (`test_scripts/out/qwen-server-check.json`, a copy of the profile: the pack's file stayed at
  08:45): a short code answer 74.0 and 80.5 tok/s, a 600-token file rewrite 95.4. The first answer after a cold start
  of the PC came at 27.3 tok/s with its first token after 7.4 s: the model's file was not in the file cache yet, and
  the experts came from the SSD while the engine's mapper read them (the engine loaded in 9.9 s).
- 2 MB pages stay out (above): no faster here. The "Lock pages in memory" right stays granted to the account; it does
  nothing until a program asks for locked memory, and the engine asks for none.

## 9. The attention cache in 4 bits, for a card too small for the context: `--kv-q4` (2026-10-20, branch `poly/kv-q4`; measured by the session that wrote it)

Steps 1 to 8 made writing faster on this card. This step is for a card too small for the context its user needs (a
4 GB gaming card, or this laptop past 144K): an attention cache of fewer bytes a token, so that the context fits
where `--kv-q8` does not. It rounds the cache, so it changes answers; it is not offered for speed (the lossy knobs
for speed were turned down in step 6), and the question here is what it costs for the context it buys. Opt-in; no
default and no config was changed.

**What it does** (commit efc0ace5: `src/poly/kernels.cu`, `include/strata/poly/kernels.hpp`, `src/poly/qwen35.cpp/.hpp`,
`src/poly/engine.cpp/.hpp`, `src/poly/main.cpp`; Qwen3.5 / 3.6 only, as `--kv-q8`):

- `--kv-q4`: the values in Q4_0, as ggml's `quantize_row_q4_0` makes it (an fp16 scale a 32 values: the value farthest
  from zero over -8; a 4-bit code, min(15, trunc(x / scale + 8.5)), read back as (code - 8) * scale), and the keys in
  Q8_0 as `--kv-q8` has them. `--kv-q4-keys q4`: the keys in Q4_0 too. The two layouts the brief named: KIVI and
  KVQuant found the keys the more sensitive half.
- Bytes a token for Qwen3.6, whose 10 attention layers and draft block have 2 KV heads of 256 values each (11 x 2
  rows of 512 values):

  | Cache | Bytes a token | Of fp16's |
  | --- | ---: | ---: |
  | fp16 (no flag) | 22,528 | 100% |
  | `--kv-q8` | 11,968 | 53% |
  | `--kv-q4` (keys Q8_0, values Q4_0) | 9,152 | 41% |
  | `--kv-q4-keys q4` (keys and values Q4_0) | 6,336 | 28% |

  (The brief put both in Q4_0 at about 9.4 KB; it is 6.2 KB, 4.5 bits a value.)
- A row is its codes, eight to a 32-bit word in the order 0, 2, 4, 6, 1, 3, 5, 7, then its scales: `(word >> 4j) &
  0x000F000F` OR 0x6400 is the pair of values 2j and 2j + 1 as two halves of 1024 + code, so that `--attn-split`'s
  reader (`Q4Rows::halves16()`) turns 16 codes into the products' halves with a shift, a mask, a subtraction and a
  product each pair. A code goes into those products as (code - 8) * scale rounded to a half, as a Q8_0 code does with
  `--kv-q8`; the other kernels sum floats of the exact values.
- Every attention path the daily config uses reads it: an answer's attention (the old kernels and `--attn-split`), a
  prompt batch's (`attend_batch_q4()`: a block of rows to fp16 before cuBLAS's products, as `--kv-q8`'s),
  `--elastic-prefill`'s wider reads (the same batch path), the draft block (one more attention layer) and parked
  conversations (`save_card_rows()` / `load_card_rows()` go cache by cache, since the keys' and the values' rows now
  differ in size). The kernels take the keys' and the values' reader as two types; with one type they are the same
  code as before.
- `--tokens @FILE` and `--follow @FILE`: the ids from a file, for the yardstick's long prompts (below).

**Checks.**

- `poly_kv_q4_test` (new, needs a card; build target of the same name): `store_q4()` against ggml's Q4_0 on the host,
  code for code and scale for scale, on 4,800 blocks with awkward ones among them (zeros, a tie of + and - at the
  largest, one far value): 0 differ, and the values read back are 8.2% off their floats (RMS).
  Then `attend_q4()` (the old kernels and split) and `attend_batch_q4()` for both layouts, in Qwen3.6's shape (16
  heads of 256, 2 groups, 20,000 rows) and one of 128-value heads, against the same attention in doubles over the
  values the rows give back: the old kernels within 1.4e-6, split and the batch within 2.5e-3 of results up to 3.1, a
  token alone in split the same bits as among 2 or 7 others. `poly_attend_split_test` still passes.
- One call of the attention (`poly_kv_q4_test --time`), the position on the card as an answer's recorded layers run:

  | Position, tokens | `--kv-q8` split | `--kv-q4` split | `--kv-q4-keys q4` split |
  | --- | ---: | ---: | ---: |
  | 37,433, 1 | 167 us (244 GB/s) | 138 us (226 GB/s) | 113 us (192 GB/s) |
  | 37,433, 4 | 257 us | 216 us | 206 us |
  | 139,056, 1 | 550 us (275 GB/s) | 453 us (256 GB/s) | 356 us (225 GB/s) |
  | 139,056, 4 | 795 us | 763 us | 717 us |
  | 139,056, 8 | 1,538 us | 1,443 us | 1,442 us |

  One token reads the smaller cache faster (17 to 35% less time); with 4 and 8 tokens a window the kernel's products
  take more of the time than the cache, and the gain is 4 to 20%.
- **The default path** (no `--kv-q4`): this build against the installed step-8 engine (kv-q8 at 7d9924fd, the same
  engine source as this branch's base), `test_scripts/quality.py ref` with the expert cache pinned (`--tail-skip 0
  --expert-swaps 0 --vram-reserve-mib 1024`), with `--kv-q8` and with the fp16 cache, two runs of each build: the
  second runs of the two builds gave the same top-20 probabilities byte for byte at all 1,500 steps, with either cache.
  The first runs differed from them on some prompts, from step 7 to 30 on, and each build's first run differed from
  its own second run at the same steps: the runs' timing, not the change.
- Parked conversations (a script of the session's, the engine on its own): conversation A (the explanation prompt,
  1,985 tokens, 64 written), then B (1,342 tokens), which parks A, then A's prompt with 199 more tokens after it, which
  restores A's 1,984 rows and checkpoint from the conversation cache and reads the rest; against a second engine that
  kept A live and was given the same follow-up. With `--window 1`, the expert cache pinned and `--cpu-share 0` (with
  the CPU's share of a prompt batch set by timing, two runs of the live sequence wrote different follow-ups, 40 of 64
  tokens the same, and the installed engine's parked and live ones differed as well): the same 64 tokens with
  `--kv-q8`, `--kv-q4` and `--kv-q4-keys q4` (A parked in 87.4, 82.4 and 77.3 MB).

**The yardstick at long contexts.** `test_scripts/quality.py --long` (new): the reference is the fp16 cache, and a
setting follows the reference's greedy answers (`--follow`) and prints its top 20 at every step, as in step 6. Three
prompts, 500 steps each: `prompt-code-2k.ids` (1,976 tokens), `prompt-engine-37k.ids` (37,433) and `make_prompt.py
100000` (100,056). They are longer than a command line holds, so the engine reads them from files (`--tokens @FILE`,
`--follow @FILE`). Every setting runs with `--max-context 147456 --attn-split --threads 8 --graph-experts --tail-skip
0` and `--no-draft --prompt-batch 256 --elastic-prefill 0`: with the draft block's rows and batches of 512, the fp16
cache did not fit the card at 100K (no expert was left, and Windows put the card's memory in system RAM: the first
try was stopped after 13 minutes); without them it read 100K at 518 tok/s, with no expert left in VRAM. The draft
block only guesses, so the model's probabilities are the same sums without it, and the width of the prompt's calls,
which moves them, is 256 for every setting. While these runs went, the engine's GPU memory in system RAM (Windows'
GPU Process Memory counters) stayed at its pinned buffers' 1.3 to 1.8 GB. Two runs of each setting, the settings in
turns:

| Setting | Top-1 changed | Of them where the reference had 60% or more | Mean KL | KL p99 | PPL of the reference answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 again (the floor) | 0.60% | 0 | 0.00057 | 0.012 | 1.2507 (the reference: 1.2497) |
| `--kv-q8` | 2.13%, 2.33% | 0, 0 | 0.00363, 0.00380 | 0.041, 0.042 | 1.2538, 1.2544 |
| `--kv-q4` | 3.27%, 3.27% | **4, 3** | 0.00783, 0.00791 | 0.094, 0.096 | 1.2558, 1.2561 |
| `--kv-q4-keys q4` | 4.20%, 4.00% | **4, 3** | 0.01271, 0.01212 | 0.135, 0.116 | 1.2641, 1.2636 |

By the prompt's length (the two runs' mean; top-1 changed, of them 60% or more, mean KL):

| Setting | 1,976 tokens | 37,433 tokens | 100,056 tokens |
| --- | --- | --- | --- |
| fp16 again | 0.0%, 0, 0.00015 | 1.8%, 0, 0.00155 | 0.0%, 0, 0.00000 |
| `--kv-q8` | 2.6%, 0, 0.00379 | 2.2%, 0, 0.00359 | 1.9%, 0, 0.00377 |
| `--kv-q4` | 3.3%, 2, 0.00792 | 3.5%, 1.5, 0.00811 | 3.0%, 0, 0.00760 |
| `--kv-q4-keys q4` | 4.2%, 1, 0.01234 | 3.7%, 2, 0.01189 | 4.4%, 0.5, 0.01303 |

At 100K the floor is 0: no expert is left in VRAM there, so every expert runs on the CPU in every run, and two runs
give the same bits.

The same on the five agent prompts of step 6 (1,342 to 3,749 tokens, 300 steps each), with the daily flags of
`quality.py` (the draft block, batches of 512, `--elastic-prefill 1024`), a reference of their own with the fp16
cache, one run a setting:

| Setting | Top-1 changed | Of them where the reference had 60% or more | Mean KL | KL p99 | PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 again (the floor) | 0.67% | 0 | 0.00074 | 0.012 | 1.1662 (the reference: 1.1656) |
| `--kv-q8` | 1.20% | 1 | 0.00246 | 0.029 | 1.1691 |
| `--kv-q4` | 1.87% | 1 | 0.00399 | 0.046 | 1.1700 |
| `--kv-q4-keys q4` | 2.73% | 3 | 0.00899 | 0.106 | 1.1758 |

**The tokens the reference was sure of**, every one in the 4,500 steps of each setting (the reference's probability
-> the setting's top token and its probability):

- `--kv-q8`: once, in the explanation of `guesses.hpp`: `` ` >` `` 68% -> `` ` >=` `` 77%.
- `--kv-q4`: 8 times. Prose about the code: `.` 62% -> ` to` 56%; `` ` ` `` 62% -> ` a` 60% (both runs, the 2K
  prompt); ` pointer` 62% -> ` is` 50%; ` are` 61% -> ` will` 55% and 58% (37K); in the explanation of `guesses.hpp`
  `` ` /` `` 68% -> `` ` *` `` 49% (an operator in a formula).
- `--kv-q4-keys q4`: 10 times. `` `, `` 65% -> `` `) `` 53% (both runs); ` iter` 65% -> ` first` 48%; ` pointer` 62% ->
  ` is` 56% and 54%; ` Expert` 65% -> ` Adaptive` 48% (100K); `in` 63% -> `out` 50% (a name in code); in the
  explanation of `guesses.hpp` ` used` 60% -> ` asked` 44%, `` ` /` `` 68% -> `` ` *` `` **81%**, `` ` >` `` 68% ->
  `` ` >=` `` 56%.

What the numbers say:

- **`--kv-q8` is not free either**, which had not been measured: against the fp16 cache it moves the probabilities 3
  to 7 times as far as a second fp16 run (a mean KL of 0.0025 to 0.0038), about half as far as step 6's `--tail-skip 6`
  (0.0060), and it changed one confident token in 4,500 steps.
- **`--kv-q4` moves them 1.6 to 2.1 times as far as `--kv-q8`** (0.0040 to 0.0079) and changed 8 confident tokens in
  4,500 steps, 0.18%; **`--kv-q4-keys q4` 3.3 to 3.7 times** (0.0090 to 0.0127), and changed 10. Every confident change
  of the long prompts was at 61 to 65% in the reference; one of the agent prompts' was a token the 4-bit keys were 81%
  sure of instead, `*` for `/` in a formula.
- **The cost does not grow with the context**: the same at 2K, 37K and 100K for each format. A longer context has
  more rows to average over, and the rounding's errors do not add up across them here.
- **Here the keys are about as sensitive as the values**: values 8 -> 4 bits (`--kv-q8` -> `--kv-q4`) add 0.0042 of KL
  for 2,816 bytes a token, keys 8 -> 4 bits (`--kv-q4` -> `--kv-q4-keys q4`) 0.0045 for the same bytes; per byte saved
  the two layouts cost the same. So both are kept: `--kv-q4` (keys in Q8_0) costs a half to two thirds of what
  `--kv-q4-keys q4` costs and fits 1.31 times `--kv-q8`'s context, `--kv-q4-keys q4` 1.89 times. Qwen3.6 norms each key
  head (`attn_k_norm`) before the rotary embedding, which may be why its keys are not the outliers KIVI and KVQuant
  found elsewhere (not measured).

**Speed** (`test_scripts/run_serve.py`, the daily flags `--threads 8 --attn-split --graph-experts --elastic-prefill
1024 --suffix-drafts 256 --cpu-multi`, `--max-context 147456 --prompt-batch 512 --tail-skip 6`, the three caches in
turns, two rounds in reversed order):

| Prompt | `--kv-q8`: tok/s; ms a window | `--kv-q4` | `--kv-q4-keys q4` |
| --- | --- | --- | --- |
| 37,433 tokens, 250 written | 61.4, 64.8; 35.9, 33.1 | 66.4, 61.1; 32.9, 33.1 | 58.8, 61.9; 33.4, 34.7 |
| 139,056 tokens, 400 written | 46.6, 46.5; 50.0, 47.1 | **51.4, 53.2; 41.7, 41.0** | **55.1, 56.8; 38.1, 38.4** |

Reading the 37K prompt took 36.6 to 37.3 s with each; the 139K one 250.9 and 247.7 s with `--kv-q8`, 235.3 and 234.6
with `--kv-q4`, 227.6 and 226.6 with `--kv-q4-keys q4`.

- **After 37K: no change** beyond the spread. An answer's windows had 2.0 to 2.3 tokens on average here, and with
  more than one token the attention's time is its products' more than the cache's bytes (the kernel table above); the
  expert cache gave 50 to 53% of the experts with each.
- **After 139K: x1.12 and x1.20** (tokens/s; ms a window 48.6 -> 41.4 and 38.3). Not the attention: the expert cache.
  With `--kv-q8` the cache of 139K tokens leaves the card 12% of the experts an answer meets (12.3% and 11.7%), with
  `--kv-q4` 30% (29.8, 30.8), with `--kv-q4-keys q4` 42% (42.5, 42.1): the smaller caches leave 0.4 and 0.8 GB more VRAM
  to the experts. The prompt is read 6 and 9% faster for the same reason.
- That is a speed bought with the answers, which this PC turned down in step 6; it is noted, not proposed.

**How long a context fits the 6 GB card** (`test_scripts/fit_run.py`, new: a prompt that fills the context, 256 tokens
written after it, then the experts the expert cache still held; the daily flags):

| Cache, context | Prompt | Read in | Writing after it | Experts in VRAM after it (at the start) |
| --- | ---: | ---: | ---: | ---: |
| fp16, 72K (`--max-context 73728`) | 73,184 tokens | 106.0 s | 47.8 tok/s, 46.0 ms a window | 141 (964) |
| `--kv-q8`, 144K (147,456: the daily config) | 146,912 | 269.5 s | 43.1 tok/s, 50.6 ms | 35 (962) |
| `--kv-q4`, 184K (188,416) | 187,872 | 393.8 s | 47.0 tok/s, 45.6 ms | 105 (977) |
| `--kv-q4-keys q4`, 256K (262,144: all the model was trained for) | 261,600 | 702.7 s | 43.9 tok/s, 45.0 ms | 140 (985) |

The GPU memory in system RAM stayed at the pinned buffers' 1.76 to 1.81 GB in all four: nothing was paged out. As the
bytes a token go, `--kv-q4` would keep `--kv-q8`'s 35 experts at about 200K, and `--kv-q4-keys q4` holds the model's
whole context with more to spare than `--kv-q8` has at 144K. The 184K and 256K prompts take 6.6 and 11.7 minutes to
read the first time (a conversation that goes on is read from its last checkpoint).

**The 4 GB card, simulated.** `test_scripts/vram_hold.py` (new) holds VRAM in a second process with `cudaMalloc`
through the CUDA runtime of the `.venv` (ctypes), in pieces of 64 MiB each written once and checked against the card's
free VRAM, so that what it holds is VRAM, not system RAM; it stops short where less is free than a piece and 128 MiB,
rather than let the driver put the rest in system RAM. It frees what it holds on Ctrl+C, Ctrl+Break and at the end of
`--seconds N`, and the driver frees a process's VRAM when it ends in any way. `--card MIB` holds what makes the card
that size.

With 2,048 MiB held, CUDA reported 3,082 MiB free of 6,143 to the engine (5,130 without: Windows' desktop and the
driver take about 1 GB of this laptop's card before any program does), about what a 4 GB card leaves free once the
desktop runs on it. Qwen3.6 UD-Q4_K_M, the daily flags, a prompt that fills the context and 256 tokens written:

| Context, cache | Loaded with | Prompt read | Writing | GPU memory in system RAM |
| --- | --- | ---: | ---: | ---: |
| 32K, `--kv-q4` | no expert in VRAM (the dense weights only) | 39.5 s | 55.4 tok/s, 39.3 ms a window | 1,442 MB (the pinned buffers) |
| 32K, `--kv-q4-keys q4` | the same | 39.2 s | 53.5 tok/s, 39.4 ms | 1,378 MB |
| 32K, `--kv-q8` | the same | 39.2 s | 50.0 tok/s, 39.9 ms | 1,442 MB |
| 32K, fp16 | the same | | stopped: "graph experts: the card gave up waiting for the host" | |
| 64K, `--kv-q4` | the same | 91.5 s | 31.0 tok/s, 58.0 ms | 1,884 MB |
| 64K, `--kv-q8` | the same | 93.1 s | 19.1 tok/s, 99.9 ms | 2,058 MB |

- **It works as a 4 GB card**: the engine sized itself to what was free, read every prompt in calls of 512 (the wider
  calls of `--elastic-prefill` found no VRAM to borrow), and left 0 MiB free at the end of each run.
- **At 32K the 8-bit cache fits as well** (one run each, 10% apart in writing speed, where two runs of one cache were 5
  to 8% apart at 37K on the whole card). So a 4 GB card does not need `--kv-q4` at 32K; setup's estimate, which says
  such a card holds 8K with `--kv-q8` and nothing longer, is too low here and is to be fitted from the 4 GB
  measurements.
- **At 64K neither fits**: the GPU memory in system RAM grew by 0.4 GB (`--kv-q4`) and 0.6 GB (`--kv-q8`), Windows
  paging the card's memory, and writing slowed to 31.0 and 19.1 tok/s. The 4-bit cache was slowed less.
- Ctrl+Break freed the 2,048 MiB at once (5,130 MiB free again, exit code 0), and so did ending the holder's process.
  The holder took no system RAM beyond its Python process.
- Not run here: the UD-Q3_K_M preset (its file is in `D:\FastStrata-data\models\qwen36-UD-Q3_K_M\` for the session
  that measures the 4 GB card), contexts between 32K and 64K, a second run of each.

**Setup** (`poly_kv_plan()`, the place marked `FASTSTRATA KV-Q4`): the cache of the most bits that fits the context
asked for by setup's estimate: fp16, then `--kv-q8`, then `--kv-q8 --prompt-batch 512`, then `--kv-q4` (with batches
of 512), then `--kv-q4-keys q4`. Where it takes a 4-bit cache it says so, with what it costs from the tables above,
and asks (`--yes` takes it; no keeps `--kv-q8` and the warning that the context does not fit). The 8-bit cache's
question now says what it costs too. The estimate is the one `--kv-q8` was fitted with (the VRAM a preset takes beside
its experts, measured on an RTX 5090, and the rows past 32K at the cache's size), so it gives this laptop UD-Q4_K_M
`--kv-q4` past 146K up to 170K, and `--kv-q4-keys q4` up to 207K. It is conservative for the 4-bit caches: by the
fit runs above a token takes about its cache's bytes in VRAM on this card (the experts given up were within 6% of the
cache's bytes in all four), where the estimate counts 4 KB more a token, fitted so that `--kv-q8` stops at 144K; so
from 171K to about 195K it offers `--kv-q4-keys q4` where `--kv-q4` fitted (184K with 105 experts left). A 4 GB card
gets `--kv-q8 --prompt-batch 512` at 8K and nothing that fits by the estimate at 16K or more, where it offers the 4-bit
cache with the warning, while the simulation above ran 32K with `--kv-q8`. The estimate is to be fitted again from a
4 GB card's measurements, and with them the 4-bit caches' slope. New cases in
`tools/test_setup_speedups.py`: the laptop past 144K, a 4 GB card, an 8 GB card that never needs it, an AMD card that
does not get it, the bytes each cache saves; all of `tools/test_setup_*.py` pass.

**Recommendation.**

- **Which layout:** both are kept, and they cost the same per byte saved. **`--kv-q4` (keys in Q8_0)** is the one to
  take first: it costs 1.6 to 2.1 times what `--kv-q8` costs and fits 1.31 times its context. `--kv-q4-keys q4` costs
  3.3 to 3.7 times `--kv-q8` and fits 1.89 times its context; it is for where `--kv-q4` does not fit.
- **On this laptop (6 GB):** keep `--kv-q8` at 144K, as the daily config has it. `--kv-q4` is worth its cost only for a
  context of 145K to about 195K (184K measured, with 105 experts left in VRAM), `--kv-q4-keys q4` only past that (the
  model's whole 256K measured, 140 experts left). It also writes 12 to 20% faster after 139K, but that is speed bought
  with the answers.
- **On a 4 GB card:** at 32K the 8-bit cache fitted the simulated card as well, so the 4-bit cache is not needed
  there; whether it buys anything between 32K and 64K (where neither fitted) is for the 4 GB measurements. On cards of
  8 GB and more: not needed (setup's estimate fits `--kv-q8` to 256K on 8 GB).
- **Plainly, what it costs:** against the 16-bit cache, `--kv-q4` changed a token that cache was 60% or more sure of
  8 times in 4,500 steps of answers, about once in 560 tokens written; `--kv-q8` once in 4,500. Every such change on
  the long prompts was at 61 to 65%, but one of the agent prompts' was an operator in a formula (`*` for `/`), and with
  `--kv-q4-keys q4` a name in code (`out` for `in`). For someone who turns down settings that change confident
  tokens, it is a setting to take only when the context does not fit otherwise; setup gives it only then, and says so.
- Also new here: **`--kv-q8` itself is lossy** by the same yardstick, half of `--tail-skip 6`'s KL (0.0025 to 0.0038
  against 0.0060) and one confident change in 4,500 steps. The daily config needs it for 144K; at 72K or less the
  16-bit cache fits this card.

Not measured: a task benchmark (HumanEval, a tool call's success) with either layout; thinking on; sampled answers;
contexts past 100K in the yardstick (the fp16 reference does not fit the card there); Ornith 1.5; a real 4 GB card.
Not built: HIP and SYCL. The readers are in the file the HIP build compiles too (the split kernel is CUDA only), and
`store_q4()` takes the scale's value from a warp's ballot, whose width on AMD's 64-lane waves was not checked.

## 10. The four FastStrata presets, old against new, and a 4 GB card (2026-10-20, branch `claude/faststrata-numbers`; measured by the session that wrote it)

FastStrata keeps four presets of the strata-poly engine: Qwen3.6-35B-A3B UD-Q4_K_M and UD-Q3_K_M, and Ornith 1.5
APEX-I-QUALITY and APEX-I-COMPACT. This step measures each of them on the engine this log started from and on today's,
in turns, and the cost in answers of each smaller file against its larger sibling, and runs them on a simulated 4 GB
card. No engine, no config and no expert profile was changed.

**The two engines.**

- *Old*, "PolyStrata with the Q8 cache added": the user's build of 2026-10-17, from before step 1 (kept as
  `engine-poly\strata-poly-kvq8.exe.bak-20261018-attnsplit`, run from a copy named `.exe`), with its daily flags of
  then: `--max-context 147456 --kv-q8 --prompt-batch 512 --tail-skip 6 --conversation-cache-mib 8192` and the default
  threads (16, every logical CPU).
- *New*, FastStrata: `build-poly\strata-poly.exe`, built from kv-q8 at 150beb0b (steps 1 to 9), with today's daily
  flags: `--max-context 147456 --kv-q8 --prompt-batch 512 --tail-skip 6 --threads 8 --attn-split --graph-experts
  --elastic-prefill 1024 --suffix-drafts 256 --cpu-multi`.
- Ornith 1.5 runs without `--tail-skip` on both: its tail skip was never measured, and setup gives Ornith none.

**How it was measured.** `test_scripts/faststrata_runs.py` (new) runs everything and logs every run to
`test_scripts/out/faststrata/runs.jsonl`:

- `speed`: `test_scripts/run_serve.py` with the 1,976-token prompt (300 tokens written) and the 37,433-token one (250
  written), two rounds, the second in the reverse order of the first (2k old, 2k new, 37k old, 37k new, then back).
  `long`: the 139,056-token prompt of `test_scripts/make_prompt.py 139000`, 400 tokens written, one run each.
  `agent`: `test_scripts/agent_bench.py` (step 7's four conversations, 35 answers), old then new, and a second round
  new then old. The new engine's agent runs add the benchmark's conversation cache (`--conversation-cache-mib 12288
  --conversation-cache-slots 8`, as the user's config has), the old engine's its own `--conversation-cache-mib 8192`.
- `run_serve.py`, `fit_run.py` (`TAIL_SKIP`, 0 for no flag), `quality.py` and `agent_bench.py` (`MODEL`, `PROFILE`,
  `BASE_FLAGS` for the old build's own flags) were extended for this; every run copies the pack's expert profile
  (Qwen3.6's for both Qwen3.6 files, Ornith's for both Ornith files: the routing does not depend on the quantization).
- Before a preset's runs its file was read once into the file cache (5 to 19 s), so no run read its experts from the
  SSD. No other strata-poly ran during any run (checked before and after each one; the Gateway did not start one).
  The user's desktop ran beside: the Claude app, Task Manager and Teams took about 0.4 of a CPU core at rest, and
  44 to 49 GB of the 64.3 GB of RAM (59.9 GiB) were available with no engine running.
- **The prompts mean the same to Ornith:** its pack's `vocab.json`, `merges.txt` and `token_type.json` are Qwen3.6's
  byte for byte, with the same pre-tokenizer (`qwen35`); its chat template differs from the Qwen3.6 pack's in how it
  checks roles and writes tool arguments, but renders all 35 prompts of the agent benchmark to the same ids, and the
  2k and 37k prompts decode and encode back to the same ids with its tokenizer.

**The four files, and what FastStrata's options reach in them.**

| Preset | File | Routed experts: gate and up / down, layers | Other weights | Experts in VRAM at a start (6 GB) | `--cpu-multi` | `--graph-experts` |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Qwen3.6 UD-Q4_K_M | 22.7 GB | Q4_K / Q5_K 38, Q4_K / Q6_K 3 | Q8_0 | 962 (1.7 GiB) | 41 of 41 layers | yes |
| Qwen3.6 UD-Q3_K_M | 17.1 GB | IQ3_XXS / IQ4_XS 37, IQ3_XXS / Q6_K 2, IQ4_XS / Q6_K 1, Q3_K / Q4_K 1 | Q8_0 | 1,341 (1.7 GiB) | 0 of 41 | **no** |
| Ornith 1.5 APEX-I-QUALITY | 23.7 GB | Q6_K 10, Q5_K 10, IQ4_XS 20, Q8_0 1 | Q6_K, Q8_0 | 1,080 (1.9 GiB) | 20 of 41 | yes |
| Ornith 1.5 APEX-I-COMPACT | 17.4 GB | Q3_K 30, Q4_K 10, Q8_0 1 | Q4_K, Q6_K | 1,823 (2.4 GiB) | 10 of 41 | yes |

- `--cpu-multi` (step 8) has kernels for Q4_K, Q5_K and Q6_K; a layer whose experts are in another type takes the old
  path, the same bits either way. `--graph-experts` (step 4) needs every layer's experts in a type its list kernel
  reads (Q4_0, Q5_0, Q8_0, IQ4_NL, Q2_K to Q6_K, IQ4_XS). UD-Q3_K_M's IQ3_XXS is not one of them, so on that file the
  engine says `--graph-experts: this card or model does not take it; the layers run as without it`.
- So the Q3 preset gets steps 1, 3, 5 and 7 (8 threads, `--attn-split`, `--elastic-prefill`, `--suffix-drafts`), not
  steps 4 and 8. Both Ornith files have the draft block (`nextn_predict_layers` 1), as Qwen3.6's do.

**Writing and reading.** tok/s and ms a window as in the steps before (each run in brackets); reading is the prompt's
time as the engine reports it.

| Preset | Prompt | Old: tok/s; ms a window | New: tok/s; ms a window | Writing | Reading, old -> new |
| --- | --- | --- | --- | ---: | --- |
| Qwen3.6 UD-Q4_K_M | 1,976 tokens | 56.4 (52.5, 60.3); 38.0 | **71.8** (70.1, 73.4); 31.5 | x1.27 | 5.5 -> 3.7 s (x1.50) |
| | 37,433 tokens | 41.9 (42.1, 41.6); 47.8 | **66.9** (65.6, 68.2); 32.5 | x1.60 | 47.2 -> 36.4 s (x1.30) |
| | 139,056 tokens, one run | 24.7; 79.1 | **48.8**; 46.2 | x1.98 | 265.1 -> 248.2 s (x1.07) |
| Qwen3.6 UD-Q3_K_M | 1,976 tokens | 58.8 (57.8, 59.8); 35.4 | **66.2** (66.2, 66.3); 31.5 | x1.13 | 4.2 -> 2.8 s (x1.52) |
| | 37,433 tokens | 37.0 (31.8, 42.3); 54.1 | **58.6** (57.9, 59.4); 37.6 | x1.58 | 45.7 -> 34.8 s (x1.31) |
| | 139,056 tokens, one run | 26.4; 74.7 | **45.6**; 44.9 | x1.73 | 258.8 -> 241.2 s (x1.07) |
| Ornith 1.5 APEX-I-QUALITY | 1,976 tokens | 45.8 (44.1, 47.5); 31.0 | **51.0** (51.2, 50.9); 27.8 | x1.11 | 5.6 -> 4.1 s (x1.37) |
| | 37,433 tokens | 28.9 (31.1, 26.7); 50.5 | **39.6** (45.4, 33.8); 33.8 | x1.37 | 47.7 -> 36.7 s (x1.30) |
| | 139,056 tokens, one run | 21.5; 51.9 | **36.8**; 39.3 | x1.71 | 269.2 -> 236.9 s (x1.14) |
| Ornith 1.5 APEX-I-COMPACT | 1,976 tokens | 60.8 (58.4, 63.2); 29.0 | **63.1** (63.4, 62.9); 24.8 | x1.04 | 4.2 -> 2.8 s (x1.50) |
| | 37,433 tokens | 41.8 (42.6, 41.0); 28.4 | **56.2** (55.7, 56.7); 21.4 | x1.34 | 46.5 -> 35.2 s (x1.32) |
| | 139,056 tokens, one run | 24.6; 57.7 | **43.6**; 37.3 | x1.77 | 268.6 -> 224.6 s (x1.20) |

**The same text, the steadiest figure after short prompts.** Each engine with its daily flags wrote the reference
answers of the quality runs below (`faststrata_runs.py same`: `quality.py` with `--follow`; the five agent prompts of
1,342 to 3,749 tokens, 300 tokens each; the Qwen3.6 files follow UD-Q4_K_M's answers, the Ornith files
APEX-I-QUALITY's), two rounds in turns. Every run writes the same tokens and its guesses can come out right as
often, so the text no longer moves the speed:

| Preset | Old: tok/s, two runs | New: tok/s, two runs | New / old | Top-1 changed against the reference, old; new | Mean KL, old; new |
| --- | --- | --- | ---: | --- | --- |
| Qwen3.6 UD-Q4_K_M | 62.2, 62.3 | **72.1, 73.6** | **x1.17** | 1.73%, 2.13%; 1.40%, 1.47% | 0.0072, 0.0071; 0.0057, 0.0054 |
| Qwen3.6 UD-Q3_K_M | 67.4, 67.2 | **69.8, 68.9** | **x1.03** | 4.47%, 4.73%; 4.27%, 3.87% | 0.0234, 0.0236; 0.0238, 0.0236 |
| Ornith 1.5 APEX-I-QUALITY | 47.0, 45.7 | **52.2, 52.6** | **x1.13** | 2.80%, 3.33%; 1.44%, 1.82% | 0.0207, 0.0196; 0.0085, 0.0092 |
| Ornith 1.5 APEX-I-COMPACT | 57.8, 57.9 | **64.0, 63.2** | **x1.10** | 8.03%, 7.95%; 8.03%, 8.26% | 0.0866, 0.0884; 0.0895, 0.0904 |

- After short prompts FastStrata writes the same text x1.03 (Qwen3.6 UD-Q3_K_M) to x1.17 (UD-Q4_K_M) faster; the
  greedy 2k runs above, each engine writing its own text, gave x1.04 to x1.27.
- The new engine's answers are no further from the reference than the old one's: for Qwen3.6 UD-Q4_K_M a KL of 0.0055
  against 0.0071 (both with `--tail-skip 6`). The reference was written by the new engine, so the old one's figures
  also hold the two builds' other arithmetic (prompts read in calls of 512 against 2,048, the attention's products in
  halves with `--attn-split`); for APEX-I-QUALITY that is all of the difference (no tail skip in either), 0.020 against
  the new engine's 0.009, which is Ornith's floor (below).

What the numbers say:

- **The longer the prompt, the more FastStrata gives**, for every preset: writing x1.04 to x1.27 after 2k, x1.34 to
  x1.60 after 37k, x1.71 to x1.98 after 139k, where the old engine wrote 21.5 to 26.4 tok/s with each file. Reading
  is x1.30 to x1.32 faster at 37k for every preset (calls of 2,048 tokens, step 5) and x1.07 to x1.20 at 139k.
- **Qwen3.6 UD-Q4_K_M gains the most after short prompts** (x1.27 after 2k), being the one file that every step
  reaches. UD-Q3_K_M gains x1.13 there: no `--cpu-multi` and no `--graph-experts` on its IQ3_XXS experts.
- **On FastStrata the Q3 file is not faster than the Q4 one**, although 1,341 of its experts fit in VRAM against 962:
  66.2 against 71.8 tok/s after 2k, 58.6 against 66.9 after 37k, 45.6 against 48.8 after 139k. Its IQ3_XXS experts
  get neither step 4 nor step 8, and the CPU takes longer for each of them: on the yardstick's five prompts below, with
  7 to 9 points more of its experts served from VRAM (41 to 63% against 34 to 54%), its CPU experts took as long or
  longer a token (0.7 to 1.5 ms against 0.6 to 1.5, the engine's own count). It reads prompts faster (2.8 against 3.7 s
  at 2k, 34.8 against 36.4 at 37k). Q3 is for a PC with less RAM, not for speed.
- **Ornith's draft block is right less often**: 0.1 to 0.65 guesses right a window against 0.96 to 1.26 for Qwen3.6, and
  its tokens/s move a lot with the text (the old engine's two 2k runs of APEX-I-COMPACT got 69 and 166 guesses right in
  300 tokens). Its windows are mostly of one token, so its gain is the windows' time: 31.0 -> 27.8 ms (APEX-I-QUALITY)
  and 29.0 -> 24.8 ms (APEX-I-COMPACT) after 2k, 50.5 -> 33.8 and 28.4 -> 21.4 after 37k.
- One run at 139k each way: a second run of the same build moves by several percent (step 8 had 47.6 to 54.3 tok/s
  for one build there).

**The agent benchmark** (step 7's `test_scripts/agent_bench.py`: an OpenCode-like editing session of 15 answers with
prompts growing to 14,268 tokens, a review subagent of 2, the same task again in a new conversation, 14, and a chat of
4 as the control; one engine for all 35 answers, as a server sends them). Two rounds, old then new and new then old;
tok/s over a conversation's answers, the mean of the two runs:

| Preset | All 35 answers (each run) | edit | review | redo: the task again in a new conversation | chat: the control | Prompts read, s a run |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6 UD-Q4_K_M | 62.4 (63.1, 61.7) -> **82.8** (81.0, 84.6), **x1.33** | 62.6 -> 83.7 | 62.5 -> 82.8 | 65.3 -> **93.6**, x1.43 | 57.0 -> 68.2 | 120.0 -> 96.6 |
| Qwen3.6 UD-Q3_K_M | 68.4 (74.1, 62.8) -> **81.2** (85.8, 76.5), **x1.19** | 69.3 -> 80.8 | 62.8 -> 79.8 | 72.8 -> **91.5**, x1.26 | 63.3 -> 67.2 | 110.0 -> 87.2 |
| Ornith 1.5 APEX-I-QUALITY | 47.0 (50.7, 43.2) -> **61.8** (63.8, 59.7), **x1.32** | 48.0 -> 59.7 | 44.0 -> 56.2 | 47.9 -> **73.6**, x1.54 | 45.2 -> 53.0 | 128.9 -> 103.1 |
| Ornith 1.5 APEX-I-COMPACT | 60.4 (63.2, 57.6) -> **76.1** (78.5, 73.7), **x1.26** | 61.0 -> 74.6 | 58.5 -> 70.4 | 63.5 -> **89.4**, x1.41 | 55.8 -> 64.8 | 107.8 -> 85.0 |

- **x1.19 to x1.33 over the whole session**, and x1.26 to x1.54 where a new conversation repeats an earlier one's task:
  there the new engine's `--suffix-drafts` guesses from the first conversation, which the old engine's lookup cannot
  see across conversations (in *redo* the new engine's windows had 2.7 to 5.4 right guesses, the old one's 1.5 to 3.2).
- The chat, which has nothing to repeat, gains x1.06 to x1.20, the windows' time.
- The second round was slower than the first for both engines with three of the four files (Qwen3.6 UD-Q3_K_M: old 74.1
  -> 62.8, new 85.8 -> 76.5): runs of the whole session move 4 to 15% with the machine's state, which is why the two
  engines ran next to each other in each round. The new engine's figures for UD-Q4_K_M, 81.0 and 84.6, are 3 to 12%
  under step 7's and step 8's checks of nearly the same build and flags (87.2 to 92.0), on other days.
- The session was recorded with Qwen3.6 UD-Q4_K_M (step 7); the prompts are the same for every file, but the other
  files write other answers (Qwen3.6 UD-Q3_K_M 11 to 13 of the 35 as recorded, Ornith 4 to 6, UD-Q4_K_M 17 to 22), so
  each file is measured on its own text, and the files are not compared with each other here.

**What the smaller files cost in answers** (`test_scripts/quality.py`, step 6's yardstick: the five agent prompts, 300
steps each). The reference is the larger file with FastStrata's flags and no lossy knob (`--tail-skip 0 --suffix-drafts
256 --cpu-multi`), the smaller file follows its answers with the same flags; with `--tail-skip 6` too it is the preset as
it runs. The Ornith reference's answer to the tool prompt ended after 120 tokens, so Ornith has 1,320 steps a run.

| Qwen3.6, reference UD-Q4_K_M (PPL 1.1745) | Top-1 changed | Of them where the reference had 60% or more | Mean KL | KL p99 | PPL | tok/s writing the same text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UD-Q4_K_M again (the floor), two runs | 0.47%, 0.67% | 0, 0 | 0.00134, 0.00085 | 0.025, 0.016 | 1.1758, 1.1762 | 64.3, 64.4 |
| UD-Q4_K_M `--tail-skip 6` (the Q4 preset), one run | 1.40% | 0 | 0.00573 | 0.070 | 1.1797 | 73.1 |
| **UD-Q3_K_M**, two runs | 4.00%, 4.13% | **5, 5** | **0.02038, 0.02022** | 0.237, 0.221 | 1.1977, 1.1975 | 61.5, 63.3 |
| UD-Q3_K_M `--tail-skip 6` (the Q3 preset), two runs | 4.40%, 4.20% | 7, 5 | 0.02433, 0.02340 | 0.304, 0.304 | 1.2004, 1.1999 | 69.7, 69.9 |

| Ornith 1.5, reference APEX-I-QUALITY (PPL 1.1613) | Top-1 changed | Of them where the reference had 60% or more | Mean KL | KL p99 | PPL | tok/s writing the same text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| APEX-I-QUALITY again (the floor), two runs | 2.27%, 1.74% | 1, 4 | 0.01072, 0.00969 | 0.194, 0.192 | 1.1774, 1.1695 | 52.6, 52.3 |
| **APEX-I-COMPACT**, two runs | 7.73%, 7.65% | **26, 27** | **0.08646, 0.08879** | 1.037, 1.070 | 1.2768, 1.2790 | 63.9, 63.9 |

The tokens the reference was sure of (its probability -> the smaller file's top token and its probability):

- UD-Q3_K_M, the same five in both runs: a line break `\n` 74% -> `\n\n` 56% (the text before the tool call); ` tokens`
  74% -> ` of` 53% (the explanation); ` fast` 82% -> ` solid` 43% and 53%, `,` 63% -> `.` 52% and `,` 63% -> ` with` 47%
  and 43% (the article). With `--tail-skip 6` also, in both runs, one in code: `if (!b.exception` then `)` 61% -> `.load` 29%.
- APEX-I-COMPACT, 26 and 27, nearly all the same in both runs; among them code: `if (!b.err` then `)` 88% ->
  `.load` 34% and `.exchange` 38%; in the pytest file `fake_ctypes.c` then `_int` 70% -> `_w` 61%, `fake_ctypes` then
  `.c` 73% -> `.Win` 58%, `fake_ctypes_module` then `.Struct` 86% -> `.w` 54% and 78%, `fake_wintypes` then `.L` 63%
  -> `.H` 63%; in the explanation `tokens[` then `0` 63% -> `1` 68%; in prose ` too` 96% -> ` less` 81% and ` small`
  98% -> ` little` 51% and 65%.

What the numbers say:

- **UD-Q3_K_M costs a mean KL of 0.020** against UD-Q4_K_M: 3.5 times what the Q4 preset's own tail skip costs
  (0.0057), 15 to 24 times the floor, and more than step 9's 4-bit keys and values (0.009 to 0.013). It changed 5
  tokens the Q4 file was 60% or more sure of in 1,500 steps, once in 300 tokens, the same five in both runs and all in
  prose or punctuation; the perplexity of the Q4 file's answers rises 2.0%. With the tail skip, as the preset runs,
  0.024 and one changed token in code.
- **APEX-I-COMPACT costs 4.3 times as much as UD-Q3_K_M**: a mean KL of 0.087 against APEX-I-QUALITY, 26 or 27 sure
  tokens changed in 1,320 steps (once in 50 tokens), several of them names in code, and the perplexity up 10%. It puts
  the attention's and the shared expert's weights in Q4_K as well, where UD-Q3_K_M keeps them in Q8_0 (the table of
  files above).
- **Ornith's floor is nine times Qwen3.6's** (a KL of 0.010 against 0.0011, with 1 and 4 sure changes, one of a line
  break at 94%): two runs of the same file move its answers more. The yardstick tells less apart for Ornith.
- On the same text the smaller Qwen3.6 file is not faster (61.5 and 63.3 against 64.3 and 64.4 tok/s without the tail
  skip, 69.7 and 69.9 against 73.1 with it); APEX-I-COMPACT is x1.22 faster than APEX-I-QUALITY (63.9 against 52.3 and
  52.6).

**A 4 GB card, simulated** (`faststrata_runs.py fit`: `test_scripts/vram_hold.py --card 4096`, then step 9's
`fit_run.py` for each run, a prompt that fills the context and 256 tokens written). The holder took 2,047 MiB of VRAM
(VRAM only: it checks each piece against the card's free VRAM) and left the engine 3,082 MiB free of 6,143, as in
step 9; it gave them back at the end (5,130 MiB free again). GPU memory in system RAM: Windows' GPU Process Memory
counters (Shared Usage) of the engine, the most of the samples every 10 s; the engine's pinned buffers are in it.

| Preset, engine | Context, cache | Experts in VRAM at the start, after the answer | Prompt read in | Writing: tok/s; ms a window | GPU memory in system RAM, most |
| --- | --- | --- | ---: | --- | ---: |
| Qwen3.6 UD-Q3_K_M, FastStrata | 16K, `--kv-q8` | 0, 0 | 17.8 s (15,840 tokens) | 54.9; 37.2 | 1,250 MB |
| | 16K, `--kv-q4` | 0, 0 | 17.8 s | 56.0; 35.6 | 1,250 MB |
| | 32K, `--kv-q8`, two runs | 0, 0 | 37.3, 37.5 s (32,224 tokens) | 39.6; 54.1 and 49.8; 40.0 | 1,378 MB |
| | 32K, `--kv-q4`, two runs | 0, 0 | 37.5, 37.3 s | 48.0; 40.6 and 48.4; 39.3 | 1,378 MB |
| | 48K, `--kv-q8` | 0, 0 | 60.5 s (48,608 tokens) | 42.0; 42.4 | 1,592 MB |
| | 48K, `--kv-q4` | 0, 0 | 60.2 s | 47.9; 39.8 | 1,506 MB |
| Qwen3.6 UD-Q3_K_M, old | 32K, `--kv-q8` | 0, 0 | 39.1 s | 35.3; 57.8 | 1,436 MB |
| Qwen3.6 UD-Q4_K_M, FastStrata | 32K, `--kv-q8`, two runs | 0, 0 | first run: stopped after 293 s, "graph experts: the card gave up waiting for the host"; second: 39.3 s | 55.0; 37.4 | 1,378 MB |
| | 32K, `--kv-q8`, without `--graph-experts` | 0, 0 | 39.2 s | 53.5; 40.7 | 1,378 MB |
| Ornith 1.5 APEX-I-COMPACT, FastStrata | 32K, `--kv-q8` | 398, 146 | 39.7 s | 47.8; 32.4 | 1,634 MB |

- **Every run fitted the simulated card, and nothing was paged to system RAM.** UD-Q3_K_M leaves no room for one
  expert beside its dense weights, the draft block and the caches in the 3,082 MiB free, even at 16K, nor does
  UD-Q4_K_M at 32K (both keep those weights in Q8_0): every routed expert runs on the CPU, and `--elastic-prefill` finds no VRAM to lend ("a wider batch takes 510 MiB or more, and 0
  MiB are free"; calls of 512). APEX-I-COMPACT's dense weights are smaller (Q4_K) and leave it 398 experts at the start,
  146 after the 32K prompt. The GPU memory in system RAM stayed at 1,250 to 1,634 MB and grew with the context: these
  are the engine's pinned buffers, which on the whole card with the same 48K prompt took 1,686 MB (UD-Q3_K_M,
  `--kv-q8`, 981 of 1,341 experts left in VRAM, 52.9 tok/s). In step 9 the card's memory did go to system RAM at 64K,
  and there the figure grew 0.4 to 0.6 GB above the pinned buffers and writing fell to 19 and 31 tok/s.
- **`--kv-q8` fits 48K on a 4 GB card**, so `--kv-q4` is not needed up to there. The two caches wrote alike at 16K and
  32K (32K: 39.6 and 49.8 against 48.0 and 48.4 tok/s, two runs of one cache 26% apart); at 48K 42.0 against 47.9, one
  run each. Contexts past 48K were not run on the simulated card (step 9: 64K pages with UD-Q4_K_M).
- **Old -> new at 32K with UD-Q3_K_M: 35.3 -> 44.7 tok/s** (39.6 and 49.8), x1.27; the prompt in 39.1 -> 37.4 s.
- **On a 4 GB card UD-Q4_K_M writes faster than UD-Q3_K_M**: 55.0 and 53.5 (without `--graph-experts`) against 44.7
  tok/s at 32K. With no expert in VRAM the CPU computes them all, and Q4_K's run with `--cpu-multi` where IQ3_XXS takes
  the old path. It takes 24.7 GB of RAM against 19.3 (the RAM table below).
- **One run of UD-Q4_K_M stopped:** its prompt was read (the engine's line about it is in the log), then nothing moved
  for about four minutes (its RAM steady, 23.8 GB available), and it ended with "graph experts: the card gave up waiting
  for the host" (the card waits 5 s at most, `kTurnWaitMs`, for the host's word), the message step 9's fp16 cache
  stopped with at 32K. The second run of the same case wrote 55.0 tok/s (step 9: 50.0), and one without
  `--graph-experts` 53.5. The cause was not found.

**The engine's RAM, for a PC with 32 GB** (`test_scripts/proc_mem.py`, new: GetProcessMemoryInfo for the private
bytes and the part of them in RAM, the private working set; QueryWorkingSetEx over the views of the model's file for
its pages in the engine's working set; `--watch` samples every strata-poly every 5 s, the file's pages every 5
minutes, and `fit_run.py` samples them once after the answer). Nothing here simulated less RAM.

| Runs | Private bytes | Of them in RAM | The model file's pages in the working set | Working set |
| --- | ---: | ---: | --- | ---: |
| Qwen3.6 UD-Q3_K_M, one conversation (4 GB card 16K to 48K; the yardstick's five prompts; 2k, 37k, 139k) | 6.2 to 8.5 GB | 2.0 to 2.8 GB | 16.7 of 17.1 GB | 18.5 to 19.9 GB |
| Ornith 1.5 APEX-I-COMPACT, the same | 6.2 to 8.8 GB | 1.9 to 3.2 GB | 17.2 of 17.4 GB | 19.2 to 20.5 GB |
| Qwen3.6 UD-Q4_K_M and APEX-I-QUALITY, the same | 6.4 to 8.8 GB | 1.9 to 3.2 GB | 22.1 of 22.7 GB, 23.3 of 23.7 GB | 24.1 to 26.6 GB |
| The agent benchmark, old / new (four conversations, the parked ones in RAM), APEX-I-COMPACT | 9.6 / 10.7 GB | 4.0 / 5.1 GB | 17.2 GB | 21.3 / 22.4 GB |
| The same, Qwen3.6 UD-Q3_K_M | 9.6 / 10.5 GB | 3.9 / 4.9 GB | 16.7 GB | 20.7 / 21.7 GB |
| The same, Qwen3.6 UD-Q4_K_M; APEX-I-QUALITY | 9.5 / 10.6 GB; 9.7 / 10.9 GB | 3.9 / 5.0 GB; 4.0 / 5.2 GB | 22.1 GB; 23.3 GB | 26.1 / 27.2 GB; 27.5 / 28.6 GB |

- Private bytes are mostly committed and never touched: 6.9 GB three seconds after a start, when the whole working set
  was 4.9 GB.
  What the RAM has to hold is the private part in RAM (2 to 3 GB with one conversation, 4 to 5 GB with four, the
  parked conversations among them) and nearly all of the model's file: the engine reads every expert that is not in
  VRAM from it while it writes.
- GB here are 10^9 bytes, as the files' sizes are. A "32 GB" PC has 32 GiB, 34.4 GB, a little less of it for Windows
  where an integrated GPU reserves some; this laptop's 64 GiB show as 59.9 GiB, 64.3 GB. With no engine running,
  Windows and the user's programs had 15.6 to 20.5 GB of it in use today.
- So an engine with a 17 GB file (UD-Q3_K_M, APEX-I-COMPACT) holds 19 to 22 GB of RAM while it runs, which leaves a
  32 GB PC 12 to 15 GB for Windows and the other programs: less than this laptop's Windows and desktop used today,
  enough for a lighter one. The parked conversations can grow to `--conversation-cache-mib` (12 GiB in the daily
  config), but a conversation is parked only while 2,560 MiB of physical RAM stay free
  (`--conversation-cache-min-free-mib`). Where the other programs take more, Windows pages the
  file's experts out and the engine reads them from the SSD again as it writes (not measured here). The 23 GB files
  hold 24 to 29 GB and leave a 32 GB PC too little.
- Not measured: a PC with 32 GB. These are the engine's own figures on a 64 GB laptop.

**Summary, per preset, for the README.** RTX 3060 Laptop 6 GB, Ryzen 7 5800H, 64 GB DDR4-3200, Windows 11, measured
on 2026-10-20 by this step's runs. *Old*: PolyStrata with the Q8 cache added, the build and flags of 2026-10-17;
*FastStrata*: kv-q8 at 150beb0b with its daily flags. Each figure is the mean of the runs in the tables above (two
runs, the 139k prompt one run each way). The 4 GB card is this card with 2 GiB of its VRAM held by another process.

*Qwen3.6-35B-A3B UD-Q4_K_M (22.7 GB)*

| | Old | FastStrata | |
| --- | ---: | ---: | ---: |
| Writing after a 2k prompt | 56.4 tok/s | 71.8 tok/s | x1.27 |
| Writing the same text (five agent prompts of 1.3k to 3.7k tokens) | 62.3 tok/s | 72.9 tok/s | x1.17 |
| Writing after a 37k prompt | 41.9 tok/s | 66.9 tok/s | x1.60 |
| Writing after a 139k prompt | 24.7 tok/s | 48.8 tok/s | x1.98 |
| Reading the 37k prompt | 47.2 s | 36.4 s | x1.30 |
| Reading the 139k prompt | 265.1 s | 248.2 s | x1.07 |
| Agent benchmark, all 35 answers | 62.4 tok/s | 82.8 tok/s | x1.33 |
| The same task again in a new conversation | 65.3 tok/s | 93.6 tok/s | x1.43 |
| 4 GB card, 32K context, `--kv-q8` | not run | 55.0 tok/s (one of two runs stopped) | |
| RAM in use while it runs | 24.1 to 27.2 GB | 24.1 to 27.2 GB | |

*Qwen3.6-35B-A3B UD-Q3_K_M (17.1 GB)*

| | Old | FastStrata | |
| --- | ---: | ---: | ---: |
| Writing after a 2k prompt | 58.8 tok/s | 66.2 tok/s | x1.13 |
| Writing the same text | 67.3 tok/s | 69.4 tok/s | x1.03 |
| Writing after a 37k prompt | 37.0 tok/s | 58.6 tok/s | x1.58 |
| Writing after a 139k prompt | 26.4 tok/s | 45.6 tok/s | x1.73 |
| Reading the 37k prompt | 45.7 s | 34.8 s | x1.31 |
| Reading the 139k prompt | 258.8 s | 241.2 s | x1.07 |
| Agent benchmark, all 35 answers | 68.4 tok/s | 81.2 tok/s | x1.19 |
| The same task again in a new conversation | 72.8 tok/s | 91.5 tok/s | x1.26 |
| 4 GB card, 32K, `--kv-q8` | 35.3 tok/s | 44.7 tok/s | x1.27 |
| 4 GB card, 16K / 48K, `--kv-q8` | not run | 54.9 / 42.0 tok/s | |
| Answers against UD-Q4_K_M | | mean KL 0.020, top token changed at 4.1% of steps, 5 sure ones in 1,500 (UD-Q4_K_M's own tail skip: 0.006, 1.4%, 0) | |
| RAM in use while it runs | 18.5 to 21.7 GB | 18.5 to 21.7 GB | |

*Ornith 1.5 35B-A3B APEX-I-QUALITY (23.7 GB), no tail skip*

| | Old | FastStrata | |
| --- | ---: | ---: | ---: |
| Writing after a 2k prompt | 45.8 tok/s | 51.0 tok/s | x1.11 |
| Writing the same text | 46.4 tok/s | 52.4 tok/s | x1.13 |
| Writing after a 37k prompt | 28.9 tok/s | 39.6 tok/s | x1.37 |
| Writing after a 139k prompt | 21.5 tok/s | 36.8 tok/s | x1.71 |
| Reading the 37k prompt | 47.7 s | 36.7 s | x1.30 |
| Reading the 139k prompt | 269.2 s | 236.9 s | x1.14 |
| Agent benchmark, all 35 answers | 47.0 tok/s | 61.8 tok/s | x1.32 |
| The same task again in a new conversation | 47.9 tok/s | 73.6 tok/s | x1.54 |
| 4 GB card | not run | not run | |
| RAM in use while it runs | 25.5 to 28.6 GB | 25.5 to 28.6 GB | |

*Ornith 1.5 35B-A3B APEX-I-COMPACT (17.4 GB), no tail skip*

| | Old | FastStrata | |
| --- | ---: | ---: | ---: |
| Writing after a 2k prompt | 60.8 tok/s | 63.1 tok/s | x1.04 |
| Writing the same text | 57.9 tok/s | 63.6 tok/s | x1.10 |
| Writing after a 37k prompt | 41.8 tok/s | 56.2 tok/s | x1.34 |
| Writing after a 139k prompt | 24.6 tok/s | 43.6 tok/s | x1.77 |
| Reading the 37k prompt | 46.5 s | 35.2 s | x1.32 |
| Reading the 139k prompt | 268.6 s | 224.6 s | x1.20 |
| Agent benchmark, all 35 answers | 60.4 tok/s | 76.1 tok/s | x1.26 |
| The same task again in a new conversation | 63.5 tok/s | 89.4 tok/s | x1.41 |
| 4 GB card, 32K, `--kv-q8` | not run | 47.8 tok/s | |
| Answers against APEX-I-QUALITY | | mean KL 0.088, top token changed at 7.7% of steps, 26 to 27 sure ones in 1,320 (APEX-I-QUALITY run again: 0.010, 2.0%, 1 to 4) | |
| RAM in use while it runs | 19.2 to 22.4 GB | 19.2 to 22.4 GB | |

Read together: FastStrata is faster with every preset and every prompt, most after long prompts (x1.71 to x1.98 after
139k) and in agent sessions (x1.19 to x1.33); after short prompts x1.03 to x1.27, least with UD-Q3_K_M, whose experts
none of steps 4 and 8 reach. The smaller files buy RAM, not speed (UD-Q3_K_M is slower than UD-Q4_K_M on FastStrata,
APEX-I-COMPACT faster than APEX-I-QUALITY), and cost answers: UD-Q3_K_M moderately (a KL 3.5 times the Q4 preset's tail
skip), APEX-I-COMPACT a lot (4.3 times that again, with names in code among its changed tokens).

**Not measured:** a real 4 GB card (here a 6 GB card with VRAM held: a real one's bandwidth, PCIe link and the desktop's
share of it differ), a PC with 32 GB of RAM, other CPUs and cards; the 4 GB card with the old engine and other files
than UD-Q3_K_M, with APEX-I-QUALITY, and past 48K (UD-Q3_K_M) or 32K (the others); a second run at 139k; a task
benchmark (HumanEval, a tool call's success) for what the smaller files cost, thinking on, sampled answers, and the
long-context yardstick for them; Ornith with a tail skip; the server and the Gateway (the engine on its own); why one
run with `--graph-experts` stopped on the 4 GB card; Linux, HIP and SYCL.

## Where it stands

| | Before (16 threads, 2026-10-18) | Now (step 8) |
| --- | --- | --- |
| Writing after 1,976 tokens | 46.7 to 61.4 tok/s | 71.3 to 73.3 tok/s |
| Writing after 37,433 tokens | 41.7, 42.6 tok/s | 62.1, 64.9 tok/s |
| Writing after 139k tokens | 23 tok/s at 150k (2026-10-17) | 49.4 tok/s |
| Writing after 100k tokens (the sessions' runs) | 34.4 tok/s | 63.9 tok/s |
| The agent benchmark, all 35 answers | | 85.9 -> 87.2, 90.6 tok/s (step 7) |
| The same task again in a new conversation | | 90.1 -> 97.9, 106.7 tok/s (step 7) |
| Reading 1,976 tokens | 5.4 s | 3.6 s |
| Reading 37,433 tokens | 47 s (793 tok/s) | 36 s (1,040 tok/s) |
| Reading 139k tokens | about 270 s at 150k (2026-10-17) | 245.7 s |

Where the time goes now after a long prompt (2026-10-19, after step 7): Nsight Systems, 5 s of writing after the
139,056-token prompt (`test_scripts/make_prompt.py 139000`), the daily flags. 10.9% of the experts came from VRAM.

| Kernel | Share of the 5 s | A call |
| --- | ---: | ---: |
| `wait_host_kernel`: the card waiting for the host, nearly all of it for the CPU's experts | 35.7% | 149 us |
| `attn_split_kernel`: the attention, reading the Q8 cache at about 277 GB/s of 336 | 18.8% | 546 us |
| `native_mmvq_multi_kernel`: the layers' own matrices | 16.4% | 29 us |
| `native_small_mmvq_kernel`, `gemv_kernel` | 7.4% | |
| `native_q6_k_mmvq_kernel`: the draft block's output rows | 3.8% | 759 us |

The engine's thread spent 1.1 s of the 5 in CUDA calls (0.46 s of it waiting on streams, 0.41 s launching 6,148
graphs at 66 us). The attention is at the card's bandwidth now, so it can only go faster by reading fewer bytes
(a 4-bit cache or reading a part of it, both lossy); the CPU's experts are the largest share and can go faster
without changing any result.

Step 8 measured the CPU's experts at 75 to 84% of what the RAM gives while an answer is written and brought them
to 83 to 98% (`--cpu-multi`, the same bits): the session measured 5% faster answers after 2k, the check here 1 to 2%;
after 139k not measurably faster. It is switched on.

Next: whether other programs' memory use pushes the experts out of RAM and slows writing, and if so a way to keep
them resident (a proposed task); the CPU's experts' fixed costs a call, or the next layer's experts guessed from its
router (step 8); a task benchmark on this engine to tell the lossy settings' costs apart where the KL cannot
(HumanEval, tool-call success; until then no lossy knob beyond today's tail skip). The 4-bit attention cache is
built and measured (step 9): opt-in, for a context the card cannot hold otherwise, not deployed here. Step 10 measured
the four FastStrata presets against the engine of 2026-10-17 (x1.03 to x1.27 after short prompts, x1.71 to x1.98 after
139k, x1.19 to x1.33 in the agent benchmark), what UD-Q3_K_M and APEX-I-COMPACT cost in answers, and a simulated 4 GB
card, where `--kv-q8` fits 48K with UD-Q3_K_M.
