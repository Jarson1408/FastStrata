# DeepSeek V4 Flash (`strata-poly`)

`strata-poly`, the engine that runs GLM 5.3 Flash, also runs DeepSeek V4 Flash: llama.cpp's architecture
`deepseek4`, read from Unsloth's GGUF. The server, the web app and the API are the ones the README describes.

It is early. This project has run it on one machine, with Linux and one NVIDIA card.

## What it was measured on

| | |
| --- | --- |
| Graphics card | RTX 5090, 32 GB |
| CPU | Xeon Platinum 8470Q, 25 vCPU of it (a cloud machine) |
| RAM | 120 GiB |
| Built with | CUDA 12.8 |
| Model | `DeepSeek-V4-Flash-0731-UD-IQ2_XXS`, three files, 90.9 GB, from [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) |
| Context | 32K |

| | `strata-poly` | llama.cpp, 10 layers' experts on the card | llama.cpp, every expert on the CPU |
| --- | ---: | ---: | ---: |
| Writes answers | 48 to 53 tokens/s | 25 to 28 tokens/s | 20 to 21 tokens/s |
| Reads a prompt of 3,384 tokens | 1,003 tokens/s | 96 tokens/s | 56 tokens/s |
| Reads a prompt of 10,398 tokens | 1,147 tokens/s | | |
| Starts | 6 to 7 s with the files in the page cache | | |

All of it ran on the machine above, with 24 CPU threads. The engine's numbers are with the expert profile it
writes as it runs; the first answers after a start without one came at 42 to 43 tokens/s. For llama.cpp a small
program drives its library and its own `deepseek4` graph and writes greedy tokens, with the routed experts of the
first layers in RAM, as `--n-cpu-moe` keeps them. The experts of ten layers are what its 32 GB card had room for
beside the rest of the model at this context. That column is the faster of two runs; the other wrote 24 to 26
tokens/s and read the prompt at 72.

The README's row is PolyStrata's own benchmark through the server, the test every model of this engine gets
([BACKENDS.md](BACKENDS.md#a-prompt-of-30000-tokens)), as setup installs the model: a short chat written at 56
tokens/s, and a prompt of 29,750 tokens read whole at 1,038 to 1,040, with the first token after 29 s and the
answer after it written at 39 to 44: the second and third of three runs, as for the other models. The first, with
no expert profile yet, wrote the short chat at 56.5.

With greedy sampling the engine and llama.cpp write the same tokens until two candidates are nearly tied: 8 of 8,
64 of 64 and 10 tokens on three short prompts, 15 of 32 on the prompt of 3,384 tokens and all 32 on the one of
10,398. Only the long prompts reach far enough back to use the compressed rows, the indexer's pick and the rows of
128 positions, so those agree as well.

## What you need

- Linux and an NVIDIA card, RTX 20 or newer. It has run on 32 GB of VRAM only. About 11.5 GB go to the weights,
  the caches and the buffers before any expert, so setup stops and asks below 13.5 GB.
- RAM for the experts the card does not hold. The files are mapped and those experts are read from them while the
  model answers, so they should stay in the OS file cache: about 62 GB of the 84 GB on a 32 GB card. Setup's
  estimate for that card is 72 GB of RAM. It has run with 120 GB only.
- 91 GB of disk for the three files.

## Install

```
./setup.sh --family deepseek
```

Setup checks the PC and asks two things: the context, and whether to keep several conversations in RAM. There is
one size so far, so it does not ask for one. Then it compiles the engine, downloads the three files, checks their
SHA-256 and starts the model. The web app is at `http://127.0.0.1:8080`, and `./setup.sh` starts the model again
without asking.

- You already have the files: `./setup.sh --family deepseek --gguf-dir /folder/with/the/files`, the three files
  under their names from Unsloth's repository.
- The other options are GLM's ([GLM.md](GLM.md#install)).

On the test machine, with the files already on the disk, this took about four minutes: two for the compile, one and a
half for the files' SHA-256.

Setup writes the run config `strata-dsv4-ud-iq2_xxs.json` and `run-dsv4-ud-iq2_xxs.sh` into the PolyStrata folder.
No expert profile is shipped: the engine starts one at the first start and keeps it up to date.

## Long contexts

Setup offers contexts up to 128K; `--context` takes up to 1,048,576 (`./setup.sh --family deepseek --context 524288`),
as does `--max-context` in the run config. A context token takes 6,880 bytes of VRAM, so on the 32 GB card even
512K leaves the expert cache most of its room: 3.4 GB less than at 32K.

Two prompts made of this repository's documents, with three facts planted at 15%, 50% and 85% of them and three
questions after them, through the server with thinking off:

| Context | Prompt | Read and answered in | Facts found |
| --- | ---: | ---: | ---: |
| 128K | 116,473 tokens | 174 s | 3 of 3 |
| 512K | 487,529 tokens | 1,923 s | 3 of 3 |

A prompt is read more slowly as it grows. The indexer scores every compressed row of its layers for every token, so
its work grows with the square of the prompt, and the layers of 128 positions read more rows too. Near the end of
the 487,529-token prompt the engine read about 150 tokens per second.

## How it differs from the other models

The engine and its options are GLM's ([GLM.md](GLM.md#how-it-uses-the-card)). What is DeepSeek V4's own:

- **The attention keeps little per token.** A token reads the last 128 positions in full, and the rest of the
  conversation as compressed rows: in 21 of the 43 layers one row for every 4 positions, of which an indexer picks
  the 512 the token reads, and in 20 others one row for every 128 positions, all of which it reads. The first two
  layers read the last 128 positions only. All 64 heads share one key row, which is also the value. A context token
  takes 6,880 bytes of VRAM, so a 128K context takes 0.7 GB more than 32K.
- **The experts.** 43 layers, each with 256 routed experts of which a token uses 6, and one shared expert.
  The first three layers pick a token's experts by the token, from a table in the file. On the 32 GB card about
  2,840 of the 11,008 experts stay in VRAM at a 32K context.
- **Four residual streams**, mixed by hyper-connections, as in GLM 5.3 Flash.
- **Two layers converted at load.** In Unsloth's file the down experts of layers 26 and 42 are MXFP4, which no
  kernel here runs on the card. Every MXFP4 value in them is a small integer times a power of two that Q5_0 holds
  exactly, so the engine converts them once at load into 1.38 GiB of RAM each and runs them like the other layers.
  Before the conversion those two layers took more of a long prompt's time than the other 41 together.
- **Reading a prompt** goes 1,024 tokens at a time, as for GLM. The attention of a batch takes 8 heads of a token
  at a time, so that the key rows they share are read once for all 8.
- **Tool calls** come in DeepSeek's own format, DSML, which the server turns into OpenAI's and Anthropic's tool
  calls. Thinking is on unless a request turns it off; `"high"` adds the template's own instruction to think
  thoroughly.

## What is missing

- The other sizes in Unsloth's repository, from UD-IQ1_S (83 GB) to UD-Q3_K_XL (128 GB). The engine should read
  them, but only UD-IQ2_XXS has been run, and setup offers no other.
- Guessing ahead with a draft model. The repository has one for this model (`dspark`, 11 GB), which the engine does
  not run yet. Guesses still come from text the conversation repeats.
- Windows, by this project: setup asks before it installs it there. AMD cards: the engine compiles for them, the
  model has not been tried on one.
- Pictures. The model reads text only.
- A faster indexer for long prompts: it runs on the card's ordinary cores, not its tensor cores.
