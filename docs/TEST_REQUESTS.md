# Test requests and hardware wanted

## Wanted for FastStrata

FastStrata was measured on one laptop (RTX 3060 Laptop 6 GB, Ryzen 7 5800H, 64 GB DDR4, Windows 11) and on 4 GB
simulated on its card. A `BENCH.bat` run from any of these answers a row: how to send it is in
[COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md#faststrata-on-your-pc).

| Wanted | Why |
|---|---|
| A real 4 GB card, RTX 20 series or newer (an RTX 3050 Laptop 4 GB, for one) | The 4 GB figures come from a 6 GB card with 2 GiB held; a real card's bandwidth, PCIe link and desktop share were not measured |
| An 8 GB card (RTX 4060, RTX 3070 Laptop, RTX 2080) | Setup offers about 248K there from its estimate less 10%, marked "estimated"; a run replaces the estimate |
| A PC with 32 GB of RAM, with UD-Q3_K_M or APEX-I-COMPACT | The 17 GB presets hold 19-22 GB of RAM while they run; whether the rest is enough for Windows and an editor was not measured |
| Linux with an NVIDIA card | Every FastStrata number is from Windows 11 |
| RTX 40 and RTX 50 cards, 12 GB and more | More of the experts fit in VRAM; how much faster that makes FastStrata was not measured |

## Upstream's list

PolyStrata is built and measured on a few machines. Results from other hardware are welcome and are used: they tell us
what a setup really does and show where the engine picks something wrong. Write a report as described in
[COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md): the build, the card and PCIe width, the CPU and RAM speed, the
model file, the exact command, three runs per size, and what you did not test. Post it in an issue, or on the pull
request it is about.

This list is edited by the maintainers. An item stays until a report answers it.

| Wanted | Why | Example of an earlier answer |
|---|---|---|
| Two NVIDIA cards of different sizes (a large and a small one), in both card orders, with `--layer-split auto` | There is no unequal pair here to check the split planner; the card order and the small card's free memory changed the split | #880, #1238 |
| Three or four NVIDIA GPUs | No multi-GPU NVIDIA rig of that size here | #880 |
| Pascal (sm_61: Tesla P40, P100, GTX 10 series) | Older cards run on the fp32 path and can behave differently from newer ones | #875, #876 |
| A Linux host with 32 GB of RAM or less | The expert file tier and the resident mode depend on how much of the model fits in RAM | #1194 |
| A long soak (an hour or more of back-to-back requests) on a split or resident setup | Shows crashes, drift and heat that a short run does not | #848 |

To add a request, open a pull request that adds a row, or tell the maintainers in an issue.
