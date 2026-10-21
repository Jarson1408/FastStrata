"""FastStrata's speedups (docs/OPTIMIZATION_LOG.md): setup gives Qwen3.6-35B-A3B and Ornith 1.5 on strata-poly each one
where the PC's card, CPU and RAM and the preset take it, and only where the engine built here has it (poly_speedups(),
poly_add_speedups()); it picks the preset by the PC's RAM and VRAM (poly_pick()).  The model menu and --check offer
those two alone; the families FastStrata does not support install with --family as before, with none of the options.
Fake machines - no GPU, no compile, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_speedups
    python tools/test_setup_speedups.py --show          what setup prints and writes for the measured laptop

The laptop's install is checked against its daily config as of 2026-10-19 (DAILY below); with DAILY_CONFIG=<a
strata-qwen36-ud-q4_k_m.json>, or that file beside setup.py, against the file as well.  It is only read.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# What the engine binaries say of their options (poly_engine_flags() reads the file): this fork's names every speedup,
# upstream's release has the old ones only.  The near misses must not count.
FORK_USAGE = (b"usage: strata-poly --model FILE.gguf\n  --threads N   CPU threads\n  --cpu-multi   ...\n"
              b"  --suffix-drafts MIB  ...\n  --suffix-corpus FILE\n  --kv-whole\n  --kv-q8   Qwen3.5 / 3.6\n"
              b"  --kv-q4   Qwen3.5 / 3.6\n  --kv-q4-keys q8|q4\n"
              b"  --attn-split  ...\n  --graph-experts  ...\n  --prompt-batch N\n  --elastic-prefill N\n"
              b"  --elastic-batch N\n  --conversation-cache-mib N\n  --conversation-cache-slots N\n  --tail-skip N\n\0"
              b"--attn-split\0--kv-q8\0")
UPSTREAM_USAGE = (b"usage: strata-poly --model FILE.gguf\n  --threads N\n  --kv-whole\n  --kv-q8x\n  --kv-q4-old\n"
                  b"  --attn-split-old\n"
                  b"  --prompt-batch N\n  --conversation-cache-mib N\n  --conversation-cache-slots N\n  --tail-skip N\n")

# The laptop every speedup was measured on (docs/OPTIMIZATION_LOG.md, "The machine and the model"), as setup sees it:
# nvidia-smi's 6,144 MiB, the 59.86 GiB Windows gives of its 64 GB, 8 cores with 16 threads.
LAPTOP = {"cards": [("NVIDIA GeForce RTX 3060 Laptop GPU", 6144, "86")], "ram": 59.86, "avx2": True,
          "threads": (8, 16), "hybrid": None, "win": True}
# its daily config, strata-qwen36-ud-q4_k_m.json, 2026-10-19 (step 8): the options setup chooses, without the paths
DAILY = ["--max-context", "147456", "--conversation-cache-mib", "12288", "--tail-skip", "6", "--kv-q8",
         "--prompt-batch", "512", "--attn-split", "--threads", "8", "--conversation-cache-slots", "8", "--graph-experts",
         "--elastic-prefill", "1024", "--suffix-drafts", "256", "--cpu-multi"]
# a 32 GB laptop with an 8 GB card and an Intel CPU of 6 performance cores (two threads each) and 8 efficiency cores
HYBRID = {"cards": [("NVIDIA GeForce RTX 4060 Laptop GPU", 8188, "89")], "ram": 31.7, "avx2": True,
          "threads": (14, 20), "hybrid": (6, 8), "win": True}
# a 16 GB desktop with a 12 GB card and a Ryzen 5 5600 (6 cores, 12 threads)
PC16 = {"cards": [("NVIDIA GeForce RTX 3060", 12288, "86")], "ram": 15.8, "avx2": True, "threads": (6, 12),
        "hybrid": None, "win": True}
# Linux, an RX 7900 XTX and a Ryzen 9 7950X (16 cores, 32 threads), 64 GB
AMD = {"amd": [("AMD Radeon RX 7900 XTX", 23.98, "gfx1100")], "ram": 62.5, "avx2": True, "threads": (16, 32),
       "hybrid": None, "win": False}
# two RTX 3060 12 GB sharing the model's layers, 64 GB, 8 cores with 16 threads
TWO = {"cards": [("NVIDIA GeForce RTX 3060", 12288, "86")] * 2, "ram": 62.7, "avx2": True, "threads": (8, 16),
       "hybrid": None, "win": False}
# a gaming PC with a 4 GB card (an RTX 3050 Laptop) and 32 GB, 6 cores with 12 threads: FastStrata's smallest card
CARD4 = {"cards": [("NVIDIA GeForce RTX 3050 Laptop GPU", 4096, "86")], "ram": 31.7, "avx2": True, "threads": (6, 12),
         "hybrid": None, "win": True}
# the laptop with 12 GB of RAM: no preset fits it (each wants 23 GB or more beside its card)
SMALL = dict(LAPTOP, ram=11.8)
# a GTX 1080 Ti (Pascal, the experimental CUDA 12 engine)
PASCAL = {"cards": [("NVIDIA GeForce GTX 1080 Ti", 11264, "61")], "ram": 31.9, "avx2": True, "threads": (8, 16),
          "hybrid": None, "win": False}
# a Xeon X5670 (Westmere: SSE4.2, no AVX2; 6 cores, 12 threads) with an RTX 3060 12 GB
NO_AVX2 = {"cards": [("NVIDIA GeForce RTX 3060", 12288, "86")], "ram": 47.9, "avx2": False, "threads": (6, 12),
           "hybrid": None, "win": False}

PATHS = ("--model", "--expert-profile", "--draft-model")
CUDA_ONLY = ("--attn-split", "--graph-experts", "--elastic-prefill", "--kv-q8", "--kv-q4")
NEW = CUDA_ONLY + ("--threads", "--suffix-drafts", "--cpu-multi", "--prompt-batch", "--conversation-cache-slots")
UNSUPPORTED = ("tiel", "gemma4", "glm", "deepseek")             # strata-poly's; the Strata engine's: test_setup_menu


def options(args: list) -> dict:
    """An engine argument list as {option: value, or None for a switch}, without the files'."""
    out, i = {}, 0
    while i < len(args):
        flag, nxt = args[i], args[i + 1] if i + 1 < len(args) else None
        has = nxt is not None and not str(nxt).startswith("--")
        if flag not in PATHS:
            out[flag] = nxt if has else None
        i += 2 if has else 1
    return out


def daily_file() -> Path | None:
    """The laptop's own config, when this checkout has it or DAILY_CONFIG names it (read, never written)."""
    p = Path(os.environ["DAILY_CONFIG"]) if os.environ.get("DAILY_CONFIG") else ROOT / "strata-qwen36-ud-q4_k_m.json"
    return p if p.is_file() else None


def quiet(fn, *args):
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


def rule(machine, model=None, mode="on", ctx=32768, family="qwen36"):
    """poly_speedups() for a fake machine, as poly_setup() calls it: -> ({option: value}, notes, kv plan, size)."""
    fam = setup.FAMILIES[family]
    hip = "amd" in machine
    cards = machine.get("amd") or machine["cards"]
    vram = sum(c[1] if hip else c[1] / 1024 for c in cards) - 2.0 * (len(cards) - 1)
    model = model or setup.poly_pick(fam["models"], machine["ram"], vram, hip) or \
        setup.poly_default(fam["models"], machine["ram"], vram)
    d = fam["models"][model]
    need = setup.poly_ram_need(d, vram)
    plan = setup.poly_kv_plan(fam, d, ctx, vram)
    gpu = {"arch": cards[0][2], "archs": sorted({c[2] for c in cards}),
           "vram_gb": cards[0][1] if hip else cards[0][1] / 1024}
    conv = setup.poly_conv_mib(family, mode, machine["ram"], need)
    items, notes = setup.poly_speedups(family, mode, d, gpu, hip, list(range(len(cards))) if len(cards) > 1 else [],
                                       machine["ram"], need, machine["avx2"], machine["threads"], machine["hybrid"],
                                       ("f16", 1024) if hip else plan[:2], conv)
    return options([x for a, _why in items for x in a]), notes, plan, model


class Presets(unittest.TestCase):
    """FastStrata's presets: Qwen3.6-35B-A3B's UD-Q4_K_M and UD-Q3_K_M, Ornith 1.5's APEX-I-QUALITY and APEX-I-COMPACT,
    in the order setup picks them; step 10 of docs/OPTIMIZATION_LOG.md measured all four with FastStrata's options."""
    WANT = {"qwen36": {"UD-Q4_K_M": ("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", 22.7, 27, True, True),
                       "UD-Q3_K_M": ("Qwen3.6-35B-A3B-UD-Q3_K_M.gguf", 17.1, 21, True, False)},
            "ornith": {"APEX-I-QUALITY": ("Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf", 23.7, 28, True, True),
                       "APEX-I-COMPACT": ("Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf", 17.4, 22, True, True)}}

    def test_each_says_its_file_download_ram_and_whether_it_was_measured(self):
        self.assertEqual(setup.FASTSTRATA_FAMILIES, tuple(self.WANT))
        for family, want in self.WANT.items():
            fam = setup.FAMILIES[family]
            self.assertTrue(fam["speedups"])
            self.assertEqual(list(fam["models"]), list(want), family)
            self.assertEqual(list(fam["sha256"]), [w[0] for w in want.values()] + [fam["mmproj"]])
            for m, (file, gb, ram, measured, multi) in want.items():
                d = fam["models"][m]
                self.assertEqual((setup.model_file(fam, m, 1), d["file"]), (file, file))
                self.assertAlmostEqual(fam["sha256"][file][0] / 1e9, gb, delta=0.05)
                self.assertEqual((d["download_gb"], d["ram_gb"], d["measured"], d["cpu_multi"]),
                                 (gb, ram, measured, multi), m)
                self.assertEqual(round(setup.poly_ram_need(d, 8.0)), ram)
        self.assertNotIn("UD-Q2_K_XL", setup.FAMILIES["qwen36"]["models"])
        self.assertNotIn("APEX-I-MINI", setup.FAMILIES["ornith"]["models"])
        self.assertNotIn("tail_skip", setup.FAMILIES["ornith"])    # not measured there: none, as before
        for d in (*setup.FAMILIES["qwen36"]["models"].values(), *setup.FAMILIES["ornith"]["models"].values()):
            self.assertIn("its speed was measured", setup.preset_note(d))
        self.assertIn("not measured", setup.preset_note({"measured": False}))    # a preset not measured yet
        # UD-Q3_K_M's IQ3_XXS experts: neither --cpu-multi nor --graph-experts reads them (step 10); the others' do
        self.assertEqual([m for f in self.WANT for m, d in setup.FAMILIES[f]["models"].items()
                          if d.get("graph_experts") is False], ["UD-Q3_K_M"])
        self.assertEqual(setup.preset_note(setup.FAMILIES["gemma4"]["models"]["UD-Q4_K_M"]), "")

    def test_setup_picks_one_by_ram_and_vram(self):
        q, o = setup.FAMILIES["qwen36"]["models"], setup.FAMILIES["ornith"]["models"]
        self.assertEqual(setup.poly_pick(q, 31.7, 8188 / 1024), "UD-Q4_K_M")     # the RAM and VRAM for it
        self.assertEqual(setup.poly_pick(q, 20.0, 8188 / 1024), "UD-Q3_K_M")     # UD-Q4_K_M wants ~27 GB there
        self.assertEqual(setup.poly_pick(q, 15.8, 12.0), "UD-Q3_K_M")
        self.assertIsNone(setup.poly_pick(q, 12.0, 8188 / 1024))                  # too small: RAM
        self.assertIsNone(setup.poly_pick(q, 59.86, 6.0))                          # too small: 4.7 GB beside + 2
        self.assertEqual(setup.poly_pick(o, 31.7, 8188 / 1024), "APEX-I-QUALITY")
        self.assertEqual(setup.poly_pick(o, 20.0, 8188 / 1024), "APEX-I-COMPACT")

    def test_the_options_follow_the_preset(self):
        for family, want in self.WANT.items():
            for m in want:
                got, notes, plan, _m = rule(LAPTOP, model=m, ctx=147456, family=family)
                # the architecture's options go to every preset; --cpu-multi and --graph-experts to those whose
                # experts they read (not UD-Q3_K_M's IQ3_XXS).  UD-Q3_K_M's dense weights take 0.1 GB more VRAM
                # (setup's estimate): 144K needs the 4-bit cache there
                self.assertTrue({"--threads", "--attn-split", "--elastic-prefill", "--suffix-drafts"} <= set(got), m)
                self.assertEqual("--graph-experts" in got, m != "UD-Q3_K_M", m)
                self.assertEqual(any(n.startswith("--graph-experts left out: this size's experts are IQ3_XXS")
                                     for n in notes), m == "UD-Q3_K_M", notes)
                self.assertEqual(plan[0], "q4" if m == "UD-Q3_K_M" else "q8", m)
                self.assertIn("--kv-q4" if m == "UD-Q3_K_M" else "--kv-q8", got, m)
                self.assertEqual("--cpu-multi" in got, m != "UD-Q3_K_M", m)
                self.assertEqual(any(n.startswith("--cpu-multi left out: this size's experts") for n in notes),
                                 m == "UD-Q3_K_M", notes)

    def test_the_4_bit_cache_is_looked_for_in_the_engine(self):
        # FASTSTRATA KV-Q4 (step 9): poly_kv_plan() gives it, and only an engine that has it gets it
        self.assertIn("FASTSTRATA KV-Q4", inspect.getsource(setup.poly_kv_plan))
        self.assertIn("--kv-q4", setup.POLY_SPEEDUP_FLAGS)
        self.assertIn("--kv-q4-keys", setup.POLY_SPEEDUP_FLAGS)


class Rule(unittest.TestCase):
    """poly_speedups() on each fake machine."""

    def test_the_laptop(self):
        got, notes, plan, _m = rule(LAPTOP, model="UD-Q4_K_M", ctx=147456)
        self.assertEqual(plan, ("q8", 512, True))                  # 144K needs Q8_0 caches and batches of 512
        want = {k: v for k, v in options(DAILY).items() if k not in ("--max-context", "--conversation-cache-mib",
                                                                     "--tail-skip")}
        self.assertEqual(got, want)
        self.assertEqual(notes, [])

    def test_the_laptop_at_32k_keeps_the_16_bit_cache(self):
        got, _notes, plan, _m = rule(LAPTOP, model="UD-Q4_K_M")
        self.assertEqual(plan, ("f16", 1024, True))
        self.assertNotIn("--kv-q8", got)
        self.assertNotIn("--kv-q4", got)
        self.assertNotIn("--prompt-batch", got)
        self.assertIn("--attn-split", got)

    def test_a_hybrid_laptop(self):
        got, notes, _plan, model = rule(HYBRID)
        self.assertEqual(model, "UD-Q4_K_M")
        # the hybrid rule writes --threads (hybrid_pool_workers: 6 - 1 + 8 / 2 = 9); one a core is not for it
        self.assertNotIn("--threads", got)
        self.assertEqual(setup.hybrid_pool_workers(HYBRID["hybrid"]), 9)
        self.assertEqual(got, {"--attn-split": None, "--graph-experts": None, "--elastic-prefill": "1024",
                               "--suffix-drafts": "256", "--cpu-multi": None})
        self.assertEqual(notes, [])

    def test_16_gb(self):
        got, notes, _plan, model = rule(PC16)
        # UD-Q3_K_M wants ~17 GB beside a 12 GB card: the index's 160 MiB stay with the model
        self.assertEqual(model, "UD-Q3_K_M")
        self.assertNotIn("--suffix-drafts", got)
        self.assertTrue(any("--suffix-drafts left out" in n for n in notes), notes)
        self.assertNotIn("--graph-experts", got)                   # IQ3_XXS experts
        self.assertNotIn("--cpu-multi", got)
        self.assertEqual(got["--threads"], "6")
        self.assertNotIn("--conversation-cache-slots", got)       # 8 GiB at most, and off by default here

    def test_amd_gets_no_cuda_option(self):
        got, notes, _plan, _m = rule(AMD, model="UD-Q4_K_M", ctx=147456)
        for flag in CUDA_ONLY + ("--prompt-batch",):
            self.assertNotIn(flag, got)
        self.assertEqual(got, {"--threads": "16", "--suffix-drafts": "256", "--cpu-multi": None,
                               "--conversation-cache-slots": "8"})
        self.assertTrue(any("neither built nor run for AMD" in n for n in notes), notes)

    def test_pascal_gets_no_attn_split(self):
        got, notes, _plan, _m = rule(PASCAL)
        self.assertNotIn("--attn-split", got)
        self.assertNotIn("--graph-experts", got)                   # Volta (7.0) on
        self.assertIn("--elastic-prefill", got)                    # the engine checks the card's VMM itself
        self.assertTrue(any("--attn-split left out" in n and "sm_61" in n for n in notes), notes)
        volta = {**PASCAL, "cards": [("Tesla V100-PCIE-16GB", 16384, "70")]}
        got, _notes, _plan, _m = rule(volta)
        self.assertNotIn("--attn-split", got)
        self.assertIn("--graph-experts", got)

    def test_a_layer_split_gets_no_graphs_or_elastic_reads(self):
        got, notes, _plan, _m = rule(TWO)
        self.assertNotIn("--graph-experts", got)
        self.assertNotIn("--elastic-prefill", got)
        self.assertIn("--attn-split", got)
        self.assertTrue(any("they run on one card" in n for n in notes), notes)

    def test_no_avx2(self):
        got, notes, _plan, _m = rule(NO_AVX2, model="UD-Q4_K_M")
        self.assertNotIn("--cpu-multi", got)
        self.assertIn("--cpu-multi left out: this CPU has no AVX2", notes)
        self.assertIn("--attn-split", got)

    def test_the_laptop_past_144k_gets_the_4_bit_cache(self):
        # 160K and 168K do not fit in 8 bits (none was left at 159K); with --kv-q4 they do
        for ctx in (163840, 172032):
            got, notes, plan, _m = rule(LAPTOP, model="UD-Q4_K_M", ctx=ctx)
            self.assertEqual(plan, ("q4", 512, True), ctx)
            self.assertIn("--kv-q4", got)
            self.assertNotIn("--kv-q4-keys", got)
            self.assertNotIn("--kv-q8", got)
            self.assertEqual(got["--prompt-batch"], "512")
            self.assertEqual(notes, [])
        # past that the keys go to 4 bits too, and past 200K nothing fits by the estimate
        for ctx, fits in ((180000, True), (204800, True), (215000, False), (262144, False)):
            got, _notes, plan, _m = rule(LAPTOP, model="UD-Q4_K_M", ctx=ctx)
            self.assertEqual(plan, ("q4k4", 512, fits), ctx)
            self.assertEqual(got["--kv-q4-keys"], "q4")
            self.assertNotIn("--kv-q4", got)

    def test_a_4_gb_card(self):
        # by setup's estimate (VRAM beside the experts measured on an RTX 5090) a 4 GB card holds 8K with the 8-bit
        # cache and batches of 512, and nothing longer even in 4 bits: the 4-bit cache is offered, with the warning.
        # The simulated 4 GB card (docs/OPTIMIZATION_LOG.md step 9) ran 32K with --kv-q8: the estimate is low there,
        # and is to be fitted again from a 4 GB card's measurements (this case then changes with it)
        for ctx, want in ((8192, ("q8", 512, True)), (16384, ("q4k4", 512, False)), (32768, ("q4k4", 512, False))):
            for model in ("UD-Q4_K_M", "UD-Q3_K_M"):
                got, _notes, plan, _m = rule(CARD4, model=model, ctx=ctx)
                self.assertEqual(plan, want, (model, ctx))
                self.assertEqual(got.get("--kv-q4-keys"), "q4" if want[0] == "q4k4" else None, ctx)
                self.assertEqual("--kv-q8" in got, want[0] == "q8", ctx)

    def test_a_card_under_5_gb_gets_no_graph_experts(self):
        # step 10: on the 6 GB card with 2 GiB held one run of two stopped with "graph experts: the card gave up
        # waiting for the host" (cause not found); it gave 55.0 against 53.5 tok/s there.  The rest stays
        for model in ("UD-Q4_K_M", "UD-Q3_K_M"):
            got, notes, _plan, _m = rule(CARD4, model=model)
            self.assertNotIn("--graph-experts", got, model)
            self.assertTrue({"--threads", "--attn-split", "--elastic-prefill"} <= set(got), model)
        got, notes, _plan, _m = rule(CARD4, model="UD-Q4_K_M")
        self.assertIn("--cpu-multi", got)
        self.assertTrue(any(n.startswith("--graph-experts left out: this card has 4 GB of VRAM") for n in notes), notes)
        # the 6 GB laptop keeps it, and so does a card of 5 GB
        self.assertIn("--graph-experts", rule(LAPTOP, model="UD-Q4_K_M")[0])
        card5 = {**CARD4, "cards": [("NVIDIA GeForce RTX 3050 Laptop GPU (5 GB)", 5120, "86")]}
        self.assertIn("--graph-experts", rule(card5, model="UD-Q4_K_M")[0])

    def test_an_8_gb_card_never_needs_the_4_bit_cache(self):
        for ctx in (131072, 163840, 262144):
            self.assertNotIn("--kv-q4", rule(HYBRID, model="UD-Q4_K_M", ctx=ctx)[0], ctx)

    def test_amd_gets_no_4_bit_cache(self):
        amd6 = {**AMD, "amd": [("AMD Radeon RX 5600M", 6.0, "gfx1010")]}
        got, notes, plan, _m = rule(amd6, model="UD-Q4_K_M", ctx=163840)
        self.assertEqual(plan[0], "q4")
        self.assertNotIn("--kv-q4", got)
        self.assertTrue(any("--kv-q4" in n and "AMD" in n for n in notes), notes)

    def test_ornith_takes_qwen36s_options(self):
        got, notes, plan, model = rule(LAPTOP, family="ornith", ctx=147456, model="APEX-I-QUALITY")
        self.assertEqual(plan, ("q8", 512, True))
        want = {k: v for k, v in options(DAILY).items() if k not in ("--max-context", "--conversation-cache-mib",
                                                                     "--tail-skip")}
        self.assertEqual((got, notes), (want, []))

    def test_off_and_the_families_faststrata_does_not_support(self):
        self.assertEqual(rule(LAPTOP, model="UD-Q4_K_M", mode="off", ctx=147456)[:2], ({}, []))
        for family in UNSUPPORTED:
            with self.subTest(family):
                fam = setup.FAMILIES[family]
                d = next(iter(setup.family_models(family).values()))
                self.assertNotIn("speedups", fam)
                self.assertEqual(setup.poly_speedups(family, "on", d, {"arch": "86"}, False, [], 256, 30, True,
                                                     (8, 16), None, ("q8", 512), 12288), ([], []))
                self.assertEqual(setup.poly_conv_mib(family, "on", 256, 30), setup.POLY_CONV_CACHE_MIB)

    def test_threads(self):
        self.assertEqual(setup.poly_smt_threads((8, 16), None), 8)
        self.assertIsNone(setup.poly_smt_threads((8, 8), None))      # no SMT: the engine's count
        self.assertIsNone(setup.poly_smt_threads((14, 20), (6, 8)))  # hybrid: hybrid_pool_workers()
        self.assertIsNone(setup.poly_smt_threads(None, None))

    def test_conversation_cache(self):
        need = setup.poly_ram_need(setup.FAMILIES["qwen36"]["models"]["UD-Q4_K_M"], 6.0)
        self.assertAlmostEqual(need, 28.7, delta=0.05)
        self.assertEqual(setup.poly_conv_mib("qwen36", "on", 59.86, need), 12288)   # the laptop
        self.assertEqual(setup.poly_conv_mib("qwen36", "on", 47.8, need), 8192)     # 48 GB: 19 GB beside it
        self.assertEqual(setup.poly_conv_mib("qwen36", "off", 59.86, need), 8192)


class Estimate(unittest.TestCase):
    """poly_fit_gb() against what the 6 GB laptop measured with --kv-q8 --prompt-batch 512."""
    fam = setup.FAMILIES["qwen36"]
    d = fam["models"]["UD-Q4_K_M"]

    def left(self, ctx, kv=True, batch=512):
        return 6.0 - setup.poly_fit_gb(self.fam, self.d, ctx, kv, batch)

    def test_the_measured_points(self):
        self.assertAlmostEqual(self.left(147456), 0.28, delta=0.01)      # 144K: 0.17 GiB of experts stayed
        self.assertGreaterEqual(self.left(147456), setup.POLY_EXPERT_FLOOR_GB)
        for n in (159000, 162816):                                       # 159K: none was left
            self.assertLess(self.left(n), setup.POLY_EXPERT_FLOOR_GB)
        self.assertLess(self.left(147456, False, 1024), 0)              # 16-bit caches: far over the card
        self.assertEqual(setup.poly_kv_plan(self.fam, self.d, 163840, 6.0), ("q4", 512, True))
        self.assertEqual(setup.poly_kv_plan(self.fam, self.d, 204800, 6.0), ("q4k4", 512, True))
        self.assertEqual(setup.poly_kv_plan(self.fam, self.d, 262144, 6.0), ("q4k4", 512, False))

    def test_an_8_gb_card_needs_nothing_at_131k(self):
        self.assertEqual(setup.poly_kv_plan(self.fam, self.d, 131072, 8188 / 1024), ("f16", 1024, True))
        self.assertEqual(setup.poly_kv_plan(self.fam, self.d, 131072, 6.0), ("q8", 512, True))

    def test_the_4_bit_cache_saves_its_bytes(self):
        # 9,152 bytes a token (keys Q8_0, values Q4_0) where 11,968 and 22,528; True and False are still "q8", "f16"
        self.assertEqual(setup.POLY_KV_Q4_SAVES, 22528 - 9152)
        for ctx in (65536, 147456, 200000):
            self.assertAlmostEqual(self.left(ctx, "q4") - self.left(ctx, "q8"), (ctx - 32768) * 2816 / 2**30)
            self.assertEqual(self.left(ctx, True), self.left(ctx, "q8"))
            self.assertEqual(self.left(ctx, False, 1024), self.left(ctx, "f16", 1024))
            self.assertAlmostEqual(self.left(ctx, "q4k4") - self.left(ctx, "q4"), (ctx - 32768) * 2816 / 2**30)


class Engine(unittest.TestCase):
    def test_the_flags_in_a_binary(self):
        with tempfile.TemporaryDirectory() as t:
            fork, up = Path(t) / "fork.exe", Path(t) / "up.exe"
            fork.write_bytes(b"\x00MZ" * 100 + FORK_USAGE)
            up.write_bytes(UPSTREAM_USAGE)
            self.assertEqual(setup.poly_engine_flags(fork), set(setup.POLY_SPEEDUP_FLAGS))
            self.assertEqual(setup.poly_engine_flags(up), {"--threads", "--prompt-batch", "--conversation-cache-slots"})
            self.assertEqual(setup.poly_engine_flags(Path(t) / "none.exe"), set())

    def test_an_engine_without_them_gets_none_of_them(self):
        with tempfile.TemporaryDirectory() as t:
            exe = Path(t) / setup.POLY_EXE
            exe.write_bytes(UPSTREAM_USAGE)
            items = [(["--threads", "8"], "a"), (["--attn-split"], "b"), (["--suffix-drafts", "256"], "c")]
            args = ["--model", "m.gguf"]
            _code, out = quiet(setup.poly_add_speedups, args, items, [], exe, "UD-Q4_K_M", True)
            self.assertEqual(args, ["--model", "m.gguf", "--threads", "8"])
            self.assertIn("does not have them", out)
            self.assertIn("--attn-split --suffix-drafts", out)

    def test_cpu_threads_on_linux(self):
        with tempfile.TemporaryDirectory() as t:
            for c in range(16):                                         # 8 cores, two threads each
                topo = Path(t, "devices", "system", "cpu", f"cpu{c}", "topology")
                topo.mkdir(parents=True)
                (topo / "physical_package_id").write_text("0\n")
                (topo / "core_id").write_text(f"{c % 8}\n")
            with mock.patch.object(setup, "WIN", False), \
                    mock.patch.object(setup.os, "sched_getaffinity", lambda pid: set(range(16)), create=True):
                self.assertEqual(setup.cpu_threads(t), (8, 16))
                (Path(t) / "fs" / "cgroup").mkdir(parents=True)
                (Path(t) / "fs" / "cgroup" / "cpu.max").write_text("max 100000\n")
                self.assertEqual(setup.cpu_threads(t), (8, 16))
                (Path(t) / "fs" / "cgroup" / "cpu.max").write_text("400000 100000\n")   # a quota: the engine's count
                self.assertIsNone(setup.cpu_threads(t))


class Install(unittest.TestCase):
    """setup.main() for a strata-poly model on a fake machine, every outside effect mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data").mkdir()
        self.built = []

    def tearDown(self):
        self.tmp.cleanup()

    def install(self, machine, argv=(), family="qwen36", engine=FORK_USAGE, answers=None):
        """answers: None = --yes; else {words of a question: its answer} (Enter for the others), the questions in
        self.asked.  engine: the usage text the built engine's binary carries.  family None: from the menu."""
        self.asked = []
        hip = "amd" in machine

        def fake_input(prompt=""):
            if answers is None:
                raise AssertionError(f"asked {prompt!r}")
            self.asked.append(prompt)
            return next((v for k, v in answers.items() if k in prompt), "")

        found = [{"index": i, "name": n, "vram_gb": mib / 1024, "arch": arch, "driver": "617.42"}
                 for i, (n, mib, arch) in enumerate(machine.get("cards", []))]
        amd = [{"index": i, "name": n, "vram_gb": gb, "arch": arch, "driver": "amdgpu", "vendor": "amd"}
               for i, (n, gb, arch) in enumerate(machine.get("amd", []))]

        def fake_build(gpu, *a, **k):
            self.built.append(("hip" if hip else "cuda", gpu.get("archs")))
            eng = self.t / setup.ENGINE_POLY_DIR
            eng.mkdir(exist_ok=True)
            (eng / setup.POLY_EXE).write_bytes(engine)
            (eng / "BUILD.json").write_text(json.dumps({"source": "local-hip", "backend": "hip"} if hip else
                                                       {"source": "local", "cuda_dirs": ["/cuda/bin"]}))
            return eng

        def fake_download(url, dst, what=None):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "WIN", machine["win"]),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "OLD_GPUS", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "amd_gpus", lambda *a, **k: [dict(g) for g in amd]),
            mock.patch.object(setup, "amd_device_access_problem", lambda *a, **k: None),
            mock.patch.object(setup, "amd_pci_devices", lambda *a, **k: []),
            mock.patch.object(setup, "pcie_link", lambda i: None),
            mock.patch.object(setup, "gpu_drives_display", lambda g: False),
            mock.patch.object(setup, "desktop_on_card", lambda g: (False, "")),
            mock.patch.object(setup, "ram_gb", lambda: machine["ram"]),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", machine["avx2"], False)),
            mock.patch.object(setup, "cpu_floor", lambda avx2: "" if avx2 else "none"),
            mock.patch.object(setup, "cpu_cores", lambda: machine["hybrid"]),
            mock.patch.object(setup, "cpu_threads", lambda: machine["threads"]),
            mock.patch.object(setup, "page_file_gb", lambda: 64.0),
            mock.patch.object(setup, "free_gb", lambda p: 500.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", mock.Mock(side_effect=AssertionError("the Qwen engine"))),
            # the release has no engine of this fork's source (its asset is named after the source's hash)
            mock.patch.object(setup, "get_poly_prebuilt", lambda gpu: False),
            mock.patch.object(setup, "build_engine_poly", fake_build),
            mock.patch.object(setup, "build_engine_poly_hip", fake_build),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "whole_shard", lambda s: False),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda *a: None),
            mock.patch.object(setup, "run", lambda cmd, *a, **k: None),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"start-{tag}.bat"),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.object(sys, "argv", ["setup.py", *(["--family", family] if family else []),
                                            *(["--yes"] if answers is None else []), "--no-start", "--details",
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", fake_input),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)
        cfg = next(iter(sorted(self.t.glob("strata-*.json"))), None)
        return code, out, (json.loads(cfg.read_text()) if cfg else None)

    def laptop(self, *argv, **kw):
        return self.install(LAPTOP, ["--model", "UD-Q4_K_M", "--context", "147456", *argv], **kw)

    def test_the_laptop_gets_its_daily_config(self):
        code, out, cfg = self.laptop()
        self.assertEqual(code, 0, out)
        self.assertEqual(options(cfg["args"]), options(DAILY))
        self.assertEqual(Path(cfg["exe"]), self.t / setup.ENGINE_POLY_DIR / setup.POLY_EXE)
        self.assertEqual(self.built, [("cuda", ["86"])])            # compiled from this source
        self.assertEqual(cfg["speedups"], "on")
        self.assertIn("size: UD-Q4_K_M; its speed was measured with FastStrata's options", out)
        self.assertIn("FastStrata's speedups (docs/OPTIMIZATION_LOG.md: measured on an RTX 3060 Laptop 6 GB, a Ryzen 7 "
                      "5800H, 64 GB, with Qwen3.6-35B-A3B UD-Q4_K_M in step 10", out)
        self.assertNotIn("left out", out)
        self.assertNotIn("not supported", out)
        mine = daily_file()
        if mine is not None:                                        # the laptop's own file, read only
            self.assertEqual(options(cfg["args"]), options(json.loads(mine.read_text(encoding="utf-8-sig"))["args"]))

    def test_the_q3_preset_gets_neither_cpu_multi_nor_graph_experts(self):
        code, out, cfg = self.install(LAPTOP, ["--model", "UD-Q3_K_M", "--context", "147456"])
        self.assertEqual(code, 0, out)
        got = options(cfg["args"])
        # 144K in 8 bits, as step 10 ran it on the 6 GB laptop (the estimate alone, which takes its dense weights for
        # 0.1 GB more, would give the 4-bit cache); its IQ3_XXS experts take neither --cpu-multi nor --graph-experts
        self.assertEqual(set(got) - {"--max-context", "--conversation-cache-mib", "--tail-skip"},
                         set(options(DAILY)) - {"--max-context", "--conversation-cache-mib", "--tail-skip",
                                                "--cpu-multi", "--graph-experts"})
        self.assertIn("--kv-q8", got)
        self.assertIn("attention cache: 8 bits, as measured on a 6 GB card at 144K", out)
        self.assertIn("size: UD-Q3_K_M; its speed was measured with FastStrata's options", out)
        self.assertIn("with Qwen3.6-35B-A3B UD-Q3_K_M in step 10", out)
        self.assertIn("--cpu-multi left out: this size's experts", out)
        self.assertIn("--graph-experts left out: this size's experts are IQ3_XXS", out)

    def test_a_4_gb_card(self):
        # named, as setup's estimate finds the card too small; no --graph-experts under 5 GB (step 10)
        code, out, cfg = self.install(CARD4, ["--model", "UD-Q4_K_M", "--context", "8192"])
        self.assertEqual(code, 0, out)
        got = options(cfg["args"])
        self.assertNotIn("--graph-experts", got)
        self.assertTrue({"--threads", "--attn-split", "--elastic-prefill", "--cpu-multi", "--kv-q8"} <= set(got), got)
        self.assertIn("--graph-experts left out: this card has 4 GB of VRAM", out)

    def test_the_laptop_takes_the_8_bit_cache_without_being_asked(self):
        """FASTSTRATA UX: 144K in 8 bits is the 6 GB laptop's measured setting, not a question, and the laptop is
        not told it is too small (it asked "Go on anyway?" and "Keep the attention cache in 8 bits?" before, on an
        estimate made for a 32 GB card)."""
        code, out, cfg = self.laptop(answers={})
        self.assertEqual(code, 0, out)
        self.assertEqual([q.split("?")[0].strip() for q in self.asked],
                         ["Is the desktop on the NVIDIA card", "Do you want images", "How long may an answer be",
                          "Keep conversations in RAM"])
        self.assertIn("--kv-q8", cfg["args"])
        self.assertIn("attention cache: 8 bits, so 144K fits this card", out)
        self.assertNotIn("Go on anyway", out)

    def test_off_writes_the_config_from_before(self):
        code, out, cfg = self.laptop("--speedups", "off")
        self.assertEqual(code, 0, out)
        self.assertEqual(options(cfg["args"]), {"--max-context": "147456", "--conversation-cache-mib": "8192",
                                                "--tail-skip": "6"})
        self.assertNotIn("FastStrata's speedups (", out)
        self.assertNotIn("8 bits", out)
        # a setup run again keeps the choice, and the earlier config's conversation cache
        code, out, cfg = self.laptop()
        self.assertEqual(code, 0, out)
        self.assertEqual((cfg["speedups"], options(cfg["args"])["--conversation-cache-mib"]), ("off", "8192"))
        self.assertNotIn("--attn-split", cfg["args"])
        code, out, cfg = self.laptop("--speedups", "on")
        self.assertEqual(code, 0, out)
        self.assertIn("--attn-split", cfg["args"])
        self.assertNotIn("--conversation-cache-slots", cfg["args"])   # 8 GiB, as it was kept

    def test_an_engine_of_upstream_source_gets_none_of_them(self):
        code, out, cfg = self.laptop(engine=UPSTREAM_USAGE)
        self.assertEqual(code, 0, out)
        for flag in CUDA_ONLY + ("--suffix-drafts", "--cpu-multi"):
            self.assertNotIn(flag, cfg["args"])
        self.assertEqual(options(cfg["args"])["--threads"], "8")      # the ones it has stay
        self.assertIn("--attn-split --graph-experts --elastic-prefill --suffix-drafts --cpu-multi", out)

    def test_a_hybrid_laptop(self):
        code, out, cfg = self.install(HYBRID)
        self.assertEqual(code, 0, out)
        # UD-Q4_K_M: ~27 GB of RAM beside the 8 GB card, which leaves the conversation cache's 8 GB no room (off)
        self.assertEqual(cfg["model_name"], "qwen3.6-35b-a3b-ud-q4_k_m")
        # the context: the largest that keeps the cache in 16 or 8 bits on this card, up to 144K (FASTSTRATA UX)
        self.assertEqual(options(cfg["args"]), {"--max-context": "147456", "--tail-skip": "6", "--threads": "9",
                                                "--attn-split": None, "--graph-experts": None,
                                                "--elastic-prefill": "1024", "--suffix-drafts": "256",
                                                "--cpu-multi": None})

    def test_16_gb(self):
        code, out, cfg = self.install(PC16)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["model_name"], "qwen3.6-35b-a3b-ud-q3_k_m")
        self.assertEqual(options(cfg["args"]), {"--max-context": "147456", "--tail-skip": "6", "--threads": "6",
                                                "--attn-split": None, "--elastic-prefill": "1024"})
        self.assertIn("--suffix-drafts left out", out)
        self.assertIn("--graph-experts left out: this size's experts are IQ3_XXS", out)

    def test_amd(self):
        code, out, cfg = self.install(AMD, ["--model", "UD-Q4_K_M", "--context", "131072"])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.built, [("hip", ["gfx1100"])])
        for flag in CUDA_ONLY + ("--prompt-batch",):
            self.assertNotIn(flag, cfg["args"])
        self.assertEqual(options(cfg["args"]), {"--max-context": "131072", "--conversation-cache-mib": "12288",
                                                "--tail-skip": "6", "--threads": "16", "--suffix-drafts": "256",
                                                "--cpu-multi": None, "--conversation-cache-slots": "8",
                                                "--kv-whole": None})

    def test_two_cards(self):
        code, out, cfg = self.install(TWO, ["--gpus", "0,1"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        got = options(cfg["args"])
        self.assertNotIn("--graph-experts", got)
        self.assertNotIn("--elastic-prefill", got)
        self.assertIn("--attn-split", got)
        self.assertIn("they run on one card", out)

    def test_no_avx2(self):
        code, out, cfg = self.install(NO_AVX2, ["--model", "UD-Q4_K_M"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("--cpu-multi", cfg["args"])
        self.assertIn("--cpu-multi left out: this CPU has no AVX2", out)

    def test_ornith(self):
        code, out, cfg = self.install(LAPTOP, ["--model", "APEX-I-QUALITY", "--context", "147456"], family="ornith")
        self.assertEqual(code, 0, out)
        self.assertEqual(options(cfg["args"]), {k: v for k, v in options(DAILY).items() if k != "--tail-skip"})
        self.assertEqual(cfg["speedups"], "on")
        self.assertIn("size: APEX-I-QUALITY; its speed was measured with FastStrata's options", out)
        self.assertIn("with Ornith 1.5 35B-A3B APEX-I-QUALITY in step 10", out)
        self.assertNotIn("not supported", out)

    def test_the_families_faststrata_does_not_support_install_as_before(self):
        for family, model in (("tiel", "UD-Q4_K_XL"), ("gemma4", "UD-Q4_K_M")):
            with self.subTest(family):
                for p in self.t.glob("strata-*.json"):
                    p.unlink()
                code, out, cfg = self.install(LAPTOP, ["--model", model, "--context", "147456"], family=family)
                self.assertEqual(code, 0, out)
                self.assertIn("is not supported by FastStrata", out)
                got = options(cfg["args"])
                for flag in NEW:
                    self.assertNotIn(flag, got)
                self.assertEqual(got["--conversation-cache-mib"], "8192")
                self.assertNotIn("speedups", cfg)
                self.assertNotIn("FastStrata's speedups", out)
                self.assertNotIn("8 bits", out)

    def test_the_menu(self):
        # FASTSTRATA UX: one menu of the four presets, each judged for this PC; Enter takes the recommended one.  The
        # Strata engine is neither downloaded nor built (get_prebuilt raises here).  FASTSTRATA MULTI: the menu is
        # step 2's second way, "Choose the models myself" (tools/test_setup_multi.py has the first)
        code, out, cfg = self.install(HYBRID, family=None, answers={"Which way": "2"})
        self.assertEqual(code, 0, out)
        menu = [ln for ln in out.splitlines() if ln[:5].strip().rstrip(")").isdigit()]
        self.assertEqual([ln.strip()[:2] for ln in menu[:2]], ["1)", "2)"])            # the two ways
        self.assertTrue(menu[2].strip().startswith("1) Qwen3.6-35B-A3B Q4 (UD-Q4_K_M)"), menu)
        self.assertTrue(menu[3].strip().startswith("2) Qwen3.6-35B-A3B Q3 (UD-Q3_K_M)"), menu)
        self.assertTrue(menu[4].strip().startswith("3) Ornith 1.5 35B-A3B Quality (APEX-I-QUALITY)"), menu)
        self.assertTrue(menu[5].strip().startswith("4) Ornith 1.5 35B-A3B Compact (APEX-I-COMPACT)"), menu)
        # an 8 GB card: no measurement, so setup's estimate less 10%, said to be one
        self.assertIn("fits this PC, up to 248K context (estimated); 66.9 tok/s; the most exact   (safe)", out)
        asked = [q.split("?")[0].strip() for q in self.asked]
        self.assertEqual(asked[:3], ["Is the desktop on the NVIDIA card", "Which way", "Which ones"])
        self.assertNotIn("Size", asked)
        self.assertEqual(asked[3:5], ["Do you want images", "Which context"])
        for title in ("Qwen3.8-Flash-Next", "GLM 5.3 Flash", "DeepSeek", "Gemma 4", "Tiel Coder"):
            self.assertFalse(any(title in ln for ln in menu), title)
        self.assertEqual(cfg["model_name"], "qwen3.6-35b-a3b-ud-q4_k_m")
        self.assertEqual(self.built, [("cuda", ["89"])])

    def test_a_pc_too_small_is_told_so(self):
        # 12 GB of RAM: no preset fits; --yes alone stops (the menu's pick is not the user's), --model with --yes goes on
        code, out, cfg = self.install(SMALL, family=None)
        self.assertNotEqual(code, 0)
        self.assertIn("too big for this PC's RAM", out)
        self.assertIn("--yes installs it anyway", out)
        self.assertIsNone(cfg)
        code, out, cfg = self.install(SMALL, ["--model", "UD-Q3_K_M", "--context", "32768"])
        self.assertEqual(code, 0, out)
        self.assertIn("going on with UD-Q3_K_M, as you chose", out)

    def test_check(self):
        code, out, cfg = self.install(HYBRID, ["--check"], family=None)
        self.assertEqual(code, 0, out)
        lines = [ln.strip() for ln in out.splitlines()]
        listed = [ln.split(":")[0] for ln in lines if ln.startswith(("Qwen3.6-35B-A3B ", "Ornith 1.5 35B-A3B "))]
        self.assertEqual(listed, ["Qwen3.6-35B-A3B UD-Q4_K_M", "Qwen3.6-35B-A3B UD-Q3_K_M",
                                  "Ornith 1.5 35B-A3B APEX-I-QUALITY", "Ornith 1.5 35B-A3B APEX-I-COMPACT"])
        self.assertIn("its speed was measured with FastStrata's options",
                      next(ln for ln in lines if ln.startswith("Qwen3.6-35B-A3B UD-Q4_K_M:")))
        for other in ("Gemma 4", "Tiel Coder", "GLM", "DeepSeek", "Q2_0 ", "IQ3_XXS "):
            self.assertFalse(any(ln.startswith(other) for ln in lines), other)
        self.assertIn("This PC can run FastStrata.", out)
        # the 6 GB laptop runs them, as measured (it was "too small" by the 32 GB card's estimate)
        code, out, cfg = self.install(LAPTOP, ["--check"], family=None)
        self.assertEqual(code, 0, out)
        self.assertIn("Qwen3.6-35B-A3B UD-Q4_K_M: fits this PC, up to 144K context", out)
        code, out, cfg = self.install(SMALL, ["--check"], family=None)
        self.assertEqual(code, 1, out)
        self.assertIn("This PC is too small for FastStrata's presets", out)
        # another family named: its own check, as before, each line saying what FastStrata supports
        code, out, cfg = self.install(HYBRID, ["--check"], family="gemma4")
        self.assertEqual(code, 0, out)
        self.assertIn("Gemma 4 26B-A4B is not supported by FastStrata", out)
        self.assertTrue(all(ln.endswith("not supported by FastStrata") for ln in out.splitlines()
                            if ln.strip().startswith("Gemma 4 26B-A4B ")))


def show():
    """What setup prints and writes for the laptop of docs/OPTIMIZATION_LOG.md (everything mocked, as in the tests)."""
    t = Install("test_the_laptop_gets_its_daily_config")
    t.setUp()
    try:
        code, out, cfg = t.laptop()
    finally:
        t.tearDown()
    print(out)
    print(f"exit {code}; the config's engine options:")
    print(" ".join(x if x.startswith("--") or len(x) < 40 else Path(x).name for x in cfg["args"]))


if __name__ == "__main__":
    if "--show" in sys.argv:
        show()
    else:
        unittest.main()
