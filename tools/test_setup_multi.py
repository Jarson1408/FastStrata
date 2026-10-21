"""Tests for step 2's two ways and several models at once (FASTSTRATA MULTI): the recommended preset installed without
more questions, or one or several chosen, each with its own context; the engine built once, each model downloaded,
prepared and written in turn, a model that fails left while the others are kept; the gateway once for all of them, the
first started first; --model more than once, --models and --context MODEL=N; and a copy set up like an earlier install
(an update unzipped beside it), which now takes every FastStrata model of it and offers the gateway, or takes over the
one it had.  setup.main() on fake machines, every outside effect mocked: no GPU, no downloads, nothing written outside
a temporary folder.

    python -m unittest tools.test_setup_multi
    python tools/test_setup_multi.py --show      (step 2 of both ways on the 6 GB laptop, as the screen shows it)
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# the laptop FastStrata was measured on, as setup sees it: nvidia-smi's 6,144 MiB, the 59.86 GiB Windows gives of 64 GB
LAPTOP = {"cards": [("NVIDIA GeForce RTX 3060 Laptop GPU", 6144, "86")], "ram": 59.86}
# an 8 GB card and 32 GB: no measurement, setup's estimate less 10%
HYBRID = {"cards": [("NVIDIA GeForce RTX 4060 Laptop GPU", 8188, "89")], "ram": 31.7}
# the laptop with 12 GB of RAM: no preset fits
SMALL = dict(LAPTOP, ram=11.8)
ENGINE = (b"usage: strata-poly\n  --threads N\n  --cpu-multi\n  --suffix-drafts MIB\n  --kv-q8\n  --kv-q4\n"
          b"  --kv-q4-keys q8|q4\n  --attn-split\n  --graph-experts\n  --prompt-batch N\n  --elastic-prefill N\n"
          b"  --conversation-cache-mib N\n  --conversation-cache-slots N\n  --tail-skip N\n")
GATEWAY = ("Install the FastStrata Gateway (the app between OpenCode and the model: tool calls, model swaps, project "
           "memory)")
Q4, Q3, QUALITY, COMPACT = ("strata-qwen36-ud-q4_k_m.json", "strata-qwen36-ud-q3_k_m.json",
                            "strata-ornith-apex-i-quality.json", "strata-ornith-apex-i-compact.json")


def arg(cfg: dict, flag: str):
    a = cfg["args"]
    return a[a.index(flag) + 1] if flag in a else None


class Machine:
    """setup.main() on a fake machine in a temporary FastStrata folder; run() again in the same folder is setup run
    again there.  `earlier`: another FastStrata folder (an install from before), for previous_config()."""

    def __init__(self, machine=LAPTOP):
        self.machine = machine
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name) / "FastStrata"
        (self.t / "data").mkdir(parents=True)
        self.earlier = Path(self.tmp.name) / "FastStrata-old"
        self.earlier.mkdir()
        self.prev = None                               # the earlier install's config previous_config() finds
        self.settings = Path(self.tmp.name) / "gateway-settings.json"
        self.broken = ()                               # file names whose download fails

    def close(self):
        self.tmp.cleanup()

    def config(self, name):
        p = self.t / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    def run(self, argv=(), answers=None, start=False):
        """answers: None = --yes; else {words of a question: its answer} (Enter for the others).  start: no
        --no-start (adoption only happens without it).  -> (exit code, printed text); self.asked, self.built,
        self.started, self.app."""
        self.asked, self.built = [], []
        self.started, self.app = mock.Mock(return_value=0), mock.Mock(return_value=0)

        def fake_input(prompt=""):
            if answers is None:
                raise AssertionError(f"asked {prompt!r} with --yes")
            self.asked.append(prompt)
            answer = next((v for k, v in answers.items() if k in prompt), "")
            if SHOW:
                print(f"{prompt}{answer}")
            return answer

        found = [{"index": i, "name": n, "vram_gb": mib / 1024, "arch": arch, "driver": "617.42"}
                 for i, (n, mib, arch) in enumerate(self.machine["cards"])]

        def fake_build(gpu, *a, **k):
            self.built.append(gpu.get("archs"))
            eng = self.t / setup.ENGINE_POLY_DIR
            eng.mkdir(exist_ok=True)
            (eng / setup.POLY_EXE).write_bytes(ENGINE)
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "cuda_dirs": ["/cuda/bin"]}))
            return eng

        def fake_download(url, dst, what=None):
            if dst.name in self.broken:
                setup.fail(f"the download of {dst.name} stopped (connection reset)", "run setup again")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_gateway(dest=None):
            gw = self.t / "gateway"
            (gw / "gateway").mkdir(parents=True, exist_ok=True)
            (gw / "gateway" / "app.py").write_text("", encoding="utf-8")
            (gw / setup.GATEWAY_MARK).write_text(setup.GATEWAY_COMMIT, encoding="utf-8")
            return gw

        real_out = setup.out
        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "WIN", True),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "OLD_GPUS", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "load_settings", lambda: {}),
            mock.patch.object(setup, "save_settings", lambda s: None),
            mock.patch.object(setup, "previous_config", lambda *a: self.prev),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "amd_gpus", lambda *a, **k: []),
            mock.patch.object(setup, "pcie_link", lambda i: None),
            mock.patch.object(setup, "gpu_drives_display", lambda g: False),
            mock.patch.object(setup, "ram_gb", lambda: self.machine["ram"]),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, False)),
            mock.patch.object(setup, "cpu_cores", lambda: None),
            mock.patch.object(setup, "cpu_threads", lambda: (8, 16)),
            mock.patch.object(setup, "page_file_gb", lambda: 64.0),
            mock.patch.object(setup, "free_gb", lambda p: self.machine.get("free_gb", 500.0)),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", mock.Mock(side_effect=AssertionError("the Qwen engine"))),
            mock.patch.object(setup, "get_poly_prebuilt", lambda gpu: False),
            mock.patch.object(setup, "build_engine_poly", fake_build),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "whole_shard", lambda s: False),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda *a: None),
            mock.patch.object(setup, "run", lambda cmd, *a, **k: None),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"run-{tag.lower()}.bat"),
            mock.patch.object(setup, "start", self.started),
            # the gateway: its folder in this FastStrata folder, its settings in the temporary folder, OpenCode found
            mock.patch.object(setup, "GATEWAY_DIR", self.t / "gateway"),
            mock.patch.object(setup, "fetch_gateway", fake_gateway),
            mock.patch.object(setup, "gateway_env", lambda gw: Path("py")),
            mock.patch.object(setup, "gateway_settings_path", lambda: self.settings),
            mock.patch.object(setup, "find_opencode", lambda: "opencode"),
            mock.patch.object(setup, "out", lambda cmd: "1.0" if cmd[:1] == ["opencode"] else real_out(cmd)),
            mock.patch.object(setup, "connect_opencode", lambda py, gw, yes: True),
            mock.patch.object(setup, "make_shortcut", lambda gw: None),
            mock.patch.object(setup, "start_gateway_app", self.app),
            mock.patch.object(sys, "argv", ["setup.py", *([] if start else ["--no-start"]),
                                            *(["--yes"] if answers is None else []),
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", fake_input),
        ]
        buf = io.StringIO()
        code = None
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            with contextlib.redirect_stdout(sys.stdout if SHOW else buf):
                try:
                    code = setup.main()
                except SystemExit as e:
                    code = e.code
        self.questions = [q.split("?")[0].strip() for q in self.asked]
        return code, buf.getvalue()

    def gateway_settings(self):
        return json.loads(self.settings.read_text(encoding="utf-8")) if self.settings.is_file() else None

    def write_earlier(self, name, context, gateway=False):
        """A run config of the earlier install (and its gateway/ folder), as setup wrote it there."""
        p = self.earlier / name
        p.write_text(json.dumps({"exe": "strata-poly.exe", "args": ["--model", "m.gguf", "--max-context", str(context),
                                                                    "--conversation-cache-mib", "12288"],
                                 "port": 8080}), encoding="utf-8")
        if gateway:
            (self.earlier / "gateway").mkdir(exist_ok=True)
            (self.earlier / "gateway" / setup.GATEWAY_MARK).write_text(setup.GATEWAY_COMMIT, encoding="utf-8")
        return p


SHOW = False


class Case(unittest.TestCase):
    machine = LAPTOP

    def setUp(self):
        self.m = Machine(self.machine)
        self.addCleanup(self.m.close)


class TheRecommendedWay(Case):
    def test_enter_installs_the_recommended_preset_and_its_context(self):
        """The 6 GB laptop: Q4 with 144K, as measured there; no question about the model or the context."""
        code, out = self.m.run(answers={})
        self.assertEqual(code, 0, out)
        self.assertIn("Recommended for this PC (NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB of VRAM, 60 GB of RAM):", out)
        self.assertIn("Qwen3.6-35B-A3B Q4 (UD-Q4_K_M) with a 144K context, download 23 GB", out)
        self.assertIn("fits this PC, up to 144K context (measured on a 6 GB card); 66.9 tok/s; the most exact", out)
        self.assertEqual(self.m.questions, ["Which way", "Do you want images", "Keep conversations in RAM", GATEWAY])
        self.assertNotIn("Which models?", out)
        self.assertEqual(arg(self.m.config(Q4), "--max-context"), "147456")
        self.assertEqual([p.name for p in self.m.t.glob("strata-*.json")], [Q4])

    def test_yes_takes_the_first_way(self):
        code, out = self.m.run()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.m.asked, [])
        self.assertEqual(arg(self.m.config(Q4), "--max-context"), "147456")
        self.assertNotIn("Model 1 of", out)
        self.assertIn("Step 5: downloading Qwen3.6-35B-A3B UD-Q4_K_M ===", out)    # one model: the steps as before
        self.assertIn("Step 7: writing the start script ===", out)

    def test_an_8_gb_card_gets_144k_of_its_estimated_248k(self):
        self.m.machine = HYBRID
        code, out = self.m.run()
        self.assertEqual(code, 0, out)
        self.assertIn("with a 144K context", out)
        self.assertEqual(arg(self.m.config(Q4), "--max-context"), "147456")

    def test_a_context_given_is_the_one_shown_and_written(self):
        code, out = self.m.run(["--context", "64K"])
        self.assertEqual(code, 0, out)
        self.assertIn("Qwen3.6-35B-A3B Q4 (UD-Q4_K_M) with a 64K context", out)
        self.assertEqual(arg(self.m.config(Q4), "--max-context"), "65536")

    def test_a_pc_too_small_still_stops_with_yes(self):
        self.m.machine = SMALL
        code, out = self.m.run()
        self.assertNotEqual(code, 0)
        self.assertIn("too big for this PC's RAM", out)
        self.assertIsNone(self.m.config(Q3))


class ChoosingSeveral(Case):
    def test_two_models_each_with_its_own_context(self):
        code, out = self.m.run(answers={"Which way": "2", "Which ones": "1,3",
                                        "Which context for Qwen3.6-35B-A3B Q4": "3"})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.m.questions, ["Which way", "Which ones", "Do you want images",
                                            "Which context for Qwen3.6-35B-A3B Q4", "Keep conversations in RAM",
                                            "Which context for Ornith 1.5 35B-A3B Quality", GATEWAY])
        self.assertEqual(arg(self.m.config(Q4), "--max-context"), "65536")
        self.assertEqual(arg(self.m.config(QUALITY), "--max-context"), "147456")
        # images and the conversation cache: one answer for both
        self.assertEqual(out.count("One answer for every model chosen"), 2)
        self.assertEqual(arg(self.m.config(QUALITY), "--conversation-cache-mib"), "12288")
        self.assertNotIn("vision", self.m.config(QUALITY))
        # RAM and VRAM judged for each on its own: one runs at a time
        self.assertIn("Each on its own: one runs at a time", out)
        # the downloads added up before any starts, the engine built once
        self.assertIn("Downloads: Qwen3.6-35B-A3B Q4 23 GB + Ornith 1.5 35B-A3B Quality 24 GB = 46 GB", out)
        self.assertIn("Disk: ~48 GB needed in", out)
        self.assertLess(out.index("Downloads:"), out.index("Step 3"))
        self.assertEqual(self.m.built, [["86"]])
        self.assertIn("Step 5: downloading Qwen3.6-35B-A3B UD-Q4_K_M (1 of 2) ===", out)
        self.assertIn("Step 5: downloading Ornith 1.5 35B-A3B APEX-I-QUALITY (2 of 2) ===", out)
        self.assertIn("Step 7: writing Ornith 1.5 35B-A3B APEX-I-QUALITY's start script ===", out)
        self.assertIn("Models:           Qwen3.6-35B-A3B UD-Q4_K_M (64K), Ornith 1.5 35B-A3B APEX-I-QUALITY (144K)",
                      out)
        self.assertIn("run-qwen36-ud-q4_k_m.bat, run-ornith-apex-i-quality.bat", out)

    def test_enter_at_which_ones_is_the_recommended_one(self):
        code, out = self.m.run(answers={"Which way": "2"})
        self.assertEqual(code, 0, out)
        self.assertIn("Which models?", out)
        self.assertEqual([p.name for p in self.m.t.glob("strata-*.json")], [Q4])
        self.assertIn("Which context", self.m.questions)        # asked: this way chooses the context too

    def test_a_wrong_answer_is_asked_again(self):
        # in order, whatever the question: 2, two wrong answers, then Q3 and Compact; Enter for the rest
        code, out = self.m.run(answers=_Seq(iter(["2", "1,7", "x", "2 4"])))
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("please answer with numbers from 1 to 4"), 2)
        self.assertTrue(self.m.config(Q3) and self.m.config(COMPACT))

    def test_the_recommended_one_is_started_first_by_the_gateway(self):
        """Typed 3,1: Q4, the recommended one, is set up first and is the gateway's first model; Ornith follows."""
        code, out = self.m.run(answers={"Which way": "2", "Which ones": "3,1"})
        self.assertEqual(code, 0, out)
        self.assertLess(out.index("Model 1 of 2: Qwen3.6-35B-A3B Q4"), out.index("Model 2 of 2: Ornith"))
        s = self.m.gateway_settings()
        self.assertEqual([m["id"] for m in s["models"]], ["qwen3.6-35b-a3b", "ornith-1.5-35b-a3b"])
        self.assertEqual(s["polystrata_config"], Q4)
        self.assertEqual([m["n_ctx"] for m in s["models"]], [147456, 147456])

    def test_without_the_recommended_one_the_first_typed_is_first(self):
        code, out = self.m.run(answers={"Which way": "2", "Which ones": "4,2"})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.m.gateway_settings()["polystrata_config"], COMPACT)

    def test_the_gateway_and_the_start_come_once_for_all_of_them(self):
        code, out = self.m.run(answers={"Which way": "2", "Which ones": "1,2"}, start=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("Step G:"), 1)
        self.assertEqual(self.m.questions.count(GATEWAY), 1)
        self.m.app.assert_called_once()                         # "Start FastStrata now, in the Gateway app?": Enter
        self.m.started.assert_not_called()

    def test_no_gateway_starts_the_first_model(self):
        code, out = self.m.run(answers={"Which way": "2", "Which ones": "2,1", "Install the FastStrata Gateway": "n"},
                               start=True)
        self.assertEqual(code, 0, out)
        self.m.started.assert_called_once()
        self.assertEqual(self.m.started.call_args[0][0].name, Q4)

    def test_too_little_disk_for_all_of_them_stops_before_any_download(self):
        self.m.machine = dict(LAPTOP, free_gb=40.0)             # one fits, both do not
        code, out = self.m.run(answers={"Which way": "2", "Which ones": "1,3"})
        self.assertEqual(code, 1, out)
        self.assertIn("not enough free disk space", out)
        self.assertIn("need ~48 GB for the 2 models", out)
        self.assertNotIn("Step 3", out)
        self.assertEqual(list(self.m.t.glob("strata-*.json")), [])


class AModelThatFails(Case):
    def test_the_others_are_kept_and_the_gateway_gets_them(self):
        self.m.broken = ("Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf",)
        code, out = self.m.run(answers={"Which way": "2", "Which ones": "3,1"}, start=True)
        self.assertEqual(code, 1, out)
        self.assertIn("[X]  the download of Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf stopped", out)
        self.assertNotIn("Setup stopped", out)
        self.assertIn("Ornith 1.5 35B-A3B APEX-I-QUALITY was not installed (the reason is above). What is done is "
                      "kept: START-HERE.bat --setup --model APEX-I-QUALITY adds it", out)
        self.assertIsNotNone(self.m.config(Q4))
        self.assertIsNone(self.m.config(QUALITY))
        self.assertEqual([m["id"] for m in self.m.gateway_settings()["models"]], ["qwen3.6-35b-a3b"])
        self.m.started.assert_not_called()
        self.m.app.assert_not_called()
        self.assertFalse(setup.SOFT_FAIL)

        # run again as it says: the model is added, the gateway serves both
        self.m.broken = ()
        code, out = self.m.run(["--setup", "--model", "APEX-I-QUALITY"], answers={})
        self.assertEqual(code, 0, out)
        self.assertNotIn("Which way", self.m.questions)
        self.assertIsNotNone(self.m.config(QUALITY))
        self.assertIn("The FastStrata Gateway is installed here", out)
        self.assertEqual(sorted(m["id"] for m in self.m.gateway_settings()["models"]),
                         ["ornith-1.5-35b-a3b", "qwen3.6-35b-a3b"])

    def test_all_of_them_failing_stops_as_before(self):
        self.m.broken = ("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf")
        code, out = self.m.run(answers={"Which way": "2", "Which ones": "1,4"})
        self.assertEqual(code, 1, out)
        self.assertIn("Setup stopped. Fix the items above and run it again", out)
        self.assertNotIn("Step G", out)

    def test_one_model_stops_setup_as_before(self):
        self.m.broken = ("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",)
        code, out = self.m.run()
        self.assertEqual(code, 1, out)
        self.assertIn("Setup stopped. Fix the item above and run it again - everything already done is kept and "
                      "skipped.", out)


class TheOptions(Case):
    def test_models_and_a_context_for_each(self):
        code, out = self.m.run(["--models", "UD-Q4_K_M,apex-i-quality", "--context", "65536",
                                "--context", "APEX-I-QUALITY=147456"])
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(self.m.config(Q4), "--max-context"), "65536")
        self.assertEqual(arg(self.m.config(QUALITY), "--max-context"), "147456")
        self.assertIn("models: Qwen3.6-35B-A3B Q4 (UD-Q4_K_M), Ornith 1.5 35B-A3B Quality (APEX-I-QUALITY)", out)
        self.assertNotIn("Which models?", out)

    def test_model_given_twice_with_yes_takes_each_ones_recommended_context(self):
        code, out = self.m.run(["--model", "UD-Q3_K_M", "--model", "APEX-I-COMPACT"])
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(self.m.config(Q3), "--max-context"), "147456")
        self.assertEqual(arg(self.m.config(COMPACT), "--max-context"), "147456")

    def test_models_without_yes_ask_each_ones_context(self):
        code, out = self.m.run(["--models", "UD-Q4_K_M,APEX-I-COMPACT", "--context", "APEX-I-COMPACT=32K"],
                               answers={})
        self.assertEqual(code, 0, out)
        self.assertEqual([q for q in self.m.questions if q.startswith("Which")],
                         ["Which context for Qwen3.6-35B-A3B Q4"])
        self.assertEqual(arg(self.m.config(COMPACT), "--max-context"), "32768")

    def test_the_contexts_read(self):
        known = ["UD-Q4_K_M", "APEX-I-QUALITY"]
        self.assertEqual(setup.parse_contexts(None, known), (None, {}))
        self.assertEqual(setup.parse_contexts(["147456"], known), (147456, {}))
        self.assertEqual(setup.parse_contexts(["144K"], known), (147456, {}))
        self.assertEqual(setup.parse_contexts(["ud-q4_k_m=147456,APEX-I-QUALITY=64k"], known),
                         (None, {"UD-Q4_K_M": 147456, "APEX-I-QUALITY": 65536}))
        for bad in (["lots"], ["UD-Q9=1000"], ["UD-Q4_K_M=big"]):
            with self.subTest(bad), self.assertRaises(ValueError):
                setup.parse_contexts(bad, known)

    def test_wrong_options_are_refused_before_anything_runs(self):
        for argv, said in ((["--models", "UD-Q4_K_M,Q2_0"], "several models at once: FastStrata's presets"),
                           (["--models", "UD-Q4_K_M,NOPE"], "--models: no model NOPE"),
                           (["--model", "UD-Q4_K_M", "--context", "UD-Q3_K_M=65536"],
                            "UD-Q3_K_M is not among the models named"),
                           (["--family", "qwen36", "--models", "UD-Q4_K_M,APEX-I-QUALITY"], "of --family qwen36"),
                           (["--context", "lots"], "a number of tokens")):
            with self.subTest(argv), contextlib.redirect_stderr(io.StringIO()) as err:
                code, _out = self.m.run(argv)
            self.assertEqual(code, 2)
            self.assertIn(said, err.getvalue())

    def test_one_model_and_one_context_are_as_before(self):
        code, out = self.m.run(["--model", "UD-Q4_K_M", "--context", "147456"])
        self.assertEqual(code, 0, out)
        self.assertIn("[ok] model: Qwen3.6-35B-A3B", out)
        self.assertNotIn("Recommended for this PC", out)
        self.assertEqual(arg(self.m.config(Q4), "--max-context"), "147456")

    def test_help_names_them(self):
        with mock.patch.object(sys, "argv", ["setup.py", "--help"]), \
                contextlib.redirect_stdout(io.StringIO()) as buf, self.assertRaises(SystemExit):
            setup.main()
        text = " ".join(buf.getvalue().split())
        self.assertIn("--models A,B", text)
        self.assertIn("--context N|MODEL=N", text)
        self.assertIn("given more than once, several of them", text)


class SetupRunAgain(Case):
    def test_the_recommended_one_installed_skips_the_first_question(self):
        code, out = self.m.run()
        self.assertEqual(code, 0, out)
        code, out = self.m.run(["--setup"], answers={"Which ones": "3"})
        self.assertEqual(code, 0, out)
        self.assertNotIn("Which way", self.m.questions)
        self.assertIn("The recommended one, Qwen3.6-35B-A3B Q4 (UD-Q4_K_M), is installed here.", out)
        self.assertIn("1) Qwen3.6-35B-A3B Q4 (UD-Q4_K_M), installed here (144K)", out)
        self.assertIsNotNone(self.m.config(QUALITY))


class AnUpdateBesideAnEarlierInstall(Case):
    """An update unzipped beside an earlier install (no model config of its own, setup run without options) is set up
    like it: before, the gateway was never offered there (a.yes was set for the models' questions)."""

    def test_the_gateway_is_offered(self):
        self.m.prev = self.m.write_earlier(Q4, 147456)
        code, out = self.m.run(answers={"Install the FastStrata Gateway": "y"}, start=True)
        self.assertEqual(code, 0, out)
        self.assertIn("Found your earlier install in", out)
        self.assertEqual(self.m.questions, [GATEWAY])
        self.assertEqual([m["id"] for m in self.m.gateway_settings()["models"]], ["qwen3.6-35b-a3b"])
        self.m.app.assert_called_once()

    def test_declined_it_starts_the_model(self):
        self.m.prev = self.m.write_earlier(Q4, 147456)
        code, out = self.m.run(answers={"Install the FastStrata Gateway": "n"}, start=True)
        self.assertEqual(code, 0, out)
        self.assertIn("skipped; later:", out)
        self.m.started.assert_called_once()
        self.assertIsNone(self.m.gateway_settings())

    def test_an_earlier_gateway_is_taken_over_without_asking(self):
        self.m.prev = self.m.write_earlier(Q4, 147456, gateway=True)
        code, out = self.m.run(answers={}, start=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.m.asked, [])
        self.assertIn(f"Your earlier install ({self.m.earlier / 'gateway'}) has the FastStrata Gateway: this copy gets "
                      "it too", out)
        s = self.m.gateway_settings()
        self.assertEqual(Path(s["polystrata_dir"]), self.m.t)
        self.m.app.assert_called_once()

    def test_the_gateway_settings_serving_it_count_as_its_gateway(self):
        self.m.prev = self.m.write_earlier(Q4, 147456)
        self.m.settings.write_text(json.dumps({"polystrata_dir": str(self.m.earlier), "models": []}), encoding="utf-8")
        code, out = self.m.run(answers={}, start=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.m.asked, [])
        self.assertIn("has the FastStrata Gateway: this copy gets it too", out)

    def test_every_faststrata_model_of_it_comes_along(self):
        self.m.write_earlier(COMPACT, 65536)
        self.m.prev = self.m.write_earlier(Q4, 147456, gateway=True)     # the one used last
        code, out = self.m.run(answers={}, start=True)
        self.assertEqual(code, 0, out)
        self.assertIn("(qwen36-ud-q4_k_m, ornith-apex-i-compact)", out)
        self.assertEqual(arg(self.m.config(Q4), "--max-context"), "147456")
        self.assertEqual(arg(self.m.config(COMPACT), "--max-context"), "65536")
        self.assertIn("Downloads: Qwen3.6-35B-A3B Q4 23 GB + Ornith 1.5 35B-A3B Compact 17 GB", out)
        s = self.m.gateway_settings()
        self.assertEqual(s["polystrata_config"], Q4)
        self.assertEqual([m["n_ctx"] for m in s["models"]], [147456, 65536])
        self.assertEqual(self.m.asked, [])

    def test_with_yes_it_is_still_left_to_gateway_on(self):
        """An AI's or a script's --yes: the gateway only when named, as for any install."""
        self.m.prev = self.m.write_earlier(Q4, 147456)
        code, out = self.m.run(["--yes"], answers={}, start=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.m.asked, [])
        self.assertIn("FastStrata Gateway for OpenCode: not set up (--yes)", out)
        self.m.started.assert_called_once()


class TheQuestionOfSeveralNumbers(unittest.TestCase):
    def ask(self, *typed, default=1, count=4):
        typed = list(typed)
        with mock.patch("builtins.input", lambda prompt="": typed.pop(0)), \
                contextlib.redirect_stdout(io.StringIO()) as buf:
            return setup.ask_numbers("Which ones?", count, default, False), buf.getvalue()

    def test_one_several_and_enter(self):
        self.assertEqual(self.ask("3")[0], [3])
        self.assertEqual(self.ask("1,3")[0], [1, 3])
        self.assertEqual(self.ask(" 4 , 2;4 ")[0], [4, 2])
        self.assertEqual(self.ask("", default=2)[0], [2])

    def test_a_number_outside_is_asked_again(self):
        got, said = self.ask("5", "0,1", "2")
        self.assertEqual(got, [2])
        self.assertEqual(said.count("please answer with numbers from 1 to 4"), 2)

    def test_yes_gives_the_default(self):
        self.assertEqual(setup.ask_numbers("Which ones?", 4, 3, True), [3])


class _Seq(dict):
    """answers in order, whatever the question (Machine.run reads a dict: every question matches the empty key)."""

    def __init__(self, it):
        super().__init__()
        self.it = it

    def items(self):
        return [("", next(self.it, ""))]


def show():
    """Step 2 of both ways on the 6 GB laptop, as the screen shows it (the answers typed after each question)."""
    global SHOW
    SHOW = True
    for title, answers in (("The recommended one (Enter)", {}),
                           ("Choosing several: 2, then 1,3, Q4 at 64K, Ornith Quality at Enter (144K)",
                            {"Which way": "2", "Which ones": "1,3", "Which context for Qwen3.6-35B-A3B Q4": "3"})):
        m = Machine(LAPTOP)
        try:
            print(f"\n######## {title}\n")
            m.run(answers=answers)
        finally:
            m.close()


if __name__ == "__main__":
    if "--show" in sys.argv:
        show()
    else:
        unittest.main()
