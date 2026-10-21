"""Tests for what setup says of FastStrata's presets on a PC (FASTSTRATA UX): the fit from what was measured, the
preset it recommends, the context it offers, and how its questions show their answers. No network, no GPU.

    python -m unittest tools.test_setup_faststrata_fit
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402


class TestTheFitIsWhatWasMeasured(unittest.TestCase):
    def test_the_6_gb_laptop_runs_q4_at_144k_in_8_bits(self):
        """The test PC: an RTX 3060 Laptop 6 GB and 64 GB, its daily config 144K with --kv-q8 (step 10). The estimate
        made for a 32 GB card called it too small for any preset."""
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q4_K_M", 59.9, 6.0), (147456, "q8"))

    def test_every_preset_runs_144k_on_the_6_gb_laptop(self):
        """Step 10 ran all four with --max-context 147456 and --kv-q8 there, a 139K prompt each; the estimate alone
        gave UD-Q3_K_M 128K."""
        for family, model in setup.FASTSTRATA_PRESETS:
            with self.subTest(preset=model):
                self.assertEqual(setup.faststrata_fit(family, model, 59.9, 6.0), (147456, "q8"))

    def test_a_4_gb_card_gets_the_context_measured_on_one(self):
        """A simulated 4 GB card (3,082 MiB free to the engine), --kv-q8: UD-Q3_K_M at 16K to 48K, UD-Q4_K_M and
        APEX-I-COMPACT at 32K; the estimate fits none of them there."""
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q3_K_M", 31.7, 4.0), (49152, "q8"))
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q4_K_M", 31.7, 4.0), (32768, "q8"))
        self.assertEqual(setup.faststrata_fit("ornith", "APEX-I-COMPACT", 31.7, 4.0)[0], 32768)

    def test_an_8_gb_card_gets_the_estimate_less_10_percent(self):
        """No 8 GB card was measured: setup's estimate (about 276K would fit) less 10%, said to be an estimate."""
        self.assertEqual(setup.faststrata_fit_info("qwen36", "UD-Q4_K_M", 31.7, 8.0), (253952, "q8", ""))
        self.assertIn("up to 248K context (estimated)",
                      setup.faststrata_fit_text("qwen36", "UD-Q4_K_M", 31.7, 8.0))

    def test_a_card_that_holds_more_keeps_the_whole_window(self):
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q4_K_M", 64.0, 12.0), (262144, "f16"))

    def test_a_measurement_holds_only_for_the_vram_it_had(self):
        """The 6 GB laptop ran 144K with the whole card free and 0.1 to 0.2 GB left: a 6 GB card with 0.3 GB taken by
        a desktop gets the estimate, not the measurement."""
        self.assertIn("(measured on a 6 GB card)", setup.faststrata_fit_text("qwen36", "UD-Q4_K_M", 59.9, 6.0))
        ctx, _cache, measured = setup.faststrata_fit_info("qwen36", "UD-Q4_K_M", 59.9, 5.7)
        self.assertEqual(measured, "")
        self.assertLess(ctx, 147456)

    def test_within_its_size_the_measurement_wins_over_a_larger_estimate(self):
        est = setup.faststrata_estimate("ornith", "APEX-I-COMPACT", 6.0)
        self.assertGreater(est, 147456)
        self.assertEqual(setup.faststrata_fit_info("ornith", "APEX-I-COMPACT", 59.9, 6.0),
                         (147456, "q8", "a 6 GB card"))

    def test_too_little_ram_fits_nothing(self):
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q4_K_M", 12.0, 6.0), (0, ""))
        self.assertIn("too big for this PC's RAM", setup.faststrata_fit_text("qwen36", "UD-Q4_K_M", 12.0, 6.0))

    def test_a_2_gb_card_fits_nothing(self):
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q3_K_M", 64.0, 2.0), (0, ""))
        self.assertEqual(setup.faststrata_fit_text("qwen36", "UD-Q3_K_M", 64.0, 2.0), "too big for this card")


class TestTheDesktop(unittest.TestCase):
    def test_a_screen_on_the_nvidia_card_is_found(self):
        """The test laptop with a monitor in its HDMI port: the built-in screen on the Radeon, the other on the card."""
        both = (["AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 3060 Laptop GPU"],
                ["AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 3060 Laptop GPU"])
        with mock.patch.object(setup, "display_adapters", return_value=both):
            on, why = setup.desktop_on_card({"index": 0})
        self.assertTrue(on)
        self.assertIn("the others on AMD Radeon(TM) Graphics", why)

    def test_a_desktop_on_the_processors_graphics_leaves_the_card_free(self):
        only = (["AMD Radeon(TM) Graphics"], ["AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 3060 Laptop GPU"])
        with mock.patch.object(setup, "display_adapters", return_value=only):
            self.assertEqual(setup.desktop_on_card({"index": 0}),
                             (False, "the desktop is on AMD Radeon(TM) Graphics"))

    def test_nothing_found_asks_nvidia_smi(self):
        with mock.patch.object(setup, "display_adapters", return_value=None), \
             mock.patch.object(setup, "out", return_value="Enabled\n"):
            self.assertTrue(setup.desktop_on_card({"index": 0})[0])
        with mock.patch.object(setup, "display_adapters", return_value=None), \
             mock.patch.object(setup, "out", return_value=""):
            self.assertEqual(setup.desktop_on_card({"index": 0}), (None, ""))

    def test_the_safe_context_is_10_percent_below_the_longest(self):
        self.assertEqual(setup.faststrata_contexts(147456)[1], 147456)
        self.assertEqual(setup.faststrata_contexts(147456, desktop=True)[1], 131072)   # 129.6K, down to 8K steps
        self.assertEqual(setup.faststrata_contexts(32768, desktop=True)[1], 24576)
        options, safe = setup.faststrata_contexts(253952, desktop=True)                # an 8 GB card: 144K stays
        self.assertEqual(safe, 147456)
        self.assertIn(253952, options)


class TestTheRecommendation(unittest.TestCase):
    def test_q4_where_it_runs_a_long_context(self):
        self.assertEqual(setup.faststrata_pick(59.9, 6.0), ("qwen36", "UD-Q4_K_M"))
        self.assertEqual(setup.faststrata_pick(31.7, 8.0), ("qwen36", "UD-Q4_K_M"))

    def test_none_when_nothing_fits(self):
        self.assertIsNone(setup.faststrata_pick(12.0, 6.0))

    def test_within_a_family(self):
        self.assertEqual(setup.faststrata_pick(59.9, 6.0, ["ornith"]), ("ornith", "APEX-I-QUALITY"))

    def test_every_preset_has_its_menu_line(self):
        presets = {(f, m) for f in setup.FASTSTRATA_FAMILIES for m in setup.FAMILIES[f]["models"]}
        self.assertEqual(presets, set(setup.FASTSTRATA_PRESETS))
        for key, pr in setup.FASTSTRATA_PRESETS.items():
            with self.subTest(preset=key):
                self.assertTrue(pr["name"] and pr["quality"] and pr["tok_s_37k"] > 0)


class TestQuestionsSayWhatEnterGives(unittest.TestCase):
    def test_yes_or_no(self):
        self.assertEqual(setup.ask_hint(["y", "n"], "y"), "[Y/n]")
        self.assertEqual(setup.ask_hint(["y", "n"], "n"), "[y/N]")

    def test_a_number(self):
        self.assertEqual(setup.ask_hint(["1", "2", "3"], "2"), "[2]")


if __name__ == "__main__":
    unittest.main()
