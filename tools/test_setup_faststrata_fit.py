"""Tests for what setup says of FastStrata's presets on a PC (FASTSTRATA UX): the fit from what was measured, the
preset it recommends, the context it offers, and how its questions show their answers. No network, no GPU.

    python -m unittest tools.test_setup_faststrata_fit
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402


class TestTheFitIsWhatWasMeasured(unittest.TestCase):
    def test_the_6_gb_laptop_runs_q4_at_144k_in_8_bits(self):
        """The test PC: an RTX 3060 Laptop 6 GB and 64 GB, its daily config 144K with --kv-q8 (step 10). The estimate
        made for a 32 GB card called it too small for any preset."""
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q4_K_M", 59.9, 6.0), (147456, "q8"))

    def test_a_4_gb_card_gets_the_context_measured_on_one(self):
        """A simulated 4 GB card (3,082 MiB free to the engine), --kv-q8: UD-Q3_K_M at 16K to 48K, UD-Q4_K_M and
        APEX-I-COMPACT at 32K; the estimate fits none of them there."""
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q3_K_M", 31.7, 4.0), (49152, "q8"))
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q4_K_M", 31.7, 4.0), (32768, "q8"))
        self.assertEqual(setup.faststrata_fit("ornith", "APEX-I-COMPACT", 31.7, 4.0)[0], 32768)

    def test_an_8_gb_card_takes_the_whole_window(self):
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q4_K_M", 31.7, 8.0)[0], 262144)

    def test_too_little_ram_fits_nothing(self):
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q4_K_M", 12.0, 6.0), (0, ""))
        self.assertIn("too big for this PC's RAM", setup.faststrata_fit_text("qwen36", "UD-Q4_K_M", 12.0, 6.0))

    def test_a_2_gb_card_fits_nothing(self):
        self.assertEqual(setup.faststrata_fit("qwen36", "UD-Q3_K_M", 64.0, 2.0), (0, ""))
        self.assertEqual(setup.faststrata_fit_text("qwen36", "UD-Q3_K_M", 64.0, 2.0), "too big for this card")


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
