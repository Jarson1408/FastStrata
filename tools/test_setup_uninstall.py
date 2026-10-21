"""Tests for setup.py --uninstall (FASTSTRATA UNINSTALL): the gateway's settings folder and the desktop shortcut go to
a bin, OpenCode's config loses the provider `faststrata` alone with a .bak first, what another FastStrata folder owns
is left unless the user says so, and nothing changes while the app runs or when the answer is no.  Every path is in a
temporary folder: the Recycle Bin, the desktop, the user's home and the app's port are stand-ins.

    python -m unittest tools.test_setup_uninstall
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402

OPENCODE = {"$schema": "https://opencode.ai/config.json",
            "model": "faststrata/qwen3.6-35b-a3b",
            "mcp": {"search": {"type": "remote", "url": "https://example.com/mcp", "headers": {"key": "secret"}}},
            "provider": {"faststrata": {"npm": "@ai-sdk/openai-compatible", "name": "FastStrata",
                                        "options": {"baseURL": "http://localhost:18000/v1"},
                                        "models": {"qwen3.6-35b-a3b": {"name": "Qwen3.6 35B-A3B UD-Q4_K_M"}}},
                         "other": {"npm": "@ai-sdk/openai-compatible", "models": {"m": {}}}}}


def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()) as buf:
        result = fn(*a, **k)
    return result, buf.getvalue()


class TheTrash(unittest.TestCase):
    def test_an_item_goes_to_the_home_trash_with_its_old_place(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(Path(tmp) / "data")}):
            item = Path(tmp) / "cfg" / "faststrata-gateway"
            item.mkdir(parents=True)
            (item / "settings.json").write_text("{}", encoding="utf-8")
            where = setup.to_home_trash(item, now="2026-10-21T12:00:00")
            self.assertFalse(item.exists())
            self.assertEqual((where / "settings.json").read_text(encoding="utf-8"), "{}")
            info = (Path(tmp) / "data" / "Trash" / "info" / "faststrata-gateway.trashinfo").read_text(encoding="utf-8")
            self.assertTrue(info.startswith("[Trash Info]\nPath="))
            self.assertIn("faststrata-gateway", info)
            self.assertIn("DeletionDate=2026-10-21T12:00:00", info)

    def test_a_second_item_of_the_same_name_does_not_replace_the_first(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(Path(tmp) / "data")}):
            for text in ("one", "two"):
                f = Path(tmp) / "faststrata.desktop"
                f.write_text(text, encoding="utf-8")
                setup.to_home_trash(f)
            files = Path(tmp) / "data" / "Trash" / "files"
            self.assertEqual((files / "faststrata.desktop").read_text(encoding="utf-8"), "one")
            self.assertEqual((files / "faststrata.2.desktop").read_text(encoding="utf-8"), "two")
            self.assertTrue((Path(tmp) / "data" / "Trash" / "info" / "faststrata.2.desktop.trashinfo").is_file())

    def test_a_path_still_there_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "x"
            f.write_text("x", encoding="utf-8")
            with mock.patch.object(setup, "to_recycle_bin", lambda p: None), \
                    mock.patch.object(setup, "to_home_trash", lambda p: None):
                with self.assertRaises(OSError):
                    setup.trash(f)


class OpenCodesConfig(unittest.TestCase):
    def test_the_provider_and_its_default_model_go_and_the_rest_stays(self):
        new, gone, refs = setup.without_faststrata(OPENCODE)
        self.assertNotIn("faststrata", new["provider"])
        self.assertEqual(new["provider"]["other"], OPENCODE["provider"]["other"])
        self.assertEqual(new["mcp"], OPENCODE["mcp"], "a key of another program's stays as it was")
        self.assertNotIn("model", new)
        self.assertEqual(gone, ["provider.faststrata", "model (faststrata/qwen3.6-35b-a3b)"])
        self.assertEqual(refs, [])
        self.assertIn("faststrata", OPENCODE["provider"], "the config given is not changed")

    def test_an_empty_provider_map_goes_and_other_mentions_are_listed(self):
        data = {"provider": {"faststrata": {}}, "small_model": "other/m",
                "agent": {"build": {"model": "faststrata/qwen3.6-35b-a3b-q3"}}}
        new, gone, refs = setup.without_faststrata(data)
        self.assertNotIn("provider", new)
        self.assertEqual(new["small_model"], "other/m")
        self.assertEqual(refs, ["agent.build.model"])
        self.assertEqual(gone, ["provider.faststrata"])

    def test_comments_are_counted_and_a_url_is_no_comment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "opencode.jsonc"
            p.write_text('{\n  // mine\n  "$schema": "https://opencode.ai/config.json", /* two */\n'
                         '  "provider": {"faststrata": {}},\n}\n', encoding="utf-8")
            data, comments = setup.read_jsonc(p)
            self.assertEqual(comments, 2)
            self.assertEqual(data["$schema"], "https://opencode.ai/config.json")

    def test_only_configs_with_the_provider_are_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / ".config" / "opencode"
            d.mkdir(parents=True)
            (d / "opencode.jsonc").write_text('{"provider": {"other": {}}}', encoding="utf-8")
            (d / "opencode.json").write_text(json.dumps(OPENCODE), encoding="utf-8")
            found = setup.opencode_with_faststrata(Path(tmp))
            self.assertEqual([p.name for p, _, _ in found], ["opencode.json"])

    def test_the_file_before_is_kept_and_a_kept_bak_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "opencode.json"
            p.write_text(json.dumps(OPENCODE), encoding="utf-8")
            (Path(tmp) / "opencode.json.bak").write_text("the user's own", encoding="utf-8")
            bak = setup.write_opencode(p, {"a": 1}, now="20261021-120000")
            self.assertEqual(bak.name, "opencode.json.bak-20261021-120000")
            self.assertEqual(json.loads(bak.read_text(encoding="utf-8")), OPENCODE)
            self.assertEqual((Path(tmp) / "opencode.json.bak").read_text(encoding="utf-8"), "the user's own")
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), {"a": 1})
            self.assertFalse((Path(tmp) / "opencode.json.tmp").exists())


class TheUninstall(unittest.TestCase):
    """uninstall() on a fake PC: home, settings, desktop shortcut, this FastStrata folder and the bin in a temp folder."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "FastStrata"
        (self.root / "gateway").mkdir(parents=True)
        self.home = self.tmp / "home"
        self.oc = self.home / ".config" / "opencode" / "opencode.json"
        self.oc.parent.mkdir(parents=True)
        self.oc.write_text(json.dumps(OPENCODE, indent=2), encoding="utf-8")
        self.settings = self.tmp / "appdata" / "faststrata-gateway" / "settings.json"
        self.settings.parent.mkdir(parents=True)
        self.write_settings(self.root)
        self.link = self.tmp / "Desktop" / "FastStrata.lnk"
        self.link.parent.mkdir()
        self.link.write_text("lnk", encoding="utf-8")
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        models = self.tmp / "PolyStrata-data" / "models"
        models.mkdir(parents=True)
        (models / "q.gguf").write_bytes(b"x" * 1000)
        (self.root / "strata-qwen36-ud-q4_k_m.json").write_text(json.dumps(
            {"exe": "strata-poly.exe", "args": ["--model", str(models / "q.gguf"), "--max-context", "147456"]}),
            encoding="utf-8")
        self.workdir = self.root / "gateway"
        self.running = False
        for name, value in (("ROOT", self.root), ("listening", lambda port: self.running),
                            ("faststrata_shortcuts", lambda: [(self.link, self.workdir)] if self.link.exists() else []),
                            ("trash", self.to_bin), ("load_settings", lambda: {}), ("other_installs", lambda s: [])):
            p = mock.patch.object(setup, name, value)
            p.start()
            self.addCleanup(p.stop)

    def write_settings(self, folder: Path, port: int = 18000):
        self.settings.write_text(json.dumps({"polystrata_dir": str(folder), "port": port}), encoding="utf-8")

    def to_bin(self, path: Path):
        shutil.move(str(path), str(self.bin / path.name))

    def run_it(self, yes=True, answers=()):
        a = argparse.Namespace(yes=yes)
        replies = iter(answers)
        with mock.patch("builtins.input", lambda *_: next(replies)):
            return quiet(setup.uninstall, a, settings=self.settings, home=self.home)

    def test_everything_of_this_folder_goes_and_opencode_keeps_the_rest(self):
        code, text = self.run_it()
        self.assertEqual(code, 0)
        self.assertFalse(self.settings.parent.exists())
        self.assertFalse(self.link.exists())
        self.assertEqual(sorted(p.name for p in self.bin.iterdir()), ["FastStrata.lnk", "faststrata-gateway"])
        data = json.loads(self.oc.read_text(encoding="utf-8"))
        self.assertNotIn("faststrata", data["provider"])
        self.assertIn("other", data["provider"])
        self.assertEqual(data["mcp"], OPENCODE["mcp"])
        self.assertEqual(json.loads((self.oc.parent / "opencode.json.bak").read_text(encoding="utf-8")), OPENCODE)
        self.assertIn(str(self.root), text, "it says which folder is left to delete")
        self.assertIn("the model files, 0.0 GB", text)

    def test_no_changes_nothing(self):
        before = self.oc.read_text(encoding="utf-8")
        code, text = self.run_it(yes=False, answers=["n"])
        self.assertEqual(code, 0)
        self.assertIn("Nothing was changed", text)
        self.assertTrue(self.settings.is_file())
        self.assertTrue(self.link.exists())
        self.assertEqual(self.oc.read_text(encoding="utf-8"), before)
        self.assertEqual(list(self.bin.iterdir()), [])

    def test_enter_is_no(self):
        self.run_it(yes=False, answers=[""])
        self.assertTrue(self.settings.is_file())

    def test_a_running_app_stops_it_before_anything(self):
        self.running = True
        before = self.oc.read_text(encoding="utf-8")
        code, text = self.run_it()
        self.assertEqual(code, 1)
        self.assertIn("is running", text)
        self.assertTrue(self.settings.is_file())
        self.assertTrue(self.link.exists())
        self.assertEqual(self.oc.read_text(encoding="utf-8"), before)

    def test_what_another_faststrata_folder_owns_is_left_unless_the_user_says_so(self):
        other = self.tmp / "FastStrata-2"
        (other / "gateway").mkdir(parents=True)
        self.write_settings(other)
        self.workdir = other / "gateway"
        self.run_it(yes=True)                          # --yes: the recommended answer, no
        self.assertTrue(self.settings.is_file())
        self.assertTrue(self.link.exists())
        self.assertNotIn("faststrata", json.loads(self.oc.read_text(encoding="utf-8"))["provider"])
        self.write_settings(other)
        self.run_it(yes=False, answers=["y", "y", "y"])
        self.assertFalse(self.settings.parent.exists())
        self.assertFalse(self.link.exists())

    def test_settings_of_a_folder_that_is_gone_are_this_ones_to_take(self):
        self.write_settings(self.tmp / "deleted-FastStrata")
        self.run_it(yes=True)
        self.assertFalse(self.settings.parent.exists())

    def test_a_jsonc_with_comments_is_left_with_yes(self):
        self.oc.unlink()
        jsonc = self.oc.with_name("opencode.jsonc")
        text = '{\n  // my notes\n' + json.dumps(OPENCODE, indent=2)[1:]
        jsonc.write_text(text, encoding="utf-8")
        code, out = self.run_it(yes=True)
        self.assertEqual(code, 0)
        self.assertEqual(jsonc.read_text(encoding="utf-8"), text)
        self.assertIn("left as it is", out)
        self.assertFalse(self.settings.parent.exists(), "the rest still goes")

    def test_nothing_to_take_says_so(self):
        shutil.rmtree(self.settings.parent)
        self.link.unlink()
        self.oc.write_text(json.dumps({"provider": {"other": {}}}), encoding="utf-8")
        code, text = self.run_it(yes=False, answers=[])
        self.assertEqual(code, 0)
        self.assertIn("Nothing of FastStrata's is outside this folder", text)


class TheCommandLine(unittest.TestCase):
    def test_uninstall_runs_before_the_data_folder_moves_anything(self):
        def no(*a, **k):
            raise AssertionError("data_folder() ran")
        with mock.patch.object(sys, "argv", ["setup.py", "--uninstall", "--yes"]), \
                mock.patch.object(setup, "data_folder", no), \
                mock.patch.object(setup, "uninstall", lambda a: 7):
            code, text = quiet(setup.main)
        self.assertEqual(code, 7)
        self.assertIn(f"FastStrata {setup.FASTSTRATA_VERSION}", text)

    def test_uninstall_bat_runs_it(self):
        bat = (ROOT / "UNINSTALL.bat").read_text(encoding="utf-8")
        self.assertIn("setup.py --uninstall %*", bat)


if __name__ == "__main__":
    unittest.main()
