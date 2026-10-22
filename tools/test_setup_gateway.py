"""Tests for setup.py's FastStrata Gateway step (FASTSTRATA GATEWAY): the pinned gateway's check and unpacking, the
models and settings it gets, OpenCode found or not, its config written through the gateway, and the desktop shortcut.
No network, no GPU, nothing written outside temporary folders: the download, pip, PowerShell and the gateway's export
are stand-ins.

    python -m unittest tools.test_setup_gateway
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402

FILES = [("gateway/app.py", b"print('app')\n"), ("gateway/__init__.py", b""), ("README.md", b"# FastStrata Gateway\n"),
         ("start_gui.bat", b"@echo off\r\n")]


def gateway_zip(path: Path, files=FILES, top="FastStrata-Gateway-abc123") -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(top + "/", "")
        for rel, data in files:
            z.writestr(f"{top}/{rel}", data)
    return path


def config(folder: Path, tag: str, context: int) -> Path:
    p = folder / f"strata-{tag}.json"
    p.write_text(json.dumps({"exe": "strata-poly.exe", "args": ["--model", "m.gguf", "--max-context", str(context)]}),
                 encoding="utf-8")
    return p


def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()) as buf:
        result = fn(*a, **k)
    return result, buf.getvalue()


class TestThePinnedGatewayIsChecked(unittest.TestCase):
    def test_the_hash_is_of_the_files_not_their_order(self):
        self.assertEqual(setup.gateway_files_sha256(FILES), setup.gateway_files_sha256(list(reversed(FILES))))
        changed = [(r, d + b" ") if r == "README.md" else (r, d) for r, d in FILES]
        self.assertNotEqual(setup.gateway_files_sha256(FILES), setup.gateway_files_sha256(changed))

    def test_the_pinned_files_are_unpacked_and_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "gateway"
            (dest / ".venv").mkdir(parents=True)
            (dest / ".venv" / "keep.txt").write_text("env", encoding="utf-8")
            (dest / "old.py").write_text("old", encoding="utf-8")
            archive = gateway_zip(Path(tmp) / "gw.zip")
            setup.unpack_gateway(archive, dest, expect=setup.gateway_files_sha256(FILES))
            self.assertEqual((dest / "gateway" / "app.py").read_bytes(), b"print('app')\n")
            self.assertEqual((dest / "start_gui.bat").read_bytes(), b"@echo off\r\n")
            self.assertFalse((dest / "old.py").exists(), "an earlier gateway's files go")
            self.assertTrue((dest / ".venv" / "keep.txt").exists(), "its Python environment stays")
            self.assertEqual((dest / setup.GATEWAY_MARK).read_text(encoding="utf-8").strip(), setup.GATEWAY_COMMIT)

    def test_other_files_are_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "gateway"
            archive = gateway_zip(Path(tmp) / "gw.zip")
            with self.assertRaises(SystemExit):
                quiet(setup.unpack_gateway, archive, dest, expect="0" * 64)
            self.assertFalse(dest.exists())
            self.assertFalse(archive.exists(), "a refused archive is not reused by the next run")

    def test_a_broken_download_is_said_and_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "gw.zip"
            archive.write_bytes(b"<html>rate limited</html>")
            with self.assertRaises(SystemExit):
                _, said = quiet(setup.unpack_gateway, archive, Path(tmp) / "gateway")
            self.assertFalse(archive.exists())

    def test_an_installed_pinned_gateway_is_not_downloaded_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "gateway").mkdir()
            (dest / "gateway" / "app.py").write_text("", encoding="utf-8")
            (dest / setup.GATEWAY_MARK).write_text(setup.GATEWAY_COMMIT, encoding="utf-8")
            with mock.patch.object(setup, "download") as dl:
                quiet(setup.fetch_gateway, dest)
            dl.assert_not_called()

    def test_the_pin_is_a_full_commit_and_a_sha256(self):
        self.assertRegex(setup.GATEWAY_COMMIT, r"^[0-9a-f]{40}$")
        self.assertRegex(setup.GATEWAY_FILES_SHA256, r"^[0-9a-f]{64}$")


class TestTheModelsTheGatewayGets(unittest.TestCase):
    def test_every_faststrata_preset_has_the_gateways_id(self):
        presets = {(f, m) for f in setup.FASTSTRATA_FAMILIES for m in setup.FAMILIES[f]["models"]}
        self.assertEqual(presets, set(setup.GATEWAY_MODELS))

    def test_the_window_comes_from_the_run_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            q4 = config(Path(tmp), "qwen36-ud-q4_k_m", 147456)
            orn = config(Path(tmp), "ornith-apex-i-quality", 65536)
            models = {m["id"]: m for m in setup.gateway_models([q4, orn])}
        self.assertEqual(models["qwen3.6-35b-a3b"]["n_ctx"], 147456)
        self.assertEqual(models["qwen3.6-35b-a3b"]["target"], "strata-qwen36-ud-q4_k_m.json")
        self.assertEqual(models["qwen3.6-35b-a3b"]["max_output"], 16384)
        self.assertEqual(models["ornith-1.5-35b-a3b"]["max_output"], 8192)
        for m in models.values():
            self.assertEqual(m["engine"], "polystrata")
            self.assertEqual(m["upstream"], setup.ENGINE_URL)

    def test_a_small_window_keeps_room_for_the_prompt(self):
        """A 4 GB card's 16K: an answer of 16,384 tokens would be the whole window, and the gateway refuses a
        model whose reserve is its window."""
        with tempfile.TemporaryDirectory() as tmp:
            m, = setup.gateway_models([config(Path(tmp), "qwen36-ud-q3_k_m", 16384)])
        self.assertEqual(m["context_reserve"], 4096)
        self.assertLess(m["context_reserve"], m["n_ctx"])

    def test_a_model_faststrata_does_not_serve_is_left_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(setup.gateway_models([config(Path(tmp), "gemma4-ud-q4_k_m", 65536)]), [])


class TestTheGatewaysSettings(unittest.TestCase):
    def test_faststrata_models_are_replaced_and_the_rest_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "faststrata-gateway" / "settings.json"
            path.parent.mkdir()
            before = {"port": 18100, "lodestar_enabled": True,
                      "models": [{"id": "old", "engine": "polystrata", "upstream": "x"},
                                 {"id": "far", "engine": "remote", "upstream": "http://far"}]}
            path.write_text(json.dumps(before), encoding="utf-8")
            new = [{"id": "qwen3.6-35b-a3b", "engine": "polystrata"}]
            setup.write_gateway_settings(new, "strata-qwen36-ud-q4_k_m.json", path=path, now="20261020")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([m["id"] for m in data["models"]], ["qwen3.6-35b-a3b", "far"])
            self.assertEqual(data["port"], 18100, "the user's port stays")
            self.assertTrue(data["lodestar_enabled"], "the user's switch stays")
            self.assertEqual(data["polystrata_dir"], str(setup.ROOT))
            self.assertEqual(data["polystrata_config"], "strata-qwen36-ud-q4_k_m.json")
            self.assertEqual(data["upstream"], setup.ENGINE_URL)
            self.assertEqual(json.loads((path.parent / "settings.json.bak-20261020").read_text(encoding="utf-8")),
                             before)

    def test_a_new_pc_gets_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "faststrata-gateway" / "settings.json"
            setup.write_gateway_settings([], "strata-x.json", path=path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["active_preset"], setup.GATEWAY_PRESET)
            self.assertEqual(data["port"], 18000)
            self.assertFalse(data["lodestar_enabled"], "LODESTAR starts off")

    def test_it_is_the_gateways_own_folder(self):
        with mock.patch.dict(os.environ, {"APPDATA": r"C:\Users\user\AppData\Roaming"}):
            if setup.WIN:
                self.assertEqual(setup.gateway_settings_path(),
                                 Path(r"C:\Users\user\AppData\Roaming\faststrata-gateway\settings.json"))


class TestOpenCodeIsFoundWhereItsInstallersPutIt(unittest.TestCase):
    def test_on_the_path(self):
        with mock.patch.object(setup.shutil, "which", return_value="/usr/bin/opencode"):
            self.assertEqual(setup.find_opencode(), "/usr/bin/opencode")

    def test_its_own_installers_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / ".opencode" / "bin" / ("opencode.exe" if setup.WIN else "opencode")
            exe.parent.mkdir(parents=True)
            exe.write_text("", encoding="utf-8")
            with mock.patch.object(setup.shutil, "which", return_value=None), \
                 mock.patch.object(setup.Path, "home", return_value=Path(tmp)):
                self.assertEqual(setup.find_opencode(), str(exe))

    def test_none_when_it_is_nowhere(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(setup.shutil, "which", return_value=None), \
             mock.patch.object(setup.Path, "home", return_value=Path(tmp)), \
             mock.patch.dict(os.environ, {"APPDATA": tmp}):
            self.assertIsNone(setup.find_opencode())

    def test_without_npm_or_scoop_nothing_is_run(self):
        with mock.patch.object(setup.shutil, "which", return_value=None), \
             mock.patch.object(setup, "run") as run:
            result, said = quiet(setup.install_opencode)
        self.assertIsNone(result)
        run.assert_not_called()
        self.assertIn("https://opencode.ai", said)


class TestOpenCodesConfigIsWrittenThroughTheGateway(unittest.TestCase):
    def _done(self, code, payload):
        return subprocess.CompletedProcess([], code, stdout=json.dumps(payload) + "\n", stderr="")

    def test_written(self):
        with mock.patch.object(setup.subprocess, "run",
                               return_value=self._done(0, {"written": True, "models": ["m"], "path": "p"})) as run:
            self.assertTrue(quiet(setup.connect_opencode, Path("py"), Path("gw"), False)[0])
        self.assertIn("--write", run.call_args[0][0])
        self.assertNotIn("--drop-comments", run.call_args[0][0])

    def test_comments_are_dropped_only_when_the_user_says_so(self):
        answers = [self._done(3, {"path": "p.jsonc", "comments": 2, "written": False}),
                   self._done(0, {"written": True, "models": ["m"], "path": "p.jsonc", "backup": "p.jsonc.bak"})]
        with mock.patch.object(setup.subprocess, "run", side_effect=answers) as run, \
             mock.patch.object(setup, "ask", return_value="y"):
            self.assertTrue(quiet(setup.connect_opencode, Path("py"), Path("gw"), False)[0])
        self.assertIn("--drop-comments", run.call_args_list[1][0][0])

        with mock.patch.object(setup.subprocess, "run", side_effect=answers[:1]) as run, \
             mock.patch.object(setup, "ask", return_value="n"):
            self.assertFalse(quiet(setup.connect_opencode, Path("py"), Path("gw"), False)[0])
        self.assertEqual(run.call_count, 1)

    def test_with_yes_the_comments_are_kept(self):
        """--yes takes the recommended answer, and losing a user's comments is not one."""
        with mock.patch.object(setup.subprocess, "run",
                               return_value=self._done(3, {"path": "p", "comments": 1})) as run:
            self.assertFalse(quiet(setup.connect_opencode, Path("py"), Path("gw"), True)[0])
        self.assertEqual(run.call_count, 1)


@unittest.skipUnless(setup.WIN, "the Windows shortcut")
class TestTheDesktopShortcut(unittest.TestCase):
    def test_it_points_at_the_gateways_app_without_a_console(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "FastStrata.lnk"
            link.write_text("", encoding="utf-8")
            done = subprocess.CompletedProcess([], 0, stdout=str(link) + "\n", stderr="")
            gw = Path(r"C:\Games\Fast'Strata\gateway")
            with mock.patch.object(setup.subprocess, "run", return_value=done) as run:
                self.assertEqual(setup.make_shortcut(gw), link)
        script = run.call_args[0][0][-1]
        self.assertIn("pythonw.exe", script)
        self.assertIn("-m gateway.gui", script)
        self.assertIn("Fast''Strata", script, "a quote in the path is doubled for PowerShell")

    def test_a_failure_is_said_not_raised(self):
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="access denied")
        with mock.patch.object(setup.subprocess, "run", return_value=failed):
            result, said = quiet(setup.make_shortcut, Path(r"C:\FastStrata\gateway"))
        self.assertIsNone(result)
        self.assertIn("access denied", said)


class TestThePythonTheGatewayNeeds(unittest.TestCase):
    """A friend's PC (2026-10-22) had Python older than 3.12 and no newer one: the gateway was left out with a
    warning. Now Windows gets Python 3.12 for the user's account, asked, and Linux is told the command."""

    def setUp(self):
        self.gw = Path(tempfile.mkdtemp()) / "gateway"
        self.gw.mkdir()

    def _env(self, win, found, installed=None, answer="y", yes=False):
        asked = []

        def ask(q, choices, default, yes_):
            asked.append(q)
            return default if yes_ else answer

        out = io.StringIO()
        with mock.patch.object(setup, "WIN", win),              mock.patch.object(setup, "gateway_python", return_value=found),              mock.patch.object(setup, "install_gateway_python", return_value=installed) as inst,              mock.patch.object(setup, "ask", ask),              mock.patch.object(setup, "run", lambda *a, **k: None),              redirect_stdout(out):
            py = setup.gateway_env(self.gw, yes)
        return py, out.getvalue(), asked, inst.called

    def test_a_new_enough_python_is_used_without_asking(self):
        py, text, asked, installed = self._env(True, r"C:\Py313\python.exe")
        self.assertIsNotNone(py)
        self.assertEqual(asked, [])
        self.assertFalse(installed)

    def test_windows_without_one_installs_python_3_12_when_the_user_says_yes(self):
        py, text, asked, installed = self._env(True, None, installed=r"C:\Py312\python.exe")
        self.assertTrue(installed)
        self.assertIsNotNone(py)
        self.assertIn("Install Python 3.12 for your account", asked[0])
        self.assertIn("Python for the gateway:", text)

    def test_yes_installs_it_without_a_question(self):
        py, text, asked, installed = self._env(True, None, installed=r"C:\Py312\python.exe", answer="n", yes=True)
        self.assertTrue(installed)
        self.assertIsNotNone(py)

    def test_no_leaves_the_gateway_out_and_says_how_to_add_it(self):
        py, text, asked, installed = self._env(True, None, answer="n")
        self.assertIsNone(py)
        self.assertFalse(installed)
        self.assertIn("--gateway on", text)

    def test_linux_is_told_the_command(self):
        py, text, asked, installed = self._env(False, None)
        self.assertIsNone(py)
        self.assertEqual(asked, [])
        self.assertIn("sudo apt install python3.12 python3.12-venv", text)


class TestTheStep(unittest.TestCase):
    def setUp(self):
        # never the checkout's own gateway/ folder: an install made from this checkout would change what is asked
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(setup, "GATEWAY_DIR", Path(tmp.name) / "gateway")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _args(self, **kw):
        base = dict(gateway="on", gateway_dir=None, opencode="skip", no_shortcut=True, yes=True, no_start=True)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_it_sets_up_the_installed_models_and_skips_what_it_is_told(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            gw = home / "gw"
            (gw / "gateway").mkdir(parents=True)
            (gw / "gateway" / "app.py").write_text("", encoding="utf-8")
            q4 = config(home, "qwen36-ud-q4_k_m", 147456)
            settings = home / "settings.json"
            with mock.patch.object(setup, "installed_configs", return_value=[q4]), \
                 mock.patch.object(setup, "gateway_env", return_value=Path("py")), \
                 mock.patch.object(setup, "gateway_settings_path", return_value=settings), \
                 mock.patch.object(setup, "find_opencode", return_value=None), \
                 mock.patch.object(setup, "install_opencode") as install, \
                 mock.patch.object(setup, "connect_opencode", return_value=True) as connect, \
                 mock.patch.object(setup, "make_shortcut") as shortcut:
                result, _ = quiet(setup.faststrata_gateway, self._args(gateway_dir=str(gw)))
            self.assertEqual(result, gw.resolve())
            install.assert_not_called()
            shortcut.assert_not_called()
            connect.assert_called_once()
            data = json.loads(settings.read_text(encoding="utf-8"))
            self.assertEqual([m["id"] for m in data["models"]], ["qwen3.6-35b-a3b"])

    def test_yes_does_not_install_opencode_unless_asked_to(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "gateway").mkdir()
            (home / "gateway" / "app.py").write_text("", encoding="utf-8")
            q4 = config(home, "qwen36-ud-q4_k_m", 147456)
            for wanted, calls in (("ask", 0), ("install", 1)):
                with mock.patch.object(setup, "installed_configs", return_value=[q4]), \
                     mock.patch.object(setup, "gateway_env", return_value=Path("py")), \
                     mock.patch.object(setup, "gateway_settings_path", return_value=home / "s.json"), \
                     mock.patch.object(setup, "find_opencode", return_value=None), \
                     mock.patch.object(setup, "install_opencode", return_value=None) as install, \
                     mock.patch.object(setup, "connect_opencode", return_value=True):
                    quiet(setup.faststrata_gateway, self._args(gateway_dir=str(home), opencode=wanted))
                self.assertEqual(install.call_count, calls, wanted)

    def test_yes_alone_leaves_it_out(self):
        """--yes is an AI's or a script's setup (docs/AI_SETUP.md): the gateway, OpenCode and a desktop shortcut are
        changes to the PC it asks for by name, --gateway on."""
        with tempfile.TemporaryDirectory() as tmp:
            q4 = config(Path(tmp), "qwen36-ud-q4_k_m", 147456)
            with mock.patch.object(setup, "installed_configs", return_value=[q4]), \
                 mock.patch.object(setup, "fetch_gateway") as fetch:
                result, said = quiet(setup.faststrata_gateway, self._args(gateway=None, yes=True))
        self.assertIsNone(result)
        fetch.assert_not_called()
        self.assertIn("--gateway on", said)

    def test_an_installed_gateway_is_refreshed_without_asking(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            gw = home / "gateway"
            (gw / "gateway").mkdir(parents=True)
            (gw / "gateway" / "app.py").write_text("", encoding="utf-8")
            (gw / setup.GATEWAY_MARK).write_text(setup.GATEWAY_COMMIT, encoding="utf-8")
            q4 = config(home, "qwen36-ud-q4_k_m", 147456)
            with mock.patch.object(setup, "installed_configs", return_value=[q4]), \
                 mock.patch.object(setup, "GATEWAY_DIR", gw), \
                 mock.patch.object(setup, "ask", side_effect=AssertionError("asked")), \
                 mock.patch.object(setup, "gateway_env", return_value=Path("py")), \
                 mock.patch.object(setup, "gateway_settings_path", return_value=home / "s.json"), \
                 mock.patch.object(setup, "find_opencode", return_value="opencode"), \
                 mock.patch.object(setup, "out", return_value="1.0"), \
                 mock.patch.object(setup, "connect_opencode", return_value=True):
                result, said = quiet(setup.faststrata_gateway, self._args(gateway=None, yes=False))
            self.assertEqual(result, gw)
            self.assertIn("installed here", said)

    def test_nothing_to_serve_is_said(self):
        with mock.patch.object(setup, "installed_configs", return_value=[]):
            result, said = quiet(setup.faststrata_gateway, self._args())
        self.assertIsNone(result)
        self.assertIn("none is installed", said)

    def test_declined_it_installs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            q4 = config(Path(tmp), "qwen36-ud-q4_k_m", 147456)
            with mock.patch.object(setup, "installed_configs", return_value=[q4]), \
                 mock.patch.object(setup, "ask", return_value="n"), \
                 mock.patch.object(setup, "fetch_gateway") as fetch:
                result, said = quiet(setup.faststrata_gateway, self._args(gateway=None, yes=False))
        self.assertIsNone(result)
        fetch.assert_not_called()
        self.assertIn("--gateway on", said)


if __name__ == "__main__":
    unittest.main()
