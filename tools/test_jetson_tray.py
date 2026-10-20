"""No-GPU checks for the opt-in Jetson tray lifecycle."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("jetson_tray", Path(__file__).with_name("jetson_tray.py"))
tray = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tray)


class JetsonTrayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "engine").write_text("engine")
        (self.root / "pack").mkdir()
        (self.root / "native.gguf").write_text("model")
        self.config = self.root / "strata-test.json"
        self.config.write_text(json.dumps({"exe": str(self.root / "engine"),
                                           "args": ["--pack", str(self.root / "pack"),
                                                    "--native", str(self.root / "native.gguf")],
                                           "model_name": "test-model"}))
        for name, value in (("ROOT", self.root), ("STATE", self.root / "config" / "state.json"),
                            ("UNIT_FILE", self.root / "units" / "strata-jetson.service"),
                            ("AUTOSTART", self.root / "autostart" / "tray.desktop")):
            patch = mock.patch.object(tray, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def test_models_and_selection(self):
        self.assertTrue(tray.models()[0]["ready"])
        with mock.patch.object(tray, "service_active", return_value=False):
            tray.select_model(self.config.name)
        self.assertEqual(tray.selected_model()["name"], "test-model")
        self.assertEqual(tray.STATE.stat().st_mode & 0o777, 0o600)

    def test_cannot_switch_while_running(self):
        with mock.patch.object(tray, "service_active", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "pause"):
                tray.select_model(self.config.name)

    def test_install_keeps_service_paused(self):
        with mock.patch.object(tray, "systemctl") as systemctl:
            systemctl.return_value.returncode = 0
            tray.install()
        self.assertEqual(systemctl.call_args.args, ("daemon-reload",))
        unit = tray.UNIT_FILE.read_text()
        self.assertIn("ExecStart=", unit)
        self.assertIn(f"WorkingDirectory={self.root}\n", unit)
        self.assertNotIn("Restart=", unit)
        self.assertIn("tray", tray.AUTOSTART.read_text())

    def test_status_rejects_wrong_server(self):
        with mock.patch.object(tray, "unit_state", return_value="active"), \
             mock.patch.object(tray, "request", return_value={"service": "other", "model": "other"}):
            self.assertEqual(tray.status()["state"], "error")

    def test_paused_status_does_not_contact_api(self):
        with mock.patch.object(tray, "unit_state", return_value="inactive"), \
             mock.patch.object(tray, "port_open", return_value=False), \
             mock.patch.object(tray, "request") as request:
            self.assertEqual(tray.status()["state"], "paused")
            request.assert_not_called()

    def test_paused_status_reports_port_conflict(self):
        with mock.patch.object(tray, "unit_state", return_value="inactive"), \
             mock.patch.object(tray, "port_open", return_value=True):
            self.assertEqual(tray.status()["state"], "error")

    def test_load_requires_owned_ready_server(self):
        with mock.patch.object(tray, "unit_state", return_value="active"), \
             mock.patch.object(tray, "status", return_value={"state": "error", "error": "wrong server"}), \
             mock.patch.object(tray, "request") as request:
            with self.assertRaisesRegex(RuntimeError, "wrong server"):
                tray.action("load")
            request.assert_not_called()

    def test_agent_terminal_uses_source_script_and_repo_workspace(self):
        script = self.root / "tools" / "strata_agent.sh"
        script.parent.mkdir()
        script.write_text("#!/bin/bash\n")
        self.assertEqual(tray.agent_terminal_command(),
                         ["xterm", "-T", "Strata Agent", "-e", "bash", str(script),
                          "--workspace", str(self.root)])

    def test_agent_terminal_requires_script(self):
        with self.assertRaisesRegex(RuntimeError, "agent script not found"):
            tray.agent_terminal_command()


if __name__ == "__main__":
    unittest.main()
