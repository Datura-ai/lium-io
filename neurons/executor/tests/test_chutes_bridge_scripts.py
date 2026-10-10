import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


BRIDGE_BIN_DIR = Path(__file__).resolve().parents[1] / "lium-bridge" / "bin"


class TestChutesBridgeScripts(unittest.TestCase):
    def _write_executable(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        path.chmod(0o755)

    def _render_script(self, workdir: Path, script_name: str) -> tuple[Path, Path, Path]:
        bridge_dir = workdir / "opt" / "lium-bridge"
        log_file = workdir / "var" / "log" / "lium-bridge.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        script = (BRIDGE_BIN_DIR / script_name).read_text(encoding="utf-8")
        script = script.replace('BRIDGE_DIR="/opt/lium-bridge"', f'BRIDGE_DIR="{bridge_dir}"')
        script = script.replace(
            'LOG_FILE="/var/log/lium-bridge.log"',
            f'LOG_FILE="{log_file}"',
        )

        script_path = bridge_dir / "bin" / script_name
        self._write_executable(script_path, script)
        return script_path, bridge_dir, log_file

    def test_bridgectl_does_not_log_hotkey_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            bridgectl_path, bridge_dir, log_file = self._render_script(workdir, "bridgectl")

            self._write_executable(
                bridge_dir / "bin" / "setup-chutes",
                """#!/usr/bin/env bash
                echo '{"ok": true, "state": "installed_stopped"}'
                """,
            )

            stub_dir = workdir / "stub-bin"
            self._write_executable(
                stub_dir / "sudo",
                """#!/usr/bin/env bash
                exec "$@"
                """,
            )

            completed = subprocess.run(
                [str(bridgectl_path), "setup", "--hotkey-seed", "super-secret-seed"],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PATH": f"{stub_dir}:{os.environ['PATH']}",
                },
                check=False,
            )

            self.assertEqual(completed.returncode, 0)
            self.assertIn('"ok": true', completed.stdout)

            log_text = log_file.read_text(encoding="utf-8")
            self.assertIn("Running verb=setup", log_text)
            self.assertNotIn("super-secret-seed", log_text)

    def test_start_chutes_unhealthy_agent_sets_error_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            start_path, bridge_dir, _ = self._render_script(workdir, "start-chutes")
            state_file = bridge_dir / "state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(
                json.dumps(
                    {
                        "state": "installed_stopped",
                        "gpu_verified": False,
                        "node_name": "gpu-node-01",
                        "last_error": None,
                    }
                ),
                encoding="utf-8",
            )

            stub_dir = workdir / "stub-bin"
            active_flag = workdir / "k3s-active"
            self._write_executable(
                stub_dir / "systemctl",
                f"""#!/usr/bin/env bash
                if [[ "$1" == "start" && "$2" == "k3s" ]]; then
                    touch "{active_flag}"
                    exit 0
                fi

                if [[ "$1" == "is-active" && "$2" == "k3s" ]]; then
                    [[ -f "{active_flag}" ]]
                    exit $?
                fi

                exit 0
                """,
            )
            self._write_executable(
                stub_dir / "k3s",
                """#!/usr/bin/env bash
                if [[ "$1" == "kubectl" && "$2" == "get" && "$3" == "nodes" ]]; then
                    echo "gpu-node-01 Ready"
                    exit 0
                fi

                if [[ "$1" == "kubectl" && "$2" == "get" && "$3" == "pods" ]]; then
                    exit 0
                fi

                exit 0
                """,
            )
            self._write_executable(
                stub_dir / "sleep",
                """#!/usr/bin/env bash
                exit 0
                """,
            )

            completed = subprocess.run(
                [str(start_path)],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PATH": f"{stub_dir}:{os.environ['PATH']}",
                },
                check=False,
            )

            self.assertEqual(completed.returncode, 1)

            payload = json.loads(completed.stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["state"], "error")
            self.assertIn("did not become healthy", payload["error"])

            persisted_state = json.loads(state_file.read_text(encoding="utf-8"))
            self.assertEqual(persisted_state["state"], "error")
            self.assertIn("did not become healthy", persisted_state["last_error"])
            self.assertEqual(persisted_state["node_name"], "gpu-node-01")


if __name__ == "__main__":
    unittest.main(verbosity=2)
