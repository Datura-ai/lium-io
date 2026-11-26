import subprocess
from types import SimpleNamespace

import pytest

from neurons.validators.src.services.task.checks import start_gpu_monitor
from neurons.validators.src.services.task.checks.machine_spec_scrape import _binary_command
from neurons.validators.src.services.task.checks.start_gpu_monitor import StartGPUMonitorCheck
from neurons.validators.src.services.verifyx_validation_service import VerifyXValidationService


def _shell_text(marker) -> str:
    return f"/bin/true; touch {marker}; true"


class _LocalRunner:
    def __init__(self, cwd):
        self.cwd = cwd

    async def run(self, command, **_):
        done = subprocess.run(["sh", "-c", command], cwd=self.cwd, capture_output=True, text=True, timeout=20)
        return SimpleNamespace(
            stdout=done.stdout,
            stderr=done.stderr,
            success=done.returncode == 0,
            exit_code=done.returncode,
            error_type=None,
            command_id="c",
        )


def test_verifyx_command_keeps_executor_reported_paths_as_single_arguments(tmp_path):
    executor = SimpleNamespace(python_path=_shell_text(tmp_path / "a"), root_dir=_shell_text(tmp_path / "b"))

    command = VerifyXValidationService()._verifyx_command(executor, SimpleNamespace(seed=1, cipher_text="c"), "lib")
    subprocess.run(["sh", "-c", command], cwd=tmp_path, capture_output=True, timeout=20)

    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()


def test_scrape_command_keeps_the_upload_directory_as_a_single_argument(tmp_path):
    command = _binary_command(_shell_text(tmp_path / "a"), "scrape.sh")
    subprocess.run(["sh", "-c", command], cwd=tmp_path, capture_output=True, timeout=20)

    assert not (tmp_path / "a").exists()


@pytest.mark.asyncio
async def test_gpu_monitor_start_keeps_executor_reported_values_as_single_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(start_gpu_monitor, "render_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(start_gpu_monitor, "CheckResult", lambda **kwargs: SimpleNamespace(**kwargs))
    ctx = SimpleNamespace(
        runner=_LocalRunner(tmp_path),
        executor=SimpleNamespace(
            python_path=_shell_text(tmp_path / "a"),
            root_dir=f"/root$(touch {tmp_path / 'b'})",
            uuid=_shell_text(tmp_path / "c"),
        ),
        config=SimpleNamespace(
            validator_keypair=SimpleNamespace(sign=lambda data: b"\x01", ss58_address="5validator"),
            compute_rest_app_url="http://backend",
        ),
    )

    await StartGPUMonitorCheck().run(ctx)

    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()
    assert not (tmp_path / "c").exists()
