import asyncio
import sys
from pathlib import Path

import pytest

from services import file_encrypt_service
from services.file_encrypt_service import FileEncryptService, ScrapeBinaryBuild

# stands in for PyInstaller: a child process that takes a moment and writes the named binary
FAKE_BUILD = "import sys, time, pathlib; time.sleep(0.2); pathlib.Path(sys.argv[1], sys.argv[2]).write_text('bin')"


@pytest.fixture
def job_files_root(monkeypatch, tmp_path):
    monkeypatch.setattr(file_encrypt_service, "JOB_FILES_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def builds(monkeypatch):
    started = []

    def command(_source, dist, _work, name):
        started.append(name)
        return [sys.executable, "-c", FAKE_BUILD, dist, name]

    monkeypatch.setattr(file_encrypt_service, "_pyinstaller_command", command)
    return started


@pytest.mark.asyncio
async def test_concurrent_stdin_failures_wait_on_one_build(job_files_root, builds):
    directory = job_files_root / "cycle-now"
    directory.mkdir()
    build = ScrapeBinaryBuild(directory, "tmpabc", "print(1)", lambda: ())

    await asyncio.gather(*(build.ensure() for _ in range(5)))
    await build.ensure()

    assert builds == ["tmpabc"]
    assert (directory / "tmpabc").read_text() == "bin"


@pytest.mark.asyncio
async def test_earlier_job_files_go_only_after_the_build(job_files_root, builds):
    old, held, current = (job_files_root / name for name in ("cycle-old", "cycle-held", "cycle-now"))
    for directory in (old, held, current):
        directory.mkdir()
        (directory / "binary").write_text("bin")
    build = ScrapeBinaryBuild(current, "tmpabc", "print(1)", lambda: {str(held)})

    pending = asyncio.ensure_future(build.ensure())
    await asyncio.sleep(0.05)
    still_there_while_building = old.exists()
    await pending

    assert still_there_while_building
    assert not old.exists()
    assert held.exists() and current.exists()


def test_preparing_a_cycle_builds_nothing(job_files_root, monkeypatch):
    def no_build(*_args):
        raise AssertionError("a cycle start must not build the binary")

    monkeypatch.setattr(file_encrypt_service, "_pyinstaller_command", no_build)
    monkeypatch.setattr(FileEncryptService, "make_binary_file", no_build)
    (job_files_root / "cycle-empty").mkdir()
    (job_files_root / "cycle-built").mkdir()
    (job_files_root / "cycle-built" / "binary").write_text("bin")

    files = FileEncryptService(ssh_service=None).prepare_cycle_job_files(lambda: ())

    assert files.machine_scrape_source
    assert list(Path(files.tmp_directory).iterdir()) == []
    assert isinstance(files.binary_build, ScrapeBinaryBuild)
    # an empty earlier directory holds nothing anyone reads; one with a binary waits for the next build
    assert not (job_files_root / "cycle-empty").exists()
    assert (job_files_root / "cycle-built").exists()
