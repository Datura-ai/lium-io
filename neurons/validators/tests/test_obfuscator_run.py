"""A failed obfuscator run raises and leaves no earlier build's output to be read."""

import subprocess
from pathlib import Path

import pytest

from services import file_encrypt_service
from services.file_encrypt_service import FileEncryptService


def test_a_failed_obfuscator_run_raises_and_removes_the_stale_output(tmp_path, monkeypatch):
    stale = Path(file_encrypt_service.__file__).parent / ".." / "miner_jobs/obfuscated_machine_scrape.py"
    existed = stale.exists()
    previous = stale.read_text() if existed else None
    stale.write_text("# output of an earlier build\n")
    service = FileEncryptService.__new__(FileEncryptService)
    fresh = []
    monkeypatch.setattr(
        service, "fresh_job_files_directory", lambda *_: fresh.append(1) or str(tmp_path), raising=False
    )

    def fail(command, **kwargs):
        assert kwargs.get("check") is True and not stale.exists()
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(file_encrypt_service.subprocess, "run", fail)
    try:
        with pytest.raises(subprocess.CalledProcessError):
            service.ecrypt_miner_job_files()
        # the previous cycle's directory stays for the express lane (review 4579627721)
        assert not fresh
    finally:
        if existed:
            stale.write_text(previous)
        else:
            stale.unlink(missing_ok=True)


def test_a_failed_build_keeps_the_previous_cycles_directory_and_drops_the_new_one(tmp_path, monkeypatch):
    output = Path(file_encrypt_service.__file__).parent / ".." / "miner_jobs/obfuscated_machine_scrape.py"
    existed = output.exists()
    previous_output = output.read_text() if existed else None
    monkeypatch.setattr(file_encrypt_service, "JOB_FILES_ROOT", tmp_path)
    previous_cycle = tmp_path / "cycle-previous"
    previous_cycle.mkdir()
    service = FileEncryptService.__new__(FileEncryptService)
    monkeypatch.setattr(file_encrypt_service.subprocess, "run", lambda *a, **k: output.write_text("x = 1\n"))
    monkeypatch.setattr(service, "generate_key_mappings", lambda: ({}, "key"), raising=False)

    def fail_build(*_):
        raise RuntimeError("build failed")

    monkeypatch.setattr(service, "make_binary_file", fail_build, raising=False)
    try:
        with pytest.raises(RuntimeError):
            service.ecrypt_miner_job_files()

        assert [entry.name for entry in tmp_path.iterdir()] == ["cycle-previous"]
    finally:
        if existed:
            output.write_text(previous_output)
        else:
            output.unlink(missing_ok=True)
