"""DAH-2794: with ENABLE_SCRAPE_SOURCE_DELIVERY the scrape is not frozen and not uploaded —
it is piped to the executor's own interpreter over the SSH channel's stdin.

The old path builds a ~13 MB PyInstaller onefile every cycle and SFTP-puts it to every
executor; behind one shared uplink that is what times out. These tests pin the two halves of
the new path: the validator produces runnable source, and the runner actually forwards it.
"""

import ast

import pytest

from neurons.validators.src.services.file_encrypt_service import (
    KEYS_FOR_ENCRYPTION_KEY_GENERATION,
    FileEncryptService,
)
from neurons.validators.src.services.task.runner import SSHCommandRunner


def test_source_delivery_carries_runnable_source_alongside_the_binary(monkeypatch):
    # Arrange
    # The obfuscator and the key substitution are the parts under test; freezing the result
    # costs ~40s of PyInstaller and is exercised by the flag-off path in production.
    monkeypatch.setattr(FileEncryptService, "make_binary_file", lambda self, tmp, path: "scrape")
    service = FileEncryptService(ssh_service=None)

    # Act
    files = service.ecrypt_miner_job_files()

    # Assert
    ast.parse(files.machine_scrape_source)
    # The key mapping is what makes every cycle's payload unique, and the encryption key is
    # the concatenation of those random names — both must survive the shortcut.
    assert files.encrypt_key == "".join(
        files.all_keys[key] for key in KEYS_FOR_ENCRYPTION_KEY_GENERATION
    )
    for original_key in KEYS_FOR_ENCRYPTION_KEY_GENERATION:
        assert original_key not in files.machine_scrape_source


@pytest.mark.asyncio
async def test_runner_forwards_stdin_text_to_the_remote_process(mock_ssh_client):
    # Arrange
    runner = SSHCommandRunner(mock_ssh_client)

    # Act
    result = await runner.run("/usr/bin/python -I -", stdin_text="print('scrape')")

    # Assert
    assert result.success is True
    mock_ssh_client.run.assert_awaited_once_with(
        "/usr/bin/python -I -", input="print('scrape')"
    )
