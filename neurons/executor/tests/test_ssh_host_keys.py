"""DAH-5729: ``ssh_host_keys.sh`` keeps the executor's sshd host keys across a recreated container.

Each "container" is a fresh SSH_ROOT (the image ships no host keys); the "volume" is SSH_HOST_KEYS_DIR. The real
script and the real ssh-keygen run.
"""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ssh_host_keys.sh"


def start(container: Path, volume: Path) -> str:
    """Run the script as run.sh does at container start; the ed25519 public key sshd then serves."""
    subprocess.run(
        ["sh", str(SCRIPT)],
        env={"PATH": "/usr/bin:/bin", "SSH_ROOT": str(container), "SSH_HOST_KEYS_DIR": str(volume)},
        check=True,
        capture_output=True,
    )
    return (container / "etc/ssh/ssh_host_ed25519_key.pub").read_text().split()[1]


def test_a_recreated_container_serves_the_same_host_key(tmp_path):
    volume = tmp_path / "volume"
    first = start(tmp_path / "container-1", volume)
    assert start(tmp_path / "container-2", volume) == first
    private = (tmp_path / "container-2/etc/ssh/ssh_host_ed25519_key").read_bytes()
    assert private == (tmp_path / "container-1/etc/ssh/ssh_host_ed25519_key").read_bytes()


def test_the_first_start_on_an_empty_volume_keeps_the_keys_the_container_has(tmp_path):
    container = tmp_path / "container"
    (container / "etc/ssh").mkdir(parents=True)
    subprocess.run(["ssh-keygen", "-A", "-f", str(container)], check=True, capture_output=True)
    before = (container / "etc/ssh/ssh_host_ed25519_key.pub").read_text().split()[1]
    volume = tmp_path / "volume"
    assert start(container, volume) == before
    assert start(tmp_path / "recreated", volume) == before


def test_another_machine_has_another_host_key(tmp_path):
    assert start(tmp_path / "a", tmp_path / "volume-a") != start(tmp_path / "b", tmp_path / "volume-b")
