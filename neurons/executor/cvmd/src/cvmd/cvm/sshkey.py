"""Is the guest up, and what is its SSH host key?

The launch report has to carry the guest's SSH host-key fingerprint, and reading it doubles as
the readiness signal: a key can only be collected once sshd inside the CVM is answering on the
forwarded port, which means the guest booted, the network device came up, and the port forward
works. A plain TCP accept proves much less — QEMU's user-mode forward accepts before anything
inside the guest is listening.

`ssh-keyscan` does the protocol work. Writing an SSH transport here to avoid a subprocess would
be a second implementation of a handshake whose only job is to produce a value openssh already
prints, and cvmd's dependencies stay deliberately small.
"""

import base64
import hashlib
import logging
import socket
import subprocess

logger = logging.getLogger(__name__)

KEYSCAN = "ssh-keyscan"
KEYSCAN_TIMEOUT_SECONDS = 10


def fingerprint_of(base64_key: str) -> str:
    """The OpenSSH SHA256 fingerprint of a base64-encoded host key blob.

    Same construction `ssh-keygen -l` prints: sha256 over the raw key blob, base64, no padding.
    """
    blob = base64.b64decode(base64_key, validate=True)
    digest = base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    return f"SHA256:{digest}"


def _parse_keyscan(output: str) -> str | None:
    """Pull the first host key out of ssh-keyscan output.

    Lines are `host keytype base64key`; comments start with '#'. Any single key identifies the
    guest, so the first usable line wins rather than the run failing on an unexpected key type.
    """
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            return fingerprint_of(parts[2])
        except (ValueError, base64.binascii.Error):
            continue
    return None


def read_host_key(host: str, port: int) -> str | None:
    """Fingerprint of the guest's SSH host key, or None if it cannot be read yet.

    None is "not ready or not readable", never an error: this runs in a poll loop where the
    normal answer for the first several minutes is "not yet".
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, host and port are host config
            [KEYSCAN, "-T", str(KEYSCAN_TIMEOUT_SECONDS), "-p", str(port), host],
            capture_output=True,
            text=True,
            timeout=KEYSCAN_TIMEOUT_SECONDS + 5,
            check=False,
        )
    except FileNotFoundError:
        logger.warning(
            "%s is not installed, so no SSH host-key fingerprint can be reported; readiness "
            "falls back to a TCP accept",
            KEYSCAN,
        )
        return None
    except subprocess.TimeoutExpired:
        return None
    return _parse_keyscan(result.stdout)


def accepts_connection(host: str, port: int, *, timeout: float = 5.0) -> bool:
    """Does anything accept a TCP connection here? The weaker readiness signal."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
