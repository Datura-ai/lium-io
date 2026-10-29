"""Per-pod SSH host key derivation.

A reboot recreates the pod's container, so a host key generated inside the container changes on
every reboot and the renter's SSH client refuses the pod. Deriving the key from the pod id gives
the recreated container the key the renter already trusts.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# distinct from the volume passphrase info, so the shared master secret never yields the same bytes
_HKDF_INFO = b"lium-pod-ssh-host-key/v1"


@dataclass(frozen=True)
class PodSshHostKey:
    private_openssh: str
    public_openssh: str


def derive_pod_ssh_host_key(master_secret: str, pod_id: str) -> PodSshHostKey:
    if not pod_id:
        raise ValueError("pod_id is required to derive a pod SSH host key")
    seed = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=pod_id.encode("utf-8"),
        info=_HKDF_INFO,
    ).derive(master_secret.encode("utf-8"))
    key = Ed25519PrivateKey.from_private_bytes(seed)
    private_openssh = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_openssh = key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode("ascii")
    return PodSshHostKey(private_openssh=private_openssh, public_openssh=public_openssh)
