"""Per-volume gocryptfs passphrase derivation."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

if TYPE_CHECKING:
    from core.config import Settings

_HKDF_INFO = b"lium-gocryptfs/v1"


@dataclass(frozen=True)
class VolumeKeyMaterial:
    passphrase: str
    key_id: str


def volume_encryption_enabled(settings: Settings) -> bool:
    return bool(settings.VOLUME_MASTER_SECRET)


def derive_volume_passphrase(master_secret: str, key_id: str) -> str:
    if not master_secret:
        raise ValueError("VOLUME_MASTER_SECRET is required to derive volume passphrases")
    if not key_id:
        raise ValueError("key_id is required to derive volume passphrases")
    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=key_id.encode("utf-8"),
        info=_HKDF_INFO,
    ).derive(master_secret.encode("utf-8"))
    return base64.urlsafe_b64encode(derived).decode("ascii").rstrip("=")


class VolumeKeyDeriver:
    def __init__(self, master_secret: str) -> None:
        self._master_secret = master_secret

    @classmethod
    def from_settings(cls, settings: Settings) -> VolumeKeyDeriver:
        return cls(master_secret=settings.VOLUME_MASTER_SECRET)

    def material(self, key_id: str) -> VolumeKeyMaterial:
        return VolumeKeyMaterial(
            passphrase=derive_volume_passphrase(self._master_secret, key_id),
            key_id=key_id,
        )
