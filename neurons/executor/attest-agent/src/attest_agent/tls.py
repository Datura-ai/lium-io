"""The agent's TLS identity — generated inside the CVM, and never leaving it.

The key is created on first boot **inside the guest**, so the private half exists only in
encrypted CVM memory and on the CVM's encrypted disk. Neither the provider nor the platform
ever sees it, which is what makes hashing the public half into `report_data` mean something:
a relay attacker can copy the certificate but cannot terminate TLS with it.

It is persisted rather than regenerated per start so that a restarted agent keeps the same
identity, and a verifier that pinned the key across a rental does not have to re-pin after
every restart. It is *not* the CVM's sealing key and does not need to be — it identifies a
channel, not data at rest.

Self-signed, and deliberately so. A CA-signed certificate would prove a name; what a
verifier needs here is that this exact key is inside an attested TDX guest holding these
exact GPUs, and only the quote can say that. The certificate is a container for the key.
"""

import datetime
import ipaddress
import logging
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)

# Ten years. The certificate's validity window is not a security control here — the quote
# is — and an expiry inside a rental would break the validator's only channel into the CVM
# for a reason that has nothing to do with trust.
_VALIDITY_DAYS = 3650

_KEY_MODE = 0o600
_CERT_MODE = 0o644


@dataclass(frozen=True)
class TlsIdentity:
    cert_path: Path
    key_path: Path
    public_key_der: bytes

    @property
    def public_key_hex(self) -> str:
        return self.public_key_der.hex()


def _write(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def _public_key_der(certificate: x509.Certificate) -> bytes:
    """The SubjectPublicKeyInfo DER — one canonical encoding of the key.

    Taken from the certificate rather than from the private key object so that what is
    hashed into `report_data` is exactly what a client sees on the wire. Two encodings that
    differ by a byte are two different identities, and the verifier only has the wire one.
    """
    return certificate.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def load_or_create(
    cert_path: Path, key_path: Path, *, common_name: str = "lium-attest-agent"
) -> TlsIdentity:
    """Return the agent's TLS identity, generating one only if there is none.

    A half-present pair — a certificate with no key, or the reverse — is regenerated rather
    than half-used: it means an interrupted first boot, and continuing from it would give
    the agent an identity it cannot actually serve.
    """
    if cert_path.is_file() and key_path.is_file():
        certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
        logger.info("using the existing TLS identity at %s", cert_path)
        return TlsIdentity(cert_path, key_path, _public_key_der(certificate))

    if cert_path.exists() or key_path.exists():
        logger.warning("only half of the TLS pair is present; generating a new one")

    key = ec.generate_private_key(ec.SECP384R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=_VALIDITY_DAYS))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(common_name), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA384())
    )

    _write(
        key_path,
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
        _KEY_MODE,
    )
    _write(cert_path, certificate.public_bytes(serialization.Encoding.PEM), _CERT_MODE)
    logger.info("generated a new TLS identity at %s", cert_path)
    return TlsIdentity(cert_path, key_path, _public_key_der(certificate))
