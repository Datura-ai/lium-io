"""Configuration, from the environment.

Environment variables rather than a config file, because the agent's configuration IS the
compose block the platform injects — anything it reads from a file inside the CVM would be
something the customer could edit, and every one of these settings decides how the agent
identifies itself.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from attest_agent.tls import TlsIdentity, load_or_create

DEFAULT_PORT = 8451
DEFAULT_STATE_DIR = Path("/var/lib/lium-attest-agent")

# A TDX quote takes well under a second on healthy hardware. The budget is generous so a
# loaded guest does not fail a trust check it would have passed, and bounded so a wedged
# guest agent answers 503 rather than holding the verifier open.
DEFAULT_QUOTE_TIMEOUT_SECONDS = 30

# How many recent nonces the agent remembers its answer for. Sized for a verifier that
# retries, not for history: a few thousand entries is far more than any real challenge rate
# and small enough that the map cannot become a memory lever for whoever can reach it.
DEFAULT_NONCE_CACHE_SIZE = 4096


@dataclass(frozen=True)
class Config:
    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT
    state_dir: Path = DEFAULT_STATE_DIR
    quote_timeout_seconds: int = DEFAULT_QUOTE_TIMEOUT_SECONDS
    nonce_cache_size: int = DEFAULT_NONCE_CACHE_SIZE

    @property
    def cert_path(self) -> Path:
        return self.state_dir / "tls" / "cert.pem"

    @property
    def key_path(self) -> Path:
        return self.state_dir / "tls" / "key.pem"

    def tls_identity(self) -> TlsIdentity:
        return load_or_create(self.cert_path, self.key_path)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def load_config() -> Config:
    config = Config(
        host=os.environ.get("ATTEST_AGENT_HOST", "0.0.0.0"),
        port=_int("ATTEST_AGENT_PORT", DEFAULT_PORT),
        state_dir=Path(os.environ.get("ATTEST_AGENT_STATE_DIR", str(DEFAULT_STATE_DIR))),
        quote_timeout_seconds=_int("ATTEST_AGENT_QUOTE_TIMEOUT", DEFAULT_QUOTE_TIMEOUT_SECONDS),
        nonce_cache_size=_int("ATTEST_AGENT_NONCE_CACHE", DEFAULT_NONCE_CACHE_SIZE),
    )
    if config.port <= 0 or config.port > 65535:
        raise ValueError(f"ATTEST_AGENT_PORT must be in 1-65535, got {config.port}")
    if config.quote_timeout_seconds <= 0:
        raise ValueError("ATTEST_AGENT_QUOTE_TIMEOUT must be positive")
    if config.nonce_cache_size <= 0:
        raise ValueError("ATTEST_AGENT_NONCE_CACHE must be positive")
    return config
