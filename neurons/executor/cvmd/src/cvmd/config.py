"""cvmd configuration — TOML file at /etc/cvmd/config.toml, with env-var overrides for tests."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("/etc/cvmd/config.toml")

# Real request bodies are a `kind` plus three hashes — hundreds of bytes. The cap exists so an
# unauthenticated caller cannot make cvmd hold arbitrary memory before a single check runs.
DEFAULT_MAX_BODY_BYTES = 64 * 1024

# The freshness window the request timestamp must fall inside, in seconds.
DEFAULT_SKEW_SECONDS = 60


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or internally inconsistent.

    Always fatal: cvmd refuses to start rather than run on a half-understood config.
    """


@dataclass(frozen=True)
class Config:
    authorized_clients: Path
    state_dir: Path
    host: str = "0.0.0.0"
    port: int = 8443
    tls_cert: Path | None = None
    tls_key: Path | None = None
    skew_seconds: int = DEFAULT_SKEW_SECONDS
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES

    @property
    def tls_enabled(self) -> bool:
        return self.tls_cert is not None and self.tls_key is not None


def _env_override(key: str) -> str | None:
    return os.environ.get(f"CVMD_{key.upper()}")


def _get(table: dict, key: str, default=None):
    """Read one setting: env var wins over the TOML table, which wins over the default."""
    override = _env_override(key)
    if override is not None:
        return override
    return table.get(key, default)


def _as_path(value) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _as_int(value, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be an integer, got {value!r}") from exc


def load_config(path: Path | None = None) -> Config:
    """Load the daemon config. Any problem here is fatal — see ConfigError."""
    path = path or Path(_env_override("config_path") or DEFAULT_CONFIG_PATH)

    table: dict = {}
    if path.exists():
        try:
            with path.open("rb") as handle:
                table = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read config at {path}: {exc}") from exc

    authorized_clients = _as_path(_get(table, "authorized_clients"))
    state_dir = _as_path(_get(table, "state_dir"))
    if authorized_clients is None:
        raise ConfigError("`authorized_clients` is required")
    if state_dir is None:
        raise ConfigError("`state_dir` is required")

    tls_cert = _as_path(_get(table, "tls_cert"))
    tls_key = _as_path(_get(table, "tls_key"))
    if (tls_cert is None) != (tls_key is None):
        raise ConfigError("`tls_cert` and `tls_key` must be set together or not at all")

    max_body_bytes = _as_int(
        _get(table, "max_body_bytes", DEFAULT_MAX_BODY_BYTES), "max_body_bytes"
    )
    skew_seconds = _as_int(_get(table, "skew_seconds", DEFAULT_SKEW_SECONDS), "skew_seconds")
    if max_body_bytes <= 0:
        raise ConfigError("`max_body_bytes` must be positive")
    if skew_seconds <= 0:
        raise ConfigError("`skew_seconds` must be positive")

    return Config(
        authorized_clients=authorized_clients,
        state_dir=state_dir,
        host=str(_get(table, "host", "0.0.0.0")),
        port=_as_int(_get(table, "port", 8443), "port"),
        tls_cert=tls_cert,
        tls_key=tls_key,
        skew_seconds=skew_seconds,
        max_body_bytes=max_body_bytes,
    )
