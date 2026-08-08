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

# How long a launch waits for the guest to become reachable before giving up. A TDX guest with
# a large memory backing takes minutes to boot, so this is generous by design; the request is
# held for the whole window (see LaunchConfig).
DEFAULT_LAUNCH_TIMEOUT_SECONDS = 900

# dstack's default key-provider port. lium-cvm.sh relies on the same default.
DEFAULT_KEY_PROVIDER_PORT = 3443


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or internally inconsistent.

    Always fatal: cvmd refuses to start rather than run on a half-understood config.
    """


@dataclass(frozen=True)
class LaunchConfig:
    """Everything the launch path needs that is not in the request.

    Sizing is **provider configuration**, never an API field: a cvmd request names which
    software stack to run, and the host decides how big it is. That was settled while
    reviewing DAH-2575 and is why `CreateCvmRequest` carries no vcpu/memory/disk/gpu.

    Every field is optional because a node can legitimately be installed before it has a
    catalog — refusing to start would take `/v1/state` down with it. `missing()` names what a
    launch would need, so an incomplete config fails at the point of use with a specific
    reason rather than at import with a stack trace. A *malformed* value is still fatal at
    startup: absent and wrong are different problems.
    """

    dstack_scripts_dir: Path | None = None
    catalog_path: Path | None = None
    run_dir: Path | None = None
    key_provider_port: int = DEFAULT_KEY_PROVIDER_PORT
    launch_timeout_seconds: int = DEFAULT_LAUNCH_TIMEOUT_SECONDS

    vcpus: int | None = None
    memory: str | None = None
    disk: str | None = None
    gpus: tuple[str, ...] = ()
    ports: tuple[str, ...] = ()
    env_file: Path | None = None
    # The guest-side port carrying SSH. When set, cvmd reads the guest's host key off the
    # forwarded port and treats that as proof the CVM is up. When unset, readiness falls back
    # to a plain TCP accept and the launch report's fingerprint is null — honest degradation
    # rather than a guessed port.
    ssh_guest_port: int | None = None
    pin_numa: bool = False
    hugepages: bool = False

    # Required for a launch. `gpus` is deliberately absent: a CVM with no GPUs is a legal
    # configuration, so an empty list cannot be distinguished from an unset one and must not
    # block a launch.
    _REQUIRED = (
        "dstack_scripts_dir",
        "catalog_path",
        "run_dir",
        "vcpus",
        "memory",
        "disk",
        "ports",
    )

    def missing(self) -> list[str]:
        """Names of the settings a launch needs and does not have."""
        return [name for name in self._REQUIRED if not getattr(self, name)]


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
    launch: LaunchConfig = LaunchConfig()

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


def _as_optional_int(value, key: str) -> int | None:
    if value is None or value == "":
        return None
    return _as_int(value, key)


def _as_optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_tuple(value, key: str) -> tuple[str, ...]:
    """A list in TOML, or a comma-separated string from an env override.

    Both spellings exist because config.toml is written by Ansible while the env vars are how
    tests and operators poke one setting without editing the file.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(f"{key} must be a list of strings, got {item!r}")
            if item.strip():
                items.append(item.strip())
        return tuple(items)
    raise ConfigError(f"{key} must be a list of strings or a comma-separated string")


def _as_bool(value, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off", ""):
        return False
    raise ConfigError(f"{key} must be a boolean, got {value!r}")


def _load_launch(table: dict) -> LaunchConfig:
    """Read the DAH-2576 launch settings. Absent is fine; malformed is fatal."""
    launch = LaunchConfig(
        dstack_scripts_dir=_as_path(_get(table, "dstack_scripts_dir")),
        catalog_path=_as_path(_get(table, "catalog_path")),
        run_dir=_as_path(_get(table, "run_dir")),
        key_provider_port=_as_int(
            _get(table, "key_provider_port", DEFAULT_KEY_PROVIDER_PORT), "key_provider_port"
        ),
        launch_timeout_seconds=_as_int(
            _get(table, "launch_timeout_seconds", DEFAULT_LAUNCH_TIMEOUT_SECONDS),
            "launch_timeout_seconds",
        ),
        vcpus=_as_optional_int(_get(table, "cvm_vcpus"), "cvm_vcpus"),
        memory=_as_optional_str(_get(table, "cvm_memory")),
        disk=_as_optional_str(_get(table, "cvm_disk")),
        gpus=_as_tuple(_get(table, "cvm_gpus"), "cvm_gpus"),
        ports=_as_tuple(_get(table, "cvm_ports"), "cvm_ports"),
        env_file=_as_path(_get(table, "cvm_env_file")),
        ssh_guest_port=_as_optional_int(_get(table, "cvm_ssh_guest_port"), "cvm_ssh_guest_port"),
        pin_numa=_as_bool(_get(table, "cvm_pin_numa"), "cvm_pin_numa"),
        hugepages=_as_bool(_get(table, "cvm_hugepages"), "cvm_hugepages"),
    )
    if launch.vcpus is not None and launch.vcpus <= 0:
        raise ConfigError("`cvm_vcpus` must be positive")
    if launch.key_provider_port <= 0:
        raise ConfigError("`key_provider_port` must be positive")
    if launch.launch_timeout_seconds <= 0:
        raise ConfigError("`launch_timeout_seconds` must be positive")
    return launch


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
        launch=_load_launch(table),
    )
