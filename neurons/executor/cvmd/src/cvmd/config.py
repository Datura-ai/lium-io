"""cvmd configuration — TOML file at /etc/cvmd/config.toml, with env-var overrides for tests."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from bittensor.sp_core import Keypair

from cvmd.catalog import CatalogConfig
from cvmd.cvm import ports

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

# How long a graceful poweroff is given before cvmd signals the process group.
#
# A large-memory TDX guest returns its RAM to the host as it exits, and that is slow — 43
# minutes was measured for a 1.13 TB guest on this fleet. So this is a floor for ordinary
# sizes, not a budget for the largest ones; setting it too low turns a legitimate slow exit
# into a SIGKILL partway through. See roles/cvmd/README.md for the measured per-hardware-class
# values these two defaults come from.
DEFAULT_TEARDOWN_TIMEOUT_SECONDS = 600

# How long the four release conditions get, counted from the moment the CVM was asked to stop.
#
# A separate budget from the one above because it measures a different thing: that one bounds
# how long the guest is given to power itself off, this one bounds how long the *host* takes to
# get its hardware back afterwards. The two are not proportional — a guest that exits in seconds
# can leave its memory draining for tens of minutes.
DEFAULT_TEARDOWN_VERIFY_TIMEOUT_SECONDS = 1800

# How much of the guest's configured memory must be back before the node counts as free.
#
# Below 1.0 because MemAvailable is the kernel's own estimate and a host reads a little short of
# a round number even when nothing is held. It is a tolerance on a measurement, not a licence to
# launch into memory that has not come back — see `cvm/release.py:memory_returned`.
DEFAULT_TEARDOWN_MEMORY_TOLERANCE = 0.9

# dstack's default key-provider port. lium-cvm.sh relies on the same default.
DEFAULT_KEY_PROVIDER_PORT = 3443

# How often the host asks the backend for the current catalog. It is the upper bound on how long
# a revocation takes to reach a node that nobody pushes to, so it is minutes rather than hours;
# the platform can also make it immediate with `POST /v1/catalog/refresh`.
DEFAULT_CATALOG_REFRESH_SECONDS = 300

DEFAULT_CATALOG_FETCH_TIMEOUT_SECONDS = 30


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
    run_dir: Path | None = None
    # Where a renter order's compose and scripts are staged before they are measured (DAH-2580).
    # Daemon-owned, defaulted off `state_dir` for the same reason the catalog's working files
    # are: a host that named its state directory has already said where these go, and another
    # required setting is another chance to point one of them at a directory Ansible also writes.
    # Deliberately NOT under `run_dir`: that directory is scanned for stray CVM disks, and a
    # staging directory appearing there would read as a guest nobody has a record of.
    renter_dir: Path | None = None
    key_provider_port: int = DEFAULT_KEY_PROVIDER_PORT
    launch_timeout_seconds: int = DEFAULT_LAUNCH_TIMEOUT_SECONDS
    teardown_timeout_seconds: int = DEFAULT_TEARDOWN_TIMEOUT_SECONDS
    teardown_verify_timeout_seconds: int = DEFAULT_TEARDOWN_VERIFY_TIMEOUT_SECONDS
    teardown_memory_tolerance: float = DEFAULT_TEARDOWN_MEMORY_TOLERANCE

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
    # The catalog is deliberately absent: it is no longer a host setting but a signed document
    # from the platform, so "this host has no catalog" is a `CatalogError` naming which of the
    # catalog settings or which manifest is missing — a far more specific answer than a list of
    # unset keys. See `catalog/store.py:CatalogConfig.missing`.
    _REQUIRED = (
        "dstack_scripts_dir",
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
    catalog: CatalogConfig = CatalogConfig()

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


def _as_float(value, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be a number, got {value!r}") from exc


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


def _load_launch(table: dict, state_dir: Path) -> LaunchConfig:
    """Read the DAH-2576 launch settings. Absent is fine; malformed is fatal."""
    launch = LaunchConfig(
        dstack_scripts_dir=_as_path(_get(table, "dstack_scripts_dir")),
        run_dir=_as_path(_get(table, "run_dir")),
        renter_dir=_as_path(_get(table, "renter_dir")) or state_dir / "renter",
        key_provider_port=_as_int(
            _get(table, "key_provider_port", DEFAULT_KEY_PROVIDER_PORT), "key_provider_port"
        ),
        launch_timeout_seconds=_as_int(
            _get(table, "launch_timeout_seconds", DEFAULT_LAUNCH_TIMEOUT_SECONDS),
            "launch_timeout_seconds",
        ),
        teardown_timeout_seconds=_as_int(
            _get(table, "teardown_timeout_seconds", DEFAULT_TEARDOWN_TIMEOUT_SECONDS),
            "teardown_timeout_seconds",
        ),
        teardown_verify_timeout_seconds=_as_int(
            _get(
                table,
                "teardown_verify_timeout_seconds",
                DEFAULT_TEARDOWN_VERIFY_TIMEOUT_SECONDS,
            ),
            "teardown_verify_timeout_seconds",
        ),
        teardown_memory_tolerance=_as_float(
            _get(table, "teardown_memory_tolerance", DEFAULT_TEARDOWN_MEMORY_TOLERANCE),
            "teardown_memory_tolerance",
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
    # Parsed with the launch path's own parser, not re-described here. Without this a mapping
    # like "tcp:0.0.0.0:0:22" is well-formed TOML, so the daemon starts looking configured and
    # refuses every launch at the first request — the failure mode the whole fail-at-startup
    # rule exists to avoid. It is also what lets Ansible check the rendered file by running
    # `load_config` against it, the same way it checks the catalog with `load_catalog`.
    try:
        ports.parse_all(launch.ports)
    except ports.PortError as exc:
        raise ConfigError(f"`cvm_ports`: {exc}") from exc

    if launch.vcpus is not None and launch.vcpus <= 0:
        raise ConfigError("`cvm_vcpus` must be positive")
    if launch.key_provider_port <= 0:
        raise ConfigError("`key_provider_port` must be positive")
    if launch.launch_timeout_seconds <= 0:
        raise ConfigError("`launch_timeout_seconds` must be positive")
    if launch.teardown_timeout_seconds <= 0:
        raise ConfigError("`teardown_timeout_seconds` must be positive")
    if launch.teardown_verify_timeout_seconds <= 0:
        raise ConfigError("`teardown_verify_timeout_seconds` must be positive")
    # Above 1.0 would demand more memory back than the guest was ever given, so no teardown on
    # this host could ever complete — a value that turns every node FAILED, accepted at startup.
    if not 0 < launch.teardown_memory_tolerance <= 1:
        raise ConfigError(
            "`teardown_memory_tolerance` must be greater than 0 and at most 1, got "
            f"{launch.teardown_memory_tolerance}"
        )
    return launch


def _load_catalog(table: dict, state_dir: Path) -> CatalogConfig:
    """Read the DAH-2578 catalog settings.

    Two of the four paths default off `state_dir` rather than being required: they are
    daemon-owned working files, so a host that names its state directory has already said where
    they go, and two more settings in every config.toml would be two more chances to write one
    of them somewhere Ansible also manages.

    `signer` and `images_dir` have no defaults on purpose. A guessed signer is a trusted key
    nobody chose, and a guessed image directory is a path whose contents decide what a node
    attests as. Unset means this host launches nothing, and says which setting is why.
    """
    signer = _as_optional_str(_get(table, "catalog_signer"))
    catalog = CatalogConfig(
        cache_path=_as_path(_get(table, "catalog_cache_path"))
        or state_dir / "catalog" / "manifest.json",
        seed_path=_as_path(_get(table, "catalog_seed_path")),
        signer=signer,
        images_dir=_as_path(_get(table, "catalog_images_dir")),
        materialize_dir=_as_path(_get(table, "catalog_materialize_dir"))
        or state_dir / "catalog" / "artifacts",
        manifest_url=_as_optional_str(_get(table, "catalog_manifest_url")),
        refresh_seconds=_as_int(
            _get(table, "catalog_refresh_seconds", DEFAULT_CATALOG_REFRESH_SECONDS),
            "catalog_refresh_seconds",
        ),
        fetch_timeout_seconds=_as_int(
            _get(table, "catalog_fetch_timeout_seconds", DEFAULT_CATALOG_FETCH_TIMEOUT_SECONDS),
            "catalog_fetch_timeout_seconds",
        ),
    )

    # Checked at startup rather than at the first fetch. A mistyped ss58 in config.toml would
    # otherwise sit there looking configured and refuse every manifest the backend sends,
    # reported as a signature failure — which reads as an attack, not as a typo.
    if signer is not None:
        try:
            Keypair(ss58_address=signer)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"`catalog_signer` is not a valid ss58 address: {exc}") from exc

    if catalog.manifest_url is not None and signer is None:
        raise ConfigError(
            "`catalog_manifest_url` is set but `catalog_signer` is not; a manifest this host "
            "cannot check the signature of is not a catalog"
        )
    if catalog.refresh_seconds <= 0:
        raise ConfigError("`catalog_refresh_seconds` must be positive")
    if catalog.fetch_timeout_seconds <= 0:
        raise ConfigError("`catalog_fetch_timeout_seconds` must be positive")
    return catalog


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
        launch=_load_launch(table, state_dir),
        catalog=_load_catalog(table, state_dir),
    )
