import importlib.util
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from scalecodec.utils.ss58 import is_valid_ss58_address

from core.logger import get_logger

logger = get_logger(__name__)

# The validator whose signatures this executor trusts (SSH-key uploads, /ping,
# /hardware_utilization, container metrics and logs). Deliberately not an
# environment variable: a host operator must not be able to repoint the executor
# at another validator at runtime. Non-prod images bake a different anchor in at
# build time — docker_build.sh writes core/config_override.py.
_BUILTIN_VALIDATOR_HOTKEY_SS58 = "5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p"


def _resolve_validator_hotkey() -> str:
    if importlib.util.find_spec("core.config_override") is None:
        logger.info("Validator trust anchor: built-in %s", _BUILTIN_VALIDATOR_HOTKEY_SS58)
        return _BUILTIN_VALIDATOR_HOTKEY_SS58
    # An override module is present, so this is a non-prod build. Anything wrong
    # with it (import error, missing name, mistyped address) is a broken build;
    # failing here beats silently trusting the built-in anchor of another
    # environment, or rejecting every signature at runtime.
    from core.config_override import _VALIDATOR_HOTKEY_SS58 as override

    if not is_valid_ss58_address(override):
        raise RuntimeError(
            f"core.config_override._VALIDATOR_HOTKEY_SS58 is not an ss58 address: {override!r}"
        )
    logger.warning("Validator trust anchor: OVERRIDDEN by core.config_override -> %s", override)
    return override


VALIDATOR_HOTKEY_SS58 = _resolve_validator_hotkey()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    PROJECT_NAME: str = "compute-subnet-executor"

    INTERNAL_PORT: int = Field(env="INTERNAL_PORT", default=8001)
    SSH_PORT: int = Field(env="SSH_PORT", default=2200)
    SSH_PUBLIC_PORT: Optional[int] = Field(env="SSH_PUBLIC_PORT", default=None)

    MINER_HOTKEY_SS58_ADDRESS: str = Field(env="MINER_HOTKEY_SS58_ADDRESS")
    DEFAULT_MINER_HOTKEY: str = Field(
        env="DEFAULT_MINER_HOTKEY",
        default="5D4jX4TqUkZwNwKAjjYrbk2FHFNN2U1TgFF6ZMuNPnjnKJVU"
    )

    RENTING_PORT_RANGE: Optional[str] = Field(env="RENTING_PORT_RANGE", default=None)
    RENTING_PORT_MAPPINGS: Optional[str] = Field(env="RENTING_PORT_MAPPINGS", default=None)

    ENV: str = Field(env="ENV", default="dev")

    # Compute-app (backend) base URL used to look up the default "cache template"
    # docker image for this host's GPU. Defaults to the same backend the
    # validators use; set to an empty value to disable the on-boot cache pre-pull.
    COMPUTE_REST_API_URL: Optional[str] = Field(
        env="COMPUTE_REST_API_URL", default="https://lium.io/api"
    )
    # How often (seconds) the executor re-checks the template's remote digest and
    # re-pulls when it has changed. Defaults to 15 minutes.
    CACHE_TEMPLATE_REFRESH_SECONDS: int = Field(
        env="CACHE_TEMPLATE_REFRESH_SECONDS", default=15 * 60
    )

    ENABLE_TDX_ATTESTATION: bool = Field(env="ENABLE_TDX_ATTESTATION", default=False)
    TDX_QUOTE_TIMEOUT: int = Field(env="TDX_QUOTE_TIMEOUT", default=60)
    SSH_HOST_KEY_PATH: str = Field(env="SSH_HOST_KEY_PATH", default="/etc/ssh/ssh_host_ed25519_key.pub")

    # G1 phase-0 — NVIDIA CC GPU evidence emission. Collected ONLY when this flag
    # is on AND the executor runs inside a dstack CVM (socket below exists): in
    # host-mode pynvml would attest a non-CC bare-metal GPU, which must never be
    # emitted as confidential-compute evidence (topology guard).
    ENABLE_GPU_ATTESTATION: bool = Field(env="ENABLE_GPU_ATTESTATION", default=False)
    # NVIDIA attestation SDK arch tag routing the evidence at NRAS ("HOPPER", "BLACKWELL").
    GPU_ATTESTATION_ARCH: str = Field(env="GPU_ATTESTATION_ARCH", default="HOPPER")
    # dstack guest marker; also what the dstack SDK talks to for TDX quotes.
    DSTACK_SOCKET_PATH: str = Field(env="DSTACK_SOCKET_PATH", default="/var/run/dstack.sock")
    # G3 enforcement phase: reject SSH-key uploads without a validator attestation
    # nonce. Leave off until the validator fleet mints nonces (migration order:
    # executors accept optional nonce first, then validators send, then this).
    REQUIRE_ATTESTATION_NONCE: bool = Field(env="REQUIRE_ATTESTATION_NONCE", default=False)
    # DAH-3200: the backend signs `timestamp = int(time.time())` into every /containers/{name}
    # request (utilization and logs). A signed request older or newer than this many seconds is
    # refused, so a captured one cannot be replayed for the life of the container. Symmetric so a
    # host clock that runs ahead is treated like one that runs behind; wide enough for NTP drift.
    CONTAINER_SIGNATURE_MAX_AGE_SECONDS: int = Field(env="CONTAINER_SIGNATURE_MAX_AGE_SECONDS", default=300)

    # liumd phase 1 (DAH-2834): `POST /verify` runs the verification suite locally from one
    # validator-signed intent instead of 25–40 SSH commands. Off: the route answers 404 and
    # `/version` does not advertise it, so validators keep the SSH path. On: only the signed,
    # nonce'd, time-windowed intent of the pinned validator hotkey is accepted.
    EXECUTOR_LOCAL_VERIFY_ENABLED: bool = Field(env="EXECUTOR_LOCAL_VERIFY_ENABLED", default=False)
    # Longest a single /verify call may run before answering with what finished (seconds).
    LOCAL_VERIFY_MAX_DEADLINE_SECONDS: int = Field(env="LOCAL_VERIFY_MAX_DEADLINE_SECONDS", default=600)
    # How far the intent's issued_at may be from this host's clock (seconds, either way).
    LOCAL_VERIFY_INTENT_WINDOW_SECONDS: int = Field(env="LOCAL_VERIFY_INTENT_WINDOW_SECONDS", default=120)


settings = Settings()
