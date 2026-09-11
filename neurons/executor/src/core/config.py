from typing import Optional
from bittensor_wallet import Keypair
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# This hotkey is used to verify the validator signature. 
# This shouldn't be overridden by the environment variable. 
VALIDATOR_HOTKEY_SS58 = "5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p"
try:
    from core.config_override import _VALIDATOR_HOTKEY_SS58 
    VALIDATOR_HOTKEY_SS58 = _VALIDATOR_HOTKEY_SS58
except Exception:
    pass

# DAH-3394: the validator hotkey is being rotated. A release that trusts only the new hotkey
# would be refused by every executor that has not restarted onto it yet, so this release accepts
# two: `current` (above) and `next`, the hotkey the validator swaps to. `next` is empty until the
# new hotkey exists; it is then set here or in config_override at build time — like `current`,
# never from the environment: the set of signers an executor trusts is fixed by the image, not by
# whoever writes its .env. With `next` empty the executor behaves exactly as before.
VALIDATOR_NEXT_HOTKEY_SS58 = ""
try:
    from core.config_override import _VALIDATOR_NEXT_HOTKEY_SS58
    VALIDATOR_NEXT_HOTKEY_SS58 = _VALIDATOR_NEXT_HOTKEY_SS58
except ImportError:
    # no override module (the default build) or one that names only `current`
    pass


def _validator_hotkeys(current: str, next_: str) -> dict[str, str]:
    """Key id -> ss58 of every hotkey whose signature the executor accepts, `current` first.

    Each address is parsed here, at import: a mistyped `next` must stop the executor (and CI) now,
    not surface as a 401 on the first request after the chain swap.
    """
    hotkeys = {"current": current}
    if next_.strip() and next_.strip() != current:
        hotkeys["next"] = next_.strip()
    for key_id, ss58 in hotkeys.items():
        try:
            Keypair(ss58_address=ss58)
        except ValueError as exc:
            raise ValueError(f"the {key_id} validator hotkey is not a valid ss58 address") from exc
    return hotkeys


VALIDATOR_HOTKEYS_SS58: dict[str, str] = _validator_hotkeys(VALIDATOR_HOTKEY_SS58, VALIDATOR_NEXT_HOTKEY_SS58)

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
    # DAH-3394: an ssh key the validator installed through /upload_ssh_key is removed by the
    # executor itself this many seconds after the upload when no /remove_ssh_key came for it,
    # so a leaked or forgotten upload is worth at most this window. The validator's own flows fit:
    # a verification job authenticates with its key for up to ~820 s (tasks capped at
    # JOB_TIME_OUT - 120 s, the RoCE sweep up to JOB_TIME_OUT - 80 s) and removes it right
    # after; a rental create keeps ONE established ssh session for the image pull, which sshd
    # does not re-check against authorized_keys. Do not set below 900.
    EXECUTOR_UPLOADED_KEY_TTL_S: int = Field(env="EXECUTOR_UPLOADED_KEY_TTL_S", default=900, ge=1)
    # How often the purge looks at authorized_keys; a key lives at most TTL + this.
    EXECUTOR_UPLOADED_KEY_PURGE_INTERVAL_S: int = Field(env="EXECUTOR_UPLOADED_KEY_PURGE_INTERVAL_S", default=60, ge=1)


settings = Settings()
