from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from scalecodec.utils.ss58 import is_valid_ss58_address

WATCHTOWER_ENDPOINT_URL: str = "https://lium.io/api/watchtower/digest"
WATCHTOWER_VALIDATOR_HOTKEY: str = "5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p"
# The hotkey the validator rotates to (DAH-3394 in the executor: `VALIDATOR_NEXT_HOTKEY_SS58`).
# While it is set, a digest signed by either hotkey is accepted, so the fleet's updaters keep
# following releases through the swap. Empty means unset: only the current hotkey is trusted.
# Like the current hotkey it is a build-time constant (here, or `config_override.py` written by
# `docker_build.sh`), never an environment variable: the signers an updater trusts are fixed by
# its image, not by whoever writes its `.env`.
WATCHTOWER_VALIDATOR_HOTKEY_NEXT: str = ""

try:
    from config_override import WATCHTOWER_ENDPOINT_URL, WATCHTOWER_VALIDATOR_HOTKEY  # noqa: F401, F811
except ImportError:
    pass
try:
    from config_override import WATCHTOWER_VALIDATOR_HOTKEY_NEXT  # noqa: F401, F811
except ImportError:
    # no override module (the prod build) or one written before the rotation (current only)
    pass


def validator_hotkeys(current: str, next_: str) -> dict[str, str]:
    """Key id -> ss58 of every hotkey whose digest signature the updater accepts, `current` first.

    Each address is checked here, at import: a mistyped `next` stops the updater (and CI) now,
    not as a fleet that silently refuses every signed digest after the chain swap.
    """
    hotkeys = {"current": current}
    if next_.strip() and next_.strip() != current:
        hotkeys["next"] = next_.strip()
    for key_id, ss58 in hotkeys.items():
        if not is_valid_ss58_address(ss58, valid_ss58_format=42):  # 42 = the Bittensor ss58 prefix
            raise ValueError(f"the {key_id} validator hotkey is not a valid ss58 address: {ss58!r}")
    return hotkeys


WATCHTOWER_VALIDATOR_HOTKEYS: dict[str, str] = validator_hotkeys(
    WATCHTOWER_VALIDATOR_HOTKEY, WATCHTOWER_VALIDATOR_HOTKEY_NEXT
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    WATCHTOWER_ENABLED: bool = Field(env="WATCHTOWER_ENABLED", default=True)
    WATCHTOWER_IMAGE: str = Field(env="WATCHTOWER_IMAGE", default="daturaai/compute-subnet-executor-runner")
    WATCHTOWER_INTERVAL: int = Field(env="WATCHTOWER_INTERVAL", default=300)
    WATCHTOWER_ENV_FILE_PATH: str = Field(env="WATCHTOWER_ENV_FILE_PATH", default="~/.env")


settings = Settings()
