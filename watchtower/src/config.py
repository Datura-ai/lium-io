from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# `project.version` in pyproject.toml (tests/test_watchtower.py holds the two equal). Sent as the
# User-Agent of every digest request so the platform can count which watchtower versions poll it —
# the fleet has no other watchtower-version signal (the executor's update-status report carries none).
WATCHTOWER_VERSION: str = "1.2.0"
WATCHTOWER_USER_AGENT: str = f"lium-watchtower/{WATCHTOWER_VERSION}"

WATCHTOWER_ENDPOINT_URL: str = "https://lium.io/api/watchtower/digest"
# The validator hotkey whose signature on the digest is trusted, and the one the Lium validator swaps
# to (owner, 22 Sep 2026). Both are checked until the swap release drops the first; a build with a
# config_override names its own pair (or only the first).
WATCHTOWER_VALIDATOR_HOTKEY: str = "5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p"
WATCHTOWER_VALIDATOR_NEXT_HOTKEY: str = "5DZhu7LLGGc7qRa8ZPFArt7KV2XEKMTr5Q7ZuM9LNdTaoNfK"

try:
    from config_override import WATCHTOWER_ENDPOINT_URL, WATCHTOWER_VALIDATOR_HOTKEY  # noqa: F401, F811
except ImportError:
    pass
else:
    # an override that names only the first hotkey (a staging build) trusts one signer
    try:
        from config_override import WATCHTOWER_VALIDATOR_NEXT_HOTKEY  # noqa: F811
    except ImportError:
        WATCHTOWER_VALIDATOR_NEXT_HOTKEY = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    WATCHTOWER_ENABLED: bool = Field(env="WATCHTOWER_ENABLED", default=True)
    WATCHTOWER_IMAGE: str = Field(env="WATCHTOWER_IMAGE", default="daturaai/compute-subnet-executor-runner")
    WATCHTOWER_INTERVAL: int = Field(env="WATCHTOWER_INTERVAL", default=300)
    WATCHTOWER_ENV_FILE_PATH: str = Field(env="WATCHTOWER_ENV_FILE_PATH", default="~/.env")


settings = Settings()
