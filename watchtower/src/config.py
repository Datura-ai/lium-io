from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

WATCHTOWER_ENDPOINT_URL: str = "https://lium.io/api/watchtower/digest"
WATCHTOWER_VALIDATOR_HOTKEY: str = "5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p"

try:
    from config_override import WATCHTOWER_ENDPOINT_URL, WATCHTOWER_VALIDATOR_HOTKEY  # noqa: F401, F811
except ImportError:
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    WATCHTOWER_ENABLED: bool = Field(env="WATCHTOWER_ENABLED", default=True)
    WATCHTOWER_IMAGE: str = Field(env="WATCHTOWER_IMAGE", default="daturaai/compute-subnet-executor-runner")
    WATCHTOWER_INTERVAL: int = Field(env="WATCHTOWER_INTERVAL", default=300)
    WATCHTOWER_ENV_FILE_PATH: str = Field(env="WATCHTOWER_ENV_FILE_PATH", default="~/.env")


settings = Settings()
