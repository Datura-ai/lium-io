from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    WATCHTOWER_ENABLED: bool = Field(env="WATCHTOWER_ENABLED", default=True)
    WATCHTOWER_IMAGE: str = Field(env="WATCHTOWER_IMAGE", default="daturaai/compute-subnet-executor-runner")
    WATCHTOWER_INTERVAL: int = Field(env="WATCHTOWER_INTERVAL", default=300)
    WATCHTOWER_ENDPOINT_URL: str = Field(env="WATCHTOWER_ENDPOINT_URL")
    WATCHTOWER_VALIDATOR_HOTKEY: str = Field(env="WATCHTOWER_VALIDATOR_HOTKEY")


settings = Settings()
