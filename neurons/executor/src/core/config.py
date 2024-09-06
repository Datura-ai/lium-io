from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict()
    PROJECT_NAME: str = "compute-subnet-executor"

    IP_ADDRESS: str = Field(env="IP_ADDRESS")
    PORT: int = Field(env="PORT", default=8001)
    SSH_PORT: int = Field(env="SSH_PORT", default=22)

    MINER_IP_ADDRESS: str = Field(env="MINER_IP_ADDRESS")
    MINER_HOTKEY_SS58_ADDRESS: str = Field(env="MINER_HOTKEY_SS58_ADDRESS")

    class Config:
        env_file = ".env"


settings = Settings()
