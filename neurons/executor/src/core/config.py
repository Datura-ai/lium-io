from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    PROJECT_NAME: str = "compute-subnet-executor"

    INTERNAL_PORT: int = Field(env="INTERNAL_PORT", default=8001)
    SSH_PORT: int = Field(env="SSH_PORT", default=2200)
    SSH_PUBLIC_PORT: Optional[int] = Field(env="SSH_PUBLIC_PORT", default=None)
    SSH_HOST_KEY_PATH: Optional[str] = Field(
        env="SSH_HOST_KEY_PATH", default="/etc/ssh/ssh_host_ed25519_key.pub"
    )

    MINER_HOTKEY_SS58_ADDRESS: str = Field(env="MINER_HOTKEY_SS58_ADDRESS")
    DEFAULT_MINER_HOTKEY: str = Field(
        env="DEFAULT_MINER_HOTKEY",
        default="5F2CpjmnJ3byqeKGNGqgL7PvJQGVeR4DYPY448TM8HS29xQb"
    )
    VALIDATOR_HOTKEY_SS58_ADDRESS: Optional[str] = Field(env="VALIDATOR_HOTKEY_SS58_ADDRESS", default=None)
    ALLOWED_HOTKEY_SS58_ADDRESS: str = Field(env="ALLOWED_HOTKEY_SS58_ADDRESS", default="5E1nK3myeWNWrmffVaH76f2mCFCbe9VcHGwgkfdcD7k3E8D1")

    RENTING_PORT_RANGE: Optional[str] = Field(env="RENTING_PORT_RANGE", default=None)
    RENTING_PORT_MAPPINGS: Optional[str] = Field(env="RENTING_PORT_MAPPINGS", default=None)

    ENV: str = Field(env="ENV", default="dev")
    ENABLE_TDX_ATTESTATION: bool = Field(env="ENABLE_TDX_ATTESTATION", default=False)
    TDX_QUOTE_TIMEOUT: int = Field(env="TDX_QUOTE_TIMEOUT", default=60)

    DB_URI: str = Field(env="DB_URI")


settings = Settings()
