from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# This hotkey is used to verify the validator signature. 
# This shouldn't be overridden by the environment variable. 
VALIDATOR_HOTKEY_SS58 = "5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p"
try:
    from core.config_override import _VALIDATOR_HOTKEY_SS58 
    VALIDATOR_HOTKEY_SS58 = _VALIDATOR_HOTKEY_SS58
except Exception as e:
    pass

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
    ALLOWED_HOTKEY_SS58_ADDRESS: str = Field(env="ALLOWED_HOTKEY_SS58_ADDRESS", default="5E1nK3myeWNWrmffVaH76f2mCFCbe9VcHGwgkfdcD7k3E8D1")

    RENTING_PORT_RANGE: Optional[str] = Field(env="RENTING_PORT_RANGE", default=None)
    RENTING_PORT_MAPPINGS: Optional[str] = Field(env="RENTING_PORT_MAPPINGS", default=None)

    CHUTES_BRIDGE_ENABLED: bool = Field(env="CHUTES_BRIDGE_ENABLED", default=False)
    CHUTES_BRIDGE_SSH_HOST: Optional[str] = Field(env="CHUTES_BRIDGE_SSH_HOST", default=None)
    CHUTES_BRIDGE_SSH_PORT: int = Field(env="CHUTES_BRIDGE_SSH_PORT", default=22)
    CHUTES_BRIDGE_SSH_USER: str = Field(env="CHUTES_BRIDGE_SSH_USER", default="lium-bridge")
    CHUTES_BRIDGE_SSH_KEY_PATH: str = Field(
        env="CHUTES_BRIDGE_SSH_KEY_PATH",
        default="/run/secrets/chutes_bridge_key",
    )
    CHUTES_BRIDGE_CONNECT_TIMEOUT_SEC: int = Field(
        env="CHUTES_BRIDGE_CONNECT_TIMEOUT_SEC",
        default=10,
    )
    CHUTES_BRIDGE_COMMAND_TIMEOUT_SEC: int = Field(
        env="CHUTES_BRIDGE_COMMAND_TIMEOUT_SEC",
        default=300,
    )

    ENV: str = Field(env="ENV", default="dev")

    DB_URI: str = Field(env="DB_URI")

    ENABLE_TDX_ATTESTATION: bool = Field(env="ENABLE_TDX_ATTESTATION", default=False)
    TDX_QUOTE_TIMEOUT: int = Field(env="TDX_QUOTE_TIMEOUT", default=60)
    SSH_HOST_KEY_PATH: str = Field(env="SSH_HOST_KEY_PATH", default="/etc/ssh/ssh_host_ed25519_key.pub")


settings = Settings()
