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

    RENTING_PORT_RANGE: Optional[str] = Field(env="RENTING_PORT_RANGE", default=None)
    RENTING_PORT_MAPPINGS: Optional[str] = Field(env="RENTING_PORT_MAPPINGS", default=None)

    ENV: str = Field(env="ENV", default="dev")

    DB_URI: str = Field(env="DB_URI")

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


settings = Settings()
