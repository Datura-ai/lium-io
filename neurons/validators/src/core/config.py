import argparse
import pathlib
from enum import Enum
from typing import TYPE_CHECKING

import bittensor
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from bittensor import Wallet


class FeatureFlag(str, Enum):
    """Feature flag names for type-safe access."""
    VERIFYX_NETWORK_VALIDATION = "verifyx_network_validation"


class DebugSettings(BaseSettings):
    """Debug configuration - all flags default to False/None.
    
    Set via environment variables prefixed with DEBUG_ (e.g., DEBUG_SKIP_STAKE_CHECKS=true).
    Use .env for local development (git-ignored).
    """
    model_config = SettingsConfigDict(env_prefix="DEBUG_", env_file=".env", extra="ignore")

    ENABLED: bool = Field(default=False, description="Enable debug mode")
    USE_LOCAL_MINER: bool = Field(default=False, description="Use local miner")
    MINER_HOTKEY: str | None = Field(default=None, description="Miner hotkey")
    MINER_COLDKEY: str | None = Field(default=None, description="Miner coldkey")
    MINER_UID: int | None = Field(default=None, description="Miner UID")
    MINER_ADDRESS: str | None = Field(default=None, description="Miner address")
    MINER_PORT: int | None = Field(default=None, description="Miner port")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    PROJECT_NAME: str = "compute-subnet-validator"

    BITTENSOR_WALLET_DIRECTORY: pathlib.Path = Field(
        env="BITTENSOR_WALLET_DIRECTORY",
        default=pathlib.Path("~").expanduser() / ".bittensor" / "wallets",
    )
    BITTENSOR_WALLET_NAME: str = Field(env="BITTENSOR_WALLET_NAME")
    BITTENSOR_WALLET_HOTKEY_NAME: str = Field(env="BITTENSOR_WALLET_HOTKEY_NAME")
    BITTENSOR_NETUID: int = Field(env="BITTENSOR_NETUID", default=51)
    BITTENSOR_CHAIN_ENDPOINT: str | None = Field(env="BITTENSOR_CHAIN_ENDPOINT", default=None)
    BITTENSOR_NETWORK: str = Field(env="BITTENSOR_NETWORK", default="finney")
    SUBTENSOR_EVM_RPC_URL: str | None = Field(env="SUBTENSOR_EVM_RPC_URL", default=None)

    SQLALCHEMY_DATABASE_URI: str = Field(env="SQLALCHEMY_DATABASE_URI")
    ASYNC_SQLALCHEMY_DATABASE_URI: str = Field(env="ASYNC_SQLALCHEMY_DATABASE_URI")

    DEBUG: bool = Field(env="DEBUG", default=False)

    INTERNAL_PORT: int = Field(env="INTERNAL_PORT", default=8000)
    BLOCKS_FOR_JOB: int = 75  # 15 minutes
    JOB_TIME_OUT: int = 60 * 15  # 15 minutes

    REDIS_HOST: str = Field(env="REDIS_HOST", default="localhost")
    REDIS_PORT: int = Field(env="REDIS_PORT", default=6379)
    COMPUTE_APP_URI: str = Field(env="COMPUTE_APP_URI", default="wss://lium.io")
    COMPUTE_REST_API_URL: str | None = Field(
        env="COMPUTE_REST_API_URL", default="https://lium.io/api"
    )
    MINER_PORTAL_REST_API_URL: str = Field(
        env="MINER_PORTAL_REST_API_URL", default="https://provider-api.lium.io/"
    )
    TAO_PRICE_API_URL: str = Field(env="TAO_PRICE_API_URL", default="https://api.coingecko.com/api/v3/coins/bittensor")
    COLLATERAL_DAYS: int = 7
    ENV: str = Field(env="ENV", default="dev")

    PORTION_FOR_UPTIME: float = 1
    UPTIME_REQUIRED_MINUTES: int = 60 * 24 * 14 # 14 days

    PORTION_FOR_SYSBOX: float = 0.2

    TIME_DELTA_FOR_EMISSION: float = 0.01

    # Read version from version.txt
    VERSION: str = (pathlib.Path(__file__).parent / ".." / ".." / "version.txt").read_text().strip()

    BURNERS: list[int] = [4, 206, 207, 208]
    NEW_BURNERS: list[int] = [187, 188, 189, 190, 191, 192, 193, 194, 195, 196]
    ENABLE_NEW_BURN_LOGIC: bool = True

    ENABLE_NO_COLLATERAL: bool = True
    ENABLE_VERIFYX: bool = True
    SKIP_RENTAL_VERIFICATION: bool = Field(env="SKIP_RENTAL_VERIFICATION", default=False)

    COLLATERAL_CONTRACT_ADDRESS: str = Field(
        env='COLLATERAL_CONTRACT_ADDRESS', default='0x8A4023FdD1eaA7b242F3723a7d096B6CC693c7C6'
    )
    CONTRACT_VERSIONS: dict = {
        "1.0.2": {
            "address": "0x8A4023FdD1eaA7b242F3723a7d096B6CC693c7C6",
            "info": "3rd version: Fixed 'ExecutorNotOwned' error",
        },
    }
    FEATURE_FLAGS: dict[str, bool] = {
        FeatureFlag.VERIFYX_NETWORK_VALIDATION: False,  # If it's True - then bad internet connection will raise error on synthetic job
    }

    # GPU types that will be excluded in collateral checks
    COLLATERAL_EXCLUDED_GPU_TYPES: list[str] = [
        "NVIDIA B200"
    ]

    debug: DebugSettings = Field(default_factory=DebugSettings)

    def get_bittensor_wallet(self) -> "Wallet":
        if not self.BITTENSOR_WALLET_NAME or not self.BITTENSOR_WALLET_HOTKEY_NAME:
            raise RuntimeError("Wallet not configured")
        wallet = bittensor.wallet(
            name=self.BITTENSOR_WALLET_NAME,
            hotkey=self.BITTENSOR_WALLET_HOTKEY_NAME,
            path=str(self.BITTENSOR_WALLET_DIRECTORY),
        )
        wallet.hotkey_file.get_keypair()  # this raises errors if the keys are inaccessible
        return wallet

    def get_latest_contract_version(self) -> str:
        return max(self.CONTRACT_VERSIONS.keys())

    def get_bittensor_config(self) -> bittensor.config:
        parser = argparse.ArgumentParser()
        # bittensor.wallet.add_args(parser)
        # bittensor.subtensor.add_args(parser)
        # bittensor.axon.add_args(parser)

        if self.BITTENSOR_NETWORK:
            if "--subtensor.network" in parser._option_string_actions:
                parser._handle_conflict_resolve(
                    None,
                    [("--subtensor.network", parser._option_string_actions["--subtensor.network"])],
                )

            parser.add_argument(
                "--subtensor.network",
                type=str,
                help="network",
                default=self.BITTENSOR_NETWORK,
            )

        if self.BITTENSOR_CHAIN_ENDPOINT:
            if "--subtensor.chain_endpoint" in parser._option_string_actions:
                parser._handle_conflict_resolve(
                    None,
                    [
                        (
                            "--subtensor.chain_endpoint",
                            parser._option_string_actions["--subtensor.chain_endpoint"],
                        )
                    ],
                )

            parser.add_argument(
                "--subtensor.chain_endpoint",
                type=str,
                help="chain endpoint",
                default=self.BITTENSOR_CHAIN_ENDPOINT,
            )

        return bittensor.config(parser)

    def get_debug_miner(self) -> dict:
        if not self.debug.MINER_ADDRESS or not self.debug.MINER_PORT:
            raise RuntimeError("Debug miner not configured")

        miner = type("Miner", (object,), {})()
        miner.hotkey            = self.debug.MINER_HOTKEY
        miner.coldkey           = self.debug.MINER_COLDKEY
        miner.uid               = self.debug.MINER_UID
        miner.axon_info         = type("AxonInfo", (object,), {})()
        miner.axon_info.ip      = self.debug.MINER_ADDRESS
        miner.axon_info.port    = self.debug.MINER_PORT
        miner.axon_info.is_serving = True
        return miner


settings = Settings()
