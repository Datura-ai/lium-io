import argparse
import pathlib
from typing import TYPE_CHECKING

import bittensor
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from bittensor_wallet import bittensor_wallet


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
    DEBUG_MINER_HOTKEY: str | None = Field(env="DEBUG_MINER_HOTKEY", default=None)
    DEBUG_MINER_COLDKEY: str | None = Field(env="DEBUG_MINER_COLDKEY", default=None)
    DEBUG_MINER_UID: int | None = Field(env="DEBUG_MINER_UID", default=None)
    DEBUG_MINER_ADDRESS: str | None = Field(env="DEBUG_MINER_ADDRESS", default=None)
    DEBUG_MINER_PORT: int | None = Field(env="DEBUG_MINER_PORT", default=None)
    
    INTERNAL_PORT: int = Field(env="INTERNAL_PORT", default=8000)
    BLOCKS_FOR_JOB: int = 75  # 15 minutes
    JOB_TIME_OUT: int = 60 * 15  # 15 minutes

    REDIS_HOST: str = Field(env="REDIS_HOST", default="localhost")
    REDIS_PORT: int = Field(env="REDIS_PORT", default=6379)
    COMPUTE_APP_URI: str = Field(env="COMPUTE_APP_URI", default="wss://lium.io")
    COMPUTE_REST_API_URL: str | None = Field(
        env="COMPUTE_REST_API_URL", default="https://lium.io/api"
    )
    TAO_PRICE_API_URL: str = Field(env="TAO_PRICE_API_URL", default="https://api.coingecko.com/api/v3/coins/bittensor")
    COLLATERAL_DAYS: int = 7
    ENV: str = Field(env="ENV", default="dev")

    PORTION_FOR_UPTIME: float = 0.05

    PORTION_FOR_SYSBOX: float = 0.2

    TIME_DELTA_FOR_EMISSION: float = 0.01

    # Read version from version.txt
    VERSION: str = (pathlib.Path(__file__).parent / ".." / ".." / "version.txt").read_text().strip()

    BURNERS: list[int] = [4, 206, 207, 208]

    ENABLE_COLLATERAL_CONTRACT: bool = True
    ENABLE_NEW_INCENTIVE_ALGO: bool = False
    ENABLE_VERIFYX: bool = True

    COLLATERAL_CONTRACT_ADDRESS: str = Field(
        env='COLLATERAL_CONTRACT_ADDRESS', default='0xfB0FEAf1aB5d3788B40F97076ae0104bFbbdC124'
    )
    OLD_COLLATERAL_CONTRACT_ADDRESS: str | None = Field(
        env='OLD_COLLATERAL_CONTRACT_ADDRESS', default='0x999F9A49A85e9D6E981cad42f197349f50172bEB'
    )
    CONTRACT_VERSIONS: dict = {
        "1.0.0": {
            "address": "0x999F9A49A85e9D6E981cad42f197349f50172bEB",
            "info": "1st Version of the collateral contract",
        },
        "1.0.1": {
            "address": "0xfB0FEAf1aB5d3788B40F97076ae0104bFbbdC124",
            "info": "2nd version: Implemented partial slashing",
        },
        "1.0.2": {
            "address": "0x8A4023FdD1eaA7b242F3723a7d096B6CC693c7C6",
            "info": "3rd version: Fixed 'ExecutorNotOwned' error",
        },
    }

    # GPU types that will be excluded in collateral checks
    COLLATERAL_EXCLUDED_GPU_TYPES: list[str] = [
        "NVIDIA B200"
    ]

    def get_bittensor_wallet(self) -> "bittensor_wallet":
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
        if not self.DEBUG_MINER_ADDRESS or not self.DEBUG_MINER_PORT:
            raise RuntimeError("Debug miner not configured")

        miner = type("Miner", (object,), {})()
        miner.hotkey            = self.DEBUG_MINER_HOTKEY
        miner.coldkey           = self.DEBUG_MINER_COLDKEY
        miner.uid               = self.DEBUG_MINER_UID
        miner.axon_info         = type("AxonInfo", (object,), {})()
        miner.axon_info.ip      = self.DEBUG_MINER_ADDRESS
        miner.axon_info.port    = self.DEBUG_MINER_PORT
        miner.axon_info.is_serving = True
        return miner


settings = Settings()
