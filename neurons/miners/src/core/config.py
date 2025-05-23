from typing import TYPE_CHECKING
import argparse
import pathlib

import bittensor
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from bittensor_wallet import bittensor_wallet


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    PROJECT_NAME: str = "compute-subnet-miner"

    BITTENSOR_WALLET_DIRECTORY: pathlib.Path = Field(
        env="BITTENSOR_WALLET_DIRECTORY",
        default=pathlib.Path("~").expanduser() / ".bittensor" / "wallets",
    )
    BITTENSOR_WALLET_NAME: str = Field(env="BITTENSOR_WALLET_NAME")
    BITTENSOR_WALLET_HOTKEY_NAME: str = Field(env="BITTENSOR_WALLET_HOTKEY_NAME")
    BITTENSOR_NETUID: int = Field(env="BITTENSOR_NETUID")
    BITTENSOR_CHAIN_ENDPOINT: str | None = Field(env="BITTENSOR_CHAIN_ENDPOINT", default=None)
    BITTENSOR_NETWORK: str = Field(env="BITTENSOR_NETWORK")

    SQLALCHEMY_DATABASE_URI: str = Field(env="SQLALCHEMY_DATABASE_URI")

    EXTERNAL_IP_ADDRESS: str = Field(env="EXTERNAL_IP_ADDRESS")
    INTERNAL_PORT: int = Field(env="INTERNAL_PORT", default=8000)
    EXTERNAL_PORT: int = Field(env="EXTERNAL_PORT", default=8000)
    ENV: str = Field(env="ENV", default="dev")
    DEBUG: bool = Field(env="DEBUG", default=False)

    MIN_ALPHA_STAKE: int = Field(env="MIN_ALPHA_STAKE", default=10)
    MIN_TOTAL_STAKE: int = Field(env="MIN_TOTAL_STAKE", default=20000)

    REQUIRED_TAO_COLLATERAL: float = 0.001

    COLLATERAL_CONTRACT_ADDRESSES: dict[str, str] = {
        "local": "0xc30c6Fefb37c8599aD6e048178BeD4300f067470",
        "dev": "0x8911acCB78363B3AD6D955892Ba966eb6869A2e6",
        "prod": "",
    }

    COLLATERAL_CONTRACT_NETWORK_MAP: dict[str, str] = {
        "local": "local",
        "dev": "test",
        "prod": "finney",
    }

    DEBUG_CONTRACT_MINERS: [str] = [
        "5Df8qGLMd19BXByefGCZFN57fWv6jDm5hUbnQeUTu2iqNBhT",
        "5Dtbwfafi4cyiDwH5HBFEAWJA913EB6G1rX7wBnfcXwiPssR",
        "5ECBM9caBAyJPBjVtfw4WGwdytacrZWvvt6i3T8GnqtByRFM"
    ]

    DEBUG_COLLATERAL_CONTRACT: bool = True
    
    ETHEREUM_MINER_KEY: str = Field(env="ETHEREUM_MINER_KEY", default=None)

    @property
    def COLLATERAL_CONTRACT_ADDRESS(self) -> str:
        """Returns the collateral contract address based on the current environment."""
        return self.COLLATERAL_CONTRACT_ADDRESSES.get(self.ENV, self.COLLATERAL_CONTRACT_ADDRESSES["prod"])

    @property
    def COLLATERAL_CONTRACT_NETWORK(self) -> str:
        """Returns the collateral contract address based on the current environment."""
        return self.COLLATERAL_CONTRACT_NETWORK_MAP.get(self.ENV, self.COLLATERAL_CONTRACT_NETWORK_MAP["prod"])
        
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


settings = Settings()
