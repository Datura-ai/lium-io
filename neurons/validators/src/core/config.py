import argparse
import pathlib
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Literal

import bittensor
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from bittensor import Wallet

from lium_core.shared_config import SharedConfigClient
from incentive.config import IncentiveConfig


class FeatureFlag(str, Enum):
    """Feature flag names for type-safe access."""
    VERIFYX_NETWORK_VALIDATION = "verifyx_network_validation"


class VerifyXSettings(BaseSettings):
    """VerifyX configuration - configurable thresholds for validation.

    Set via environment variables prefixed with VERIFYX_ (e.g., VERIFYX_MEMORY_MIN_TEST_GB=16).
    Use .env for local development (git-ignored).
    """
    model_config = SettingsConfigDict(env_prefix="VERIFYX_", env_file=".env", extra="ignore")

    # Memory configuration
    MEMORY_ALLOCATION_PERCENTAGE: int = Field(
        default=75,
        description="Percentage of RAM to allocate for testing"
    )
    MEMORY_MIN_TEST_GB: int = Field(
        default=8,
        description="Minimum RAM required in GB"
    )
    MEMORY_MAX_TEST_GB: int = Field(
        default=128,
        description="Maximum RAM to test in GB"
    )

    # Storage configuration
    STORAGE_MIN_AVAILABLE_GB: int = Field(
        default=100,
        description="Minimum storage space required in GB"
    )
    STORAGE_THROUGHPUT_TEST_GB: int = Field(
        default=5,
        description="Size of data to test storage throughput in GB"
    )

    # Network configuration
    NETWORK_TIMEOUT_SECONDS: int = Field(
        default=120,
        description="Timeout for network tests in seconds"
    )
    NETWORK_MIN_DOWNLOAD_SPEED_MBPS: float = Field(
        default=50.0,
        description="Minimum required network download speed in Mbps"
    )


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

    SKIP_STORAGE_CHECK: bool = Field(default=False, description="Skip storage check")


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

    INTERNAL_PORT: int = Field(env="INTERNAL_PORT", default=8000)
    BLOCKS_FOR_JOB: int = 75  # 15 minutes
    JOB_TIME_OUT: int = 60 * 15  # 15 minutes

    REDIS_HOST: str = Field(env="REDIS_HOST", default="localhost")
    REDIS_PORT: int = Field(env="REDIS_PORT", default=6379)
    REDIS_USERNAME: str | None = Field(env="REDIS_USERNAME", default=None)
    REDIS_PASSWORD: str | None = Field(env="REDIS_PASSWORD", default=None)
    COMPUTE_APP_URI: str = Field(env="COMPUTE_APP_URI", default="wss://lium.io/api")
    COMPUTE_REST_API_URL: str | None = Field(
        env="COMPUTE_REST_API_URL", default="https://lium.io/api"
    )
    COMPUTE_REST_API_URL_EXTERNAL: str | None = Field(
        env="COMPUTE_REST_API_URL_EXTERNAL", default="https://lium.io/api"
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
    PORTION_FOR_SYSBOX_UNRENTED: float = 1
    PORTION_FOR_SYSBOX_RENTED: float = 1
    SYSBOX_RENTED_CUTOFF: datetime = datetime(2026, 4, 3, 12, 0, 0)
    DISCORD_INCENTIVE_CUTOFF: datetime = datetime(2026, 6, 15, 12, 0, 0)

    # Minimum NVIDIA driver requirement. Compared as a dotted version tuple against the
    # executor's reported gpu.driver (e.g. "580.95.05"). 580.65.06 is the r580 floor that
    # ships CUDA 13.0. Before MIN_DRIVER_CUTOFF providers get a grace period (no penalty);
    # on/after the cutoff, non-compliant unrented executors are fully gated (multiplier 0).
    # Currently-rented executors are always exempt.
    MIN_NVIDIA_DRIVER_VERSION: str = Field(env="MIN_NVIDIA_DRIVER_VERSION", default="580.65.06")
    MIN_DRIVER_CUTOFF: datetime = datetime(2026, 6, 30, 12, 0, 0)

    TIME_DELTA_FOR_EMISSION: float = 0.01

    # Read version from version.txt
    VERSION: str = (pathlib.Path(__file__).parent / ".." / ".." / "version.txt").read_text().strip()

    BURNERS: list[int] = [4, 206, 207, 208]
    # Duplicate entries are slot weights: UID 47 holds the aggregated share of 5 retired burners.
    NEW_BURNERS: list[int] = [187, 188, 189, 190, 191, 47, 47, 47, 47, 47]
    ENABLE_NEW_BURN_LOGIC: bool = True

    ENABLE_NO_COLLATERAL: bool = True
    ENABLE_VERIFYX: bool = True
    ENABLE_INSPECTOR: bool = False
    INSPECTOR_ENSURE_COLLECTOR_ON_RENTED_CHECK: bool = Field(
        env="INSPECTOR_ENSURE_COLLECTOR_ON_RENTED_CHECK",
        default=False,
    )
    SKIP_RENTAL_VERIFICATION: bool = Field(env="SKIP_RENTAL_VERIFICATION", default=False)
    SKIP_COLLATERAL_PENALTY: bool = Field(env="SKIP_COLLATERAL_PENALTY", default=True)
    DRY_RUN: bool = Field(env="DRY_RUN", default=False, description="Run validation without publishing scores/weights")
    CONTAINER_CLEANUP_DRY_RUN: bool = Field(env="CONTAINER_CLEANUP_DRY_RUN", default=False, description="Dry run mode for stale container cleanup")

    # DAH-2272: raise the asyncssh logger to DEBUG (debug level 2) so the SSH
    # handshake (banner / key exchange / auth) is logged per connection. Enabled
    # by default while DAH-2272 is under investigation, so the handshake trace is
    # captured whenever the rare slow-connect transient hits; set
    # SSH_DEBUG_LOGGING=false (env) to silence it without a deploy. Lives on the
    # main settings (not DebugSettings, which is local-dev only) so it is
    # configurable in staging/prod where the real slow connects happen.
    SSH_DEBUG_LOGGING: bool = Field(env="SSH_DEBUG_LOGGING", default=True, description="Enable verbose asyncssh SSH handshake debug logging")

    # DAH-2250 — unrented incentive soft price limit. When True, an unrented executor
    # whose price_per_gpu exceeds market p90 * SOFT_LIMIT_PRICE_RATE loses the unrented
    # rental incentive while staying active. When False, the breach is only logged
    # (shadow mode) so prod impact can be observed before enforcing.
    ENABLE_UNRENTED_SOFT_PRICE_LIMIT: bool = Field(env="ENABLE_UNRENTED_SOFT_PRICE_LIMIT", default=False)

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
    
    # TDX Attestation settings
    ENABLE_TDX_ATTESTATION: bool = Field(env="ENABLE_TDX_ATTESTATION", default=False)
    TDX_VERIFIER_URL: str | None = Field(env="TDX_VERIFIER_URL", default=None)
    ENABLE_ATTESTATION_WHITELIST: bool = Field(env="ENABLE_ATTESTATION_WHITELIST", default=False)

    # Use REST API instead of WebSocket for miner communication
    USE_REST_API: bool = Field(env="USE_REST_API", default=False)

    # DAH-2211 — custom-dockerfile pod build tunables (validator side).
    # These mirror the spec keys `features.custom_dockerfile_pod.*`; the route
    # is authoritative for the size cap but the validator double-checks it as
    # defense-in-depth.
    CUSTOM_DOCKERFILE_BUILD_TIMEOUT_SECONDS: int = Field(
        env="CUSTOM_DOCKERFILE_BUILD_TIMEOUT_SECONDS", default=1200,
        description="Hard timeout for `docker build` of a custom-dockerfile pod (seconds).",
    )
    CUSTOM_DOCKERFILE_MAX_BYTES: int = Field(
        env="CUSTOM_DOCKERFILE_MAX_BYTES", default=65_536,
        description="Defense-in-depth cap on dockerfile_content size; route is authoritative.",
    )
    # DAH-2211 — internet-enabled custom builds run inside a throwaway sysbox
    # Docker-in-Docker container (NOT on the host daemon) so the build can
    # `--network` out for deps while a build-time escape stays confined to an
    # unprivileged-on-host (user-namespaced) container. Egress is firewalled
    # host-side to block cloud metadata + RFC1918. See the DAH-2211 build flow.
    CUSTOM_DOCKERFILE_DIND_IMAGE: str = Field(
        env="CUSTOM_DOCKERFILE_DIND_IMAGE", default="daturaai/dind:0.0.1",
        description="Sysbox DinD image used to build custom-dockerfile pods in isolation.",
    )
    CUSTOM_DOCKERFILE_DIND_CPUS: str = Field(
        env="CUSTOM_DOCKERFILE_DIND_CPUS", default="4",
        description="--cpus limit for the throwaway DinD build container.",
    )
    CUSTOM_DOCKERFILE_DIND_MEMORY: str = Field(
        env="CUSTOM_DOCKERFILE_DIND_MEMORY", default="8g",
        description="--memory limit for the throwaway DinD build container.",
    )
    CUSTOM_DOCKERFILE_DIND_READY_TIMEOUT_SECONDS: int = Field(
        env="CUSTOM_DOCKERFILE_DIND_READY_TIMEOUT_SECONDS", default=60,
        description="Max seconds to wait for the inner DinD dockerd to become ready.",
    )
    CUSTOM_DOCKERFILE_EGRESS_BLOCK_CIDRS: str = Field(
        env="CUSTOM_DOCKERFILE_EGRESS_BLOCK_CIDRS",
        default="169.254.0.0/16,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16",
        description=(
            "Comma-separated CIDRs the custom build must NOT reach (cloud metadata + "
            "RFC1918). Enforced host-side in the DOCKER-USER chain, scoped to the DinD "
            "container IP. Build still has full public-internet egress."
        ),
    )

    DEPLOY_ENV: Literal["PROD", "LOCAL", "STAGE"] = Field(env="DEPLOY_ENV", default="PROD")

    debug: DebugSettings = Field(default_factory=DebugSettings)
    verifyx: VerifyXSettings = Field(default_factory=VerifyXSettings)
    incentive: IncentiveConfig = Field(default_factory=IncentiveConfig)

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

    def get_redis_connection_url(self) -> str:
        if self.REDIS_USERNAME and self.REDIS_PASSWORD:
            return f"redis://{self.REDIS_USERNAME}:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{int(self.REDIS_PORT)}"
        return f"redis://{self.REDIS_HOST}:{int(self.REDIS_PORT)}"

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
shared_client = SharedConfigClient(
    api_url=f"{settings.COMPUTE_REST_API_URL}/v1/shared-config"
)
