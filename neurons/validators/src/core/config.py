import pathlib
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Literal

import bittensor
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from bittensor import Wallet

from incentive.config import IncentiveConfig
from lium_core.shared_config import DEFAULT_SHARED_CONFIG, SharedConfigClient


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
    FORCED_VALIDATION_INTERNAL_TOKEN: str | None = Field(
        env="FORCED_VALIDATION_INTERNAL_TOKEN", default=None
    )

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
    # DAH-2313: reject unrented executors without sysbox so they never appear on the network.
    REQUIRE_SYSBOX_FOR_UNRENTED: bool = Field(env="REQUIRE_SYSBOX_FOR_UNRENTED", default=True)
    DISCORD_INCENTIVE_CUTOFF: datetime = datetime(2026, 6, 15, 12, 0, 0)

    # DAH-2265: cached-template requirement. Before the cutoff the CachedTemplateVerificationCheck
    # is purely advisory (publishes recommended_image_cached / recommended_image_digest_match to
    # executor.specs, never changes score). On/after the cutoff that check turns critical: an
    # unrented executor that has not pre-pulled the recommended default image, or holds stale
    # content under the same tag, fails verification early (score 0, reason surfaced to the
    # provider). Fails open: an unmeasured signal (None) is never penalised.
    # The cutoff has elapsed and CachedTemplateVerificationCheck is fatal again, so the gate is
    # live: unrented executors are enforced, rented ones are exempt (the check runs after the
    # tenant short-circuit).
    CACHED_TEMPLATE_CUTOFF: datetime = datetime(2026, 7, 14, 12, 0, 0)

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
    # All burn emission slots route to UID 47 (Datura payout key).
    NEW_BURNERS: list[int] = [47, 47, 47, 47, 47, 47, 47, 47, 47, 47]
    # Expected coldkeys for burner UIDs. Burn weights are withheld when coldkey mismatches.
    BURNER_COLDKEYS: dict[int, str] = {
        47: "5G694c15wAu1LKi9rpSQqJjpBfg4K1oiBxEm5QSVdVZAfp9f",
    }
    ENABLE_NEW_BURN_LOGIC: bool = True

    ENABLE_NO_COLLATERAL: bool = True
    ENABLE_VERIFYX: bool = True
    ENABLE_INSPECTOR: bool = True
    INSPECTOR_ENSURE_COLLECTOR_ON_RENTED_CHECK: bool = Field(
        env="INSPECTOR_ENSURE_COLLECTOR_ON_RENTED_CHECK",
        default=True,
    )
    SKIP_RENTAL_VERIFICATION: bool = Field(env="SKIP_RENTAL_VERIFICATION", default=False)
    # ISSUE-050 filler liveness. CHECK_ENABLED is the master switch: shadow mode runs the SSH
    # probe + backend re-check and logs the verdict, but never withholds incentive; switching it
    # off disables the probe entirely. ENFORCEMENT (only effective while CHECK_ENABLED is on)
    # additionally fails the fatal check — no unrented incentive — when a host killed the Lium
    # filler container. Rollout: shadow first, flip enforcement after the shadow data is clean.
    FILLER_LIVENESS_CHECK_ENABLED: bool = Field(env="FILLER_LIVENESS_CHECK_ENABLED", default=True)
    FILLER_LIVENESS_ENFORCEMENT_ENABLED: bool = Field(env="FILLER_LIVENESS_ENFORCEMENT_ENABLED", default=False)
    SKIP_COLLATERAL_PENALTY: bool = Field(env="SKIP_COLLATERAL_PENALTY", default=True)
    DRY_RUN: bool = Field(env="DRY_RUN", default=False, description="Run validation without publishing scores/weights")
    CONTAINER_CLEANUP_DRY_RUN: bool = Field(env="CONTAINER_CLEANUP_DRY_RUN", default=False, description="Dry run mode for stale container cleanup")
    DUPLICATE_EXECUTOR_DRY_RUN: bool = Field(env="DUPLICATE_EXECUTOR_DRY_RUN", default=True, description="Observe mode: detect duplicate executors but don't penalize")

    # DAH-2272: when on, raise the asyncssh logger to DEBUG (debug level 2) so the
    # SSH handshake (banner / key exchange / auth) is logged per connection, and
    # emit the per-connect ``ssh_connect_phase_timing`` breakdown (see
    # ``services.ssh_connect_timing``). Off by default now that the slow-connect
    # investigation is closed — this instrumentation is kept in the tree so it can
    # be re-enabled without a code change by setting SSH_DEBUG_LOGGING=true (env)
    # for a future investigation. Lives on the main settings (not DebugSettings,
    # which is local-dev only) so it is configurable in staging/prod where the
    # real slow connects happen.
    SSH_DEBUG_LOGGING: bool = Field(env="SSH_DEBUG_LOGGING", default=False, description="Enable verbose asyncssh SSH handshake debug logging and per-connect phase timing")

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

    # G4 — TCB/advisory enforcement (CVM attestation gap remediation).
    # Off = warn-only telemetry: violations are logged with structured reasons but do
    # not reject the quote. On = fail closed with AttestationError naming the value.
    # Also gates the minimal-G5 hooks: fail-closed on an omitted quote from a
    # previously-attested (CVM-ratcheted) executor, and the CVM score gate.
    ENABLE_TCB_ENFORCEMENT: bool = Field(env="ENABLE_TCB_ENFORCEMENT", default=False)
    # Comma-separated allowlists. TCB statuses other than these reject (when enforced);
    # advisory IDs (e.g. INTEL-SA-XXXXX) not listed here reject (when enforced).
    TDX_ALLOWED_TCB_STATUSES: str = Field(env="TDX_ALLOWED_TCB_STATUSES", default="UpToDate")
    TDX_ALLOWED_ADVISORY_IDS: str = Field(env="TDX_ALLOWED_ADVISORY_IDS", default="")

    # G3 — freshness/anti-replay nonce channel. When enabled the validator mints a
    # 32-byte challenge per attestation event, folds it into the signed SSH-key
    # request, and requires the executor to echo it in TDX report_data[32:64].
    ENABLE_ATTESTATION_NONCE: bool = Field(env="ENABLE_ATTESTATION_NONCE", default=False)
    # A nonce-bound attestation event is required at most once per this interval per
    # miner (freshness = "bounded, provably recent", not "every wave") — GPU evidence
    # collection is too expensive to force fleet-wide every cycle.
    TDX_REATTEST_INTERVAL_SECONDS: int = Field(env="TDX_REATTEST_INTERVAL_SECONDS", default=3600)
    # Issued nonce validity window; a quote echoing an older nonce is stale.
    ATTESTATION_NONCE_TTL_SECONDS: int = Field(env="ATTESTATION_NONCE_TTL_SECONDS", default=600)

    # G1 — NVIDIA GPU confidential-compute attestation (validator-side verification).
    # Off = observe-only: nvidia_payload, when present, is verified and logged but
    # never affects the outcome. On = a CVM executor (verified TDX quote) must supply
    # GPU evidence bound to the issued nonce; any failure raises AttestationError.
    ENABLE_GPU_ATTESTATION_ENFORCEMENT: bool = Field(
        env="ENABLE_GPU_ATTESTATION_ENFORCEMENT", default=False
    )
    # Hosted NRAS endpoint (interim verification locus; a local nv-verifier removes
    # this outbound SPOF before fleet enforcement — see remediation plan CD4).
    NVIDIA_NRAS_URL: str = Field(
        env="NVIDIA_NRAS_URL", default="https://nras.attestation.nvidia.com/v3/attest/gpu"
    )
    # Optional comma-separated allowlist of NRAS `hwmodel` claims (e.g. "GH100 A01 GSP BROM").
    # Empty = no hwmodel constraint. Substring-prefix match per entry.
    GPU_ATTESTATION_ALLOWED_HWMODELS: str = Field(env="GPU_ATTESTATION_ALLOWED_HWMODELS", default="")

    # N3 — rental host-key policy. Off preserves today's behavior (a non-attestation
    # error while building the policy falls back to unpinned SSH). On = fail closed:
    # no host-key policy ⇒ no rental SSH.
    ENABLE_RENTAL_HOSTKEY_FAIL_CLOSED: bool = Field(
        env="ENABLE_RENTAL_HOSTKEY_FAIL_CLOSED", default=False
    )

    # G2 — monotonic version floor for the measured compose (runner) releases.
    # Whitelisted compose hashes carry a version; hashes below the floor are rejected
    # ("≥ approved version", newest-wins — never a rolling window). 0 accepts every
    # whitelisted hash (today's behavior).
    TDX_MINIMUM_COMPOSE_VERSION: int = Field(env="TDX_MINIMUM_COMPOSE_VERSION", default=0)

    def get_allowed_tcb_statuses(self) -> set[str]:
        return {s.strip() for s in self.TDX_ALLOWED_TCB_STATUSES.split(",") if s.strip()}

    def get_allowed_advisory_ids(self) -> set[str]:
        return {s.strip() for s in self.TDX_ALLOWED_ADVISORY_IDS.split(",") if s.strip()}

    def get_allowed_gpu_hwmodels(self) -> set[str]:
        return {s.strip() for s in self.GPU_ATTESTATION_ALLOWED_HWMODELS.split(",") if s.strip()}

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
        wallet = bittensor.Wallet(
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

    def get_bittensor_config(self) -> bittensor.Config:
        # bittensor >=10.3.2 ignores argparse defaults (BT_NO_PARSE_CLI_ARGS is on by
        # default), so assign values directly or Subtensor silently falls back to finney
        config = bittensor.Config()

        if self.BITTENSOR_NETWORK:
            config.subtensor.network = self.BITTENSOR_NETWORK

        if self.BITTENSOR_CHAIN_ENDPOINT:
            config.subtensor.chain_endpoint = self.BITTENSOR_CHAIN_ENDPOINT

        return config

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


def get_total_burn_emission() -> float:
    """Burn-emission share from shared config, guarded for on-chain use (DAH-2274).

    ``SharedConfigClient`` already falls back to the packaged lium-core default when
    the backend is unreachable, so ``shared_client.config.total_burn_emission`` is
    always a valid float. The only extra guard is rejecting a structurally valid but
    out-of-range served value (e.g. an operator typo) before it reaches on-chain
    weights, falling back to the lium-core default.
    """
    value = shared_client.config.total_burn_emission
    if 0.0 <= value <= 1.0:
        return value

    # Lazy import avoids a circular dependency (core.utils imports settings).
    from core.utils import _m, get_logger

    get_logger(__name__).warning(
        _m(
            "total_burn_emission out of range; using lium-core default",
            extra={"served_value": value, "fallback": DEFAULT_SHARED_CONFIG.total_burn_emission},
        )
    )
    return DEFAULT_SHARED_CONFIG.total_burn_emission
