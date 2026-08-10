import base64
import binascii
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Annotated, Any

import aiohttp
import asyncssh
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends

from core.config import settings
from core.utils import _m, get_extra_info
from services.const import TDX_WHITELIST
from services.cvm_whitelist import CvmWhitelistSource
from services.redis_service import RedisService

logger = logging.getLogger(__name__)

# Redis set of executor UUIDs that have ever presented a valid TDX quote.
# Minimal-G5 ratchet: once an executor is known-CVM, an omitted quote is treated
# as a bypass attempt (fail-closed under ENABLE_TCB_ENFORCEMENT), not as a
# bare-metal node.
TDX_ATTESTED_EXECUTOR_SET = "tdx_attested_executors"

# Redis key prefix recording the last nonce-bound attestation event per miner,
# used to bound re-attestation cadence (G3: freshness = provably recent, not
# every wave).
TDX_LAST_ATTESTATION_EVENT_PREFIX = "tdx_last_attestation_event"

# The separator the in-CVM agent joins GPU UUIDs with before digesting them (DAH-2579,
# attest-agent/identity.py). A character no NVIDIA UUID can contain, so no two different GPU
# sets can produce the same joined string. Reproduced here rather than imported: the agent is a
# separate package that does not ship with the validator, and the value is part of the wire
# contract, so a copy that drifts must fail a test rather than silently accept a wrong set.
GPU_UUID_SEPARATOR = ","


class AttestationError(RuntimeError):
    """Raised when attestation or host verification fails."""


@dataclass
class AttestationNonce:
    """Validator-minted challenge binding one attestation event (G3 + G1).

    The hex value rides inside the signed SSHPubKeySubmitRequest, is folded by
    the executor into TDX report_data[32:64] and into the NVIDIA GPU-evidence
    nonce, and is asserted on the way back. One nonce is minted per attestation
    event (a validation request to one miner) and fans out to that miner's
    executors; per-executor uniqueness comes from the identity half of
    report_data, freshness from the nonce half.
    """

    value_hex: str
    issued_at: float = field(default_factory=time.time)

    @classmethod
    def issue(cls) -> "AttestationNonce":
        return cls(value_hex=secrets.token_hex(32))

    @property
    def value_bytes(self) -> bytes:
        return bytes.fromhex(self.value_hex)

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.issued_at) > ttl_seconds


@dataclass
class HostPolicyResult:
    """Outcome of prepare_host_policy.

    attestation_passed enforces the G1 invariant: it requires the TDX digest AND
    that GPU verification did not fail (a verified-bad GPU can never surface as
    passed; None = GPU verification not performed).
    """

    known_hosts: asyncssh.SSHKnownHosts | None = None
    attestation_digest: str | None = None
    tee_type: str | None = None
    gpu_attestation_passed: bool | None = None
    # DAH-2581 — which report_data recipe the quote satisfied, so a rollout can be watched
    # rather than guessed at: "how much of the fleet still answers v1" is the question that
    # decides when v1 can be dropped from the accepted list.
    report_data_recipe: str | None = None
    # DAH-2582 — the hardware-bound identities this attestation produced. Every one of them is
    # signed by something the node does not control, which is what makes a collision between
    # two executors provable rather than merely suspicious.
    instance_id: str | None = None
    device_id: str | None = None
    gpu_ueids: list[str] = field(default_factory=list)

    @property
    def attestation_passed(self) -> bool:
        return self.attestation_digest is not None and self.gpu_attestation_passed is not False

    def identities(self) -> dict[str, list[str]]:
        """The identities the uniqueness check compares across the fleet, by class.

        Grouped by class rather than flattened so a collision names *what* collided — one GPU
        answering for two executors is a different fraud from one CVM doing so, and the message
        an operator reads should say which.
        """
        return {
            "cvm_instance_id": [self.instance_id] if self.instance_id else [],
            "cvm_device_id": [self.device_id] if self.device_id else [],
            "gpu_ueid": list(self.gpu_ueids),
        }


class AttestationService:
    """
    Validates executor authenticity for Intel TDX deployments by verifying a pre-generated
    quote supplied alongside the executor's SSH host key.

    If attestation is disabled (default), this service becomes a no-op and simply returns
    the host key policy when possible.
    """

    REPORT_PREFIX = b"SSH_HOST_KEY:"

    def __init__(
        self,
        redis_service: Annotated[RedisService | None, Depends(RedisService)] = None,
    ) -> None:
        self.enabled: bool = bool(
            settings.ENABLE_TDX_ATTESTATION and settings.TDX_VERIFIER_URL)
        self.verifier_url: str | None = settings.TDX_VERIFIER_URL
        self.quote_timeout: int = 60
        # Injected by FastAPI DI or passed explicitly (ioc / validator core).
        # Used for the minimal-G5 CVM ratchet and the re-attest cadence; both
        # degrade gracefully (log-only) when Redis is absent (e.g. unit tests).
        self.redis_service = redis_service
        # DAH-2581 — the platform's signed catalog, refreshed on a timer and held in memory.
        # One per service instance rather than a module global so a test can substitute one
        # without leaking a manifest into every other test in the process.
        self.whitelist_source = CvmWhitelistSource()
        # DAH-2582 — the GPU ueids decoded from the last evidence set this service checked.
        # Held on the instance because `_check_gpu_claims` is where the claims are already
        # decoded and re-decoding them elsewhere would be a second place to keep in step. Read
        # immediately by `prepare_host_policy`, never across executors.
        self._last_gpu_ueids: list[str] = []

    def _should_verify(self, executor: ExecutorSSHInfo) -> bool:
        return self.enabled and bool(executor.tdx_quote)

    # ------------------------------------------------------------------
    # G3 — attestation-event cadence + nonce issuance
    # ------------------------------------------------------------------

    @property
    def nonce_enabled(self) -> bool:
        return self.enabled and settings.ENABLE_ATTESTATION_NONCE

    async def maybe_issue_nonce(
        self, miner_hotkey: str, *, rented: bool = False
    ) -> AttestationNonce | None:
        """Mint a fresh attestation nonce when this miner is due for a
        nonce-bound attestation event.

        Freshness is bounded by an interval rather than demanded every validation
        wave — GPU evidence collection is seconds per node, so the cadence is
        load-shed (see remediation plan §4). The event timestamp is recorded at
        issuance so a persistently failing executor cannot force per-wave
        collection. Without Redis the interval cannot be tracked, so a nonce is
        issued every wave (safe, just not load-shed).

        DAH-2581: a miner holding a rented CVM is re-attested on the far tighter
        TDX_RENTAL_REATTEST_INTERVAL_SECONDS (900 s). A rental is exactly when a
        node has something to gain from swapping the stack underneath a proof it
        gave once at launch, and it is also when the customer is relying on that
        proof — so the two cadences are sized for two different risks rather than
        one being a compromise between them.

        The cadence is per miner and the *tighter* one wins, because the key is
        shared: a miner with one rented executor and ten idle ones gets the rental
        cadence for all of them. That is the safe direction — it over-attests idle
        nodes rather than under-attesting a rented one.
        """
        if not self.nonce_enabled:
            return None

        interval = (
            settings.TDX_RENTAL_REATTEST_INTERVAL_SECONDS
            if rented
            else settings.TDX_REATTEST_INTERVAL_SECONDS
        )

        if self.redis_service is not None:
            key = f"{TDX_LAST_ATTESTATION_EVENT_PREFIX}:{miner_hotkey}"
            try:
                raw = await self.redis_service.get(key)
                last = float(raw.decode() if isinstance(raw, bytes) else raw) if raw else 0.0
                if (time.time() - last) < interval:
                    return None
                await self.redis_service.set(key, str(time.time()))
                logger.info(_m(
                    "Issuing a nonce-bound attestation event",
                    extra=get_extra_info({
                        "miner_hotkey": miner_hotkey,
                        "rented": rented,
                        "interval_seconds": interval,
                        "since_last_seconds": int(time.time() - last) if last else None,
                    }),
                ))
            except Exception as exc:
                logger.warning(_m(
                    "Failed to read/record attestation-event cadence; issuing nonce",
                    extra=get_extra_info({"miner_hotkey": miner_hotkey, "error": str(exc)}),
                ))

        return AttestationNonce.issue()

    def _expected_report_data(self, host_key: str) -> str:
        digest = hashlib.sha256(self.REPORT_PREFIX +
                                host_key.encode("utf-8")).hexdigest()
        return f"0x{digest}"

    # ------------------------------------------------------------------
    # DAH-2581 — report_data recipes, accepted in lockstep
    # ------------------------------------------------------------------

    @staticmethod
    def gpu_uuid_digest(uuids: list[str]) -> bytes:
        """The digest the in-CVM agent takes over the GPU set it holds.

        Sorted, so it depends on which GPUs are present rather than on the order NVML happened
        to enumerate them in — an ordering that varies with PCI topology and would otherwise
        make the same node attest differently after a reboot. A CVM with no GPUs digests to the
        sha256 of the empty string rather than to a special case, so every quote carries a GPU
        binding of the same shape.

        This mirrors `attest-agent/identity.py:gpu_uuid_digest`. The two must agree exactly or a
        correct node fails verification, which is why the separator is pinned as a constant.
        """
        return hashlib.sha256(GPU_UUID_SEPARATOR.join(sorted(uuids)).encode()).digest()

    def _bound_gpu_digest(self, executor: ExecutorSSHInfo) -> bytes | None:
        """The GPU digest to fold into the v2 identity, or None if it cannot be established.

        The node supplies it, so it is never trusted on its own — and it does not need to be.
        The digest is inside the hardware-signed report_data, so a node that supplies one set
        and holds another produces a quote whose identity half this validator will not match.
        What the value is *for* is telling the validator which set to expect; what makes it
        evidence is that the hardware signed it.

        `scraped_uuids` is the independent check, applied when this validator has already
        collected the node's GPU UUIDs. A mismatch means the claim and the machine disagree,
        which is refused rather than papered over — the whole point of FR-G6 is that a node
        cannot claim GPUs it does not have.
        """
        claimed = (executor.gpu_uuid_digest or "").strip().lower()
        if not claimed:
            return None
        try:
            return bytes.fromhex(claimed)
        except ValueError:
            logger.warning(_m(
                "Ignoring a GPU UUID digest that is not hex",
                extra=get_extra_info({
                    "executor": f"{executor.address}:{executor.port}",
                    "gpu_uuid_digest": claimed,
                }),
            ))
            return None

    def _expected_identity_digests(
        self,
        executor: ExecutorSSHInfo,
    ) -> list[tuple[str, bytes]]:
        """Every identity half this validator will accept, as (recipe, 32 bytes).

        A recipe change is a lockstep change — the node produces one value and the validator
        compares against it — so a validator that accepted only the new one would fail every
        node in the fleet the moment it shipped. Both are accepted through the transition and v1
        is dropped from the configured list only once every node produces v2.

            v1  sha256("SSH_HOST_KEY:" | host_key)
            v2  sha256("SSH_HOST_KEY:" | host_key | gpu_uuid_digest)

        v2 is the same identity material with the GPU set folded in, which is what makes "which
        GPUs is this CVM holding" a hardware-signed claim instead of an asserted one. The renter
        form substitutes the agent's TLS public key for the SSH host key, on the same shape, and
        is produced by `attest-agent/identity.py`.
        """
        material = self.REPORT_PREFIX + (executor.ssh_host_key or "").encode("utf-8")
        digests: list[tuple[str, bytes]] = []
        for recipe in settings.get_report_data_recipes():
            if recipe == "v1":
                digests.append(("v1", hashlib.sha256(material).digest()))
            elif recipe == "v2":
                gpu_digest = self._bound_gpu_digest(executor)
                if gpu_digest is None:
                    # Not a failure: a node that has not started producing v2 simply has nothing
                    # to bind. It still has to satisfy some accepted recipe, so this narrows the
                    # accepted set rather than widening it.
                    logger.info(_m(
                        "Skipping report_data recipe v2: this executor supplied no GPU digest",
                        extra=get_extra_info({"executor": f"{executor.address}:{executor.port}"}),
                    ))
                    continue
                digests.append(("v2", hashlib.sha256(material + gpu_digest).digest()))
            else:
                logger.warning(_m(
                    "Ignoring an unknown report_data recipe",
                    extra=get_extra_info({"recipe": recipe}),
                ))
        return digests

    async def _call_verifier(self, quote_json: str, executor: ExecutorSSHInfo) -> dict:
        if not self.verifier_url:
            raise AttestationError("TDX verifier URL is not configured")

        url = self.verifier_url.rstrip("/")
        if not url.endswith("/verify"):
            url = f"{url}/verify"

        logger.info(f"[VERIFIER REQUEST] Posting to {url} for {executor.address}:{executor.port}")

        timeout = aiohttp.ClientTimeout(total=self.quote_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                data=quote_json.encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status >= 400:
                    snippet = await response.text()
                    snippet = snippet[:200] if snippet else "<empty>"
                    raise AttestationError(
                        f"Verifier request returned {response.status} for executor {executor.address}:{executor.port}: {snippet}"
                    )
                try:
                    return await response.json()
                except aiohttp.ContentTypeError as exc:
                    raise AttestationError(
                        "Verifier response was not JSON") from exc

    @staticmethod
    def _normalise_report_data(value: str | None) -> bytes | None:
        if not value:
            return None
        token = value.strip().lower()
        if token.startswith("0x"):
            token = token[2:]
        try:
            return bytes.fromhex(token)
        except ValueError:
            return None

    async def _check_whitelist(
        self,
        compose_hash: str,
        os_image_hash: str,
        executor: ExecutorSSHInfo,
        kind: str | None = None,
    ) -> bool:
        """Check the attestation digest against the whitelist.

        DAH-2581: the platform's signed catalog is consulted first when this validator holds
        one. It is the same document every CVM host verifies before launching, so "what a host
        may run" and "what this validator accepts" stop being two lists that drift — and a
        revocation reaches both without a redeploy.

        The static list stays as the fallback, deliberately. A validator that could not reach
        the backend would otherwise reject every CVM in the fleet, which is a far worse outage
        than accepting a slightly stale set; the catalog's own expiry bounds how stale.

        G2: compose hashes in the static list carry a monotonic release version; acceptance
        requires the version to be at/above TDX_MINIMUM_COMPOSE_VERSION so a below-floor
        (rotated-back or known-bad) runner release is rejected immediately instead of aging out
        of a rolling window. The catalog carries the same idea as its own floors, applied by the
        backend before the manifest is signed.
        """
        catalog = await self._catalog()
        if catalog is not None:
            approved = catalog.approves(
                os_image_hash=os_image_hash, compose_hash=compose_hash, kind=kind
            )
            logger.info(_m(
                "Checked the attestation digest against the signed CVM catalog",
                extra=get_extra_info({
                    "executor": f"{executor.address}:{executor.port}",
                    "serial": catalog.serial,
                    "kind": kind,
                    "approved": approved,
                }),
            ))
            return approved

        if os_image_hash not in TDX_WHITELIST["OS_IMAGE_HASH"]:
            return False

        compose_version = TDX_WHITELIST["COMPOSE_HASH"][settings.DEPLOY_ENV].get(compose_hash)
        if compose_version is None:
            return False

        if compose_version < settings.TDX_MINIMUM_COMPOSE_VERSION:
            logger.warning(_m(
                "Attestation rejected: measured compose release below the minimum version floor",
                extra=get_extra_info({
                    "executor": f"{executor.address}:{executor.port}",
                    "reason": "runner_below_floor",
                    "compose_hash": compose_hash,
                    "compose_version": compose_version,
                    "minimum_version": settings.TDX_MINIMUM_COMPOSE_VERSION,
                }),
            ))
            return False

        return True

    async def _catalog(self):
        """The signed CVM catalog in force, or None when this validator has none.

        None is the honest answer for three different situations — the feature is off, the
        backend has never been reachable, or the manifest in hand has expired — and all three
        mean the same thing to the caller: fall back to the static list rather than approve
        nothing.
        """
        source = self.whitelist_source
        if source is None or not source.enabled:
            return None
        await source.refresh()
        return source.current()

    # ------------------------------------------------------------------
    # G4 — TCB status / advisory enforcement
    # ------------------------------------------------------------------

    def _collect_tcb_violations(self, details: dict) -> list[tuple[str, Any]]:
        """Collect (reason_code, offending_value) pairs from the verifier response.

        Checked explicitly rather than leaning on the aggregate is_valid flag:
        tcb_status, advisory_ids, event_log_verified, os_image_hash_verified.
        The TDX debug attribute is not currently surfaced by dstack-verifier
        (absent from its response schema); when it appears it must be asserted
        off here — tracked as a plan follow-up.
        """
        violations: list[tuple[str, Any]] = []

        tcb_status = details.get("tcb_status")
        if tcb_status not in settings.get_allowed_tcb_statuses():
            violations.append(("tcb_status_not_allowed", tcb_status))

        advisory_ids = details.get("advisory_ids") or []
        unexpected_advisories = sorted(set(advisory_ids) - settings.get_allowed_advisory_ids())
        if unexpected_advisories:
            violations.append(("advisory_present", unexpected_advisories))

        if not details.get("event_log_verified"):
            violations.append(("event_log_not_verified", details.get("event_log_verified")))

        if not details.get("os_image_hash_verified"):
            violations.append(("os_image_hash_not_verified", details.get("os_image_hash_verified")))

        return violations

    def _enforce_tcb(self, details: dict, executor: ExecutorSSHInfo) -> None:
        violations = self._collect_tcb_violations(details)
        if not violations:
            return

        extra = get_extra_info({
            "executor": f"{executor.address}:{executor.port}",
            "violations": [
                {"reason": reason, "value": value} for reason, value in violations
            ],
            "enforced": settings.ENABLE_TCB_ENFORCEMENT,
        })

        if settings.ENABLE_TCB_ENFORCEMENT:
            reason, value = violations[0]
            logger.warning(_m("TCB enforcement rejected TDX quote", extra=extra))
            raise AttestationError(
                f"TCB enforcement failed for executor {executor.address}:{executor.port}: "
                f"{reason}={value!r}"
                + (f" (+{len(violations) - 1} more violations)" if len(violations) > 1 else "")
            )

        # Warn-only telemetry window: size the blast radius before enforcing.
        logger.warning(_m("TCB enforcement warning (not enforced)", extra=extra))

    def _validate_verifier_response(
        self,
        verifier_payload: dict,
        executor: ExecutorSSHInfo,
        nonce: AttestationNonce | None = None,
    ) -> str | None:
        """Check the verifier's answer. Returns which report_data recipe matched.

        The identity half is compared as the whole first 32 bytes rather than as a prefix. Both
        spellings accept the same quotes today — a sha256 is exactly 32 bytes — but "the first
        32 bytes are this digest" is what the recipe actually says, and a prefix comparison
        would keep passing if a future recipe shortened it.
        """
        details = verifier_payload.get("details")
        if not isinstance(details, dict):
            raise AttestationError(
                "Verifier response missing details section")

        is_valid = bool(verifier_payload.get("is_valid"))
        quote_verified = bool(details.get("quote_verified"))

        returned_bytes = self._normalise_report_data(
            details.get("report_data"))
        matched_recipe: str | None = None
        if returned_bytes is not None and len(returned_bytes) >= 32:
            for recipe, expected in self._expected_identity_digests(executor):
                if returned_bytes[:32] == expected:
                    matched_recipe = recipe
                    break

        if not (is_valid and quote_verified and matched_recipe is not None):
            raise AttestationError(
                f"Verifier rejected TDX quote for executor {executor.address}:{executor.port}"
            )

        # G3 — the quote must echo the issued challenge in report_data[32:64].
        if nonce is not None:
            if nonce.is_expired(settings.ATTESTATION_NONCE_TTL_SECONDS):
                logger.warning(_m(
                    "Attestation rejected: issued nonce expired before verification",
                    extra=get_extra_info({
                        "executor": f"{executor.address}:{executor.port}",
                        "reason": "nonce_stale",
                        "nonce_age_seconds": int(time.time() - nonce.issued_at),
                    }),
                ))
                raise AttestationError(
                    f"Attestation nonce expired for executor {executor.address}:{executor.port}"
                )

            echoed = returned_bytes[32:64] if returned_bytes and len(returned_bytes) >= 64 else b""
            if echoed != nonce.value_bytes:
                logger.warning(_m(
                    "Attestation rejected: report_data does not echo the issued nonce",
                    extra=get_extra_info({
                        "executor": f"{executor.address}:{executor.port}",
                        "reason": "nonce_mismatch",
                        "expected_nonce": nonce.value_hex,
                        "echoed": echoed.hex(),
                    }),
                ))
                raise AttestationError(
                    f"TDX quote for executor {executor.address}:{executor.port} does not echo "
                    "the issued attestation nonce (stale or replayed quote)"
                )

        # G4 — enforce TCB/advisory posture (warn-only unless ENABLE_TCB_ENFORCEMENT).
        self._enforce_tcb(details, executor)

        return matched_recipe

    # ------------------------------------------------------------------
    # G1 — NVIDIA GPU confidential-compute attestation (verification side)
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_jwt_claims(jwt_token: str) -> dict:
        """Decode the claims segment of a JWT without signature verification.

        The NRAS response is fetched by this validator directly over TLS, so
        transport authenticates the origin (interim posture — the local
        nv-verifier replaces NRAS before fleet enforcement, removing both the
        outbound SPOF and this trust-on-TLS).
        """
        try:
            payload_b64 = jwt_token.split(".")[1]
            padded = payload_b64 + "=" * ((4 - len(payload_b64) % 4) % 4)
            return json.loads(base64.urlsafe_b64decode(padded).decode())
        except (IndexError, ValueError, binascii.Error) as exc:
            raise AttestationError(f"Malformed NRAS attestation token: {exc}") from exc

    async def _post_nras(self, payload: dict, executor: ExecutorSSHInfo) -> Any:
        timeout = aiohttp.ClientTimeout(total=self.quote_timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(settings.NVIDIA_NRAS_URL, json=payload) as response:
                    if response.status >= 400:
                        snippet = (await response.text())[:200]
                        raise AttestationError(
                            f"NRAS returned {response.status} for executor "
                            f"{executor.address}:{executor.port}: {snippet}"
                        )
                    return await response.json()
        except AttestationError:
            raise
        except Exception as exc:
            raise AttestationError(
                f"NRAS unreachable for executor {executor.address}:{executor.port}: {exc}"
            ) from exc

    def _check_gpu_claims(
        self,
        nras_response: Any,
        expected_nonce_hex: str | None,
        executor: ExecutorSSHInfo,
    ) -> tuple[bool, list[str]]:
        """Evaluate the NRAS response: overall verdict, nonce echo, per-GPU claims.

        Returns (passed, failure_reasons). The overall verdict is the
        load-bearing check (matching the private-ml-sdk reference verifier);
        optional per-GPU claims (measres, eat_nonce, dbgstat, secboot, hwmodel)
        are asserted when present.
        """
        failures: list[str] = []
        self._last_gpu_ueids = []

        if not (isinstance(nras_response, list) and nras_response
                and isinstance(nras_response[0], list | tuple) and len(nras_response[0]) >= 2):
            return False, ["nras_response_malformed"]

        overall_claims = self._decode_jwt_claims(nras_response[0][1])
        if overall_claims.get("x-nvidia-overall-att-result") is not True:
            failures.append("gpu_att_failed:overall_result")

        if expected_nonce_hex:
            overall_nonce = overall_claims.get("eat_nonce")
            if isinstance(overall_nonce, str) and overall_nonce.lower() != expected_nonce_hex.lower():
                failures.append("gpu_att_failed:eat_nonce_mismatch")

        allowed_hwmodels = settings.get_allowed_gpu_hwmodels()
        per_gpu = nras_response[1] if len(nras_response) > 1 and isinstance(nras_response[1], dict) else {}
        for gpu_name, gpu_token in per_gpu.items():
            if not isinstance(gpu_token, str):
                continue
            claims = self._decode_jwt_claims(gpu_token)
            # DAH-2581/2582: the ueid is this GPU's hardware-attested identity — the one
            # identifier in the whole evidence set that a node cannot choose. Collected here
            # because it is where the claims are already decoded, and used by the cross-node
            # uniqueness check: the same ueid appearing under two executors is one GPU
            # answering for two, and the evidence proving it is signed.
            ueid = claims.get("ueid")
            if isinstance(ueid, str) and ueid:
                self._last_gpu_ueids.append(ueid)
            if claims.get("measres") not in (None, "success"):
                failures.append(f"gpu_att_failed:{gpu_name}:measres")
            if expected_nonce_hex:
                gpu_nonce = claims.get("eat_nonce")
                if isinstance(gpu_nonce, str) and gpu_nonce.lower() != expected_nonce_hex.lower():
                    failures.append(f"gpu_att_failed:{gpu_name}:eat_nonce_mismatch")
            dbgstat = claims.get("dbgstat")
            if dbgstat is not None and dbgstat != "disabled":
                failures.append(f"gpu_att_failed:{gpu_name}:dbgstat={dbgstat}")
            secboot = claims.get("secboot")
            if secboot is not None and secboot is not True:
                failures.append(f"gpu_att_failed:{gpu_name}:secboot={secboot}")
            hwmodel = claims.get("hwmodel")
            if allowed_hwmodels and isinstance(hwmodel, str):
                if not any(hwmodel.startswith(allowed) for allowed in allowed_hwmodels):
                    failures.append(f"gpu_att_failed:{gpu_name}:hwmodel={hwmodel}")

        return not failures, failures

    async def _verify_gpu(
        self,
        executor: ExecutorSSHInfo,
        nonce: AttestationNonce | None,
    ) -> bool | None:
        """Verify the executor's NVIDIA CC GPU evidence.

        Enforcement (ENABLE_GPU_ATTESTATION_ENFORCEMENT) applies on attestation
        events (a nonce was issued): missing/unbound/failed evidence raises
        AttestationError. Outside events, or with enforcement off, verification
        is observe-only: the result is logged and recorded but never raises.

        Returns True (verified), False (verified-bad), or None (not performed /
        undeterminable). Never returns True unless NRAS accepted the evidence.
        """
        enforce_event = settings.ENABLE_GPU_ATTESTATION_ENFORCEMENT and nonce is not None
        payload_raw = executor.nvidia_payload
        log_extra = {"executor": f"{executor.address}:{executor.port}"}

        if not payload_raw:
            if enforce_event:
                logger.warning(_m(
                    "GPU attestation rejected: evidence missing on attestation event",
                    extra=get_extra_info({**log_extra, "reason": "payload_missing"}),
                ))
                raise AttestationError(
                    f"Executor {executor.address}:{executor.port} did not provide NVIDIA GPU "
                    "evidence while GPU attestation enforcement is enabled"
                )
            return None

        try:
            payload = json.loads(payload_raw)
            if not isinstance(payload, dict) or "evidence_list" not in payload:
                raise ValueError("missing evidence_list")
        except (ValueError, TypeError) as exc:
            if enforce_event:
                raise AttestationError(
                    f"Malformed NVIDIA GPU evidence from executor {executor.address}:{executor.port}: {exc}"
                ) from exc
            logger.warning(_m(
                "Malformed NVIDIA GPU evidence (observe-only)",
                extra=get_extra_info({**log_extra, "reason": "payload_malformed", "error": str(exc)}),
            ))
            return False

        # On an attestation event the GPU evidence must be bound to the issued
        # nonce — a self-generated (phase-0) nonce is replayable and rejected.
        payload_nonce = payload.get("nonce")
        if nonce is not None and payload_nonce != nonce.value_hex:
            if enforce_event:
                logger.warning(_m(
                    "GPU attestation rejected: evidence nonce is not the issued challenge",
                    extra=get_extra_info({
                        **log_extra,
                        "reason": "gpu_nonce_unbound",
                        "expected_nonce": nonce.value_hex,
                        "payload_nonce": payload_nonce,
                    }),
                ))
                raise AttestationError(
                    f"NVIDIA GPU evidence from executor {executor.address}:{executor.port} is not "
                    "bound to the issued attestation nonce"
                )
            logger.info(_m(
                "GPU evidence nonce differs from issued challenge (observe-only, likely phase-0 self-nonce)",
                extra=get_extra_info({**log_extra, "payload_nonce": payload_nonce}),
            ))

        expected_nonce_hex = nonce.value_hex if (nonce is not None and payload_nonce == nonce.value_hex) else (
            payload_nonce if isinstance(payload_nonce, str) else None
        )

        try:
            nras_response = await self._post_nras(payload, executor)
            passed, failures = self._check_gpu_claims(nras_response, expected_nonce_hex, executor)
        except AttestationError as exc:
            if enforce_event:
                raise
            logger.warning(_m(
                "GPU attestation verification undeterminable (observe-only)",
                extra=get_extra_info({**log_extra, "reason": "nras_unreachable", "error": str(exc)}),
            ))
            return None

        if passed:
            logger.info(_m(
                "NVIDIA GPU attestation verified",
                extra=get_extra_info({
                    **log_extra,
                    "nonce_bound": bool(nonce is not None and payload_nonce == nonce.value_hex),
                    "gpu_count": len(nras_response[1]) if len(nras_response) > 1 and isinstance(nras_response[1], dict) else None,
                }),
            ))
            return True

        logger.warning(_m(
            "NVIDIA GPU attestation failed",
            extra=get_extra_info({**log_extra, "reason": "gpu_att_failed", "failures": failures}),
        ))
        if enforce_event:
            raise AttestationError(
                f"NVIDIA GPU attestation failed for executor {executor.address}:{executor.port}: "
                + "; ".join(failures)
            )
        return False

    async def _verify_tdx(
        self,
        executor: ExecutorSSHInfo,
        nonce: AttestationNonce | None = None,
        expectations=None,
        kind: str | None = None,
    ) -> tuple[str | None, str | None, dict]:
        """Verify TDX quote and return (attestation_digest, tee_type, facts).

        Validates the quote string, calls the external verifier, extracts
        os_image_hash / compose_hash from the response, and checks the whitelist
        when ENABLE_ATTESTATION_WHITELIST is set.

        Raises AttestationError if the quote is missing, the verifier rejects it,
        or the digest is not in the whitelist when whitelisting is enabled.
        """
        quote = executor.tdx_quote
        if not isinstance(quote, str) or not quote.strip():
            raise AttestationError(
                f"Executor {executor.address}:{executor.port} provided empty TDX quote"
            )

        verifier_payload = await self._call_verifier(quote.strip(), executor)
        recipe = self._validate_verifier_response(verifier_payload, executor, nonce=nonce)

        details = verifier_payload.get("details", {})
        app_info = details.get("app_info", {})
        os_image_hash = app_info.get("os_image_hash")
        compose_hash = app_info.get("compose_hash")

        if os_image_hash and compose_hash:
            attestation_digest = f"{os_image_hash}{compose_hash}"
        elif os_image_hash:
            attestation_digest = os_image_hash
        else:
            attestation_digest = None

        tee_type = "dstack/tdx"

        logger.info(_m(
            "Attestation verified",
            extra=get_extra_info({
                "executor": f"{executor.address}:{executor.port}",
                "attestation_digest": attestation_digest,
                "tee_type": tee_type,
            }),
        ))

        if settings.ENABLE_ATTESTATION_WHITELIST:
            if not attestation_digest:
                raise AttestationError(
                    f"Missing attestation digest for executor {executor.address}:{executor.port}"
                )

            is_whitelisted = await self._check_whitelist(
                compose_hash=compose_hash,
                os_image_hash=os_image_hash,
                executor=executor,
                kind=kind,
            )
            if not is_whitelisted:
                raise AttestationError(
                    f"Attestation digest {attestation_digest} with TEE type {tee_type} not in whitelist for executor {executor.address}:{executor.port}"
                )

            logger.info(_m(
                "Attestation digest verified against whitelist",
                extra=get_extra_info({
                    "executor": f"{executor.address}:{executor.port}",
                    "attestation_digest": attestation_digest,
                    "tee_type": tee_type,
                }),
            ))

        self._assert_renter_expectations(details, expectations, executor)

        logger.info(_m(
            "TDX quote verified",
            extra=get_extra_info({
                "executor": f"{executor.address}:{executor.port}",
                "report_data_recipe": recipe,
                "instance_id": app_info.get("instance_id"),
            }),
        ))

        return attestation_digest, tee_type, {
            "report_data_recipe": recipe,
            "instance_id": app_info.get("instance_id"),
            "device_id": app_info.get("device_id"),
        }

    def _assert_renter_expectations(
        self,
        details: dict,
        expectations,
        executor: ExecutorSSHInfo,
    ) -> None:
        """Check a rented CVM against what was actually sold (DAH-2581).

        Three things, and they come from three different places, which is the point:

          the OS image   is in the platform's catalog — checked by `_check_whitelist`
          the compose    is the hash the backend DERIVED for this customer's order, so a node
                         running a different stack measures differently and is caught here
          the GPUs       are the model and count the customer paid for

        The GPU check is what turns "the node delivered the hardware it sold" from a billing
        record into a verified statement. It reads the count from the attested evidence rather
        than from the node's scraped specs: specs are asserted by software the host controls,
        while the ueid list came out of signed GPU evidence.

        Enforcement is behind a flag. Off, every mismatch is logged with the same detail and
        nothing fails — which is how the blast radius gets measured before it is taken.
        """
        if expectations is None:
            return

        app_info = details.get("app_info", {}) or {}
        mismatches: list[str] = []

        expected_compose = getattr(expectations, "compose_hash", None)
        if expected_compose and app_info.get("compose_hash") != expected_compose:
            mismatches.append(
                f"compose_hash: sold {expected_compose}, measured {app_info.get('compose_hash')}"
            )

        expected_image = getattr(expectations, "os_image_hash", None)
        if expected_image and app_info.get("os_image_hash") != expected_image:
            mismatches.append(
                f"os_image_hash: sold {expected_image}, measured {app_info.get('os_image_hash')}"
            )

        expected_count = getattr(expectations, "gpu_count", None)
        if expected_count is not None and self._last_gpu_ueids:
            attested = len(self._last_gpu_ueids)
            if attested != expected_count:
                mismatches.append(f"gpu_count: sold {expected_count}, attested {attested}")

        if not mismatches:
            return

        extra = get_extra_info({
            "executor": f"{executor.address}:{executor.port}",
            "executor_uuid": executor.uuid,
            "reason": "renter_cvm_mismatch",
            "mismatches": mismatches,
            "enforced": settings.ENABLE_RENTER_CVM_VERIFICATION,
        })
        if settings.ENABLE_RENTER_CVM_VERIFICATION:
            logger.warning(_m("Renter CVM does not match what was sold", extra=extra))
            raise AttestationError(
                f"Renter CVM on executor {executor.address}:{executor.port} does not match the "
                f"order it is serving — " + "; ".join(mismatches)
            )
        logger.warning(_m("Renter CVM mismatch (not enforced)", extra=extra))

    async def _record_attested_executor(self, executor: ExecutorSSHInfo) -> None:
        """Ratchet: remember that this executor is a CVM (minimal-G5)."""
        if self.redis_service is None:
            return
        try:
            await self.redis_service.sadd(TDX_ATTESTED_EXECUTOR_SET, str(executor.uuid))
        except Exception as exc:
            logger.warning(_m(
                "Failed to record attested executor in ratchet set",
                extra=get_extra_info({"executor_uuid": executor.uuid, "error": str(exc)}),
            ))

    async def _reject_omitted_quote_if_ratcheted(self, executor: ExecutorSSHInfo) -> None:
        """Minimal-G5 fail-closed: a previously attested executor cannot silently
        drop its quote to skip verification (host-key-only bypass)."""
        if not (self.enabled and settings.ENABLE_TCB_ENFORCEMENT):
            return
        if executor.tdx_quote or self.redis_service is None:
            return
        try:
            is_ratcheted = await self.redis_service.is_elem_exists_in_set(
                TDX_ATTESTED_EXECUTOR_SET, str(executor.uuid)
            )
        except Exception as exc:
            logger.warning(_m(
                "Failed to check attested-executor ratchet set",
                extra=get_extra_info({"executor_uuid": executor.uuid, "error": str(exc)}),
            ))
            return
        if is_ratcheted:
            logger.warning(_m(
                "Attestation rejected: previously attested executor omitted its TDX quote",
                extra=get_extra_info({
                    "executor": f"{executor.address}:{executor.port}",
                    "executor_uuid": executor.uuid,
                    "reason": "quote_omitted_ratcheted",
                }),
            ))
            raise AttestationError(
                f"Executor {executor.address}:{executor.port} previously attested as a CVM "
                "but omitted its TDX quote (fail-closed)"
            )

    def _build_known_hosts(
        self,
        executor: ExecutorSSHInfo,
    ) -> asyncssh.SSHKnownHosts | None:
        """Build an asyncssh known-hosts policy from executor.ssh_host_key.

        Constructs a known_hosts entry that covers both the bare address and the
        bracketed address:port form so asyncssh accepts connections on any port.

        Returns None if the key cannot be parsed, logging a warning in that case.
        """
        hosts = [executor.address]
        try:
            port = int(executor.port)
        except Exception:  # pragma: no cover - defensive
            port = None
        if port:
            hosts.append(f"[{executor.address}]:{port}")

        known_hosts_entry = ",".join(hosts)

        try:
            return asyncssh.import_known_hosts(
                f"{known_hosts_entry} {executor.ssh_host_key.strip()}\n"
            )
        except Exception as exc:
            logger.warning(_m(
                "Failed to build executor known_hosts entry",
                extra=get_extra_info({
                    "executor": f"{executor.address}:{executor.port}",
                    "error": str(exc),
                }),
            ))
            return None

    async def prepare_host_policy(
        self,
        executor: ExecutorSSHInfo,
        nonce: AttestationNonce | None = None,
        expectations=None,
        kind: str | None = None,
    ) -> HostPolicyResult:
        """Verify TDX (and, when supplied, NVIDIA GPU) attestation and build the
        SSH known_hosts policy for the executor.

        Returns a HostPolicyResult carrying:
        - known_hosts: SSH host key policy built from the verified TDX quote
        - attestation_digest: os_image_hash + compose_hash identifying the CVM software stack
        - tee_type: "dstack/tdx", upgraded to "dstack/tdx+nvcc" when nonce-bound
          GPU evidence verified
        - gpu_attestation_passed: True / False / None (None = not performed)

        All fields are None when attestation is disabled or the executor has no
        host key. Raises AttestationError if verification is enabled but fails.
        """
        await self._reject_omitted_quote_if_ratcheted(executor)

        should_verify = self._should_verify(executor)

        if not executor.ssh_host_key:
            if should_verify:
                raise AttestationError(
                    f"Executor {executor.address}:{executor.port} missing SSH host key for attestation"
                )
            return HostPolicyResult()

        attestation_digest, tee_type = None, None
        gpu_attestation_passed: bool | None = None
        facts: dict = {}
        gpu_ueids: list[str] = []
        if should_verify:
            # GPU evidence is verified BEFORE the renter expectations are checked, because the
            # attested GPU count is one of the things they compare against — and the ueids only
            # exist once the evidence has been through NRAS. So the TDX quote is checked first
            # (it establishes the stack), then the GPUs, then the two together against the order.
            gpu_attestation_passed = await self._verify_gpu(executor, nonce)
            gpu_ueids = list(self._last_gpu_ueids)

            attestation_digest, tee_type, facts = await self._verify_tdx(
                executor, nonce=nonce, expectations=expectations, kind=kind
            )
            await self._record_attested_executor(executor)

            # tee_type only claims verified GPU CC when the evidence was bound
            # to the validator-issued challenge (an unbound pass is replayable).
            if gpu_attestation_passed is True and nonce is not None:
                tee_type = "dstack/tdx+nvcc"

        known_hosts = self._build_known_hosts(executor)
        return HostPolicyResult(
            known_hosts=known_hosts,
            attestation_digest=attestation_digest,
            tee_type=tee_type,
            gpu_attestation_passed=gpu_attestation_passed,
            report_data_recipe=facts.get("report_data_recipe"),
            instance_id=facts.get("instance_id"),
            device_id=facts.get("device_id"),
            gpu_ueids=gpu_ueids,
        )
