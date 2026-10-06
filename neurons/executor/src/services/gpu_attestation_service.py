import asyncio
import json
import logging
import os
import secrets

from core.config import settings

logger = logging.getLogger(__name__)


class GPUAttestationService:
    """Collects NVIDIA confidential-compute GPU evidence (G1 phase-0 emission).

    Mirrors the working private-ml-sdk flow (vllm-proxy quote.py): pynvml probes
    the device count, then `cc_admin.collect_gpu_evidence_remote(nonce)` (single
    GPU) or the nv_attestation_sdk REMOTE attester (multi-GPU) gathers evidence
    that the validator submits to NRAS.

    Topology guard (load-bearing): the same executor image serves both the
    in-CVM topology (GPU via VFIO passthrough inside the TDX guest) and
    host-mode (bare metal, privileged). Evidence is collected ONLY in-CVM —
    detected via the dstack guest socket — because host-mode pynvml would attest
    a non-CC bare-metal GPU. Host-mode therefore never emits evidence, it does
    not error.

    The NVIDIA verifier stack ships only in CVM runner builds (executor
    Dockerfile, build arg INSTALL_GPU_ATTESTATION=true — kept out of the pdm
    lock to isolate its fragile dependency tree); a missing stack is a
    structured error and a None result, never a crash of the SSH-key upload path.
    """

    def __init__(self) -> None:
        self.enabled: bool = bool(
            settings.ENABLE_TDX_ATTESTATION and settings.ENABLE_GPU_ATTESTATION
        )
        self.arch: str = settings.GPU_ATTESTATION_ARCH

    @staticmethod
    def _in_cvm() -> bool:
        return os.path.exists(settings.DSTACK_SOCKET_PATH)

    @staticmethod
    def _normalise_nonce(nonce: str | None) -> str | None:
        """Return a valid 32-byte hex nonce: the validator's when usable, else a
        self-generated one (phase-0 shape — emission without the nonce channel).

        A malformed validator nonce is a protocol error: return None so the
        caller skips emission (the validator fails the event closed on its side)
        rather than silently substituting a self-nonce for an issued challenge.
        """
        if nonce is None:
            return secrets.token_hex(32)
        try:
            if len(bytes.fromhex(nonce)) == 32:
                return nonce.lower()
        except ValueError:
            pass
        logger.error("Invalid attestation nonce for GPU evidence (expected 32-byte hex)")
        return None

    def _collect_evidence(self, nonce_hex: str) -> list:
        """Blocking NVIDIA evidence collection — run in a worker thread.

        Import errors are separated from collection errors so a runner image
        missing the NVIDIA stack (pre-mortem #2) is directly visible in logs.
        """
        try:
            import pynvml
            from nv_attestation_sdk import attestation
            from verifier import cc_admin
        except ImportError as exc:
            raise RuntimeError(
                f"NVIDIA attestation stack unavailable in this image (build with "
                f"INSTALL_GPU_ATTESTATION=true): {exc}"
            ) from exc

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 1:
                return cc_admin.collect_gpu_evidence_remote(nonce_hex)
            attester = attestation.Attestation()
            attester.set_name(self.arch)
            attester.set_nonce(nonce_hex)
            attester.set_claims_version("2.0")
            attester.set_ocsp_nonce_disabled(True)
            attester.add_verifier(
                dev=attestation.Devices.GPU,
                env=attestation.Environment["REMOTE"],
                url=None,
                evidence="",
            )
            # ppcie_mode is the SDK's "standalone mode", not "node runs Protected PCIe": True makes
            # cc_admin.init_nvml refuse a PPCIE driver. False works for PPCIE and per-card CC alike.
            return attester.get_evidence(options={"ppcie_mode": False})
        finally:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    async def collect(self, nonce: str | None = None) -> str | None:
        """Collect GPU evidence and return the JSON nvidia_payload, or None.

        Payload shape matches the private-ml-sdk reference exactly:
        {"nonce": <hex>, "evidence_list": [...], "arch": <arch>}.
        """
        if not self.enabled:
            return None

        if not self._in_cvm():
            logger.info(
                "GPU attestation enabled but no dstack guest detected at %s — "
                "host-mode topology, skipping GPU evidence collection",
                settings.DSTACK_SOCKET_PATH,
            )
            return None

        nonce_hex = self._normalise_nonce(nonce)
        if nonce_hex is None:
            return None

        try:
            evidences = await asyncio.to_thread(self._collect_evidence, nonce_hex)
        except Exception as exc:
            logger.error("GPU evidence collection failed: %s", exc)
            return None

        if not evidences:
            logger.error("GPU evidence collection returned no evidence")
            return None

        return json.dumps({"nonce": nonce_hex, "evidence_list": evidences, "arch": self.arch})
