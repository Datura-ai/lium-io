"""Collecting the two pieces of evidence a trust check needs, and nothing else.

A TDX quote from the dstack guest agent, and NVIDIA confidential-compute evidence for the
GPUs this CVM holds. Both are read-only operations against the guest's own devices; neither
touches the customer's workload, and the agent has no path to it (FR-F1, FR-E6).

**Degradation is reported, never hidden.** A CVM with no GPUs, or a build without the
NVIDIA verifier stack, returns `None` for GPU evidence with a stated reason — it does not
return an empty list, which a verifier would be entitled to read as "this CVM holds no
GPUs" when the truth is "this agent could not tell". The two answers lead to opposite
decisions on the validator's side, so they must not share an encoding.
"""

import hashlib
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuEvidence:
    uuids: list[str]
    evidence: list | None
    detail: str

    def report(self) -> dict:
        return {"gpu_uuids": self.uuids, "evidence": self.evidence, "detail": self.detail}


class EvidenceError(Exception):
    """A quote could not be produced. A trust check without one has nothing to check."""


def gpu_uuids() -> tuple[list[str], str]:
    """The UUIDs of the GPUs passed into this CVM, and how they were read.

    NVML rather than parsing `nvidia-smi`: the UUID is the identifier the NVIDIA
    attestation stack itself uses, and a text scrape would put a formatting change between
    the attested identity and the evidence it is supposed to match.
    """
    try:
        import pynvml
    except ImportError:
        return [], "pynvml is not installed in this build, so no GPU can be identified"

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - NVML raises its own exception family
        return [], f"NVML did not initialise ({exc}), so no GPU can be identified"

    try:
        uuids = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuids.append(uuid.decode() if isinstance(uuid, bytes) else uuid)
        return uuids, f"{len(uuids)} GPU(s) read from NVML"
    except Exception as exc:  # noqa: BLE001
        return [], f"NVML failed while enumerating devices: {exc}"
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - shutdown failure must not mask the read
            pass


def collect_gpu_evidence(uuids: list[str], nonce_hex: str) -> GpuEvidence:
    """NVIDIA CC evidence for this CVM's GPUs, bound to the caller's nonce.

    The same nonce as the TDX quote's, so the two halves of one trust check cannot be
    assembled from two different moments — which is the whole shape of a replay.
    """
    if not uuids:
        return GpuEvidence([], None, "this CVM holds no GPUs that NVML could identify")

    try:
        from nv_attestation_sdk import attestation
    except ImportError:
        return GpuEvidence(uuids, None, "the NVIDIA attestation SDK is not installed in this build")

    try:
        client = attestation.Attestation()
        client.set_name("lium-attest-agent")
        client.set_nonce(nonce_hex)
        client.add_verifier(attestation.Devices.GPU, attestation.Environment.REMOTE, "", "", "")
        return GpuEvidence(uuids, client.get_evidence(), f"evidence for {len(uuids)} GPU(s)")
    except Exception as exc:  # noqa: BLE001 - the SDK raises broadly
        # Not fatal here. The validator decides what a missing half means; an agent that
        # refused to answer at all would leave it unable to tell "no evidence" from
        # "no agent".
        logger.warning("GPU evidence collection failed: %s", exc)
        return GpuEvidence(uuids, None, f"GPU evidence collection failed: {exc}")


def tdx_quote(report_data: bytes, *, timeout: int) -> str:
    """A fresh TDX quote over exactly these 64 bytes.

    Never cached. The second half of `report_data` is the verifier's nonce, so a cached
    quote is by construction a quote for someone else's challenge — the one thing a fresh
    trust check is defined to exclude.
    """
    try:
        from dstack_sdk import DstackClient
    except ImportError as exc:
        raise EvidenceError(
            "the dstack SDK is not installed, so this agent cannot produce a quote"
        ) from exc

    try:
        client = DstackClient(timeout=timeout)
        quote = client.get_quote(report_data)
    except Exception as exc:  # noqa: BLE001 - the SDK raises broadly
        raise EvidenceError(f"the dstack guest agent did not return a quote: {exc}") from exc

    # The SDK returns an object on some versions and a mapping on others. Normalised here
    # rather than at the route so the wire shape does not change with a dependency bump.
    if hasattr(quote, "model_dump"):
        return json.dumps(quote.model_dump())
    if isinstance(quote, dict | list):
        return json.dumps(quote)
    return str(quote)


def quote_digest(quote: str) -> str:
    """A short, stable handle for a quote, for logs that must not carry the quote itself."""
    return hashlib.sha256(quote.encode()).hexdigest()[:16]
