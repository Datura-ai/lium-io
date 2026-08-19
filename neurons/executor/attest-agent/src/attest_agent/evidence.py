"""Collecting the two pieces of evidence a trust check needs, and nothing else.

A TDX quote from the dstack guest agent, and NVIDIA confidential-compute evidence for the
GPUs this CVM holds. Both are read-only operations against the guest's own devices; neither
touches the customer's workload, and the agent has no path to it (FR-F1, FR-E6).

**Degradation is reported, never hidden.** A CVM with no GPUs, or a build without the
NVIDIA verifier stack, returns `None` for GPU evidence with a stated reason — it does not
return an empty list, which a verifier would be entitled to read as "this CVM holds no
GPUs" when the truth is "this agent could not tell". The two answers lead to opposite
decisions on the validator's side, so they must not share an encoding.

**The reason is authored here, never quoted from an exception.** Every `detail` below is
returned to whoever can reach the agent's port, so a vendor exception pasted into one hands
an unauthenticated caller this guest's internals — driver versions, library paths, NVML's
own error text (CodeQL `py/stack-trace-exposure`). The exception goes to the log, which is
on the host an operator is already reading; the response keeps the distinction the validator
acts on, which is *which* of these four states this CVM is in, not what the vendor said.
"""

import hashlib
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# The architecture NRAS is asked to verify this CVM's evidence as. A property of the cards
# passed into the guest rather than of this agent, so it is configurable — with the fleet's
# current default, which is what the executor's own collector uses.
DEFAULT_GPU_ARCH = "HOPPER"


@dataclass(frozen=True)
class GpuEvidence:
    uuids: list[str]
    # The NRAS request payload — {"nonce", "evidence_list", "arch"} — not the bare evidence
    # list. See `collect_gpu_evidence`.
    payload: dict | None
    detail: str


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
        logger.warning("NVML did not initialise: %s", exc)
        return [], "NVML did not initialise, so no GPU can be identified"

    try:
        uuids = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuids.append(uuid.decode() if isinstance(uuid, bytes) else uuid)
        return uuids, f"{len(uuids)} GPU(s) read from NVML"
    except Exception as exc:  # noqa: BLE001
        logger.warning("NVML failed while enumerating devices: %s", exc)
        return [], "NVML failed while enumerating devices, so no GPU can be identified"
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - shutdown failure must not mask the read
            pass


def collect_gpu_evidence(
    uuids: list[str], nonce_hex: str, *, arch: str = DEFAULT_GPU_ARCH
) -> GpuEvidence:
    """NVIDIA CC evidence for this CVM's GPUs, bound to the caller's nonce.

    Returned as the payload NRAS takes — `{"nonce", "evidence_list", "arch"}` — rather than
    the bare evidence list. This is the half that carries the per-GPU `ueid` claims, and the
    ueids are what actually settle how many GPUs this CVM holds and whether they are genuine;
    the NVML UUIDs in the quote's identity half cannot settle either (see `identity.py`). A
    list with no nonce and no arch beside it is not submittable, so an agent returning one
    would leave that question permanently open.

    The same nonce as the TDX quote's, so the two halves of one trust check cannot be
    assembled from two different moments — which is the whole shape of a replay.

    The collection flow mirrors `executor/src/services/gpu_attestation_service.py`, single-GPU
    branch included: that flow is the one NRAS is known to accept on this fleet, and a second
    dialect of the same call is a second thing that can silently stop being accepted.
    """
    if not uuids:
        return GpuEvidence([], None, "this CVM holds no GPUs that NVML could identify")

    try:
        from nv_attestation_sdk import attestation
        from verifier import cc_admin
    except ImportError:
        return GpuEvidence(
            uuids, None, "the NVIDIA attestation stack is not installed in this build"
        )
    except Exception as exc:  # noqa: BLE001 - importing it does real work, see below
        # The verifier package opens its own log file in the process's working directory the
        # moment it is imported, so an unwritable CWD raises here rather than at collection —
        # and this agent runs unprivileged on a read-only rootfs, where most directories are.
        # The Dockerfile therefore starts it in the one writable path it has. Degraded rather
        # than raised: an exception out of this route reads to the validator as a broken
        # agent, when what is broken is one half of one answer.
        logger.warning("the NVIDIA attestation stack could not be loaded: %s", exc)
        return GpuEvidence(uuids, None, "the NVIDIA attestation stack could not be loaded")

    try:
        if len(uuids) == 1:
            # The count comes from the NVML read above rather than a second nvmlInit: one
            # enumeration per request, so the two halves cannot disagree about it.
            evidence_list = cc_admin.collect_gpu_evidence_remote(nonce_hex)
        else:
            attester = attestation.Attestation()
            attester.set_name(arch)
            attester.set_nonce(nonce_hex)
            attester.set_claims_version("2.0")
            attester.set_ocsp_nonce_disabled(True)
            attester.add_verifier(
                dev=attestation.Devices.GPU,
                env=attestation.Environment["REMOTE"],
                url=None,
                evidence="",
            )
            evidence_list = attester.get_evidence(options={"ppcie_mode": False})
    except Exception as exc:  # noqa: BLE001 - the SDK raises broadly
        # Not fatal here. The validator decides what a missing half means; an agent that
        # refused to answer at all would leave it unable to tell "no evidence" from
        # "no agent".
        logger.warning("GPU evidence collection failed: %s", exc)
        return GpuEvidence(uuids, None, "GPU evidence collection failed")

    if not evidence_list:
        return GpuEvidence(uuids, None, "GPU evidence collection returned nothing")

    return GpuEvidence(
        uuids,
        {"nonce": nonce_hex, "evidence_list": evidence_list, "arch": arch},
        f"evidence for {len(uuids)} GPU(s)",
    )


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
