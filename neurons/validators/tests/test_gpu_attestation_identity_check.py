"""DAH-4551 - GpuAttestationIdentityCheck: NVIDIA chain + nonce-bound report verified off the host, UUID<->identity pins."""
from __future__ import annotations

import base64
import datetime
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.x509.oid import NameOID
from neurons.validators.src.services.task.checks.gpu_attestation_identity import (
    ATTEST_MARKER,
    AttestationInvalid,
    GpuAttestationIdentityCheck,
    build_attest_command,
    nvidia_root,
    parse_attest_output,
    verify_gpu_evidence,
)
from neurons.validators.src.services.task.messages import GpuAttestationIdentityMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from tests.helpers import build_services, build_state, default_executor

MODULE = "neurons.validators.src.services.task.checks.gpu_attestation_identity"
FIXTURES = Path(__file__).parent / "fixtures"


# --- NVIDIA's own published sample (nvtrust local_gpu_verifier/src/verifier/samples), checked against the pinned root


def _sample():
    chain = (FIXTURES / "nvidia_hopper_cert_chain.pem").read_bytes()
    report = bytes.fromhex((FIXTURES / "nvidia_hopper_attestation_report.hex").read_text().strip())
    return chain, report, report[4:36]


def test_nvidia_sample_verifies_against_pinned_root():
    chain, report, nonce = _sample()
    got = verify_gpu_evidence(chain, report, nonce, nvidia_root())
    assert got["device_cn"] == "GH100 A01 GSP BROM"
    assert got["device_serial"] == "41431D480FE5E87274"
    assert len(got["identity"]) == 32


def test_nvidia_sample_replayed_under_another_nonce_is_rejected():
    chain, report, _ = _sample()
    with pytest.raises(AttestationInvalid, match="report_nonce_mismatch"):
        verify_gpu_evidence(chain, report, b"\x00" * 32, nvidia_root())


def test_nvidia_sample_with_a_changed_byte_is_rejected():
    chain, report, nonce = _sample()
    tampered = report[:100] + bytes([report[100] ^ 1]) + report[101:]
    with pytest.raises(AttestationInvalid, match="report_signature"):
        verify_gpu_evidence(chain, tampered, nonce, nvidia_root())


# --- synthetic chains: root -> identity -> provisioner ICA -> device -> leaf, all P-384 like NVIDIA's


def _cert(name, key, issuer_name, issuer_key):
    now = datetime.datetime(2025, 1, 1)
    return (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_name)]))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .sign(issuer_key, hashes.SHA384())
    )


class FakeCa:
    def __init__(self):
        self.root_key = ec.generate_private_key(ec.SECP384R1())
        self.root = _cert("Test Device Identity CA", self.root_key, "Test Device Identity CA", self.root_key)
        self.ica_keys = [ec.generate_private_key(ec.SECP384R1()) for _ in range(2)]
        self.icas = [
            _cert("Test Identity", self.ica_keys[0], "Test Device Identity CA", self.root_key),
            _cert("Test Provisioner ICA", self.ica_keys[1], "Test Identity", self.ica_keys[0]),
        ]

    def card(self):
        device_key = ec.generate_private_key(ec.SECP384R1())
        leaf_key = ec.generate_private_key(ec.SECP384R1())
        device = _cert("GH100 A01 GSP BROM", device_key, "Test Provisioner ICA", self.ica_keys[1])
        leaf = _cert("GH100 A01 GSP FMC LF", leaf_key, "GH100 A01 GSP BROM", device_key)
        pem = b"".join(c.public_bytes(serialization.Encoding.PEM) for c in [leaf, device, *reversed(self.icas), self.root])
        return SimpleNamespace(pem=pem, leaf_key=leaf_key)


def _report(card, nonce, signer=None):
    body = bytes([0x11, 0xE0, 0x01, 0xFF]) + nonce + b"\x00" + bytes([0x11, 0x60]) + b"measurements" * 4
    r, s = decode_dss_signature((signer or card.leaf_key).sign(body, ec.ECDSA(hashes.SHA384())))
    return body + r.to_bytes(48, "big") + s.to_bytes(48, "big")


def _row(gpu_uuid, card, nonce, **kw):
    return {
        "uuid": gpu_uuid,
        "cert_rc": 0,
        "report_rc": 0,
        "chain": base64.b64encode(card.pem).decode(),
        "report": base64.b64encode(_report(card, nonce, **kw)).decode(),
    }


def test_chain_from_another_root_is_rejected():
    ca, other = FakeCa(), FakeCa()
    card = ca.card()
    nonce = b"\x01" * 32
    with pytest.raises(AttestationInvalid, match="chain_not_nvidia_root"):
        verify_gpu_evidence(card.pem, _report(card, nonce), nonce, other.root)


def test_report_signed_by_a_key_outside_the_chain_is_rejected():
    ca = FakeCa()
    card = ca.card()
    nonce = b"\x02" * 32
    with pytest.raises(AttestationInvalid, match="report_signature"):
        verify_gpu_evidence(card.pem, _report(card, nonce, signer=ec.generate_private_key(ec.SECP384R1())), nonce, ca.root)


def _ctx(context_factory, ca, rows_for, *, gpus=("GPU-a",), redis=None, rented=False, details=None):
    """`rows_for(nonce)` is what the host prints for the validator's nonce (the last arg of the command)."""

    async def run(cmd, *args, **kwargs):
        nonce = bytes.fromhex(cmd.rsplit(" ", 1)[1])
        return SimpleNamespace(exit_status=0, stdout=ATTEST_MARKER + json.dumps(rows_for(nonce)) + "\n", stderr="")

    executor = default_executor()
    rented_data = (
        RentedExecutorsResponse(
            executors={
                executor.uuid: RentedExecutor(
                    miner_hotkey="miner-hotkey",
                    executor_ip_address=executor.address,
                    executor_ip_port=str(executor.port),
                    pods=[RentedPod(pod_id="pod-1", container_name="pod_p1")],
                )
            }
        )
        if rented
        else None
    )
    services = build_services(redis=redis) if redis is not None else build_services()
    return context_factory(
        state=build_state(
            gpu_uuids=",".join(gpus),
            rented_data=rented_data,
            gpu_details=details if details is not None else [{"uuid": u, "name": "H100", "capacity": 81559} for u in gpus],
        ),
        ssh=SimpleNamespace(run=run),
        services=services,
        executor=executor,
    )


async def _run(ctx, ca, enforce):
    with patch(f"{MODULE}.settings") as s, patch(f"{MODULE}.nvidia_root", return_value=ca.root):
        s.GPU_ATTESTATION_IDENTITY_ENFORCEMENT_ENABLED = enforce
        return await GpuAttestationIdentityCheck().run(ctx)


@pytest.mark.asyncio
async def test_verified_cards_pass_and_are_pinned(context_factory):
    ca = FakeCa()
    a, b = ca.card(), ca.card()
    ctx = _ctx(context_factory, ca, lambda n: [_row("GPU-a", a, n), _row("GPU-b", b, n)], gpus=("GPU-a", "GPU-b"))
    result = await _run(ctx, ca, enforce=True)
    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason
    assert set(result.event.what_we_saw["verified"]) == {"GPU-a", "GPU-b"}
    assert await ctx.services.redis.get("gpu_identity:uuid:GPU-a")


@pytest.mark.asyncio
@pytest.mark.parametrize("enforce", [False, True])
async def test_uuid_moved_to_another_card_fails_only_when_enforced(context_factory, enforce):
    ca = FakeCa()
    first, second = ca.card(), ca.card()
    ctx = _ctx(context_factory, ca, lambda n: [_row("GPU-a", first, n)])
    assert (await _run(ctx, ca, enforce)).passed is True

    ctx = _ctx(context_factory, ca, lambda n: [_row("GPU-a", second, n)], redis=ctx.services.redis)
    result = await _run(ctx, ca, enforce)
    assert result.passed is (not enforce)
    assert result.event.reason_code == Msg.MISMATCH.reason
    assert result.event.what_we_saw["moved"] == {"GPU-a": "uuid_changed_identity"}
    assert result.updates == ({"gpu_attestation_identity_passed": False} if enforce else {})


@pytest.mark.asyncio
async def test_card_re_added_under_a_swapped_uuid_is_flagged(context_factory):
    ca = FakeCa()
    card = ca.card()
    ctx = _ctx(context_factory, ca, lambda n: [_row("GPU-a", card, n)])
    assert (await _run(ctx, ca, enforce=True)).passed is True

    ctx = _ctx(context_factory, ca, lambda n: [_row("GPU-z", card, n)], gpus=("GPU-z",), redis=ctx.services.redis)
    result = await _run(ctx, ca, enforce=True)
    assert result.passed is False
    assert result.event.what_we_saw["moved"] == {"GPU-z": "identity_was_under:GPU-a"}


@pytest.mark.asyncio
async def test_one_card_answering_for_two_uuids_is_flagged(context_factory):
    ca = FakeCa()
    card = ca.card()
    ctx = _ctx(context_factory, ca, lambda n: [_row("GPU-a", card, n), _row("GPU-b", card, n)], gpus=("GPU-a", "GPU-b"))
    result = await _run(ctx, ca, enforce=True)
    assert result.passed is False
    assert "one_card_under_several_uuids" in result.event.what_we_saw["findings"]


@pytest.mark.asyncio
async def test_forged_report_is_flagged_and_not_pinned(context_factory):
    ca = FakeCa()
    card = ca.card()
    forger = ec.generate_private_key(ec.SECP384R1())
    ctx = _ctx(context_factory, ca, lambda n: [_row("GPU-a", card, n, signer=forger)])
    result = await _run(ctx, ca, enforce=True)
    assert result.passed is False
    assert result.event.what_we_saw["findings"] == ["attestation_invalid"]
    assert await ctx.services.redis.get("gpu_identity:uuid:GPU-a") is None


@pytest.mark.asyncio
async def test_cc_off_host_is_unavailable_and_passes_even_enforced(context_factory):
    ca = FakeCa()
    ctx = _ctx(context_factory, ca, lambda n: [{"uuid": "GPU-a", "cert_rc": 3}])  # NVML_ERROR_NOT_SUPPORTED
    result = await _run(ctx, ca, enforce=True)
    assert result.passed is True
    assert result.event.reason_code == Msg.UNAVAILABLE.reason
    assert result.event.what_we_saw["unavailable"] == {"GPU-a": "cert_rc=3,report_rc=None"}


@pytest.mark.asyncio
@pytest.mark.parametrize("rented", [False, True])
async def test_attested_card_going_dark_is_a_downgrade(context_factory, rented):
    ca = FakeCa()
    card = ca.card()
    ctx = _ctx(context_factory, ca, lambda n: [_row("GPU-a", card, n)])
    assert (await _run(ctx, ca, enforce=True)).passed is True

    ctx = _ctx(context_factory, ca, lambda n: [{"uuid": "GPU-a", "cert_rc": 3}], redis=ctx.services.redis, rented=rented)
    result = await _run(ctx, ca, enforce=True)
    assert result.passed is False
    assert result.fatal is (not rented)
    assert result.event.what_we_saw["moved"] == {"GPU-a": "attested_before_unavailable_now"}


def test_host_script_runs_and_reports_no_nvml_as_a_row():
    # this box has no libnvidia-ml: the script must still print one parseable row, not crash
    out = subprocess.run(build_attest_command(b"\x03" * 32), shell=True, capture_output=True, text=True, timeout=30)
    rows = parse_attest_output(out.stdout)
    assert len(rows) == 1 and "error" in rows[0]


# --- fallback: NVML serial / VBIOS / PCI slot plus the scrape's model and memory, for cards that cannot attest


def _hw(gpu_uuid, serial=None, vbios="96.00.74.00.01", pci=None):
    return {"uuid": gpu_uuid, "cert_rc": 3, "serial": serial, "vbios": vbios, "pci": pci, "board_part": "900-21010"}


async def _twice(context_factory, first, second, **kw):
    """Run a clean first cycle, then `second`; each is (rows, gpus, details)."""
    ca = FakeCa()
    rows, gpus, details = first
    ctx = _ctx(context_factory, ca, lambda n: rows, gpus=gpus, details=details)
    assert (await _run(ctx, ca, enforce=True)).passed is True
    rows, gpus, details = second
    ctx = _ctx(context_factory, ca, lambda n: rows, gpus=gpus, details=details, redis=ctx.services.redis)
    return await _run(ctx, ca, **kw)


@pytest.mark.asyncio
@pytest.mark.parametrize("enforce", [False, True])
async def test_same_serial_under_a_new_uuid_is_a_swap(context_factory, enforce):
    result = await _twice(
        context_factory,
        ([_hw("GPU-a", serial="1650123456789")], ("GPU-a",), None),
        ([_hw("GPU-z", serial="1650123456789")], ("GPU-z",), None),
        enforce=enforce,
    )
    assert result.passed is (not enforce)
    assert result.event.reason_code == Msg.SWAPPED.reason
    assert result.event.what_we_saw["moved"] == {"GPU-z": "serial_was_under:GPU-a"}


@pytest.mark.asyncio
async def test_every_uuid_new_in_the_same_slots_is_a_swap(context_factory):
    # consumer cards report no serial: the PCI slots + model set is what stays
    result = await _twice(
        context_factory,
        ([_hw("GPU-a", pci="00000000:17:00.0"), _hw("GPU-b", pci="00000000:65:00.0")], ("GPU-a", "GPU-b"), None),
        ([_hw("GPU-y", pci="00000000:17:00.0"), _hw("GPU-z", pci="00000000:65:00.0")], ("GPU-y", "GPU-z"), None),
        enforce=True,
    )
    assert result.passed is False
    assert result.event.reason_code == Msg.SWAPPED.reason
    assert result.event.what_we_saw["findings"] == ["uuids_changed_same_slots"]


@pytest.mark.asyncio
async def test_replacing_one_failed_card_in_place_is_not_a_swap(context_factory):
    result = await _twice(
        context_factory,
        ([_hw("GPU-a", pci="00000000:17:00.0"), _hw("GPU-b", pci="00000000:65:00.0")], ("GPU-a", "GPU-b"), None),
        ([_hw("GPU-a", pci="00000000:17:00.0"), _hw("GPU-n", pci="00000000:65:00.0")], ("GPU-a", "GPU-n"), None),
        enforce=True,
    )
    assert result.passed is True


@pytest.mark.asyncio
async def test_one_host_listing_several_models_is_inconsistent(context_factory):
    ca = FakeCa()
    gpus = ("GPU-a", "GPU-b", "GPU-c")
    details = [
        {"uuid": "GPU-a", "name": "H100", "capacity": 81559},
        {"uuid": "GPU-b", "name": "A100", "capacity": 81920},
        {"uuid": "GPU-c", "name": "RTX 4090", "capacity": 24564},
    ]
    ctx = _ctx(context_factory, ca, lambda n: [_hw(u) for u in gpus], gpus=gpus, details=details)
    result = await _run(ctx, ca, enforce=True)
    assert result.passed is False
    assert result.event.reason_code == Msg.INCONSISTENT.reason
    assert result.event.what_we_saw["findings"] == ["mixed_models_one_host"]


@pytest.mark.asyncio
async def test_uuid_changing_model_across_cycles_is_inconsistent(context_factory):
    result = await _twice(
        context_factory,
        ([_hw("GPU-a")], ("GPU-a",), [{"uuid": "GPU-a", "name": "H100", "capacity": 81559}]),
        ([_hw("GPU-a")], ("GPU-a",), [{"uuid": "GPU-a", "name": "H200", "capacity": 143771}]),
        enforce=True,
    )
    assert result.passed is False
    assert result.event.reason_code == Msg.INCONSISTENT.reason
    assert result.event.what_we_saw["hardware_changed"] == {"GPU-a": ["name", "capacity"]}


@pytest.mark.asyncio
async def test_vbios_update_and_ecc_toggle_are_recorded_not_flagged(context_factory):
    ca = FakeCa()
    ctx = _ctx(context_factory, ca, lambda n: [_hw("GPU-a", serial="1650123456789")])
    assert (await _run(ctx, ca, enforce=True)).passed is True
    redis = ctx.services.redis
    ctx = _ctx(
        context_factory,
        ca,
        lambda n: [_hw("GPU-a", serial="1650123456789", vbios="96.00.99.00.02")],
        details=[{"uuid": "GPU-a", "name": "H100", "capacity": 79647}],  # ECC on: ~2.3% less
        redis=redis,
    )
    result = await _run(ctx, ca, enforce=True)
    assert result.passed is True
    assert result.event.reason_code == Msg.UNAVAILABLE.reason
    assert json.loads(await redis.get("gpu_identity:hw:GPU-a"))["vbios"] == "96.00.99.00.02"
