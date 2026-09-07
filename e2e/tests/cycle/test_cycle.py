"""One validation cycle, run by the validator's own services against the stack's miner and executor.

`MinerService.request_job_to_miner` is what `Validator.sync` calls for every miner: sign in over REST, have the miner
install the validator's key on each executor, SSH in, upload and run the obfuscated scrape, walk the check pipeline,
build a JobResult. Nothing is mocked below the chain: the same pyarmor/PyInstaller scrape build, the same asyncssh
hop, the same checks. `publish_machine_specs` then puts the verdict on the redis channel the connector forwards to
the platform — the message a node's "verification" state on lium.io is made from.

Without a GPU (CI) the pipeline stops at the GPU model check with GPU_COUNT_ZERO: score 0, a named reason — the
exact verdict a provider with a broken driver gets. With E2E_GPU=1 the executor sees the host's GPUs and the cycle
must report them.
"""

import asyncio
import json
import time
import uuid

import pytest

from tests import lib

pytestmark = pytest.mark.timeout(900)


@pytest.fixture(scope="module")
def services():
    from services.ioc import ioc  # constructs the validator's services from the tester's env (no subtensor)

    return ioc


async def _one_cycle(ioc, miner_address=lib.MINER_IP, miner_port=lib.MINER_PORT):
    from payload_models.payloads import MinerJobRequestPayload
    from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

    payload = MinerJobRequestPayload(
        job_batch_id=f"e2e-{uuid.uuid4()}",
        miner_hotkey=lib.MINER_HOTKEY,
        miner_coldkey=lib.MINER_HOTKEY,
        miner_address=miner_address,
        miner_port=miner_port,
    )
    t0 = time.monotonic()
    encrypted_files = ioc["FileEncryptService"].ecrypt_miner_job_files()
    t_build = time.monotonic() - t0
    result = await ioc["MinerService"].request_job_to_miner(
        payload=payload,
        encrypted_files=encrypted_files,
        rented_data=RentedExecutorsResponse(executors={}),
        default_docker_image_digests={},
        executor_image_snapshot=None,
    )
    return payload, result, {"scrape_build_s": round(t_build, 1), "cycle_s": round(time.monotonic() - t0, 1)}


def test_cycle_reaches_the_executor_and_returns_a_verdict(services):
    payload, result, timing = asyncio.run(_one_cycle(services))
    lib.write_artifact("cycle-result.json", {"timing": timing, "result": {k: (v if k != "results" else [r.model_dump(mode="json") for r in v]) for k, v in result.items()}})
    assert result["miner_hotkey"] == lib.MINER_HOTKEY
    results = result["results"]
    assert len(results) == 1, [r.log_text for r in results]
    job = results[0]
    # the job ran on OUR executor (not the synthetic 1111… failure result the validator builds when the miner is unreachable)
    assert job.executor_info.uuid == lib.EXECUTOR_UUID, job.log_text
    assert job.executor_info.address == lib.EXECUTOR_IP and job.executor_info.ssh_port == lib.EXECUTOR_SSH_PORT
    assert job.job_batch_id == payload.job_batch_id
    if lib.GPU:
        assert job.gpu_count >= 1, job.log_text
        assert job.spec and job.spec.get("gpu", {}).get("count", 0) >= 1, job.log_text
        assert job.gpu_model, job.log_text
    else:
        # no GPU on this host: the scrape ran (specs came back) and the pipeline refused the node for it
        assert job.spec is not None, f"no specs came back — the scrape never ran on the executor: {job.log_text}"
        assert job.spec.get("gpu", {}).get("count", None) == 0, job.spec.get("gpu")
        assert job.score == 0 and job.gpu_count == 0
        assert "GPU_COUNT_ZERO" in job.log_text, job.log_text


def test_cycle_verdict_is_published_for_the_platform(services):
    """publish_machine_specs → MACHINE_SPEC_CHANNEL: the connector relays exactly this to the compute app."""
    import redis

    from services.redis_service import MACHINE_SPEC_CHANNEL

    r = redis.Redis(host=lib.ENV["REDIS_HOST"], port=int(lib.ENV.get("REDIS_PORT", "6379")))
    sub = r.pubsub(ignore_subscribe_messages=True)
    sub.subscribe(MACHINE_SPEC_CHANNEL)
    time.sleep(0.5)

    payload, result, _ = asyncio.run(_one_cycle(services))
    asyncio.run(services["MinerService"].publish_machine_specs(result["results"], payload.miner_hotkey, payload.miner_coldkey))

    msg = lib.wait_for(lambda: sub.get_message(timeout=1.0), timeout=30, interval=0.2, what="MACHINE_SPEC_CHANNEL message")
    body = json.loads(msg["data"])
    lib.write_artifact("machine-spec-message.json", body)
    assert body["executor_uuid"] == lib.EXECUTOR_UUID
    assert body["miner_hotkey"] == lib.MINER_HOTKEY and body["job_batch_id"] == payload.job_batch_id
    assert body["executor_ip"] == lib.EXECUTOR_IP and body["executor_ssh_port"] == lib.EXECUTOR_SSH_PORT
    for key in ("specs", "score", "log_status", "log_text", "incentive_reasons", "netuid", "batch_total"):
        assert key in body, f"{key} missing from the published verdict"
    assert body["batch_total"] == 1
    if not lib.GPU:
        assert body["score"] == 0 and "GPU_COUNT_ZERO" in body["log_text"]


def test_offline_executor_is_dropped_not_scored_and_does_not_stall_the_cycle():
    """The miner also owns an executor nobody answers on (seeded at E2E_DEAD_EXECUTOR_IP). The miner cannot install the
    key there, so it must leave that executor out of AcceptSSHKeyRequest within its own timeout — the live one is
    scored, the dead one is neither scored nor allowed to stall the batch. (Asserted on the miner reply that the cycle
    above consumed: exactly one executor, ours.)"""
    vk = lib.validator_keypair()
    _, pub = lib.ssh_keypair()
    t0 = time.monotonic()
    r = lib.http("POST", f"{lib.MINER_URL}/api/validator/ssh-pubkey-submit", headers=lib.validator_rest_headers(vk), timeout=120,
                 json={"message_type": "SSHPubKeySubmitRequest", "public_key": pub, "validator_signature": lib.sign(vk, pub), "miner_hotkey": lib.MINER_HOTKEY})
    took = time.monotonic() - t0
    assert r.status_code == 200, r.text
    uuids = [e["uuid"] for e in r.json()["executors"]]
    assert uuids == [lib.EXECUTOR_UUID], f"the dead executor must be dropped, the live one kept: {uuids}"
    assert took < 60, f"a dead executor stalled the miner reply for {took:.0f}s"
    lib.http("POST", f"{lib.MINER_URL}/api/validator/ssh-pubkey-remove", headers=lib.validator_rest_headers(vk), timeout=60,
             json={"message_type": "SSHPubKeyRemoveRequest", "public_key": pub, "validator_signature": lib.sign(vk, pub), "miner_hotkey": lib.MINER_HOTKEY})


def test_unreachable_miner_fails_the_job_fast(services):
    """No miner on that port: the validator must return its synthetic failed result, not hang the cycle."""
    t0 = time.monotonic()
    payload, result, _ = asyncio.run(_one_cycle(services, miner_address=lib.MINER_IP, miner_port=lib.MINER_PORT + 7))
    assert time.monotonic() - t0 < 300, "an unreachable miner must fail fast, not eat the cycle"
    job = result["results"][0]
    assert job.score == 0 and job.log_status == "error"
    assert job.executor_info.uuid == "11111111-1111-1111-1111-111111111111", job.executor_info
    assert "REST API" in job.log_text or "exception" in job.log_text.lower(), job.log_text


def test_sysbox_gate_names_its_reason(services):
    """REQUIRE_SYSBOX_FOR_UNRENTED is off in the stack (no sysbox on a CI box or a Lium pod). The check that enforces
    it must still be in the pipeline and, when on, refuse an unrented executor with SYSBOX_REQUIRED_MISSING — the
    verdict PERSONA_TESTS saw on staging. Asserted at the check level: flipping the flag mid-process is not supported."""
    from services.task.checks import SysboxRequiredCheck
    from services.task.messages import SysboxRequiredMessages
    from services.task.pipeline_factory import PipelineFactory

    checks = PipelineFactory.build_checks()
    assert any(isinstance(c, SysboxRequiredCheck) for c in checks), [type(c).__name__ for c in checks]
    assert SysboxRequiredMessages.MISSING.reason == "SYSBOX_REQUIRED_MISSING"
