"""liumd phase 2c (DAH-2834): the port check's DinD container started from the facts intent.

What the validator trusts from the executor here: nothing but "you may skip your own `docker run`".
The name, the port and the key pair are the validator's; the connection with the private key, the
sysbox proof inside and the removal are the validator's; a container that does not answer is a
failed probe and the probe runs as today on a proven port.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import bittensor
import pytest
from datura.requests.validator_requests import DindStep
from neurons.validators.src.services.task.checks.local_facts import LocalFactsCheck
from neurons.validators.src.services.task.checks.port_connectivity import PortConnectivityCheck
from neurons.validators.src.services.task.pipeline import _settle_background_work
from services.executor_connectivity.dind_probe import DindVerifier
from services.executor_connectivity.models import DindProbeResult, PortPair, PortProbeResult
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.port_selector import PortSelector
from services.local_verify_client import (
    CAPABILITY,
    DIND_CAPABILITY,
    LocalVerifyClient,
    StepEvidence,
    build_intent,
)
from services.local_verify_facts import LocalFacts, PreparedDind, judge_dind_step

from core.config import settings
from tests.helpers import build_context_config, build_services, build_state, make_context
from tests.test_local_verify import SPECS, FakeExecutor
from tests.test_local_verify_facts import facts_answer

# Placeholders, not key material: the private half is an opaque token the verifier hands to
# asyncssh (patched in these tests); the public line has the OpenSSH shape the wire bounds.
PRIVATE, PUBLIC = "test-private-key-token", "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGQ2b7l3kK5f5iFq3p0d9m4xX3oL0mYq6x2Ck5N4z1aB\n"


@pytest.fixture
def keypair():
    return bittensor.Keypair.create_from_uri("//Alice")


@pytest.fixture
def dind_on(monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_VERIFY_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_VERIFY_FACTS_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_VERIFY_DIND_IN_INTENT", True)


def client_factory(keypair):
    return lambda ctx: LocalVerifyClient(
        keypair, timeout_s=settings.LOCAL_VERIFY_FACTS_TIMEOUT_SECONDS, connect_timeout_s=settings.LOCAL_VERIFY_CONNECT_TIMEOUT_SECONDS
    )


def prepared(port=40000, started=True, consumed=False, name=None) -> PreparedDind:
    return PreparedDind(
        name=name or f"container_miner-hotkey_{port}",
        port=PortPair(port, port),
        private_key=PRIVATE,
        public_key=PUBLIC.strip(),
        sysbox=True,
        started=started,
        consumed=consumed,
    )


def dind_context(keypair, executor_info, *, rented_ports=(), filler_ports=(), ssh_service=True, job_batch_id="batch-1"):
    executor_info = executor_info.model_copy(update={"port_range": "40000-40003"})
    rented_executor = MagicMock()
    rented_executor.get_rented_ports.return_value = list(rented_ports)
    rented_executor.pods = []
    rented_data = MagicMock()
    rented_data.executors = {executor_info.uuid: rented_executor} if rented_ports else {}
    rented_data.get_filler_ports.return_value = list(filler_ports)
    ssh = SimpleNamespace(generate_keypair=MagicMock(return_value=(PRIVATE, PUBLIC))) if ssh_service else None
    return make_context(
        executor=executor_info,
        services=build_services(ssh=ssh),
        config=build_context_config(validator_keypair=keypair, job_batch_id=job_batch_id),
        state=build_state(specs=SPECS, rented_data=rented_data, sysbox_runtime=True),
    )


def echo(intent, **step):
    """The executor's answer with a `dind` step echoing the intent (or as overridden)."""
    answer = facts_answer(intent)
    asked = intent["steps"].get("dind")
    if asked is not None:
        answer["steps"]["dind"] = {
            "status": "ok",
            "ms": 900,
            "data": {"container_name": asked["name"], "port": asked["port"], "publish_port": asked["port"]},
            **step,
        }
    return answer


# --- the wire -------------------------------------------------------------------------------------


def test_the_phase_1_intent_is_unchanged_without_a_dind_step():
    intent = build_intent(executor_uuid="e", miner_hotkey="m", matmul=None, verifyx=None, parallel_gpu=False, deadline_s=20)
    assert "dind" not in intent["steps"]
    with_dind = build_intent(
        executor_uuid="e", miner_hotkey="m", matmul=None, verifyx=None, parallel_gpu=False, deadline_s=20,
        dind=DindStep(name="container_h_1", port=1, public_key="ssh-ed25519 A", sysbox=True),
    )
    assert with_dind["steps"]["dind"] == {"name": "container_h_1", "port": 1, "public_key": "ssh-ed25519 A", "sysbox": True}


def test_the_echo_must_match_name_and_port_exactly():
    p = prepared(started=False)
    ok = {"dind": StepEvidence(status="ok", data={"container_name": p.name, "port": 40000})}
    assert judge_dind_step(prepared(started=False), ok).started
    for steps, reason in (
        ({}, "not_answered"),
        ({"dind": StepEvidence(status="failed", error="port is already allocated")}, "failed"),
        ({"dind": StepEvidence(status="timeout")}, "timeout"),
        ({"dind": StepEvidence(status="ok", data={"container_name": "container_other_40000", "port": 40000})}, "echo_mismatch"),
        ({"dind": StepEvidence(status="ok", data={"container_name": p.name, "port": 40001})}, "echo_mismatch"),
        ({"dind": StepEvidence(status="ok", data={"container_name": p.name, "port": "40000"})}, "echo_mismatch"),
        ({"dind": StepEvidence(status="ok", data={})}, "echo_mismatch"),
    ):
        judged = judge_dind_step(prepared(started=False), steps)
        assert not judged.started and judged.reason == reason, (steps, reason)


# --- the facts check prepares it ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_facts_call_asks_for_the_validators_own_container(keypair, dind_on):
    async with FakeExecutor(keypair) as executor:
        executor.version_override = {"version": "4.2.0", "capabilities": [CAPABILITY, DIND_CAPABILITY]}
        executor.answer_override = echo
        ctx = dind_context(keypair, executor.executor_info, rented_ports=(40000,), filler_ports=(40001,))
        result = await LocalFactsCheck(client_factory(keypair)).run(ctx)

    (intent,) = executor.intents
    asked = intent["steps"]["dind"]
    assert asked == {"name": "container_miner-hotkey_40002", "port": 40002, "public_key": PUBLIC.strip(), "sysbox": True}
    dind = result.updates["state"].local_facts.dind
    assert dind.started and not dind.consumed and dind.private_key == PRIVATE and dind.port == PortPair(40002, 40002)
    assert "dind" in result.event.what_we_saw["usable"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup, why",
    [
        (lambda ex, mp: mp.setattr(settings, "LOCAL_VERIFY_DIND_IN_INTENT", False), "flag off"),
        (lambda ex, mp: setattr(ex, "version_override", {"version": "4.1.0", "capabilities": [CAPABILITY]}), "not advertised"),
    ],
)
async def test_no_dind_step_without_the_flag_or_the_capability(keypair, dind_on, monkeypatch, setup, why):
    async with FakeExecutor(keypair) as executor:
        executor.version_override = {"version": "4.2.0", "capabilities": [CAPABILITY, DIND_CAPABILITY]}
        setup(executor, monkeypatch)
        executor.answer_override = echo
        ctx = dind_context(keypair, executor.executor_info)
        result = await LocalFactsCheck(client_factory(keypair)).run(ctx)
    (intent,) = executor.intents
    assert "dind" not in intent["steps"], why
    assert result.updates["state"].local_facts.dind is None


def test_the_validators_real_key_and_a_real_hotkey_fit_the_shared_dind_bounds():
    """Both ends bound `steps.dind` from datura (`LOCAL_VERIFY_DIND_*`); a step outside them
    is a 422 on the WHOLE intent. The validator's actual `SSHService.generate_keypair()` line and a
    48-char SS58 hotkey must fit — and the check refuses to send one that does not."""
    from neurons.validators.src.services.task.checks.local_facts import (
        dind_step_fits_the_wire_bounds,
    )

    from services.ssh_service import SSHService

    _, public_key = SSHService().generate_keypair()
    hotkey = bittensor.Keypair.create_from_uri("//Alice").ss58_address
    assert len(hotkey) >= 47
    assert dind_step_fits_the_wire_bounds(f"container_{hotkey}_65535", public_key.strip())
    assert not dind_step_fits_the_wire_bounds("container_" + "x" * 101 + "_1", public_key.strip())
    assert not dind_step_fits_the_wire_bounds("container_ok_1", "ssh-ed25519 not base64!")
    assert not dind_step_fits_the_wire_bounds("container_ok_1", "ssh-ed25519 " + "A" * 1100)


@pytest.mark.asyncio
async def test_a_dind_step_outside_the_shared_bounds_is_not_sent(keypair, dind_on):
    async with FakeExecutor(keypair) as executor:
        executor.version_override = {"version": "4.2.0", "capabilities": [CAPABILITY, DIND_CAPABILITY]}
        executor.answer_override = echo
        ctx = dind_context(keypair, executor.executor_info)
        ctx.services.ssh.generate_keypair.return_value = (PRIVATE, "ssh-ed25519 not/a/key!! x")
        result = await LocalFactsCheck(client_factory(keypair)).run(ctx)
    (intent,) = executor.intents
    assert "dind" not in intent["steps"]
    assert result.updates["state"].local_facts.dind is None
    assert result.event.reason_code == "LOCAL_FACTS_OK"  # the facts still arrived


@pytest.mark.asyncio
async def test_no_dind_step_when_every_port_is_taken_or_nothing_can_mint_a_key(keypair, dind_on):
    async with FakeExecutor(keypair) as executor:
        executor.version_override = {"version": "4.2.0", "capabilities": [CAPABILITY, DIND_CAPABILITY]}
        executor.answer_override = echo
        for ctx in (
            dind_context(keypair, executor.executor_info, rented_ports=(40000, 40001), filler_ports=(40002, 40003)),
            dind_context(keypair, executor.executor_info, ssh_service=False),
            dind_context(keypair, executor.executor_info, job_batch_id=None),
        ):
            await LocalFactsCheck(client_factory(keypair)).run(ctx)
    assert all("dind" not in intent["steps"] for intent in executor.intents) and len(executor.intents) == 3


@pytest.mark.asyncio
async def test_a_step_that_fails_or_misechoes_leaves_the_probe_to_start_its_own(keypair, dind_on):
    for override, reason in (
        (lambda intent: echo(intent, status="failed", error="port is already allocated"), "failed"),
        (lambda intent: echo(intent, data={"container_name": "container_evil_40000", "port": 40000}), "echo_mismatch"),
        (lambda intent: facts_answer(intent), "not_answered"),
    ):
        async with FakeExecutor(keypair) as executor:
            executor.version_override = {"version": "4.2.0", "capabilities": [CAPABILITY, DIND_CAPABILITY]}
            executor.answer_override = override
            ctx = dind_context(keypair, executor.executor_info)
            result = await LocalFactsCheck(client_factory(keypair)).run(ctx)
        dind = result.updates["state"].local_facts.dind
        assert dind is not None and not dind.started and dind.reason == reason


@pytest.mark.asyncio
async def test_a_lost_answer_keeps_the_name_the_validator_asked_for(keypair, dind_on):
    """The executor may have started the container before the answer was lost (a 500 here). The
    asked name stays in the state, `started` False with the loss as reason, so ProviderSideLoadCheck
    excuses it instead of billing its boot to the provider and the settle step removes it; without
    this the state carried no `dind` at all and a running container of ours counted as side load."""
    async with FakeExecutor(keypair) as executor:
        executor.version_override = {"version": "4.2.0", "capabilities": [CAPABILITY, DIND_CAPABILITY]}
        executor.answer_override = lambda raw: (500, {"detail": "boom"})
        ctx = dind_context(keypair, executor.executor_info)
        result = await LocalFactsCheck(client_factory(keypair)).run(ctx)
    facts = result.updates["state"].local_facts
    assert facts.containers is None  # nothing else from the lost answer
    assert facts.dind is not None and facts.dind.name == "container_miner-hotkey_40000"
    assert not facts.dind.started and not facts.dind.consumed and facts.dind.reason


# --- the port check hands it over ---------------------------------------------------------------


class RecordingConnectivity:
    def __init__(self):
        self.calls = []

    async def verify_ports(self, ssh_client, miner_hotkey, executor_info, sysbox_runtime, **kwargs):
        self.calls.append(kwargs)
        ok = (PortPair(40002, 40002),)
        from services.executor_connectivity.models import PortVerificationResult

        return PortVerificationResult(
            selected_ports=ok, successful_ports=ok, failed_ports=(), dind_port=ok[0], dind_ok=True,
            sysbox_runtime=True, status="ok", elapsed_sec=0.1,
        )


@pytest.mark.asyncio
async def test_the_port_check_hands_over_a_started_unconsumed_container_only():
    for dind, handed in (
        (prepared(started=True), True),
        (prepared(started=False), False),
        (prepared(started=True, consumed=True), False),
        (None, False),
    ):
        connectivity = RecordingConnectivity()
        executor = MagicMock(uuid="executor-123", address="127.0.0.1", port_range="40000-40003", port_mappings=None, ssh_port=22)
        rented_data = MagicMock()
        rented_data.executors = {}
        rented_data.get_filler_ports.return_value = []
        ctx = make_context(
            executor=executor,
            services=build_services(connectivity=connectivity, redis=SimpleNamespace(renting_in_progress=AsyncMock(return_value=False))),
            state=build_state(local_facts=LocalFacts(dind=dind), rented_data=rented_data, sysbox_runtime=True),
        )
        await PortConnectivityCheck().run(ctx)
        # the parameter is always passed (the service defaults it to None); only a started,
        # unconsumed container is handed over
        assert connectivity.calls[0]["prestarted_dind"] is (dind if handed else None), dind


# --- the orchestrator and the verifier ----------------------------------------------------------


def orchestrator(dind_results):
    probe = SimpleNamespace(probe=AsyncMock(return_value=PortProbeResult(
        successful=(PortPair(40001, 40001), PortPair(40002, 40002)), failed=(PortPair(40003, 40003),)
    )))
    dind = SimpleNamespace(verify=AsyncMock(side_effect=list(dind_results)))
    return ConnectivityOrchestrator(port_selector=PortSelector(), port_probe=probe, dind_probe=dind), probe, dind


def executor_info():
    from datura.requests.miner_requests import ExecutorSSHInfo

    return ExecutorSSHInfo(uuid="executor-123", address="127.0.0.1", port=8000, ssh_username="root", ssh_port=22,
                           python_path="/usr/bin/python", root_dir="/root", port_range="40000-40003")


@pytest.mark.asyncio
async def test_the_prestarted_port_is_kept_out_of_the_batch_and_counts_when_the_probe_succeeds():
    orch, probe, dind = orchestrator([DindProbeResult(success=True, sysbox_runtime=True, port=PortPair(40000, 40000))])
    p = prepared(40000)
    result = await orch.verify(executor_info=executor_info(), miner_hotkey="miner-hotkey", sysbox_runtime=True,
                              unavailable_ports=[], ssh_client=None, prestarted_dind=p)
    batch_ports = probe.probe.await_args.args[0]
    assert PortPair(40000, 40000) not in batch_ports and len(batch_ports) == 3
    assert dind.verify.await_count == 1 and dind.verify.await_args.kwargs["prestarted"] is p
    assert dind.verify.await_args.args[0] == PortPair(40000, 40000)
    assert result.dind_ok and result.dind_port == PortPair(40000, 40000)
    assert PortPair(40000, 40000) in result.successful_ports and len(result.successful_ports) == 3


@pytest.mark.asyncio
async def test_the_validators_own_prestarted_port_is_not_a_port_taken_by_the_host():
    """The facts' `ports` step ran beside the `dind` step, so the fresh container's own port can
    appear in `published_by_docker`. That must not shrink or skip anything: the prestarted port
    is probed by the DinD probe as planned and the rest of the window as today."""
    orch, probe, dind = orchestrator([DindProbeResult(success=True, sysbox_runtime=True, port=PortPair(40000, 40000))])
    p = prepared(40000)
    result = await orch.verify(executor_info=executor_info(), miner_hotkey="miner-hotkey", sysbox_runtime=True,
                              unavailable_ports=[], ssh_client=None, prestarted_dind=p, published_ports=[40000, 40003])
    batch_ports = probe.probe.await_args.args[0]
    assert batch_ports == [PortPair(40001, 40001), PortPair(40002, 40002)]  # 40003 published; 40000 is ours
    assert result.dind_ok and result.dind_port == PortPair(40000, 40000) and result.status == "ok"


@pytest.mark.asyncio
async def test_when_the_prestarted_port_is_the_only_port_the_dind_probe_alone_decides():
    """No batch bind on a port the container already holds: the port is counted once."""
    orch, probe, dind = orchestrator([DindProbeResult(success=True, sysbox_runtime=True, port=PortPair(40003, 40003))])
    result = await orch.verify(executor_info=executor_info(), miner_hotkey="miner-hotkey", sysbox_runtime=True,
                              unavailable_ports=[40000, 40001, 40002], ssh_client=None, prestarted_dind=prepared(40003))
    probe.probe.assert_not_awaited()
    assert result.selected_ports == (PortPair(40003, 40003),)
    assert result.successful_ports == (PortPair(40003, 40003),) and result.failed_ports == ()
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_a_prestarted_container_that_fails_is_followed_by_todays_probe_on_the_freed_port():
    """taiberium, #1346: today's probe runs on the port the removal freed, not on one of the
    batch's proven ports — otherwise a failed prestart drops one port from the count and a host
    with exactly MIN_PORT_COUNT free ports fails the fatal port check for our container's fault.
    The batch's two proven ports + the freed one = three successful (the batch's own failure
    stays failed), the same count the run would reach without a prestart."""
    orch, probe, dind = orchestrator([
        DindProbeResult(success=False, sysbox_runtime=True, port=PortPair(40000, 40000)),
        DindProbeResult(success=True, sysbox_runtime=True, port=PortPair(40000, 40000)),
    ])
    result = await orch.verify(executor_info=executor_info(), miner_hotkey="miner-hotkey", sysbox_runtime=True,
                              unavailable_ports=[], ssh_client=None, prestarted_dind=prepared(40000))
    assert dind.verify.await_count == 2
    second = dind.verify.await_args_list[1]
    assert second.args[0] == PortPair(40000, 40000) and "prestarted" not in second.kwargs
    assert result.dind_ok and result.dind_port == PortPair(40000, 40000) and result.sysbox_runtime
    assert set(result.successful_ports) == {PortPair(40000, 40000), PortPair(40001, 40001), PortPair(40002, 40002)}
    assert result.failed_ports == (PortPair(40003, 40003),)


@pytest.mark.asyncio
async def test_a_failed_prestart_on_the_only_port_is_followed_by_todays_probe_on_that_freed_port():
    """The only-port branch is not a dead end: the verifier removed the container that did not
    answer the key, so the port is free again and today's `docker run` probe decides."""
    orch, probe, dind = orchestrator([
        DindProbeResult(success=False, sysbox_runtime=True, port=PortPair(40003, 40003)),
        DindProbeResult(success=True, sysbox_runtime=True, port=PortPair(40003, 40003)),
    ])
    result = await orch.verify(executor_info=executor_info(), miner_hotkey="miner-hotkey", sysbox_runtime=True,
                              unavailable_ports=[40000, 40001, 40002], ssh_client=None, prestarted_dind=prepared(40003))
    probe.probe.assert_not_awaited()
    assert dind.verify.await_count == 2
    second = dind.verify.await_args_list[1]
    assert second.args[0] == PortPair(40003, 40003) and "prestarted" not in second.kwargs
    assert result.status == "ok" and result.successful_ports == (PortPair(40003, 40003),) and result.dind_ok


@pytest.mark.asyncio
async def test_without_a_prestart_the_orchestrator_is_todays():
    orch, probe, dind = orchestrator([DindProbeResult(success=True, sysbox_runtime=True, port=PortPair(40001, 40001))])
    result = await orch.verify(executor_info=executor_info(), miner_hotkey="miner-hotkey", sysbox_runtime=True,
                              unavailable_ports=[], ssh_client=None)
    assert len(probe.probe.await_args.args[0]) == 4
    assert "prestarted" not in dind.verify.await_args.kwargs and result.dind_port == PortPair(40001, 40001)


class FakeInnerSsh:
    def __init__(self, hello_ok=True):
        self.hello_ok = hello_ok
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, cmd):
        self.commands.append(cmd)
        return SimpleNamespace(exit_status=0 if self.hello_ok else 1, stderr="")


def verifier(inner):
    v = DindVerifier(ssh_service=SimpleNamespace(generate_keypair=MagicMock(return_value=("other-private", "ssh-ed25519 OTHER"))))
    connected = []

    async def connect(host, port, pkey, log_ctx):
        connected.append(pkey)
        if isinstance(inner, Exception):
            raise inner
        return inner

    v._connect_retrying_until_sshd_answers = connect
    return v, connected


@pytest.mark.asyncio
async def test_the_verifier_skips_only_the_docker_run_and_connects_with_the_validators_key(monkeypatch):
    import asyncssh

    monkeypatch.setattr(asyncssh, "import_private_key", lambda key: f"pkey({key})")
    host_ssh = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_status=0, stderr="")))
    inner = FakeInnerSsh()
    v, connected = verifier(inner)
    p = prepared(40000)
    result = await v.verify(PortPair(40000, 40000), ssh_client=host_ssh, host="h", container_name_prefix="container_miner-hotkey",
                           sysbox=True, prestarted=p)
    assert result.success and result.sysbox_runtime
    assert p.consumed
    assert connected == [f"pkey({PRIVATE})"]
    issued = [c.args[0] for c in host_ssh.run.await_args_list]
    assert not any("docker run" in c for c in issued), "the executor already ran it"
    assert issued == ["/usr/bin/docker rm -fv container_miner-hotkey_40000"], "the removal is still ours"
    assert inner.commands == ["docker run --rm hello-world"], "the sysbox proof is still ours"


@pytest.mark.asyncio
async def test_a_prestarted_container_that_does_not_answer_the_key_is_removed_and_fails(monkeypatch):
    import asyncssh

    monkeypatch.setattr(asyncssh, "import_private_key", lambda key: key)
    host_ssh = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_status=0, stderr="")))
    v, _ = verifier(ConnectionRefusedError("refused"))
    p = prepared(40000)
    result = await v.verify(PortPair(40000, 40000), ssh_client=host_ssh, host="h", container_name_prefix="container_miner-hotkey",
                           sysbox=True, prestarted=p)
    assert not result.success and p.consumed
    assert [c.args[0] for c in host_ssh.run.await_args_list] == ["/usr/bin/docker rm -fv container_miner-hotkey_40000"]


@pytest.mark.asyncio
async def test_a_prestart_for_another_name_or_port_is_ignored_and_the_verifier_starts_its_own(monkeypatch):
    import asyncssh

    monkeypatch.setattr(asyncssh, "import_private_key", lambda key: key)
    host_ssh = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_status=0, stderr="")))
    v, connected = verifier(FakeInnerSsh())
    p = prepared(40001)
    result = await v.verify(PortPair(40000, 40000), ssh_client=host_ssh, host="h", container_name_prefix="container_miner-hotkey",
                           sysbox=True, prestarted=p)
    assert result.success and not p.consumed  # left for the pipeline's settle step to remove
    assert connected == ["other-private"]
    issued = [c.args[0] for c in host_ssh.run.await_args_list]
    assert any("docker run" in c and "container_miner-hotkey_40000" in c for c in issued)


# --- the settle step ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_started_container_no_probe_took_is_removed_when_the_pipeline_ends():
    ssh = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_status=0)))
    p = prepared(40000)
    ctx = make_context(ssh=ssh, state=build_state(local_facts=LocalFacts(dind=p)))
    await _settle_background_work(ctx)
    assert p.consumed
    ssh.run.assert_awaited_once_with("/usr/bin/docker rm -fv container_miner-hotkey_40000")

    # A name the validator asked for is removed whether or not the executor confirmed the start
    # (a lost answer may have left it running); only a consumed one, or none, is left alone.
    ssh = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_status=0)))
    unconfirmed = prepared(40001, started=False)
    ctx = make_context(ssh=ssh, state=build_state(local_facts=LocalFacts(dind=unconfirmed)))
    await _settle_background_work(ctx)
    assert unconfirmed.consumed
    ssh.run.assert_awaited_once_with("/usr/bin/docker rm -fv container_miner-hotkey_40001")

    for untouched in (prepared(consumed=True), None):
        ssh = SimpleNamespace(run=AsyncMock())
        ctx = make_context(ssh=ssh, state=build_state(local_facts=LocalFacts(dind=untouched)))
        await _settle_background_work(ctx)
        ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_settle_step_survives_a_dead_ssh():
    ssh = SimpleNamespace(run=AsyncMock(side_effect=OSError("gone")))
    p = prepared(40000)
    ctx = make_context(ssh=ssh, state=build_state(local_facts=LocalFacts(dind=p)))
    await _settle_background_work(ctx)  # no raise
    assert p.consumed
    ctx = make_context(ssh=None, state=build_state(local_facts=LocalFacts(dind=prepared(40000))))
    await _settle_background_work(ctx)


# --- the container the validator started is not the provider's load ----------------------------


def _load_context(context_factory, *, dind, host_busy_cores, dind_percent, infra=(), scrape_percent=12.0):
    """A rented host whose scrape reading is over the CPU floor (so the confirming read runs) and
    whose `docker stats` shows the renter's pod beside the port-check container booting.
    `infra` adds `(name, cpu_percent)` rows the scrape listed under Lium's infra prefixes."""
    from dataclasses import replace

    from tests.test_provider_side_load import _rented_state, confirming_runner

    scraped = [{"name": "pod_renter", "cpu_percent": 158.0}]
    scraped += [{"name": name, "cpu_percent": percent} for name, percent in infra]
    specs = {"cpu": {"count": 32}, "docker": {"host_cpu_percent": scrape_percent, "containers": scraped}}
    state = replace(_rented_state(specs), local_facts=LocalFacts(dind=dind))
    rows = [("c1", "pod_renter", 158.0), ("c2", dind.name, dind_percent)]
    rows += [(f"i{n}", name, percent) for n, (name, percent) in enumerate(infra)]
    return context_factory(state=state, runner=confirming_runner(host_busy_cores, 32, rows))


@pytest.mark.asyncio
async def test_the_validators_own_dind_boot_is_not_billed_to_the_provider(context_factory):
    """The DinD container boots (inner dockerd, sshd, sysbox — 1–3 core-seconds) while
    ProviderSideLoadCheck's confirming read runs, and the scrape listed it on no tier: it was not
    there yet. It is the validator's own name this cycle, so its cores are excused; a host at
    3.8 busy cores — renter 1.58, DinD 2.0 — leaves the provider 0.2, not 2.2."""
    from neurons.validators.src.services.task.checks.provider_side_load import ProviderSideLoadCheck

    from tests.test_provider_side_load import provider_load_gate

    ctx = _load_context(context_factory, dind=prepared(40002), host_busy_cores=3.8, dind_percent=200.0)
    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)
    assert result.passed is True
    assert result.event.what_we_saw["provider_cpu_cores"] == 0.2


@pytest.mark.asyncio
async def test_a_miner_wearing_the_dind_name_is_excused_only_up_to_the_name_tiers_cap(context_factory):
    """The name is the validator's choice but free to wear: a 9-core row under it is excused for
    CLAIMED_EXCUSE_CORES (2.0) at most, the same cap as the forged-infra-name tier — 12.58 busy
    − 1.58 renter − 2.0 = 9.0 on the provider's side, and the verdict is red."""
    from neurons.validators.src.services.task.checks.provider_side_load import ProviderSideLoadCheck

    from tests.test_provider_side_load import provider_load_gate

    ctx = _load_context(context_factory, dind=prepared(40002), host_busy_cores=12.58, dind_percent=900.0)
    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)
    assert result.passed is False
    assert result.event.what_we_saw["provider_cpu_cores"] == 9.0


@pytest.mark.asyncio
async def test_the_dind_name_and_the_infra_names_share_one_cap(context_factory):
    """taiberium, #1346: with a cap per tier a forged `dind: ok` bought 2.0 more excused cores by
    name — names hid 4.0 against a 2.0 floor. Renter 1.58, an infra-named row at 2.0 and the DinD
    row at 2.0 on a 6.58-core host: the two names excuse 2.0 together, so 3.0 stays on the
    provider's side and the verdict is red (a cap each would have left 1.0 and passed)."""
    from neurons.validators.src.services.task.checks.provider_side_load import ProviderSideLoadCheck

    from tests.test_provider_side_load import provider_load_gate

    ctx = _load_context(
        context_factory,
        dind=prepared(40002),
        host_busy_cores=6.58,
        dind_percent=200.0,
        infra=[("container_probe", 200.0)],
        scrape_percent=20.6,  # 6.59 cores − renter − the excused infra name = 3.0: over the floor, so the read runs
    )
    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)
    assert result.passed is False
    assert result.event.what_we_saw["provider_cpu_cores"] == 3.0


@pytest.mark.asyncio
async def test_a_dind_the_executor_did_not_confirm_is_still_excused_under_the_cap(context_factory):
    """`started=False` covers a lost answer as well as a refusal: the validator asked for the name,
    so a container wearing it may be ours and booting. Excused like a confirmed one — under the
    shared name cap, so a miner wearing the name still cannot hide more than the cap."""
    from neurons.validators.src.services.task.checks.provider_side_load import ProviderSideLoadCheck

    from tests.test_provider_side_load import provider_load_gate

    ctx = _load_context(
        context_factory, dind=prepared(40002, started=False), host_busy_cores=3.8, dind_percent=200.0
    )
    with provider_load_gate(enforce=True):
        result = await ProviderSideLoadCheck().run(ctx)
    assert result.passed is True
    assert result.event.what_we_saw["provider_cpu_cores"] == 0.2
