"""liumd phase 2 (DAH-2834): the fact table — the executor's read-only host facts from one early,
facts-only `POST /verify`, and the three readers that stand on them.

The lens is the phase-1 one: what does the validator trust from an executor-controlled answer?
Here the answer is "the same things it trusted from `docker ps` over SSH, bounded", and nothing
that decides a verdict — a stale container is re-aged over SSH before `docker rm`, a port is still
proven by the connect-back, the inspector digest is only observed.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import bittensor
import pytest
from neurons.validators.src.services.task.checks.inspector import InspectorRentedCheck
from neurons.validators.src.services.task.checks.local_facts import LocalFactsCheck
from neurons.validators.src.services.task.checks.local_verify import LocalVerifyCheck
from neurons.validators.src.services.task.checks.port_connectivity import PortConnectivityCheck
from neurons.validators.src.services.task.checks.stale_container_cleanup import (
    StaleContainerCleanupCheck,
)
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from services.container_cleanup import ContainerCleanup
from services.executor_connectivity.models import PortPair, PortVerificationResult
from services.local_verify_client import CAPABILITY, SCHEMA, LocalVerifyClient, StepEvidence
from services.local_verify_facts import (
    HOST_NOW_MAX,
    MAX_CONTAINERS,
    MAX_PORTS,
    HostContainer,
    LocalFacts,
    parse_containers,
    parse_created,
    parse_facts,
    parse_inspector_digest,
    parse_published_ports,
)

from core.config import settings
from tests.helpers import build_context_config, build_services, build_state, make_context
from tests.test_local_verify import EXECUTOR_UUID, SPECS, FakeExecutor

HOST_NOW = 1_800_000_000
DIGEST = "ab" * 32


def created(minutes_ago: float, now: int = HOST_NOW) -> str:
    seconds = now - int(minutes_ago * 60)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds)) + ".123456789Z"


def facts_answer(intent, *, containers=None, published=(), digest=DIGEST, now=HOST_NOW):
    steps = {
        "docker": {
            "status": "ok",
            "ms": 300,
            "data": {
                "server_version": "27.0",
                "sysbox_runtime": True,
                "containers": containers if containers is not None else [],
                "now": now,
            },
        },
        "ports": {
            "status": "ok",
            "ms": 40,
            "data": {"configured": 10, "sampled": 10, "published_by_docker": list(published)},
        },
        "inspector": {"status": "ok", "ms": 2, "data": {"lib_sha256": digest}},
    }
    return {
        "schema": SCHEMA,
        "nonce": intent["nonce"],
        "executor_uuid": intent["executor_uuid"],
        "executor_version": "4.1.0",
        "started_at": int(time.time()),
        "elapsed_ms": 350,
        "deadline_hit": False,
        "steps": steps,
        "signer": "none",
        "signature": None,
    }


@pytest.fixture
def keypair():
    return bittensor.Keypair.create_from_uri("//Alice")


@pytest.fixture
def facts_on(monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_VERIFY_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_VERIFY_FACTS_ENABLED", True)


def client_factory(keypair):
    return lambda ctx: LocalVerifyClient(
        keypair,
        timeout_s=settings.LOCAL_VERIFY_FACTS_TIMEOUT_SECONDS,
        connect_timeout_s=settings.LOCAL_VERIFY_CONNECT_TIMEOUT_SECONDS,
    )


def facts_context(keypair, executor_info, *, services=None, state=None, first_pass=True):
    return make_context(
        executor=executor_info,
        services=services or build_services(),
        config=build_context_config(validator_keypair=keypair, first_pass=first_pass),
        state=state or build_state(specs=SPECS),
    )


@pytest.fixture
def metric_log(monkeypatch):
    """The `[local_verify] outcome` lines the checks log, by module (the extras are what Loki reads)."""
    import neurons.validators.src.services.task.checks.inspector as inspector_module
    import neurons.validators.src.services.task.checks.local_facts as facts_module

    log = MagicMock()
    monkeypatch.setattr(facts_module, "logger", log)
    monkeypatch.setattr(inspector_module, "logger", log)
    return log


def outcome_lines(log, step):
    lines = [
        call.args[0].extra
        for call in log.info.call_args_list
        if str(call.args[0]) == "[local_verify] outcome"
    ]
    return [SimpleNamespace(**line) for line in lines if line.get("step") == step]


# --- the parser: closed sets and bounds ------------------------------------------------------------


def test_created_is_dockers_rfc3339_with_nine_digits_or_nothing():
    assert parse_created(created(0)) == HOST_NOW
    ts = parse_created("2026-09-09T07:15:00.000000001Z")
    assert ts == parse_created("2026-09-09T07:15:00Z") == parse_created("2026-09-09T09:15:00+02:00") == 1_788_938_100
    for bad in (None, 5, "", "yesterday", "2026-09-09", "2026-09-09T07:15:00" + "0" * 40, "0001-01-01T00:00:00Z" * 2):
        assert parse_created(bad) is None


def test_containers_take_dockers_status_set_and_name_grammar_or_the_fact_is_dropped():
    good = {"containers": [{"name": "pod_a-1.x", "status": "exited", "created": created(20), "image": "img:1"}]}
    (one,) = parse_containers(good)
    assert one == HostContainer(name="pod_a-1.x", status="exited", created_at=HOST_NOW - 1200, image="img:1")

    for bad_item in (
        {"name": "pod_a", "status": "zombie", "created": created(1)},  # not docker's set
        {"name": "pod_a", "status": ["running"], "created": created(1)},  # not even a string
        {"name": "pod a; rm -rf /", "status": "running", "created": created(1)},  # not docker's grammar
        {"name": "p" * 129, "status": "running", "created": created(1)},  # too long
        {"name": "-leading", "status": "running", "created": created(1)},
        "pod_a",  # not an object
        {"status": "running"},  # no name
    ):
        assert parse_containers({"containers": [bad_item]}) is None, bad_item

    assert parse_containers({"containers": [{"name": "n", "status": "running"} for _ in range(MAX_CONTAINERS + 1)]}) is None
    assert parse_containers({"containers": "pod_a"}) is None
    assert parse_containers({}) is None
    assert parse_containers(None) is None
    assert parse_containers({"containers": []}) == ()

    # An unparsable `created` drops the AGE, not the fact: the container is listed, unaged.
    (unaged,) = parse_containers({"containers": [{"name": "pod_b", "status": "running", "created": "soon"}]})
    assert unaged.created_at is None


def test_published_ports_are_ints_in_range_and_bounded_or_the_fact_is_dropped():
    assert parse_published_ports({"published_by_docker": [22, 40000, 40000]}) == frozenset({22, 40000})
    assert parse_published_ports({"published_by_docker": []}) == frozenset()
    for bad in ([0], [65536], ["40000"], [True], [1.5], list(range(1, MAX_PORTS + 2)), "40000", None):
        assert parse_published_ports({"published_by_docker": bad} if bad is not None else None) is None, bad


def test_inspector_digest_is_64_lowercase_hex_or_nothing():
    assert parse_inspector_digest({"lib_sha256": DIGEST}) == DIGEST
    for bad in ("AB" * 32, "ab" * 31, "ab" * 32 + "\n", 12, None):
        assert parse_inspector_digest({"lib_sha256": bad}) is None


def test_parse_facts_reads_ok_steps_only_and_needs_the_host_clock_to_age():
    steps = {
        "docker": StepEvidence(status="ok", data={"containers": [{"name": "pod_a", "status": "running", "created": created(5)}], "now": HOST_NOW}),
        "ports": StepEvidence(status="failed", data={"published_by_docker": [1]}),
        "inspector": StepEvidence(status="ok", data={"lib_sha256": DIGEST}),
    }
    facts = parse_facts(steps, capabilities={CAPABILITY}, round_trip_ms=400, executor_elapsed_ms=350)
    assert facts.can_age_containers()
    assert facts.published_ports is None  # a step that is not ok contributes nothing
    assert facts.inspector_lib_sha256 == DIGEST
    assert facts.steps == {"docker": "ok", "ports": "failed", "inspector": "ok"}

    no_clock = parse_facts(
        {"docker": StepEvidence(status="ok", data={"containers": [], "now": "1800000000"})},
        capabilities=set(), round_trip_ms=0, executor_elapsed_ms=0,
    )
    assert no_clock.containers == () and no_clock.host_now is None
    assert not no_clock.can_age_containers()  # a string clock (or a bool) is no clock

    # The clock is bounded: 0 < now < 2**40. A few-hundred-digit int parses as JSON and would
    # overflow the float division in the cleanup; here it is no clock at all.
    for bad_now in (0, -1, HOST_NOW_MAX, 10**300, True):
        facts = parse_facts(
            {"docker": StepEvidence(status="ok", data={"containers": [], "now": bad_now})},
            capabilities=set(), round_trip_ms=0, executor_elapsed_ms=0,
        )
        assert facts.host_now is None and not facts.can_age_containers(), bad_now


# --- the check -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_makes_no_call_and_leaves_no_facts(keypair, monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_VERIFY_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_VERIFY_FACTS_ENABLED", False)
    async with FakeExecutor(keypair) as executor:
        ctx = facts_context(keypair, executor.executor_info)
        result = await LocalFactsCheck(client_factory(keypair)).run(ctx)
    assert result.passed and result.event.reason_code == "LOCAL_FACTS_DISABLED"
    assert result.updates == {} and executor.intents == []


@pytest.mark.asyncio
async def test_the_facts_call_asks_for_no_gpu_step_and_leaves_bounded_facts(keypair, facts_on, metric_log):
    async with FakeExecutor(keypair) as executor:
        executor.answer_override = lambda intent: facts_answer(
            intent,
            containers=[
                {"name": "pod_old", "status": "exited", "created": created(60), "image": "x"},
                {"name": "container_filler", "status": "running", "created": created(600)},
                {"name": "unrelated", "status": "running", "created": created(5)},
            ],
            published=[40000, 40001],
        )
        ctx = facts_context(keypair, executor.executor_info)
        result = await LocalFactsCheck(client_factory(keypair)).run(ctx)

    (intent,) = executor.intents
    assert intent["steps"] == {"matmul": None, "verifyx": None, "docker": True, "ports": True, "inspector": True}
    assert intent["parallel_gpu"] is False and intent["deadline_s"] == 20
    assert result.passed and result.event.reason_code == "LOCAL_FACTS_OK"
    facts: LocalFacts = result.updates["state"].local_facts
    assert facts.can_age_containers() and len(facts.containers) == 3 and facts.host_now == HOST_NOW
    assert facts.published_ports == frozenset({40000, 40001})
    assert facts.inspector_lib_sha256 == DIGEST
    assert facts.capabilities == frozenset({CAPABILITY})
    assert result.event.what_we_saw["usable"] == ["containers", "ports", "inspector"]
    (line,) = outcome_lines(metric_log, "facts")
    assert line.outcome == "consumed" and line.reason == "ok"


@pytest.mark.asyncio
async def test_not_advertised_keeps_only_the_capability_answer(keypair, facts_on):
    async with FakeExecutor(keypair, advertise=False) as executor:
        ctx = facts_context(keypair, executor.executor_info)
        result = await LocalFactsCheck(client_factory(keypair)).run(ctx)
    assert result.passed and result.event.reason_code == "LOCAL_FACTS_SKIPPED"
    facts = result.updates["state"].local_facts
    assert facts.containers is None and facts.published_ports is None and facts.capabilities == frozenset()
    assert executor.intents == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override, reason",
    [
        (lambda intent: (500, {"detail": "boom"}), "http_error"),
        (lambda intent: {"schema": SCHEMA, "nonce": "other"}, "nonce_mismatch"),
        (
            lambda intent: {**facts_answer(intent), "steps": {"docker": {"status": "failed", "error": "daemon down"}}},
            "no_usable_fact",
        ),
    ],
)
async def test_a_refusal_or_a_malformed_answer_leaves_the_ssh_listings_in_place(
    keypair, facts_on, override, reason, metric_log
):
    async with FakeExecutor(keypair) as executor:
        executor.answer_override = override
        ctx = facts_context(keypair, executor.executor_info)
        result = await LocalFactsCheck(client_factory(keypair)).run(ctx)
    assert result.passed and result.event.reason_code == "LOCAL_FACTS_UNAVAILABLE"
    facts = result.updates["state"].local_facts
    # Either no facts at all or facts that no reader can use: the consumers behave as with None.
    assert facts.containers is None or not facts.can_age_containers()
    (line,) = outcome_lines(metric_log, "facts")
    assert line.outcome == "fallback" and line.reason == reason
    # ... and the first reader proves it: the cleanup runs its SSH listing as today.
    ssh, commands = ssh_recording({"pod_x": created(60)})
    cleanup = ContainerCleanup(stale_threshold_minutes=15)
    cleanup.prune_dangling_anonymous_volumes = AsyncMock()
    await cleanup.cleanup(ssh, None, EXECUTOR_UUID, host_facts=facts)
    assert any("docker ps -a" in c for c in commands), "the SSH listing ran"


@pytest.mark.asyncio
async def test_a_bug_in_the_check_is_a_fallback_not_a_halt(keypair, facts_on):
    def broken(ctx):
        raise RuntimeError("factory bug")

    async with FakeExecutor(keypair) as executor:
        ctx = facts_context(keypair, executor.executor_info)
        result = await LocalFactsCheck(broken).run(ctx)
    assert result.passed and result.event.what_we_saw["reason"] == "internal_error"
    assert result.updates == {}


@pytest.mark.asyncio
async def test_a_slow_executor_falls_back_inside_the_facts_budget(keypair, facts_on, monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_VERIFY_FACTS_TIMEOUT_SECONDS", 1)

    class SlowExecutor(FakeExecutor):
        async def verify(self, request):
            await asyncio.sleep(3)
            return await super().verify(request)

    async with SlowExecutor(keypair) as executor:
        executor.answer_override = lambda intent: facts_answer(intent)
        ctx = facts_context(keypair, executor.executor_info)
        started = time.perf_counter()
        result = await LocalFactsCheck(client_factory(keypair)).run(ctx)
    assert time.perf_counter() - started < 2.5
    assert result.event.reason_code == "LOCAL_FACTS_UNAVAILABLE" and result.event.what_we_saw["reason"] == "timeout"


@pytest.mark.asyncio
async def test_local_verify_reuses_the_capabilities_the_facts_call_read(keypair, facts_on, monkeypatch):
    """One /version per cycle: the GPU call reads the facts call's capability answer."""
    monkeypatch.setattr(settings, "LOCAL_VERIFY_FIRST_PASS_ONLY", False)
    version_hits = 0

    class CountingExecutor(FakeExecutor):
        async def version(self, request):
            nonlocal version_hits
            version_hits += 1
            return await super().version(request)

    async with CountingExecutor(keypair) as executor:
        executor.answer_override = lambda intent: facts_answer(intent)

        ctx = facts_context(keypair, executor.executor_info)
        facts_result = await LocalFactsCheck(client_factory(keypair)).run(ctx)
        ctx2 = ctx.model_copy(update={"state": facts_result.updates["state"]})
        # No GPU services in this context: the GPU check must fall back AFTER the capability read.
        gpu_result = await LocalVerifyCheck(client_factory(keypair)).run(ctx2)
    assert version_hits == 1
    assert gpu_result.passed


# --- reader 1: the stale-container cleanup ---------------------------------------------------------


def ssh_recording(created_by_name: dict[str, str], now: int = HOST_NOW):
    """An ssh whose `docker ps`, `inspect .Created`, `date +%s` and `rm` answer from a table and
    record every command, so a test can say which listings were and were not run."""
    commands: list[str] = []

    async def run(cmd, *args, **kwargs):
        commands.append(cmd)
        if "docker ps -a" in cmd:
            return MagicMock(exit_status=0, stdout="\n".join(created_by_name), stderr="")
        if cmd.strip() == "date +%s":
            return MagicMock(exit_status=0, stdout=str(now), stderr="")
        if "docker inspect" in cmd and "Created" in cmd:
            for name, created_str in created_by_name.items():
                if name in cmd:
                    return MagicMock(exit_status=0, stdout=str(parse_created(created_str)), stderr="")
            return MagicMock(exit_status=1, stdout="", stderr="not found")
        return MagicMock(exit_status=0, stdout="", stderr="")

    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=run)
    return ssh, commands


def host_facts(containers: dict[str, str], now: int = HOST_NOW, **kw) -> LocalFacts:
    return LocalFacts(
        containers=tuple(
            HostContainer(name=n, status="running", created_at=parse_created(c)) for n, c in containers.items()
        ),
        host_now=now,
        **kw,
    )


@pytest.mark.asyncio
async def test_cleanup_reads_candidates_and_ages_from_the_fact_and_proves_the_removal_over_ssh():
    table = {"pod_old": created(60), "pod_young": created(3), "health_check_1": created(2), "nginx": created(999)}
    ssh, commands = ssh_recording(table)
    cleanup = ContainerCleanup(stale_threshold_minutes=15)
    cleanup.prune_dangling_anonymous_volumes = AsyncMock()

    removed, names = await cleanup.cleanup(ssh, None, EXECUTOR_UUID, host_facts=host_facts(table))

    assert (removed, names) == (1, ["pod_old"])
    assert not any("docker ps -a" in c for c in commands), "the listing came from the fact"
    aged_over_ssh = [c for c in commands if "docker inspect" in c and "Created" in c]
    assert len(aged_over_ssh) == 1 and "pod_old" in aged_over_ssh[0], "only the stale one is re-aged over SSH"
    assert sum(c.strip() == "date +%s" for c in commands) == 1
    assert any("docker rm -f" in c and "pod_old" in c for c in commands)
    assert not any("nginx" in c for c in commands), "not a rental prefix: never a candidate"


@pytest.mark.asyncio
async def test_a_container_the_fact_calls_stale_but_ssh_finds_young_is_kept():
    """The executor's clock/created say 'stale'; the SSH pair says 'young'. SSH wins — the fact
    never removes anything by itself."""
    ssh, commands = ssh_recording({"pod_a": created(3)})  # SSH truth: 3 minutes old
    cleanup = ContainerCleanup(stale_threshold_minutes=15)
    cleanup.prune_dangling_anonymous_volumes = AsyncMock()
    lying = host_facts({"pod_a": created(3)}, now=HOST_NOW + 3600)  # host clock an hour ahead

    removed, names = await cleanup.cleanup(ssh, None, EXECUTOR_UUID, host_facts=lying)

    assert (removed, names) == (0, [])
    assert not any("docker rm" in c for c in commands)


@pytest.mark.asyncio
async def test_an_unaged_fact_container_falls_to_the_ssh_pair_and_rented_ones_are_never_touched():
    table = {"pod_unaged": created(60), "pod_rented": created(60)}
    ssh, commands = ssh_recording(table)
    facts = LocalFacts(
        containers=(
            HostContainer(name="pod_unaged", status="running", created_at=None),
            HostContainer(name="pod_rented", status="running", created_at=HOST_NOW - 3600),
        ),
        host_now=HOST_NOW,
    )
    rented = MagicMock()
    rented.executors = {EXECUTOR_UUID: MagicMock(pods=[MagicMock(container_name="pod_rented")])}
    rented.get_filler_containers.return_value = []
    cleanup = ContainerCleanup(stale_threshold_minutes=15)
    cleanup.prune_dangling_anonymous_volumes = AsyncMock()

    removed, names = await cleanup.cleanup(ssh, rented, EXECUTOR_UUID, host_facts=facts)

    assert (removed, names) == (1, ["pod_unaged"])
    assert any("docker inspect" in c and "pod_unaged" in c for c in commands)
    assert not any("pod_rented" in c for c in commands)


@pytest.mark.asyncio
async def test_facts_that_cannot_age_leave_the_ssh_listing_in_place():
    table = {"pod_old": created(60)}
    for facts in (None, LocalFacts(containers=(), host_now=None), LocalFacts(containers=None, host_now=HOST_NOW)):
        ssh, commands = ssh_recording(table)
        cleanup = ContainerCleanup(stale_threshold_minutes=15)
        cleanup.prune_dangling_anonymous_volumes = AsyncMock()
        removed, _ = await cleanup.cleanup(ssh, None, EXECUTOR_UUID, host_facts=facts)
        assert removed == 1
        assert any("docker ps -a" in c for c in commands), facts


@pytest.mark.asyncio
async def test_a_fact_the_cleanup_cannot_age_lands_on_the_ssh_listing_not_on_removing_nothing():
    """A bug or an odd value while aging the fact must not become "removed nothing this cycle":
    the aging runs outside the cleanup's guarded block and falls to `docker ps -a`."""
    ssh, commands = ssh_recording({"pod_old": created(60)})
    cleanup = ContainerCleanup(stale_threshold_minutes=15)
    cleanup.prune_dangling_anonymous_volumes = AsyncMock()
    odd = host_facts({"pod_old": created(60)}, now=10**400)  # past the parser, hypothetically: float overflow

    removed, names = await cleanup.cleanup(ssh, None, EXECUTOR_UUID, host_facts=odd)

    assert (removed, names) == (1, ["pod_old"])
    assert any("docker ps -a" in c for c in commands), "the SSH listing ran"


@pytest.mark.asyncio
async def test_the_stale_check_hands_the_state_facts_to_the_cleanup():
    cleanup = SimpleNamespace(
        cleanup=AsyncMock(return_value=(0, [])),
        sweep_abandoned_download_temporaries=AsyncMock(return_value=0),
        reclaim_dphn_cache_when_disk_is_tight=AsyncMock(return_value=0),
    )
    facts = host_facts({"pod_a": created(1)})
    ctx = make_context(
        services=build_services(container_cleanup=cleanup),
        state=build_state(local_facts=facts, rented_data=None),
    )
    result = await StaleContainerCleanupCheck().run(ctx)
    assert result.passed
    assert cleanup.cleanup.await_args.kwargs["host_facts"] is facts


# --- reader 2: the port selector -------------------------------------------------------------------


class RecordingConnectivity:
    def __init__(self, status="ok"):
        self.calls: list[dict] = []
        self.status = status

    async def verify_ports(self, ssh_client, miner_hotkey, executor_info, sysbox_runtime, **kwargs):
        self.calls.append(kwargs)
        ok = (PortPair(40002, 40002),)
        return PortVerificationResult(
            selected_ports=ok, successful_ports=ok, failed_ports=(), dind_port=40002, dind_ok=True,
            sysbox_runtime=True, status=self.status, elapsed_sec=0.1,
        )


def port_context(facts, *, rented_ports=(), filler_ports=(), port_range="40000-40003"):
    executor = MagicMock()
    executor.uuid = EXECUTOR_UUID
    executor.address = "127.0.0.1"
    executor.port_range = port_range
    executor.port_mappings = None
    executor.ssh_port = 22
    rented_executor = MagicMock()
    rented_executor.get_rented_ports.return_value = list(rented_ports)
    rented_executor.pods = []
    rented_data = MagicMock()
    rented_data.executors = {EXECUTOR_UUID: rented_executor} if rented_ports else {}
    rented_data.get_filler_ports.return_value = list(filler_ports)
    connectivity = RecordingConnectivity()
    ctx = make_context(
        executor=executor,
        services=build_services(
            connectivity=connectivity, redis=SimpleNamespace(renting_in_progress=AsyncMock(return_value=False))
        ),
        state=build_state(local_facts=facts, rented_data=rented_data, sysbox_runtime=True),
    )
    return ctx, connectivity


@pytest.mark.asyncio
async def test_published_ports_are_handed_to_the_selector_only_when_the_fact_is_present():
    ctx, connectivity = port_context(LocalFacts(published_ports=frozenset({40001, 40000})))
    result = await PortConnectivityCheck().run(ctx)
    assert result.passed
    assert connectivity.calls[0]["published_ports"] == [40000, 40001]

    ctx, connectivity = port_context(None)
    await PortConnectivityCheck().run(ctx)
    assert "published_ports" not in connectivity.calls[0]

    ctx, connectivity = port_context(LocalFacts(published_ports=None))
    await PortConnectivityCheck().run(ctx)
    assert "published_ports" not in connectivity.calls[0]


@pytest.mark.asyncio
async def test_the_executors_word_alone_never_skips_the_probe():
    """Every port 'published' but no rental from the backend: the probe runs — the orchestrator
    keeps the full window when the fact would empty it, and the connect-back decides."""
    ctx, connectivity = port_context(LocalFacts(published_ports=frozenset(range(40000, 40004))))
    await PortConnectivityCheck().run(ctx)
    assert len(connectivity.calls) == 1


@pytest.mark.asyncio
async def test_the_service_hands_published_ports_to_the_orchestrator_apart_from_the_unavailable_set():
    """`unavailable_ports` (backend rentals + fillers) decides WHICH window is probed; the fact
    travels separately so it can only be applied to that window."""
    from services.executor_connectivity.service import ExecutorConnectivityService

    orchestrator = SimpleNamespace(verify=AsyncMock(return_value=SimpleNamespace(
        sysbox_runtime=True, selected_ports=(), successful_ports=(), failed_ports=(), dind_port=None,
        dind_ok=False, status="ok", error=None,
    )))
    service = ExecutorConnectivityService(orchestrator)
    await service.verify_ports(None, "hk", MagicMock(), True, rented_ports=[1], filler_ports=[2], published_ports=[3, 4])
    assert orchestrator.verify.await_args.kwargs["unavailable_ports"] == [1, 2]
    assert orchestrator.verify.await_args.kwargs["published_ports"] == [3, 4]


def _orchestrator_with(selected, probed_ok):
    from services.executor_connectivity.models import DindProbeResult, PortProbeResult
    from services.executor_connectivity.orchestrator import ConnectivityOrchestrator

    selector = MagicMock()
    selector.select.return_value = list(selected)
    probe = SimpleNamespace(probe=AsyncMock(side_effect=lambda ports, **kw: PortProbeResult(
        successful=tuple(p for p in ports if p in probed_ok), failed=tuple(p for p in ports if p not in probed_ok)
    )))
    dind = SimpleNamespace(verify=AsyncMock(side_effect=lambda port, **kw: DindProbeResult(
        success=port in probed_ok, sysbox_runtime=True, port=port, log_text=""
    )))
    return ConnectivityOrchestrator(selector, probe, dind), selector, probe


@pytest.mark.asyncio
async def test_published_ports_shrink_the_probed_window_and_never_shift_it():
    """Property: the ports probed WITH the fact ⊆ the ports probed WITHOUT it. A published list
    that covered the first ports of the range must not move the 300-port window onto ports the
    validator would never have probed (an executor could then choose its own sample)."""
    window = [PortPair(p, p) for p in range(40000, 40006)]
    executor = MagicMock()

    # Without the fact: the selector's window, whole.
    orch, selector, probe = _orchestrator_with(window, probed_ok=set(window[3:]))
    await orch.verify(executor_info=executor, miner_hotkey="hk", sysbox_runtime=True, unavailable_ports=[7], ssh_client=None)
    baseline = set(probe.probe.await_args.args[0])
    assert baseline == set(window)
    selector.select.assert_called_once_with(executor, 300, {7})

    # With the fact: the SAME selection, the published ports removed from it, nothing added.
    orch, selector, probe = _orchestrator_with(window, probed_ok=set(window[3:]))
    result = await orch.verify(
        executor_info=executor, miner_hotkey="hk", sysbox_runtime=True, unavailable_ports=[7], ssh_client=None,
        published_ports=[40000, 40001, 40002, 50000],
    )
    probed = set(probe.probe.await_args.args[0])
    selector.select.assert_called_once_with(executor, 300, {7})  # the fact never reaches the selector
    assert probed == set(window[3:]) and probed <= baseline
    assert result.status == "ok" and set(result.selected_ports) == probed


@pytest.mark.asyncio
async def test_a_published_list_that_covers_the_whole_window_keeps_the_window():
    """The executor's word cannot empty the probe: every port 'published' → the full window is
    probed and the connect-back decides (today's verdict, whatever it is)."""
    window = [PortPair(p, p) for p in range(40000, 40003)]
    orch, _, probe = _orchestrator_with(window, probed_ok=set())
    result = await orch.verify(
        executor_info=MagicMock(), miner_hotkey="hk", sysbox_runtime=True, unavailable_ports=[], ssh_client=None,
        published_ports=[40000, 40001, 40002],
    )
    assert set(probe.probe.await_args.args[0]) == set(window)
    assert result.status == "no_working_ports"


@pytest.mark.asyncio
async def test_the_inspector_digest_is_logged_against_the_local_one_and_the_ssh_precheck_still_runs(metric_log):
    inspector = SimpleNamespace(
        local_checksum=DIGEST,
        validate_rented_executor=AsyncMock(return_value=SimpleNamespace(error="ssh precheck ran", message=None, diagnostics={})),
    )
    rented = MagicMock()
    rented.executors = {EXECUTOR_UUID: MagicMock(pods=[MagicMock(container_name="pod_a", pod_id="p")])}
    for reported, expected_reason in ((DIGEST, "digest_match"), ("cd" * 32, "digest_mismatch")):
        metric_log.reset_mock()
        ctx = make_context(
            executor=MagicMock(uuid=EXECUTOR_UUID),
            services=build_services(inspector=inspector),
            config=build_context_config(inspector_enabled=True),
            state=build_state(local_facts=LocalFacts(inspector_lib_sha256=reported), rented_data=rented),
        )
        result = await InspectorRentedCheck().run(ctx)
        assert result.passed and result.event.what_we_saw["error"] == "ssh precheck ran"
        (line,) = outcome_lines(metric_log, "inspector")
        assert (line.outcome, line.reason) == ("observed", expected_reason)

    metric_log.reset_mock()
    ctx = make_context(
        executor=MagicMock(uuid=EXECUTOR_UUID),
        services=build_services(inspector=inspector),
        config=build_context_config(inspector_enabled=True),
        state=build_state(local_facts=None, rented_data=rented),
    )
    await InspectorRentedCheck().run(ctx)
    assert outcome_lines(metric_log, "inspector") == []


# --- the pipeline ----------------------------------------------------------------------------------


def test_the_facts_check_runs_before_its_readers_and_after_the_verdict_checks_it_needs_nothing_from():
    ids = [c.check_id for c in PipelineFactory.build_checks()]
    facts = ids.index("executor.local_facts")
    # after the verdict checks it needs nothing from (a banned or duplicate executor makes no call) ...
    assert ids.index("gpu.validate.collateral") < facts
    assert ids.index("executor.validate.duplicate") < facts
    assert ids.index("gpu.validate.banned") < facts
    # ... and before every reader
    assert facts < ids.index("executor.cleanup.stale_containers") < ids.index("executor.validate.port_connectivity")
    assert facts < ids.index("executor.validate.inspector_rented") < ids.index("executor.local_verify")
    assert "executor.local_facts" not in [c.check_id for c in PipelineFactory.build_dry_run_checks()]
