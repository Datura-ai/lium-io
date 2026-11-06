"""DAH-2748: an availability error means we could not reach something, and hides the node."""

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import MinerJobRequestPayload
from services.attestation_service import HostPolicyResult
from services.task import service as task_service_module
from services.task.availability import (
    AVAILABILITY_CATEGORY,
    MAX_PEER_TEXT_LENGTH,
    SSH_HOST_KEY_MISMATCH_REMEDIATION,
    AvailabilityErrorCode,
    ReachSource,
    ReachTarget,
    availability_errors,
    build_availability_event,
    build_ssh_unreachable_event,
    silence_availability_errors_on_our_own_outage,
)
from services.task.models import JobResult, build_msg
from services.task.service import TaskService, _is_ssh_transport_failure


def test_ssh_unreachable_is_an_availability_error() -> None:
    # Arrange / Act
    event = build_ssh_unreachable_event(
        executor_uuid="node-1", host="1.2.3.4", port=2200, error="Not allowed at this time"
    )

    # Assert
    assert event.category == AVAILABILITY_CATEGORY
    assert event.is_availability_error is True
    assert event.reason_code == AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE
    assert event.what_we_saw["reach_source"] == "validator"
    assert event.what_we_saw["reach_target"] == "executor_ssh"
    assert event.what_we_saw["ssh_port"] == 2200
    assert "EXECUTOR_SSH_SOURCE_PIN=false" in event.remediation  # the NAT/proxy opt-out is named
    assert "EXECUTOR_SSH_SOURCE_LIMITS=false" in event.remediation  # and the per-source sshd limits one
    assert "docker compose up -d --force-recreate" in event.remediation  # .env is bind-mounted into the runner: only a recreate reads it
    assert [e.reason_code for e in availability_errors([event])] == ["EXECUTOR_SSH_UNREACHABLE"]


def test_an_ordinary_check_failure_is_not_an_availability_error() -> None:
    # Arrange: a normal verdict about the machine itself.
    event = build_msg(
        event="GPU count does not match",
        reason="GPU_COUNT_MISMATCH",
        severity="error",
        impact="Score is zero for this cycle.",
    )

    # Act / Assert
    assert event.is_availability_error is False
    assert availability_errors([event]) == []


def test_no_events_mean_no_availability_error() -> None:
    assert availability_errors(None) == []
    assert availability_errors([]) == []


def test_an_availability_error_anywhere_in_the_cycle_counts() -> None:
    # Arrange: the reachability check fails early and an ordinary verdict follows it.
    unreachable = build_ssh_unreachable_event(
        executor_uuid="node-1", host="1.2.3.4", port=2200, error="refused"
    )
    later_verdict = build_msg(
        event="GPU count does not match",
        reason="GPU_COUNT_MISMATCH",
        severity="error",
        impact="Score is zero for this cycle.",
    )

    # Act / Assert
    assert [e.reason_code for e in availability_errors([unreachable, later_verdict])] == ["EXECUTOR_SSH_UNREACHABLE"]


def test_the_class_names_who_could_not_reach_what() -> None:
    # Arrange: a container that cannot reach an image registry. A new check adds its members
    # to the two enums and the code; nothing downstream changes.
    event = build_availability_event(
        code=AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE,
        reach_source=ReachSource.CONTAINER,
        reach_target=ReachTarget.EXECUTOR_SSH,
        event_text="Container cannot reach the image registry",
        impact="The node is hidden from the market.",
        remediation="Check outbound network on the node.",
        what_we_saw={"registry": "docker.io"},
    )

    # Act / Assert
    assert event.is_availability_error is True
    assert event.what_we_saw["reach_source"] == "container"
    assert event.what_we_saw["reach_target"] == "executor_ssh"
    assert event.what_we_saw["registry"] == "docker.io"


def _task_service_that_reaches_the_ssh_connect() -> TaskService:
    """A TaskService whose only live steps are key decryption and the attestation policy."""
    service = TaskService.__new__(TaskService)
    service.ssh_service = MagicMock()
    service.ssh_service.decrypt_payload = MagicMock(return_value="decrypted-key")
    service.attestation_service = MagicMock()
    service.attestation_service.prepare_host_policy = AsyncMock(return_value=HostPolicyResult())
    return service


def _a_job_for(executor_uuid: str) -> tuple[MinerJobRequestPayload, ExecutorSSHInfo]:
    executor = ExecutorSSHInfo(
        uuid=executor_uuid,
        address="10.0.0.5",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )
    miner = MinerJobRequestPayload(
        job_batch_id="batch-1",
        miner_hotkey="5Miner",
        miner_coldkey="5Cold",
        miner_address="10.0.0.5",
        miner_port=8080,
        executors=[],
    )
    return miner, executor


async def _run_cycle(
    service: TaskService, miner: MinerJobRequestPayload, executor: ExecutorSSHInfo
) -> JobResult:
    return await service.create_task(
        miner_info=miner,
        executor_info=executor,
        keypair=MagicMock(ss58_address="5Val"),
        private_key="key",
        public_key="pub",
        encrypted_files=MagicMock(),
        rented_data=MagicMock(),
        default_docker_image_digests={},
    )


class _ShellThatRefusesToConnect:
    """Stands in for InteractiveShellService: the handshake fails, the body never runs."""

    def __init__(self, **_: object) -> None:
        pass

    async def __aenter__(self) -> "_ShellThatRefusesToConnect":
        raise asyncssh.Error(code=1, reason="Connection refused")

    async def __aexit__(self, *_: object) -> bool:
        return False


@pytest.mark.asyncio
async def test_a_cycle_that_cannot_open_ssh_scores_zero_and_reports_the_code(monkeypatch) -> None:
    """A refused SSH handshake must score zero and name the node unreachable."""
    assert _is_ssh_transport_failure(asyncssh.Error(code=1, reason="refused")) is True
    assert _is_ssh_transport_failure(ValueError("a failed check")) is False

    monkeypatch.setattr(
        task_service_module, "InteractiveShellService", _ShellThatRefusesToConnect
    )
    miner, executor = _a_job_for("node-9")

    result = await _run_cycle(_task_service_that_reaches_the_ssh_connect(), miner, executor)

    assert result.score == 0
    assert [e["reason_code"] for e in result.availability_errors] == ["EXECUTOR_SSH_UNREACHABLE"]
    assert "EXECUTOR_SSH_UNREACHABLE" in result.log_text


class _ShellAnsweredByAnotherSshd(_ShellThatRefusesToConnect):
    async def __aenter__(self) -> "_ShellAnsweredByAnotherSshd":
        raise asyncssh.HostKeyNotVerifiable("Host key is not trusted for host 1.2.3.4")


@pytest.mark.asyncio
async def test_an_untrusted_host_key_points_at_the_ssh_port_not_the_firewall(monkeypatch) -> None:
    """A provider on Discord, 11 Oct 2026: the node SSH port reached the host's own sshd, not the executor."""
    monkeypatch.setattr(task_service_module, "InteractiveShellService", _ShellAnsweredByAnotherSshd)
    miner, executor = _a_job_for("node-9")

    result = await _run_cycle(_task_service_that_reaches_the_ssh_connect(), miner, executor)

    [error] = result.availability_errors
    assert error["reason_code"] == "EXECUTOR_SSH_UNREACHABLE"
    assert error["remediation"] == SSH_HOST_KEY_MISMATCH_REMEDIATION
    assert "firewall" not in error["remediation"]


@pytest.mark.asyncio
async def test_a_refused_connect_keeps_the_firewall_advice(monkeypatch) -> None:
    monkeypatch.setattr(task_service_module, "InteractiveShellService", _ShellThatRefusesToConnect)
    miner, executor = _a_job_for("node-9")

    result = await _run_cycle(_task_service_that_reaches_the_ssh_connect(), miner, executor)

    assert "firewall" in result.availability_errors[0]["remediation"]


def _does_not_resolve(host, port, *args, **kwargs):
    raise socket.gaierror(-2, "Name or service not known")


def _resolves_to_a_private_address(host, port, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port))]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fake_getaddrinfo", "detail"),
    [
        (_does_not_resolve, "does not resolve"),
        (_resolves_to_a_private_address, "non-public address"),
    ],
)
async def test_a_node_hostname_that_fails_the_preflight_hides_the_node(
    monkeypatch, fake_getaddrinfo, detail
) -> None:
    """DAH-4035: the real shell's hostname preflight fails at the connect, so the node is unreachable.

    A generic error here would leave availability_errors unset and the node listed.
    """
    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    dialled = AsyncMock(side_effect=AssertionError("the preflight must stop the dial"))
    monkeypatch.setattr("services.interactive_shell_service.asyncssh.connect", dialled)
    miner, executor = _a_job_for("node-9")
    executor = executor.model_copy(update={"address": "gpu-node1.example.net"})

    result = await _run_cycle(_task_service_that_reaches_the_ssh_connect(), miner, executor)

    assert result.score == 0
    assert [e["reason_code"] for e in result.availability_errors] == ["EXECUTOR_SSH_UNREACHABLE"]
    assert detail in result.log_text
    dialled.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_network_error_before_the_ssh_connect_is_not_blamed_on_the_node() -> None:
    """The attestation verifier's own HTTP call raises OSError types too (aiohttp, timeouts).

    Nothing about the node's sshd was measured there, so the node must not be hidden.
    """
    service = _task_service_that_reaches_the_ssh_connect()
    service.attestation_service.prepare_host_policy = AsyncMock(
        side_effect=ConnectionRefusedError("TDX verifier is down")
    )
    miner, executor = _a_job_for("node-9")

    result = await _run_cycle(service, miner, executor)

    assert result.score == 0
    # None, not []: this cycle never reached the connect, so it must not clear what the
    # cycle before it found.
    assert result.availability_errors is None
    assert "EXECUTOR_SSH_UNREACHABLE" not in result.log_text


def test_every_failed_reachability_check_is_reported() -> None:
    """Two network checks failing in one cycle must both reach the provider."""
    # Arrange
    ssh = build_ssh_unreachable_event(
        executor_uuid="node-1", host="1.2.3.4", port=2200, error="refused"
    )
    registry = build_availability_event(
        code=AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE,
        reach_source=ReachSource.CONTAINER,
        reach_target=ReachTarget.EXECUTOR_SSH,
        event_text="Container cannot reach the image registry",
        impact="The node is hidden from the market.",
        remediation="Check outbound network on the node.",
        what_we_saw={"registry": "docker.io"},
    )
    registry.reason_code = "CONTAINER_CANNOT_REACH_DOCKER_HUB"

    # Act
    problems = availability_errors([ssh, registry, ssh])

    # Assert - both, in the order the checks ran, no repeat, each with its own details
    assert [p.reason_code for p in problems] == ["EXECUTOR_SSH_UNREACHABLE", "CONTAINER_CANNOT_REACH_DOCKER_HUB"]
    assert problems[0].reach_target == "executor_ssh"
    assert problems[0].what_we_saw["ssh_port"] == 2200
    assert problems[1].what_we_saw["registry"] == "docker.io"


@pytest.mark.asyncio
async def test_a_failure_after_the_shell_opened_reports_the_node_as_reachable() -> None:
    """DAH-2748: the connect proved the node is there, so a later crash must not keep it hidden."""
    import asyncssh
    from datura.requests.miner_requests import ExecutorSSHInfo
    from payload_models.payloads import MinerJobRequestPayload
    from services.task.service import TaskService

    class _Shell:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    service = TaskService.__new__(TaskService)
    service.ssh_service = MagicMock()
    service.ssh_service.decrypt_payload = MagicMock(return_value="key")
    service.attestation_service = MagicMock()
    service.attestation_service.prepare_host_policy = AsyncMock(
        return_value=MagicMock(
            known_hosts=None, attestation_digest=None, tee_type=None,
            gpu_attestation_passed=None, attestation_passed=False,
        )
    )
    service.pipeline_factory = MagicMock()
    # The shell opens, then the pipeline blows up with a network error of its own.
    service.pipeline_factory.build_context = AsyncMock(side_effect=asyncssh.Error(code=1, reason="later"))

    executor = ExecutorSSHInfo(
        uuid="node-9", address="10.0.0.5", port=8080, ssh_username="root", ssh_port=2200,
        python_path="/usr/bin/python3", root_dir="/root/app",
    )
    miner = MinerJobRequestPayload(
        job_batch_id="batch-1", miner_hotkey="5Miner", miner_coldkey="5Cold",
        miner_address="10.0.0.5", miner_port=8080, executors=[],
    )

    with patch("services.task.service.InteractiveShellService", return_value=_Shell()):
        result = await service.create_task(
            miner_info=miner, executor_info=executor, keypair=MagicMock(ss58_address="5Val"),
            private_key="key", public_key="pub", encrypted_files=MagicMock(),
            rented_data=MagicMock(), default_docker_image_digests={},
        )

    assert result.availability_errors == []


def _result_with(errors: list[dict] | None) -> "JobResult":
    from datura.requests.miner_requests import ExecutorSSHInfo
    from services.task.models import JobResult

    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid="node", address="10.0.0.5", port=8080, ssh_username="root", ssh_port=2200,
            python_path="/usr/bin/python3", root_dir="/root/app",
        ),
        score=0, job_score=0, job_batch_id="b", log_status="error", log_text="x",
        availability_errors=errors,
    )


def test_one_unreachable_node_among_many_is_still_reported() -> None:
    """A single node that refuses SSH is the node's problem, and it must stay hidden."""
    # Arrange
    unreachable = [{"reason_code": "EXECUTOR_SSH_UNREACHABLE"}]
    results = [_result_with(unreachable)] + [_result_with([]) for _ in range(9)]

    # Act
    silenced = silence_availability_errors_on_our_own_outage(results)

    # Assert
    assert silenced == 0
    assert results[0].availability_errors == unreachable


def test_a_cycle_that_reached_almost_nothing_reports_nothing() -> None:
    """Most of a cycle failing at the connect is our own outage; the market must not empty."""
    # Arrange
    unreachable = [{"reason_code": "EXECUTOR_SSH_UNREACHABLE"}]
    results = [_result_with(unreachable) for _ in range(8)] + [_result_with([]) for _ in range(2)]

    # Act
    silenced = silence_availability_errors_on_our_own_outage(results)

    # Assert - None, so the backend keeps whatever it already stored
    assert silenced == 10
    assert all(result.availability_errors is None for result in results)


def test_results_that_never_checked_are_not_counted() -> None:
    """A cycle that failed before the connect says nothing, so it cannot tip the share."""
    # Arrange - two real failures, four real successes, eight that never checked
    results = [
        _result_with([{"reason_code": "EXECUTOR_SSH_UNREACHABLE"}]) for _ in range(2)
    ] + [_result_with([]) for _ in range(4)] + [_result_with(None) for _ in range(8)]

    # Act
    silenced = silence_availability_errors_on_our_own_outage(results)

    # Assert - two of six checked is not "most of them"
    assert silenced == 0


def test_a_validator_with_a_few_nodes_still_reports_what_it_found() -> None:
    """The share carries no signal on a small fleet, and a fleet that small cannot empty the market."""
    # Arrange - every node of a four-node cycle refuses SSH
    unreachable = [{"reason_code": "EXECUTOR_SSH_UNREACHABLE"}]
    results = [_result_with(list(unreachable)) for _ in range(4)]

    # Act
    silenced = silence_availability_errors_on_our_own_outage(results)

    # Assert
    assert silenced == 0
    assert all(result.availability_errors == unreachable for result in results)


def test_the_peer_text_of_an_ssh_failure_is_capped() -> None:
    # Arrange / Act
    event = build_ssh_unreachable_event(
        executor_uuid="node-1", host="1.2.3.4", port=2200, error="x" * 5000
    )

    # Assert
    assert len(event.what_we_saw["error"]) == MAX_PEER_TEXT_LENGTH


class _DictRedis:
    """Honours ex/nx against the attestation service's clock."""

    def __init__(self, clock=None) -> None:
        self.kv: dict[str, tuple[str, float | None]] = {}
        self.clock = clock or (lambda: 0.0)

    async def get(self, key):
        value, expires = self.kv.get(key, (None, None))
        return None if expires is not None and self.clock() >= expires else value

    async def set(self, key, value, ex=None, nx=False):
        if nx and await self.get(key) is not None:
            return
        self.kv[key] = (value, None if ex is None else self.clock() + ex)


_PASSED = HostPolicyResult(attestation_digest="d", gpu_attestation_passed=True)
_NO_QUOTE = HostPolicyResult()


async def _wave(attestation, miner_hotkey, outcomes, executor_id=None, host_keys=None) -> dict[str, bool]:
    """One validation wave of a miner: each executor's policy outcome -> did it get past attestation."""
    nonce = await attestation.maybe_issue_nonce(miner_hotkey, executor_id)
    reached = {}
    for uuid, outcome in outcomes:
        service = _task_service_that_reaches_the_ssh_connect()
        service.attestation_service.prepare_host_policy = AsyncMock(
            **({"side_effect": outcome} if isinstance(outcome, BaseException) else {"return_value": outcome})
        )
        service.attestation_service.record_attestation_event = attestation.record_attestation_event
        service.attestation_service.has_recent_nonce_bound_pass = attestation.has_recent_nonce_bound_pass
        miner, executor = _a_job_for(uuid)
        miner.miner_hotkey = miner_hotkey
        executor.ssh_host_key = (host_keys or {}).get(uuid, f"ssh-ed25519 key-of-{uuid}")
        dialled = AsyncMock(side_effect=asyncssh.Error(code=1, reason="refused"))

        class _Shell(_ShellThatRefusesToConnect):
            async def __aenter__(self):
                await dialled()

        with patch.object(task_service_module, "InteractiveShellService", _Shell):
            await service.create_task(
                miner_info=miner,
                executor_info=executor,
                keypair=MagicMock(ss58_address="5Val"),
                private_key="key",
                public_key="pub",
                encrypted_files=MagicMock(),
                rented_data=MagicMock(),
                default_docker_image_digests={},
                attestation_nonce=nonce,
            )
        reached[uuid] = dialled.await_count == 1
    return reached


@pytest.mark.asyncio
@pytest.mark.parametrize("a_first", [True, False], ids=["A-first", "B-first"])
@pytest.mark.parametrize(
    "b_on_the_challenged_wave",
    [_NO_QUOTE, None, ConnectionRefusedError("verifier timed out")],
    ids=["missing-quote", "omitted", "verifier-timeout"],
)
async def test_a_siblings_pass_never_lets_an_executor_skip_its_own_challenge(
    monkeypatch, a_first, b_on_the_challenged_wave
) -> None:
    from core.config import settings
    from services.attestation_service import AttestationService

    monkeypatch.setattr(settings, "TDX_VERIFIER_URL", "https://verifier.example/verify")
    attestation = AttestationService(redis_service=_DictRedis())
    first = [("A", _PASSED)] + ([] if b_on_the_challenged_wave is None else [("B", b_on_the_challenged_wave)])
    await _wave(attestation, "5Miner", first if a_first else first[::-1])

    # the next wave carries no challenge for A; B shows a quote it never bound to one
    reached = await _wave(attestation, "5Miner", [("A", _PASSED), ("B", _PASSED)])

    assert reached == {"A": True, "B": False}
    assert await attestation.maybe_issue_nonce("5Miner") is not None


@pytest.mark.asyncio
async def test_an_executor_that_passed_its_own_challenge_skips_the_next_until_due(monkeypatch) -> None:
    from core.config import settings
    from services.attestation_service import AttestationService

    monkeypatch.setattr(settings, "TDX_VERIFIER_URL", "https://verifier.example/verify")
    attestation = AttestationService(redis_service=_DictRedis())
    # B is new: the express lane asks for it alone, with a challenge, while the miner is not due
    await _wave(attestation, "5Miner", [("A", _PASSED)])
    await _wave(attestation, "5Miner", [("B", _PASSED)], executor_id="B")

    assert await _wave(attestation, "5Miner", [("A", _PASSED), ("B", _PASSED)]) == {"A": True, "B": True}


@pytest.mark.asyncio
async def test_an_unreadable_attestation_record_withholds_the_run_instead_of_failing_the_node() -> None:
    import redis.exceptions

    from services.task.models import VALIDATOR_REDIS_UNAVAILABLE

    service = _task_service_that_reaches_the_ssh_connect()
    service.attestation_service.prepare_host_policy = AsyncMock(return_value=_PASSED)
    service.attestation_service.has_recent_nonce_bound_pass = AsyncMock(
        side_effect=redis.exceptions.ConnectionError("max number of clients reached")
    )
    service.attestation_service.record_attestation_event = AsyncMock()
    miner, executor = _a_job_for("node-9")

    result = await _run_cycle(service, miner, executor)

    assert result.failure_reason_code == VALIDATOR_REDIS_UNAVAILABLE
    service.attestation_service.record_attestation_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_executor_left_out_of_the_due_challenge_loses_its_earlier_pass(monkeypatch) -> None:
    from types import SimpleNamespace

    from core.config import settings
    from services import attestation_service as attestation_module

    now = [1_000_000.0]
    monkeypatch.setattr(attestation_module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(settings, "TDX_VERIFIER_URL", "https://verifier.example/verify")
    monkeypatch.setattr(settings, "TDX_REATTEST_INTERVAL_SECONDS", 3600)
    attestation = attestation_module.AttestationService(redis_service=_DictRedis(clock=lambda: now[0]))

    await _wave(attestation, "5Miner", [("A", _PASSED), ("B", _PASSED)])  # 00:00, challenged
    now[0] += 1800  # 00:30: the express lane challenges a new node C alone
    await _wave(attestation, "5Miner", [("C", _PASSED)], executor_id="C")
    now[0] += 1800  # 01:00: due again; A is left out, B passes
    await _wave(attestation, "5Miner", [("B", _PASSED)])
    now[0] += 300  # 01:05: no challenge; A comes back with its old evidence

    assert await _wave(attestation, "5Miner", [("A", _PASSED), ("B", _PASSED), ("C", _PASSED)]) == {
        "A": False,
        "B": True,
        "C": False,
    }


@pytest.mark.asyncio
async def test_another_host_key_under_a_passed_executor_id_has_not_passed(monkeypatch) -> None:
    from core.config import settings
    from services.attestation_service import AttestationService

    monkeypatch.setattr(settings, "TDX_VERIFIER_URL", "https://verifier.example/verify")
    attestation = AttestationService(redis_service=_DictRedis())
    await _wave(attestation, "5Miner", [("A", _PASSED), ("B", _PASSED)])

    reached = await _wave(
        attestation, "5Miner", [("A", _PASSED), ("B", _PASSED)], host_keys={"B": "ssh-ed25519 another-key"}
    )

    assert reached == {"A": True, "B": False}
