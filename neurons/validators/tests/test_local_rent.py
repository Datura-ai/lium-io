"""liumd deploy: the rental container made by ONE signed `POST /rent` on the executor.

Fake executor: an in-process aiohttp `/rent` that checks the signature and replay the way the
real route does and answers what the test tells it to. The validator side under test:
eligibility (nothing private on plain HTTP), the intent, the answer's reading, and
`DockerService._create_with_local_rent` — taken, not taken, fell back, freed the name.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import bittensor
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from datura.rental_spec import (
    PUBLIC_ENVIRONMENT,
    RENTAL_CONTAINER_NAME_PREFIXES,
    RENTAL_NETWORK_NAME,
    ContainerRunSpec,
    ContainerUlimit,
    DeviceMount,
    GpuDeviceRequest,
    PortBinding,
    VolumeMount,
    WireError,
    build_host_config_kwargs,
    carries_only_public_fields,
    create_and_start,
    spec_from_wire,
    spec_to_wire,
)
from datura.requests.miner_requests import ExecutorSSHInfo
from services.docker_service import DockerService
from services.local_rent_client import (
    SCHEMA,
    LocalRentClient,
    LocalRentUnavailable,
    build_intent,
    eligible,
    executor_deadline_s,
    host_key_sha256,
    parse_answer,
)
from services.local_verify_client import canonical_intent_message

from core.config import settings
from services import local_rent_client as lrc

EXECUTOR_UUID = "11111111-2222-3333-4444-555555555555"
HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExecutorHostKey0000000000000000000000000000 root@executor"


@pytest.fixture(scope="module")
def keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidator")


def _spec(**overrides) -> ContainerRunSpec:
    fields = dict(
        image="daturaai/ubuntu:24.04",
        name="pod_abc",
        environment=dict(PUBLIC_ENVIRONMENT),
        ports=(PortBinding(22, 40001), PortBinding(8888, 40002)),
        volumes=(VolumeMount("pod_abc_vol", "/root"),),
        restart_policy="unless-stopped",
        runtime="sysbox-runc",
        cap_add=("NET_ADMIN",),
        sysctls={"net.ipv4.conf.all.src_valid_mark": "1"},
        ulimits=(ContainerUlimit("memlock", -1, -1),),
        devices=(DeviceMount("/dev/net/tun", "/dev/net/tun"), DeviceMount("/dev/nvidia0")),
        device_requests=(GpuDeviceRequest(device_ids=("GPU-1",)),),
        cpu_count=8,
        memory_gb=32,
        storage_limit_gb=100,
        shm_size="16g",
        network=RENTAL_NETWORK_NAME,
    )
    fields.update(overrides)
    return ContainerRunSpec(**fields)


# --- the spec on the wire (datura) -------------------------------------------------------------


def test_the_spec_round_trips_through_the_wire():
    spec = _spec()
    wire = json.loads(json.dumps(spec_to_wire(spec)))
    assert spec_from_wire(wire) == spec


def test_the_spec_the_validator_really_builds_reaches_the_executor_whole(svc):
    """Regression (fresh review on this PR): `_build_rental_container_run_spec` names the icc-off
    rental network (DAH-3199) but the wire did not carry `network`, so an executor-made rental
    landed on docker0 with inter-container traffic on and nothing said so. The real builder's
    output, not a hand-built spec, must survive the wire field for field — and the executor's
    HostConfig for it must be the SDK path's, `network_mode` included."""
    from payload_models.payloads import ContainerCreateRequest, CustomOptions
    from services.rental_docker_sdk import GpuDockerConfig

    payload = ContainerCreateRequest(
        miner_hotkey="hk", executor_id=EXECUTOR_UUID, pod_id="abc", docker_image="daturaai/ubuntu:24.04",
        gpu_uuids=["GPU-1"], is_sysbox=True,
    )
    built = svc._build_rental_container_run_spec(
        payload=payload,
        container_name="pod_abc",
        custom_options=CustomOptions(),
        port_maps=[(22, 40001, 50001)],
        local_volume="pod_abc_vol",
        local_volume_path="/root",
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None,
        cpu_count=None,
    )
    assert built.network == RENTAL_NETWORK_NAME and carries_only_public_fields(built)
    wire = json.loads(json.dumps(spec_to_wire(built)))
    assert wire["network"] == RENTAL_NETWORK_NAME
    parsed = spec_from_wire(wire)
    assert parsed == built
    assert build_host_config_kwargs(parsed)["network_mode"] == RENTAL_NETWORK_NAME


def test_a_spec_without_a_network_builds_no_network_mode():
    """The SDK path's one create on the default bridge — the CVM quote broker (never a rental, never
    on the wire): a `network=None` spec must not put `network_mode` in the HostConfig."""
    assert "network_mode" not in build_host_config_kwargs(_spec(network=None))


def test_a_zero_limit_travels_as_no_limit_and_builds_the_same_host_config():
    """The SDK path treats a falsy cpu_count / memory_gb / storage_limit_gb as 'no limit' (a pod's
    ram_total defaults to 0); the wire must say the same thing the executor's strict parser reads,
    or every such rental would be refused 'not an integer in 1..' and fall back after a wasted call."""
    spec = _spec(cpu_count=0, memory_gb=0, storage_limit_gb=0)
    wire = json.loads(json.dumps(spec_to_wire(spec)))
    assert wire["cpu_count"] is None and wire["memory_gb"] is None and wire["storage_limit_gb"] is None
    parsed = spec_from_wire(wire)
    assert (parsed.cpu_count, parsed.memory_gb, parsed.storage_limit_gb) == (None, None, None)
    assert build_host_config_kwargs(parsed) == build_host_config_kwargs(spec)


@pytest.mark.parametrize(
    "mutate, why",
    [
        (lambda w: w.update(surprise=1), "unknown field"),
        (lambda w: w.update(name="../etc"), "name"),
        (lambda w: w.update(image=""), "image"),
        (lambda w: w["ports"].append({"container_port": 22, "host_port": 70000}), "host_port"),
        (lambda w: w["ports"].append({"container_port": 22, "host_port": 1, "protocol": "sctp"}), "protocol"),
        (lambda w: w.update(cpu_count=-1), "cpu_count"),
        (lambda w: w.update(restart_policy="forever"), "restart_policy"),
        (lambda w: w.update(network="lium rentals; rm -rf"), "network"),
        (lambda w: w["volumes"].append({"source": "x", "target": "relative"}), "target"),
        (lambda w: w.update(ports=[{"container_port": 22, "host_port": 40001}] * 600), "ports"),
        (lambda w: w.update(environment={"A" * 5000: "b"}), "environment"),
        (lambda w: w["devices"].append({"path_on_host": "/etc/passwd"}), "device"),
        (lambda w: w["devices"].append({"path_on_host": "/dev/nvidia1", "permissions": "rwx"}), "device"),
        (lambda w: w["devices"].append({"path_on_host": "/dev/nvidia1", "path_in_container": "relative"}), "device"),
    ],
)
def test_a_wire_document_outside_a_rentals_shape_is_refused(mutate, why):
    wire = spec_to_wire(_spec())
    mutate(wire)
    with pytest.raises(WireError) as exc:
        spec_from_wire(wire)
    assert why in str(exc.value)


def test_a_missing_optional_field_takes_the_specs_default():
    wire = spec_to_wire(_spec())
    for key in ("cap_add", "sysctls", "ulimits", "devices", "device_requests", "shm_size", "cpu_count", "restart_policy", "network"):
        wire.pop(key)
    parsed = spec_from_wire(wire)
    assert parsed.cap_add == () and parsed.shm_size is None and parsed.cpu_count is None
    # absent = the dataclass default: a rental restarts with the daemon; a spec that wants none says null
    assert parsed.restart_policy == "unless-stopped" and parsed.network is None
    assert spec_from_wire({**spec_to_wire(_spec()), "restart_policy": None}).restart_policy is None
    # a device request without the capabilities key is the dataclass default (gpu), not "none"
    wire = spec_to_wire(_spec())
    wire["device_requests"][0].pop("capabilities")
    assert spec_from_wire(wire).device_requests == (GpuDeviceRequest(device_ids=("GPU-1",)),)


def test_the_executors_name_rule_names_the_validators_own_prefixes():
    """The executor refuses an intent whose container name is not a rental's or a filler's
    (`RENTAL_CONTAINER_NAME_PREFIXES`, shared); the validator's own prefixes must be in that set."""
    from services.const import FILLER_CONTAINER_PREFIX, POD_CONTAINER_PREFIX

    assert POD_CONTAINER_PREFIX in RENTAL_CONTAINER_NAME_PREFIXES
    assert FILLER_CONTAINER_PREFIX in RENTAL_CONTAINER_NAME_PREFIXES
    assert DockerService.get_container_name(Mock(workload_kind=None, pod_id="abc")).startswith(POD_CONTAINER_PREFIX)


def test_only_a_spec_with_nothing_private_may_travel_on_http():
    assert carries_only_public_fields(_spec())
    assert eligible(_spec(), HOST_KEY) is None
    assert eligible(_spec(), None) == "no_host_key" and eligible(_spec(), "  ") == "no_host_key"
    assert not carries_only_public_fields(_spec(command=("bash", "-c", "echo $SECRET")))
    assert not carries_only_public_fields(_spec(entrypoint="/my/entry.sh"))
    assert not carries_only_public_fields(_spec(environment={**PUBLIC_ENVIRONMENT, "HF_TOKEN": "hf_x"}))
    assert not carries_only_public_fields(_spec(environment={}))
    assert eligible(_spec(environment={**PUBLIC_ENVIRONMENT, "JUPYTER_PASSWORD": "t"}), HOST_KEY) == "private_fields"


def test_create_and_start_issues_the_sdk_paths_exact_calls():
    """The validator's SDK path and the executor's /rent call ONE function; the create sees the
    spec's fields the way docker-py wants them and the HostConfig is `build_host_config_kwargs`."""
    api = Mock()
    api.create_host_config.side_effect = lambda **kw: {"HostConfig": kw}
    api.create_container.return_value = {"Id": "abc123"}
    spec = _spec()
    assert create_and_start(api, spec) == "abc123"
    api.create_host_config.assert_called_once_with(**build_host_config_kwargs(spec))
    kwargs = api.create_container.call_args.kwargs
    assert kwargs["image"] == spec.image and kwargs["name"] == "pod_abc" and kwargs["detach"] is True
    assert kwargs["command"] is None and kwargs["entrypoint"] is None
    assert kwargs["ports"] == [(22, "tcp"), (8888, "tcp")]
    assert kwargs["volumes"] == ["/root"]
    assert kwargs["host_config"] == {"HostConfig": build_host_config_kwargs(spec)}
    assert "labels" not in kwargs  # the validator's own create carries no label
    api.start.assert_called_once_with("pod_abc")


def test_create_and_start_hands_the_id_over_before_start_and_labels_only_when_asked():
    api = Mock()
    api.create_host_config.side_effect = lambda **kw: {"HostConfig": kw}
    api.create_container.return_value = {"Id": "abc123"}
    api.start.side_effect = RuntimeError("Bind for 0.0.0.0:40001 failed: port is already allocated")
    seen: list[str] = []
    with pytest.raises(RuntimeError):
        create_and_start(api, _spec(), labels={"lium.local_rent.nonce": "n"}, on_created=seen.append)
    assert seen == ["abc123"]  # the caller knows what to remove although start failed
    assert api.create_container.call_args.kwargs["labels"] == {"lium.local_rent.nonce": "n"}


# --- the intent and the answer -----------------------------------------------------------------


def test_the_intent_carries_the_spec_and_asks_for_sshd_only_when_the_image_ships_it():
    spec = _spec()
    with_sshd = build_intent(executor_uuid=EXECUTOR_UUID, host_key=HOST_KEY, spec=spec, deadline_s=40, ssh_host_port=40001, ssh_wait_s=10, now=1000)
    assert with_sshd["schema"] == SCHEMA and with_sshd["executor_uuid"] == EXECUTOR_UUID
    assert with_sshd["ssh_host_key_sha256"] == host_key_sha256(HOST_KEY) == host_key_sha256(f"  {HOST_KEY}\n")
    assert len(with_sshd["ssh_host_key_sha256"]) == 64
    assert with_sshd["issued_at"] == 1000 and with_sshd["expires_at"] == 1120
    assert with_sshd["deadline_s"] == 40
    assert with_sshd["steps"]["image"] is True
    assert spec_from_wire(with_sshd["steps"]["container"]) == spec
    assert with_sshd["steps"]["ready"] == {"running_timeout_s": 10, "ssh_host_port": 40001, "ssh_timeout_s": 10}
    without = build_intent(executor_uuid=EXECUTOR_UUID, host_key=HOST_KEY, spec=spec, deadline_s=40, ssh_host_port=None, ssh_wait_s=10)
    assert without["steps"]["ready"] == {"running_timeout_s": 10}
    no_wait = build_intent(executor_uuid=EXECUTOR_UUID, host_key=HOST_KEY, spec=spec, deadline_s=40, ssh_host_port=40001, ssh_wait_s=0)
    assert no_wait["steps"]["ready"] == {"running_timeout_s": 10}
    assert without["nonce"] != with_sshd["nonce"]
    # an operator's LOCAL_RENT_SSHD_WAIT_SECONDS=90 is not a 422 from the executor's `le=60` on every rent
    long_wait = build_intent(executor_uuid=EXECUTOR_UUID, host_key=HOST_KEY, spec=spec, deadline_s=40, ssh_host_port=40001, ssh_wait_s=90)
    assert long_wait["steps"]["ready"]["ssh_timeout_s"] == 60


def _answer(intent, **overrides) -> dict:
    answer = {
        "schema": SCHEMA,
        "nonce": intent["nonce"],
        "executor_uuid": intent["executor_uuid"],
        "executor_version": "4.2.0",
        "started_at": 1000,
        "elapsed_ms": 1500,
        "deadline_hit": False,
        "rolled_back": False,
        "steps": {
            "image": {"status": "ok", "ms": 10, "data": {"present": True, "digest": "sha256:abc"}},
            "container": {"status": "ok", "ms": 1200, "data": {"container_name": "pod_abc", "container_id": "c1"}},
            "ready": {"status": "ok", "ms": 300, "data": {"state": {"Running": True}, "ssh_answered": True}},
        },
    }
    answer.update(overrides)
    return answer


def test_a_good_answer_reads_as_created():
    intent = build_intent(executor_uuid=EXECUTOR_UUID, host_key=HOST_KEY, spec=_spec(), deadline_s=40, ssh_host_port=40001)
    answer = parse_answer(_answer(intent), intent=intent, round_trip_ms=1600)
    assert answer.created and not answer.may_hold_the_name
    assert answer.step("ready").data["ssh_answered"] is True
    assert answer.executor_version == "4.2.0" and answer.round_trip_ms == 1600


@pytest.mark.parametrize(
    "overrides, holds_the_name",
    [
        ({"deadline_hit": True, "rolled_back": True}, False),
        ({"rolled_back": True, "steps": {"container": {"status": "ok"}, "ready": {"status": "timeout"}}}, False),
        ({"steps": {"container": {"status": "ok"}, "ready": {"status": "failed", "error": "exit 127"}}}, True),
        ({"rolled_back": True, "steps": {"container": {"status": "failed", "error": "port is already allocated"}}}, False),
        # a create the deadline cut before the daemon answered: the executor cannot prove the name free
        ({"deadline_hit": True, "steps": {"container": {"status": "timeout"}}}, True),
        ({"rolled_back": True, "steps": {"image": {"status": "ok", "data": {"present": False}}}}, False),
        ({"steps": {"container": {"status": "ok"}}}, True),  # no ready step answered: not proven running
    ],
)
def test_anything_short_of_made_and_running_is_not_created(overrides, holds_the_name):
    intent = build_intent(executor_uuid=EXECUTOR_UUID, host_key=HOST_KEY, spec=_spec(), deadline_s=40, ssh_host_port=40001)
    answer = parse_answer(_answer(intent, **overrides), intent=intent, round_trip_ms=1)
    assert not answer.created
    assert answer.may_hold_the_name is holds_the_name


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda a: a.update(schema="lium.local_verify/1"), "schema_mismatch"),
        (lambda a: a.update(nonce="other"), "nonce_mismatch"),
        (lambda a: a.update(executor_uuid="other"), "executor_mismatch"),
        (lambda a: a.update(steps=[]), "malformed"),
    ],
)
def test_an_answer_to_another_intent_is_refused(mutate, reason):
    intent = build_intent(executor_uuid=EXECUTOR_UUID, host_key=HOST_KEY, spec=_spec(), deadline_s=40, ssh_host_port=None)
    raw = _answer(intent)
    mutate(raw)
    with pytest.raises(LocalRentUnavailable) as exc:
        parse_answer(raw, intent=intent, round_trip_ms=1)
    assert exc.value.reason == reason


# --- the client against a fake executor --------------------------------------------------------


class FakeExecutor:
    def __init__(self, keypair, *, answer=None, status=200, sleep=0.0):
        self.keypair = keypair
        self.answer = answer  # callable(intent) -> dict
        self.status = status
        self.sleep = sleep
        self.intents: list[dict] = []
        self.seen: set[str] = set()
        self.app = web.Application()
        self.app.router.add_post("/rent", self.rent)
        self.server = TestServer(self.app)

    async def __aenter__(self):
        await self.server.start_server()
        return self

    async def __aexit__(self, *exc):
        await self.server.close()

    @property
    def executor_info(self) -> ExecutorSSHInfo:
        return ExecutorSSHInfo(
            uuid=EXECUTOR_UUID,
            address=self.server.host,
            port=self.server.port,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python",
            root_dir="/root/app",
            ssh_host_key=HOST_KEY,
        )

    async def rent(self, request):
        raw = await request.json()
        if not self.keypair.verify(canonical_intent_message(raw), raw["signature"]):
            return web.json_response({"detail": "Invalid signature"}, status=401)
        if raw["nonce"] in self.seen:
            return web.json_response({"detail": "nonce already used"}, status=409)
        if raw.get("ssh_host_key_sha256") != host_key_sha256(HOST_KEY):
            return web.json_response({"detail": "Intent refused: bound to another executor"}, status=401)
        self.seen.add(raw["nonce"])
        self.intents.append(raw)
        await asyncio.sleep(self.sleep)
        if self.status != 200:
            return web.json_response({"detail": "no"}, status=self.status)
        return web.json_response(self.answer(raw) if self.answer else _answer(raw))


def _client(keypair, timeout_s=5) -> LocalRentClient:
    return LocalRentClient(keypair, timeout_s=timeout_s, connect_timeout_s=2)


def test_the_client_posts_a_signed_intent_the_executor_accepts(keypair):
    async def scenario():
        async with FakeExecutor(keypair) as executor:
            intent = build_intent(executor_uuid=EXECUTOR_UUID, host_key=HOST_KEY, spec=_spec(), deadline_s=40, ssh_host_port=40001)
            answer = await _client(keypair).rent(executor.executor_info, intent)
            return executor.intents, answer

    intents, answer = asyncio.run(scenario())
    assert len(intents) == 1 and intents[0]["schema"] == SCHEMA and "signature" in intents[0]
    assert intents[0]["steps"]["container"]["name"] == "pod_abc"
    assert answer.created


@pytest.mark.parametrize("status, reason", [(404, "not_supported"), (409, "busy_or_replay"), (401, "refused"), (500, "http_error")])
def test_every_non_200_is_a_labelled_unavailable(keypair, status, reason):
    async def scenario():
        async with FakeExecutor(keypair, status=status) as executor:
            intent = build_intent(executor_uuid=EXECUTOR_UUID, host_key=HOST_KEY, spec=_spec(), deadline_s=40, ssh_host_port=None)
            with pytest.raises(LocalRentUnavailable) as exc:
                await _client(keypair).rent(executor.executor_info, intent)
            return exc.value.reason

    assert asyncio.run(scenario()) == reason


def test_a_slow_executor_is_a_timeout(keypair):
    async def scenario():
        async with FakeExecutor(keypair, sleep=1.5) as executor:
            intent = build_intent(executor_uuid=EXECUTOR_UUID, host_key=HOST_KEY, spec=_spec(), deadline_s=40, ssh_host_port=None)
            with pytest.raises(LocalRentUnavailable) as exc:
                await _client(keypair, timeout_s=1).rent(executor.executor_info, intent)
            return exc.value.reason

    assert asyncio.run(scenario()) == "timeout"


def test_a_stranger_cannot_make_the_executor_create(keypair):
    stranger = bittensor.Keypair.create_from_uri("//Stranger")

    async def scenario():
        async with FakeExecutor(keypair) as executor:
            intent = build_intent(executor_uuid=EXECUTOR_UUID, host_key=HOST_KEY, spec=_spec(), deadline_s=40, ssh_host_port=None)
            with pytest.raises(LocalRentUnavailable) as exc:
                await _client(stranger).rent(executor.executor_info, intent)
            return exc.value.reason, executor.intents

    reason, intents = asyncio.run(scenario())
    assert reason == "refused" and intents == []


# --- DockerService: taken, not taken, fell back ------------------------------------------------


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


def _create(svc, executor_info, keypair, spec, *, image_ships_sshd=True, docker_client=None):
    return asyncio.run(
        svc._create_with_local_rent(
            executor_info=executor_info,
            keypair=keypair,
            docker_client=docker_client or Mock(remove_container=AsyncMock()),
            run_spec=spec,
            image_ships_sshd=image_ships_sshd,
            default_extra={"executor_id": EXECUTOR_UUID},
        )
    )


def _offline_executor(host_key: str | None = HOST_KEY) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=EXECUTOR_UUID, address="127.0.0.1", port=9, ssh_username="root", ssh_port=22,
        python_path="/usr/bin/python", root_dir="/root/app", ssh_host_key=host_key,
    )


def test_the_flag_ships_off_and_nothing_is_posted(svc, keypair, monkeypatch):
    assert settings.VALIDATOR_LOCAL_RENT_ENABLED is False
    posted = Mock()
    monkeypatch.setattr(lrc.LocalRentClient, "rent", posted)
    assert _create(svc, _offline_executor(), keypair, _spec()) is None
    posted.assert_not_called()


def test_a_spec_with_private_fields_never_leaves_the_tunnel(svc, keypair, monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)
    posted = Mock()
    monkeypatch.setattr(lrc.LocalRentClient, "rent", posted)
    spec = _spec(environment={**PUBLIC_ENVIRONMENT, "HF_TOKEN": "hf_secret"})
    assert _create(svc, _offline_executor(), keypair, spec) is None
    posted.assert_not_called()


def test_an_executor_without_a_pinned_host_key_gets_no_intent(svc, keypair, monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)
    posted = Mock()
    monkeypatch.setattr(lrc.LocalRentClient, "rent", posted)
    assert _create(svc, _offline_executor(host_key=None), keypair, _spec()) is None
    posted.assert_not_called()


def test_the_intent_is_bound_to_the_executors_host_key(svc, keypair, monkeypatch):
    """A fake executor with another host key refuses the intent (as the real route does), and the
    validator falls back — an intent captured on the wire creates nothing elsewhere."""
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)

    async def scenario():
        async with FakeExecutor(keypair) as executor:
            info = executor.executor_info.model_copy(update={"ssh_host_key": "ssh-ed25519 AAAA-other root@other"})
            answer = await svc._create_with_local_rent(
                executor_info=info, keypair=keypair, docker_client=Mock(),
                run_spec=_spec(), image_ships_sshd=True, default_extra={},
            )
            return answer, executor.intents

    answer, intents = asyncio.run(scenario())
    assert answer is None and intents == []


def test_a_created_answer_is_taken_and_the_sshd_wait_is_asked_for_only_when_the_image_ships_it(svc, keypair, monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_RENT_TIMEOUT_SECONDS", 20)
    monkeypatch.setattr(settings, "LOCAL_RENT_SSHD_WAIT_SECONDS", 8)

    async def scenario():
        async with FakeExecutor(keypair) as executor:
            taken = await svc._create_with_local_rent(
                executor_info=executor.executor_info, keypair=keypair, docker_client=Mock(),
                run_spec=_spec(), image_ships_sshd=True, default_extra={},
            )
            also = await svc._create_with_local_rent(
                executor_info=executor.executor_info, keypair=keypair, docker_client=Mock(),
                run_spec=_spec(), image_ships_sshd=False, default_extra={},
            )
            return taken, also, executor.intents

    taken, also, intents = asyncio.run(scenario())
    assert taken is not None and taken.created
    assert also is not None and also.created
    assert intents[0]["steps"]["ready"] == {"running_timeout_s": 10, "ssh_host_port": 40001, "ssh_timeout_s": 8}
    assert "ssh_host_port" not in intents[1]["steps"]["ready"]
    assert intents[0]["deadline_s"] == executor_deadline_s(20) == 5  # 20 − 15 rollback margin, floor 5
    assert executor_deadline_s(45) == 30


def test_the_sshd_wait_ships_off(svc, keypair, monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)
    assert settings.LOCAL_RENT_SSHD_WAIT_SECONDS == 0

    async def scenario():
        async with FakeExecutor(keypair) as executor:
            await svc._create_with_local_rent(
                executor_info=executor.executor_info, keypair=keypair, docker_client=Mock(),
                run_spec=_spec(), image_ships_sshd=True, default_extra={},
            )
            return executor.intents

    intents = asyncio.run(scenario())
    assert intents[0]["steps"]["ready"] == {"running_timeout_s": 10}


def test_an_executor_without_the_route_means_the_sdk_path(svc, keypair, monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)

    async def scenario():
        async with FakeExecutor(keypair, status=404) as executor:
            return await svc._create_with_local_rent(
                executor_info=executor.executor_info, keypair=keypair, docker_client=Mock(),
                run_spec=_spec(), image_ships_sshd=True, default_extra={},
            )

    assert asyncio.run(scenario()) is None


def test_an_unreachable_executor_means_the_sdk_path_and_the_name_is_left_alone(svc, keypair, monkeypatch):
    """A connection that was never made carried no intent: nothing over there to free."""
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)
    removed = AsyncMock()
    assert _create(svc, _offline_executor(), keypair, _spec(), docker_client=Mock(remove_container=removed)) is None
    removed.assert_not_awaited()


def test_an_old_images_422_is_a_no_route_answer_and_the_name_is_left_alone(svc, keypair, monkeypatch):
    """An executor image without the route answers the intent 422 from `MinerMiddleware` (no
    `data_to_sign`), before anything could be created: the SDK path, and no `rm` over the tunnel."""
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)

    async def scenario():
        removed = AsyncMock()
        async with FakeExecutor(keypair, status=422) as executor:
            answer = await svc._create_with_local_rent(
                executor_info=executor.executor_info, keypair=keypair,
                docker_client=Mock(remove_container=removed),
                run_spec=_spec(), image_ships_sshd=True, default_extra={},
            )
        return answer, removed

    answer, removed = asyncio.run(scenario())
    assert answer is None
    removed.assert_not_awaited()


def test_a_non_answer_after_the_intent_left_frees_the_name_before_the_sdk_run(svc, keypair, monkeypatch):
    """A total timeout, a connection that broke after the send, a 5xx or an unreadable answer: the
    executor may have made the container under the pod's name, so the fallback force-removes the
    name first (as after an unproven rollback) — or the SDK `run` would answer 'name in use'."""
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_RENT_TIMEOUT_SECONDS", 1)

    async def scenario(**executor_kw):
        removed = AsyncMock()
        async with FakeExecutor(keypair, **executor_kw) as executor:
            answer = await svc._create_with_local_rent(
                executor_info=executor.executor_info, keypair=keypair,
                docker_client=Mock(remove_container=removed),
                run_spec=_spec(), image_ships_sshd=True, default_extra={},
            )
        return answer, removed

    for executor_kw in (dict(sleep=3), dict(status=500), dict(answer=lambda intent: {"schema": "other"})):
        answer, removed = asyncio.run(scenario(**executor_kw))
        assert answer is None, executor_kw
        removed.assert_awaited_once()
        assert removed.await_args.kwargs["container_name"] == "pod_abc" and removed.await_args.kwargs["force"] is True


@pytest.mark.parametrize(
    "reason, detail, acted",
    [
        ("not_supported", "executor answered 404", False),
        ("refused", "Intent refused", False),
        ("busy_or_replay", "nonce already used", False),
        ("transport", "ClientConnectorError: Cannot connect to host", False),
        ("transport", "ClientConnectorDNSError: no such host", False),
        ("transport", "ServerDisconnectedError: ", True),
        ("timeout", "no answer within 45s", True),
        ("http_error", "status 500", True),
        ("http_error", "status 422: {'detail': 'data_to_sign missing'}", False),  # an old image's middleware, or the model check
        ("malformed", "answer is not JSON", True),
        ("schema_mismatch", "got 'x'", True),
    ],
)
def test_which_non_answers_leave_it_open_whether_the_executor_acted(reason, detail, acted):
    assert lrc.may_have_acted(reason, detail) is acted


def test_a_container_the_executor_made_but_did_not_see_running_is_removed_before_the_sdk_run(svc, keypair, monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)
    removed = AsyncMock()
    docker_client = Mock(remove_container=removed)

    def answer(intent):
        return _answer(
            intent,
            steps={"container": {"status": "ok"}, "ready": {"status": "failed", "error": "container is exited: exit 127"}},
            rolled_back=False,
        )

    async def scenario():
        async with FakeExecutor(keypair, answer=answer) as executor:
            return await svc._create_with_local_rent(
                executor_info=executor.executor_info, keypair=keypair, docker_client=docker_client,
                run_spec=_spec(), image_ships_sshd=True, default_extra={},
            )

    assert asyncio.run(scenario()) is None
    removed.assert_awaited_once()
    assert removed.await_args.kwargs["container_name"] == "pod_abc" and removed.await_args.kwargs["force"] is True


def test_a_create_the_deadline_cut_is_freed_by_name_before_the_sdk_run(svc, keypair, monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)
    removed = AsyncMock()

    def answer(intent):
        return _answer(intent, deadline_hit=True, rolled_back=False, steps={"container": {"status": "timeout"}})

    async def scenario():
        async with FakeExecutor(keypair, answer=answer) as executor:
            return await svc._create_with_local_rent(
                executor_info=executor.executor_info, keypair=keypair, docker_client=Mock(remove_container=removed),
                run_spec=_spec(), image_ships_sshd=True, default_extra={},
            )

    assert asyncio.run(scenario()) is None
    removed.assert_awaited_once()


def test_a_rolled_back_failure_leaves_the_name_alone(svc, keypair, monkeypatch):
    monkeypatch.setattr(settings, "VALIDATOR_LOCAL_RENT_ENABLED", True)
    removed = AsyncMock()

    def answer(intent):
        return _answer(
            intent,
            steps={"container": {"status": "failed", "error": "Bind for 0.0.0.0:40001 failed: port is already allocated"}},
            rolled_back=True,
        )

    async def scenario():
        async with FakeExecutor(keypair, answer=answer) as executor:
            return await svc._create_with_local_rent(
                executor_info=executor.executor_info, keypair=keypair, docker_client=Mock(remove_container=removed),
                run_spec=_spec(), image_ships_sshd=True, default_extra={},
            )

    assert asyncio.run(scenario()) is None
    removed.assert_not_awaited()

