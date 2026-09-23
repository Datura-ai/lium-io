"""DAH-3798 (ticket-0355): a rental container's memory limit keeps a reserve for the host.

The limit calculation (`services/rental_memory.py`), the Docker flags it becomes (`--memory`,
`--memory-swap` equal to it, `--oom-score-adj`), and the create flow reading the host's RAM over the
rent's SSH connection before `docker run`.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from core.config import settings
from services.rental_docker_sdk import (
    ContainerRunSpec,
    RentalDockerSdkClient,
    _build_host_config_kwargs,
)
from services.rental_memory import (
    HOST_MEMORY_PROBE_CMD,
    KIB_PER_GIB,
    SOURCE_CAP_DISABLED,
    SOURCE_CLAMPED,
    SOURCE_HOST_MINUS_RESERVE,
    SOURCE_HOST_UNKNOWN,
    SOURCE_REQUESTED,
    gpu_share_of_host,
    host_memory_reserve_gb,
    parse_host_memory_probe,
    rental_memory_limit,
    resolve_rental_memory_limit,
)
from tests.test_deploy_optimizations import (  # the create-flow harness these tests share
    _created_run_spec,
    _patch_happy,
    _payload,
    _run,
    _ssh_client,
    _ssh_result,
    svc,  # noqa: F401  (pytest fixture)
)

# ticket-0355's guest: 8x H200 CVM; 99.7 % used with 5.6 GB available puts MemTotal near 1.8 TiB.
H200_GUEST_KIB = int(1843 * KIB_PER_GIB)
DEFAULTS = {"reserve_min_gb": 4, "reserve_percent": 3.0}


def _backend_memory_gb(host_kib: int, share: float = 1.0) -> int:
    """What lium-platform sends today: int((ram.total − 2 GiB) × share / 2**20)."""
    return int((host_kib - 2 * KIB_PER_GIB) * share / KIB_PER_GIB)


# --- the limit -------------------------------------------------------------------------------------


def test_the_reserve_is_the_percent_of_host_ram_rounded_up_and_never_below_the_minimum():
    assert host_memory_reserve_gb(1843, reserve_min_gb=4, reserve_percent=3.0) == 56  # 55.29 → 56
    assert host_memory_reserve_gb(256, reserve_min_gb=4, reserve_percent=3.0) == 8  # 7.68 → 8
    assert (
        host_memory_reserve_gb(64, reserve_min_gb=4, reserve_percent=3.0) == 4
    )  # 1.92 → the minimum
    assert host_memory_reserve_gb(64, reserve_min_gb=4, reserve_percent=-5) == 4


def test_ticket_0355_guest_whole_host_rental_is_clamped_to_ram_less_the_reserve():
    # 1841: 2 GiB left for the guest, as in ticket-0355
    requested = _backend_memory_gb(H200_GUEST_KIB)

    limit = rental_memory_limit(requested, H200_GUEST_KIB, **DEFAULTS)

    assert requested == 1841
    assert (limit.limit_gb, limit.reserve_gb, limit.ceiling_gb, limit.source) == (
        1787,
        56,
        1787,
        SOURCE_CLAMPED,
    )
    assert limit.swap_off is True


def test_a_request_under_the_ceiling_is_kept_as_sent():
    limit = rental_memory_limit(200, H200_GUEST_KIB, gpu_share=1 / 8, **DEFAULTS)

    assert (limit.limit_gb, limit.ceiling_gb, limit.source) == (200, 223, SOURCE_REQUESTED)


def test_a_split_host_pod_is_clamped_to_its_gpu_share_of_ram_less_the_reserve():
    host_kib = 2048 * KIB_PER_GIB  # 8 GPUs, reserve 62 GiB
    requested = _backend_memory_gb(host_kib, share=1 / 8)  # 255

    limit = rental_memory_limit(requested, host_kib, gpu_share=1 / 8, **DEFAULTS)

    assert (requested, limit.limit_gb, limit.reserve_gb, limit.source) == (
        255,
        248,
        62,
        SOURCE_CLAMPED,
    )
    # eight such pods leave the reserve free
    assert 8 * limit.limit_gb <= 2048 - limit.reserve_gb


@pytest.mark.parametrize("requested", [None, 0])
def test_a_pod_row_sized_zero_gets_the_ceiling_instead_of_no_limit(requested):
    limit = rental_memory_limit(requested, 128 * KIB_PER_GIB, **DEFAULTS)

    assert (limit.limit_gb, limit.requested_gb, limit.source) == (
        124,
        None,
        SOURCE_HOST_MINUS_RESERVE,
    )


@pytest.mark.parametrize("host_kib", [None, 0, -1])
def test_an_unknown_host_ram_keeps_the_backend_value(host_kib):
    assert rental_memory_limit(30, host_kib, **DEFAULTS).limit_gb == 30
    assert rental_memory_limit(30, host_kib, **DEFAULTS).source == SOURCE_HOST_UNKNOWN
    assert rental_memory_limit(0, host_kib, **DEFAULTS).limit_gb is None


def test_a_host_smaller_than_its_reserve_still_gets_one_gib():
    limit = rental_memory_limit(2, 3 * KIB_PER_GIB, **DEFAULTS)

    assert (limit.limit_gb, limit.source) == (1, SOURCE_CLAMPED)


def test_gpu_share_of_host():
    assert gpu_share_of_host(["a"], 8) == 1 / 8
    assert gpu_share_of_host(["a", "b", "c", "d"], 8) == 0.5
    assert gpu_share_of_host([], 8) == 1.0  # whole host (--gpus all)
    assert gpu_share_of_host(["a"], None) == 1.0  # count unknown
    assert gpu_share_of_host(["a"] * 8, 8) == 1.0
    assert gpu_share_of_host(["a"] * 9, 8) == 1.0


def test_parse_host_memory_probe():
    assert parse_host_memory_probe("1932510208\n8\n") == (1932510208, 8)
    assert parse_host_memory_probe("1932510208\n0\n") == (1932510208, 0)
    assert parse_host_memory_probe("") == (None, None)
    assert parse_host_memory_probe("awk: not found\n8\n") == (None, 8)


def test_describe_names_the_limit_the_host_and_the_reserve():
    limit = rental_memory_limit(None, H200_GUEST_KIB, **DEFAULTS)

    assert limit.describe() == (
        "Memory limit: 1787 GiB of the host's 1843 GiB (56 GiB kept for the host, swap off)"
    )


# --- resolving over SSH ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_reads_meminfo_and_the_gpu_count_over_the_rent_connection():
    ssh = Mock()
    ssh.run = AsyncMock(return_value=_ssh_result(stdout=f"{2048 * KIB_PER_GIB}\n8\n"))

    limit = await resolve_rental_memory_limit(
        ssh, requested_gb=255, gpu_uuids=["GPU-1"], log_extra={}
    )

    ssh.run.assert_awaited_once_with(HOST_MEMORY_PROBE_CMD)
    assert (limit.limit_gb, limit.gpu_share, limit.source) == (248, 1 / 8, SOURCE_CLAMPED)


@pytest.mark.asyncio
async def test_resolve_keeps_the_backend_value_when_the_read_fails():
    ssh = Mock()
    ssh.run = AsyncMock(side_effect=OSError("channel closed"))

    limit = await resolve_rental_memory_limit(ssh, requested_gb=30, gpu_uuids=[], log_extra={})

    assert (limit.limit_gb, limit.source, limit.swap_off) == (30, SOURCE_HOST_UNKNOWN, True)


@pytest.mark.asyncio
async def test_resolve_with_the_cap_off_reads_nothing_and_changes_nothing(monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_MEMORY_CAP_ENABLED", False)
    ssh = Mock()
    ssh.run = AsyncMock()

    limit = await resolve_rental_memory_limit(ssh, requested_gb=1841, gpu_uuids=[], log_extra={})

    ssh.run.assert_not_awaited()
    assert (limit.limit_gb, limit.source, limit.cap_enabled, limit.swap_off) == (
        1841,
        SOURCE_CAP_DISABLED,
        False,
        False,
    )


def test_the_defaults_are_on_with_a_three_percent_four_gib_reserve_and_renter_first_oom():
    assert settings.RENTAL_MEMORY_CAP_ENABLED is True
    assert settings.RENTAL_MEMORY_RESERVE_MIN_GB == 4
    assert settings.RENTAL_MEMORY_RESERVE_PERCENT == 3.0
    assert settings.RENTAL_CONTAINER_OOM_SCORE_ADJ == 500


# --- the Docker flags ------------------------------------------------------------------------------


def test_host_config_sets_memory_swap_equal_to_memory_and_the_oom_score():
    kwargs = _build_host_config_kwargs(
        ContainerRunSpec(
            image="img", name="pod_x", memory_gb=1787, memory_swap_gb=1787, oom_score_adj=500
        )
    )

    assert kwargs["mem_limit"] == "1787g"
    assert kwargs["memswap_limit"] == "1787g"
    assert kwargs["oom_score_adj"] == 500
    assert "oom_kill_disable" not in kwargs


def test_host_config_without_the_cap_is_the_flags_of_before():
    kwargs = _build_host_config_kwargs(ContainerRunSpec(image="img", name="pod_x", memory_gb=30))

    assert kwargs["mem_limit"] == "30g"
    assert "memswap_limit" not in kwargs
    assert "oom_score_adj" not in kwargs


def test_host_config_never_sends_memory_swap_without_memory():
    kwargs = _build_host_config_kwargs(
        ContainerRunSpec(image="img", name="pod_x", memory_swap_gb=8)
    )

    assert "mem_limit" not in kwargs and "memswap_limit" not in kwargs


def _sdk_client_with_create_reply(reply):
    api = Mock()
    api.create_host_config = Mock(side_effect=lambda **kwargs: kwargs)
    api.create_container = Mock(return_value=reply)
    client = RentalDockerSdkClient.__new__(RentalDockerSdkClient)
    client._api_client = api
    return client, api


def test_a_memory_limit_the_daemon_discarded_is_logged_as_an_error(caplog):
    client, api = _sdk_client_with_create_reply(
        {
            "Id": "abc",
            "Warnings": [
                "Your kernel does not support memory limit capabilities or the cgroup is not mounted. "
                "Limitation discarded."
            ],
        }
    )

    with caplog.at_level("WARNING"):
        client._run_container_sync(ContainerRunSpec(image="img", name="pod_x", memory_gb=100))

    assert api.create_container.call_args.kwargs["host_config"]["mem_limit"] == "100g"
    api.start.assert_called_once_with("pod_x")
    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert errors and "rental_memory_limit_discarded" in errors[0].getMessage()


def test_a_clean_create_reply_logs_nothing(caplog):
    client, _ = _sdk_client_with_create_reply({"Id": "abc", "Warnings": None})

    with caplog.at_level("WARNING"):
        client._run_container_sync(ContainerRunSpec(image="img", name="pod_x", memory_gb=100))

    assert not [record for record in caplog.records if record.levelno >= 30]


# --- the create flow -------------------------------------------------------------------------------


def _ssh_with_meminfo(host_kib: int, gpu_count: int):
    ssh_client = _ssh_client()

    def _side(cmd, *args, **kwargs):
        if cmd == HOST_MEMORY_PROBE_CMD:
            return _ssh_result(stdout=f"{host_kib}\n{gpu_count}\n")
        return _ssh_result(exit_status=0)

    ssh_client.run = AsyncMock(side_effect=_side)
    return ssh_client


@pytest.mark.asyncio
async def test_create_runs_the_container_with_the_clamped_limit_swap_off_and_the_oom_score(
    svc,  # noqa: F811
    monkeypatch,
):
    ssh_client = _ssh_with_meminfo(H200_GUEST_KIB, 8)
    _patch_happy(svc, monkeypatch, ssh_client)

    await _run(svc, _payload(gpu_uuids=[], memory_gb=_backend_memory_gb(H200_GUEST_KIB)))

    run_spec = _created_run_spec(svc)
    assert (run_spec.memory_gb, run_spec.memory_swap_gb, run_spec.oom_score_adj) == (
        1787,
        1787,
        500,
    )
    host_config = _build_host_config_kwargs(run_spec)
    assert (
        host_config["mem_limit"],
        host_config["memswap_limit"],
        host_config["oom_score_adj"],
    ) == (
        "1787g",
        "1787g",
        500,
    )
    streamed = [call.args[0] for call in svc.stream_log.await_args_list if call.args]
    assert (
        "Memory limit: 1787 GiB of the host's 1843 GiB (56 GiB kept for the host, swap off)"
        in streamed
    )


@pytest.mark.asyncio
async def test_create_sizes_a_zero_ram_pod_row_from_the_host(svc, monkeypatch):  # noqa: F811
    ssh_client = _ssh_with_meminfo(64 * KIB_PER_GIB, 2)
    _patch_happy(svc, monkeypatch, ssh_client)

    await _run(svc, _payload(gpu_uuids=["GPU-test"], memory_gb=0))

    run_spec = _created_run_spec(svc)
    assert (run_spec.memory_gb, run_spec.memory_swap_gb) == (30, 30)  # (64 − 4) × 1/2


@pytest.mark.asyncio
async def test_create_with_the_cap_off_passes_the_backend_value_and_nothing_else(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(settings, "RENTAL_MEMORY_CAP_ENABLED", False)
    ssh_client = _ssh_with_meminfo(H200_GUEST_KIB, 8)
    _patch_happy(svc, monkeypatch, ssh_client)

    await _run(svc, _payload(gpu_uuids=[], memory_gb=1841))

    run_spec = _created_run_spec(svc)
    assert (run_spec.memory_gb, run_spec.memory_swap_gb, run_spec.oom_score_adj) == (
        1841,
        None,
        None,
    )
    assert HOST_MEMORY_PROBE_CMD not in [
        c.args[0] for c in ssh_client.run.await_args_list if c.args
    ]


def _builder_run_spec(docker_service, *, memory_gb, memory_limit, devices=()):
    from payload_models.payloads import ContainerCreateRequest, CustomOptions
    from services.rental_docker_sdk import DeviceMount, GpuDockerConfig

    payload = ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:tag",
        gpu_uuids=["g0"],
        memory_gb=memory_gb,
    )
    return docker_service._build_rental_container_run_spec(
        payload=payload,
        container_name="pod_test",
        custom_options=CustomOptions(),
        port_maps=[(22, 30022, 40022)],
        local_volume="volume_pod",
        local_volume_path="/root",
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(
            device_mounts=tuple(DeviceMount(path_on_host=d) for d in devices)
        ),
        effective_storage_limit_gb=None,
        cpu_count=None,
        memory_limit=memory_limit,
    )


def test_run_spec_without_a_resolved_limit_keeps_the_payload_value(svc):  # noqa: F811
    run_spec = _builder_run_spec(svc, memory_gb=30, memory_limit=None)

    assert (run_spec.memory_gb, run_spec.memory_swap_gb, run_spec.oom_score_adj) == (30, None, None)


def test_run_spec_with_a_resolved_limit_uses_it_with_swap_off_and_the_oom_score(svc):  # noqa: F811
    limit = rental_memory_limit(1841, H200_GUEST_KIB, **DEFAULTS)

    run_spec = _builder_run_spec(svc, memory_gb=1841, memory_limit=limit)

    assert (run_spec.memory_gb, run_spec.memory_swap_gb, run_spec.oom_score_adj) == (
        1787,
        1787,
        500,
    )


def test_rdma_memlock_follows_the_resolved_limit_for_a_zero_ram_pod_row(svc):  # noqa: F811
    limit = rental_memory_limit(0, 128 * KIB_PER_GIB, **DEFAULTS)

    run_spec = _builder_run_spec(
        svc, memory_gb=0, memory_limit=limit, devices=("/dev/infiniband/uverbs0",)
    )

    assert run_spec.memory_gb == 124
    assert [(u.name, u.soft, u.hard) for u in run_spec.ulimits] == [("memlock", -1, -1)]
