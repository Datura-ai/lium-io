"""A renter's pod leaves the host room for its own work: container logs rotate and CPU keeps a reserve."""

import asyncio
from types import SimpleNamespace

import pytest

from core.config import settings
from services.docker_service import DockerService
from services.rental_docker_sdk import ContainerRunSpec, _build_host_config_kwargs
from tests.test_deploy_optimizations import svc  # noqa: F401  (pytest fixture)
from tests.test_rental_swap_off import _run_spec


def test_host_config_rotates_the_container_log_when_log_opts_are_set():
    kwargs = _build_host_config_kwargs(
        ContainerRunSpec(image="img", name="pod_x", log_opts={"max-size": "100m", "max-file": "3"})
    )

    assert kwargs["log_config"] == {"type": "json-file", "config": {"max-size": "100m", "max-file": "3"}}


def test_host_config_without_log_opts_keeps_the_daemon_log_driver():
    assert "log_config" not in _build_host_config_kwargs(ContainerRunSpec(image="img", name="pod_x"))


def test_a_rental_container_log_is_capped_so_stdout_cannot_fill_the_host_disk(svc):  # noqa: F811
    run_spec = _run_spec(svc, 30)

    assert run_spec.log_opts == {
        "max-size": settings.RENTAL_LOG_MAX_SIZE,
        "max-file": str(settings.RENTAL_LOG_MAX_FILE),
    }


@pytest.mark.parametrize(
    "cpu_count, host_cpus, expected",
    [
        (128, 128, 128 - settings.RENTAL_HOST_CPU_RESERVE),  # whole host: keeps the reserve
        (16, 128, 16),  # a split pod already below the host is unchanged
        (128, None, 128),  # host count unknown: the backend's value
        (None, 128, None),  # no per-pod cap (CVM) stays uncapped
        (2, 1, 2),  # a host too small for the reserve keeps the backend's value
    ],
)
def test_rental_cpu_count_leaves_the_host_reserve(cpu_count, host_cpus, expected):
    assert DockerService._rental_cpu_count(cpu_count, host_cpus) == expected


def test_a_whole_host_rental_does_not_get_every_host_core(svc):  # noqa: F811
    run_spec = _run_spec(svc, 30, cpu_count=64, host_cpu_count=64)

    assert run_spec.cpu_count == 64 - settings.RENTAL_HOST_CPU_RESERVE


def test_read_host_cpu_count_counts_the_processors(svc):  # noqa: F811
    class _Ssh:
        async def run(self, cmd, check=False, timeout=None):
            return SimpleNamespace(exit_status=0, stdout="96\n")

    assert asyncio.run(svc._read_host_cpu_count(_Ssh())) == 96


def test_read_host_cpu_count_is_none_on_a_failed_read(svc):  # noqa: F811
    class _Ssh:
        async def run(self, cmd, check=False, timeout=None):
            return SimpleNamespace(exit_status=1, stdout="")

    assert asyncio.run(svc._read_host_cpu_count(_Ssh())) is None
