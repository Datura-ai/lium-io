"""A renter's pod leaves the host room for its own work: container logs rotate, and the pod gets exactly the CPU
count the backend sent, which already holds the host's core back (DAH-5951: one source for the reserve)."""

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
    "cpu_count, expected",
    [
        (127, 127),  # whole host: the backend already held the host's core back
        (16, 16),  # a split pod
        (None, None),  # no per-pod cap (the validator's probe pod, a legacy row) stays uncapped
        (0, 1),  # a 1-GPU slice of an 8-core 8-GPU host rounds to 0: floored at 1, never unlimited
    ],
)
def test_rental_cpu_count_is_the_backends_count(cpu_count, expected):
    assert DockerService._rental_cpu_count(cpu_count) == expected


def test_the_cvm_quote_broker_log_is_capped_like_a_rental():
    from services.cvm_quote_broker import quote_broker_run_spec

    assert quote_broker_run_spec().log_opts == {
        "max-size": settings.RENTAL_LOG_MAX_SIZE,
        "max-file": str(settings.RENTAL_LOG_MAX_FILE),
    }
