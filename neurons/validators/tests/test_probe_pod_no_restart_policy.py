"""The validator's own rental probe pod carries no restart policy; every other pod keeps `unless-stopped`.

A probe pod lives for one check and is removed by the probe's teardown. With `unless-stopped`, a host
stop that cut the teardown short brought the probe pod back at boot beside the node's filler on the same
GPUs, and the node stayed unreachable until the provider started the executor by hand.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import ContainerCreated, WorkloadKind
from services.docker_service import DockerService
from test_deploy_optimizations import _executor_info, _patch_happy, _payload, _ssh_client


@pytest.fixture
def svc():
    # the same services test_filler_removed_at_customer_create builds
    return DockerService(ssh_service=Mock(), redis_service=Mock(get=AsyncMock(return_value=None), mget=AsyncMock(side_effect=lambda keys: [None] * len(keys)), set=AsyncMock(), write_atomically=AsyncMock(), get_rented_machine=AsyncMock(return_value=None), rented_machines_synced=AsyncMock(return_value=True)), attestation_service=Mock())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "synthetic_probe", "expected"),
    [
        (WorkloadKind.CUSTOMER_RENTAL, True, None),
        (WorkloadKind.CUSTOMER_RENTAL, False, "unless-stopped"),
        (WorkloadKind.FILLER, False, "unless-stopped"),
    ],
    ids=["rental-probe", "customer", "filler"],
)
async def test_only_the_rental_probe_pod_has_no_restart_policy(svc, monkeypatch, kind, synthetic_probe, expected):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    specs = []
    original_build_run_spec = svc._build_rental_container_run_spec
    monkeypatch.setattr(
        svc, "_build_rental_container_run_spec", lambda **kw: specs.append(original_build_run_spec(**kw)) or specs[-1]
    )
    payload = _payload(workload_kind=kind)

    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
        beside_fillers=synthetic_probe,
        synthetic_probe=synthetic_probe,
    )

    assert isinstance(result, ContainerCreated)
    assert [spec.restart_policy for spec in specs] == [expected]
