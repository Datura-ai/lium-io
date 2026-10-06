"""A rental through the validator: ContainerCreateRequest → a container on the executor's docker → the renter's key
opens it → ContainerDeleteRequest → gone.

`MinerService.handle_container` is what the connector calls when the platform sends a ContainerCreateRequest for a
rented executor: sign in to the miner for that one executor (is_rental_request), SSH in, `docker run --gpus …` with
the renter's public keys, map the SSH port, answer ContainerCreated with the port map. On a host without GPUs the
stack's dockerd carries a no-op NVIDIA prestart hook (e2e/dind), so the same command produces a GPU-less pod and the
whole path is exercised; with E2E_GPU=1 the pod gets the host's GPUs and `nvidia-smi -L` inside it must list them.
"""

import time
import uuid

import pytest

from tests import lib

pytestmark = pytest.mark.timeout(900)


@pytest.fixture(scope="module")
def services():
    from services.ioc import ioc

    return ioc


def _create_request(pod_id: str, renter_pub: str):
    from payload_models.payloads import ContainerCreateRequest, PayloadPortMapping

    # the platform sends the ports the validator verified on this executor in earlier cycles (PortConnectivityCheck →
    # backend); the stack's executor advertises RENTING_PORT_RANGE=40000-40019, published on its address by dind
    ports = [PayloadPortMapping(internal_port=p, external_port=p) for p in range(40000, 40010)]
    return ContainerCreateRequest(
        miner_hotkey=lib.MINER_HOTKEY,
        miner_address=lib.MINER_IP,
        miner_port=lib.MINER_PORT,
        executor_id=lib.EXECUTOR_UUID,
        pod_id=pod_id,
        docker_image=lib.ENV["E2E_POD_IMAGE"],
        user_public_keys=[renter_pub],
        gpu_uuids=[],  # whole node
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=1,
        storage_limit_gb=None,  # what the platform sends for a host whose docker root is not xfs+pquota (no --storage-opt)
        is_sysbox=False,
        timestamp=int(time.time()),
        available_ports=ports,
        pod_mapping=[],
    )


def _delete_request(pod_id: str, container_name: str, volume_name: str | None):
    from payload_models.payloads import ContainerDeleteRequest

    return ContainerDeleteRequest(
        miner_hotkey=lib.MINER_HOTKEY,
        miner_address=lib.MINER_IP,
        miner_port=lib.MINER_PORT,
        executor_id=lib.EXECUTOR_UUID,
        pod_id=pod_id,
        container_name=container_name,
        local_volume=volume_name,
    )


def test_rental_creates_a_container_the_renter_can_ssh_into_then_deletes_it(services):
    from payload_models.payloads import ContainerCreated, ContainerDeleted, FailedContainerRequest

    renter_priv, renter_pub = lib.ssh_keypair()
    pod_id = str(uuid.uuid4())
    t0 = time.monotonic()
    created = lib.run(services["MinerService"].handle_container(_create_request(pod_id, renter_pub)))
    create_s = round(time.monotonic() - t0, 1)
    dump = created.model_dump(mode="json") if hasattr(created, "model_dump") else str(created)
    lib.write_artifact("rental-created.json", {"create_s": create_s, "response": dump})
    if isinstance(created, FailedContainerRequest):
        pytest.fail(f"ContainerCreateRequest failed in {create_s}s: {created.msg} (type={created.error_type} code={created.error_code})")
    assert isinstance(created, ContainerCreated), type(created)
    assert created.executor_id == lib.EXECUTOR_UUID
    assert created.container_name and created.port_maps, dump

    ssh_map = [m for m in created.port_maps if m[0] == 22]
    assert ssh_map, f"no SSH port in the map: {created.port_maps}"
    external_ssh = ssh_map[0][1]
    try:
        # the renter's key opens the pod on the executor's address — what `lium exec`/`ssh` do after `lium up`
        rc, out = lib.wait_for(
            lambda: lib.run(lib.ssh_run(lib.EXECUTOR_IP, external_ssh, "root", renter_priv, "hostname && cat /etc/os-release | head -1")),
            timeout=120, interval=3, what="ssh into the rented container",
        )
        assert rc == 0 and out.strip(), (rc, out)
        if lib.GPU:
            rc, out = lib.run(lib.ssh_run(lib.EXECUTOR_IP, external_ssh, "root", renter_priv, "nvidia-smi -L"))
            assert rc == 0 and "GPU 0" in out, (rc, out)
        else:
            rc, out = lib.run(lib.ssh_run(lib.EXECUTOR_IP, external_ssh, "root", renter_priv, "ls /dev/nvidia0 2>&1; true"))
            assert "No such file" in out, f"a GPU-less host handed out a GPU device: {out}"
    finally:
        # the platform answers ContainerCreated with ExecutorRentFinishedRequest once it has recorded the pod; the
        # connector's only action on it is clearing the executor's pending-rental flag (clients/compute_client.py) —
        # without that step a delete is refused with RentingInProgress for up to 30 min, exactly as in production
        lib.run(services["MinerService"].redis_service.remove_pending_pod(lib.MINER_HOTKEY, lib.EXECUTOR_UUID, pod_id))
        t1 = time.monotonic()
        deleted = lib.run(services["MinerService"].handle_container(_delete_request(pod_id, created.container_name, created.volume_name)))
        lib.write_artifact("rental-deleted.json", {"delete_s": round(time.monotonic() - t1, 1), "response": deleted.model_dump(mode="json") if hasattr(deleted, "model_dump") else str(deleted)})
    assert isinstance(deleted, ContainerDeleted), deleted
    with pytest.raises(Exception):
        lib.run(lib.ssh_run(lib.EXECUTOR_IP, external_ssh, "root", renter_priv, "true", timeout=15))


def test_rental_on_an_executor_the_miner_does_not_own_is_refused(services):
    from payload_models.payloads import FailedContainerRequest

    _, renter_pub = lib.ssh_keypair()
    req = _create_request(str(uuid.uuid4()), renter_pub)
    req.executor_id = str(uuid.uuid4())
    t0 = time.monotonic()
    resp = lib.run(services["MinerService"].handle_container(req))
    assert time.monotonic() - t0 < 120, "a refused rental must fail fast"
    assert isinstance(resp, FailedContainerRequest), resp
    lib.write_artifact("rental-refused.json", resp.model_dump(mode="json"))
    assert resp.msg, resp
