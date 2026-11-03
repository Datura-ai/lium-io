"""A rented pod after the create: an image whose default command would exit at once stays up, and the cycle's
SSH probe of rented pods reads what the renter's `ssh -p` would meet."""

import uuid

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from protocol.vc_protocol.validator_requests import PodSshResult
from services.pod_ssh_probe import probe_rented_pods

from services.ioc import ioc
from tests import lib
from tests.rental.test_rental import (
    _create_request,
    _release_and_delete,
    _renter_can_ssh,
    _ssh_port,
)

pytestmark = pytest.mark.timeout(900)

BARE_SHELL_IMAGE = "lium-e2e/pod-bare-shell:latest"


@pytest.fixture(scope="module")
def services():
    return ioc


def test_an_image_whose_default_command_is_a_bare_shell_stays_up(services):
    """`pytorch/pytorch` and `nvidia/cuda` default to CMD ["/bin/bash"], which exits at once without a TTY. A
    create with no start command runs `sleep infinity` instead, so the pod stays up and the renter's key opens
    it (lium-io#1529). The image is the stack's pod image with its CMD set to a bare shell, made on the
    executor's dockerd, so no registry is involved."""
    from payload_models.payloads import ContainerCreated

    builder = f"e2e-bare-shell-{uuid.uuid4().hex[:8]}"
    rc, out = lib.executor_sh(
        f"docker create --name {builder} {lib.ENV['E2E_POD_IMAGE']} >/dev/null"
        f" && docker commit --change 'CMD [\"/bin/bash\"]' {builder} {BARE_SHELL_IMAGE} >/dev/null"
        f" && docker rm {builder} >/dev/null"
        f" && docker image inspect -f '{{{{json .Config.Cmd}}}}' {BARE_SHELL_IMAGE}"
    )
    assert rc == 0 and out.strip() == '["/bin/bash"]', (rc, out)
    renter_priv, renter_pub = lib.ssh_keypair()
    pod_id = str(uuid.uuid4())
    req = _create_request(pod_id, renter_pub)
    req.docker_image = BARE_SHELL_IMAGE
    try:
        created = lib.run(services["MinerService"].handle_container(req))
        assert isinstance(created, ContainerCreated), created
        try:
            _renter_can_ssh(created, renter_priv)
            rc, out = lib.executor_sh(
                f"docker inspect -f '{{{{.State.Running}}}}' {created.container_name}"
            )
            assert rc == 0 and out.strip() == "true", (rc, out)
        finally:
            _release_and_delete(services, pod_id, created)
    finally:
        # a refused create leaves the pending flag (no ContainerCreated to release): clear it for the next case
        lib.run(
            services["MinerService"].redis_service.remove_pending_pod(
                lib.MINER_HOTKEY, lib.EXECUTOR_UUID, pod_id
            )
        )
        lib.executor_sh(
            f"docker rm -f {builder} >/dev/null 2>&1; docker rmi {BARE_SHELL_IMAGE} >/dev/null 2>&1; true"
        )


def _probe(pod_id: str, container_name: str, ssh_port: int) -> PodSshResult:
    """One run of the cycle's rented pod SSH probe over a rented list holding this one pod."""
    rented = RentedExecutorsResponse(
        executors={
            lib.EXECUTOR_UUID: RentedExecutor(
                miner_hotkey=lib.MINER_HOTKEY,
                executor_ip_address=lib.EXECUTOR_IP,
                executor_ip_port=str(lib.EXECUTOR_PORT),
                pods=[
                    RentedPod(
                        pod_id=pod_id,
                        container_name=container_name,
                        ssh_port=ssh_port,
                        status="RUNNING",
                    )
                ],
            )
        }
    )
    observations = lib.run(
        probe_rented_pods(rented, timeout=10, concurrency=4, job_batch_id=f"e2e-{uuid.uuid4()}")
    )
    [observation] = observations[lib.EXECUTOR_UUID]
    assert observation.pod_id == pod_id, observation
    return observation.result


def test_the_rented_pod_ssh_probe_reports_a_pod_whose_ssh_stopped_answering(services):
    """Each cycle probes every rented pod's mapped SSH port without a key: a pod the renter can open reads
    as a banner; once the pod's container is stopped, so its SSH port no longer answers, the probe reports
    it as not answering (lium-io#1478, #1372)."""
    from payload_models.payloads import ContainerCreated

    renter_priv, renter_pub = lib.ssh_keypair()
    pod_id = str(uuid.uuid4())
    created = lib.run(
        services["MinerService"].handle_container(_create_request(pod_id, renter_pub))
    )
    assert isinstance(created, ContainerCreated), created
    try:
        _renter_can_ssh(created, renter_priv)
        port = _ssh_port(created)
        assert _probe(pod_id, created.container_name, port) == PodSshResult.BANNER
        rc, out = lib.executor_sh(f"docker stop -t 1 {created.container_name}")
        assert rc == 0, out
        try:
            result = _probe(pod_id, created.container_name, port)
            assert result != PodSshResult.BANNER, result
        finally:
            lib.executor_sh(f"docker start {created.container_name} >/dev/null 2>&1; true")
    finally:
        _release_and_delete(services, pod_id, created)
