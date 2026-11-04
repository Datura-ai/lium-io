from services.docker_service import DockerService
from services.rental_docker_sdk import ContainerUlimit


def test_memory_limited_pod_gets_unlimited_memlock_without_rdma():
    # a partial pod gets no RDMA devices but NCCL/NVSHMEM still pin host buffers
    assert DockerService._memlock_ulimit_for(64) == (ContainerUlimit(name="memlock", soft=-1, hard=-1),)


def test_no_memory_limit_keeps_default_memlock():
    # without a cgroup ceiling locked pages are unbounded
    assert DockerService._memlock_ulimit_for(None) == ()
    assert DockerService._memlock_ulimit_for(0) == ()
