"""E7 differential harness SKELETON (DAH-2382 PR-1a, oracle §E row E7, decision D-B).

Purpose: the per-extraction-PR differential gate against a REAL docker daemon.
In PR-1a there is only ONE implementation (the monolith), so the test asserts
SELF-consistency: the probe drives the live docker-service layers, captures the
comparable surface, and pins it byte-for-byte to a golden snapshot under
``tests/fixtures/docker_oracle/``.

In extraction PRs this becomes old-vs-new: the same probe runs against the
pre-refactor facade (pinned golden) and the new package — any drift fails.

Decision D-B (2026-07-19): no docker-capable CI runner is budgeted, so this
module is env-gated exactly like ``test_rental_docker_sdk_local.py``
(``RUN_RENTAL_DOCKER_SDK_INTEGRATION=1``) and runs as a mandatory LOCAL gate
(``make e7`` in ``neurons/validators/``) before each extraction PR, with the
staging canary as the final backstop. See ``tests/integration/README_E7.md``.

Scope (PR-1a reality check): the FULL ``DockerService.create_container`` cannot
run against the local fixtures under this gate —
- the run spec always carries a GPU device request (``build_gpu_docker_config``
  emits ``count=-1`` even for zero UUIDs), which a non-NVIDIA local daemon
  rejects at ``docker run``;
- port generation, volume sizing and sshd provisioning need a redis-backed port
  state and the SSH executor endpoint, which sits behind the SEPARATE
  ``RUN_RENTAL_DOCKER_SDK_SSH_INTEGRATION`` gate (``local_ssh_docker_endpoint``);
- ``storage_limit_gb`` maps to ``--storage-opt size`` (xfs+pquota only).
So the probe drives the deepest daemon-real layer instead: the real
``CustomOptions.sanitize`` -> ``_build_rental_container_run_spec`` -> the LIVE
``_run_rental_docker_create_with_port_retry`` (creates a REAL container) ->
``docker inspect`` -> the real ``_stop_container_gracefully`` +
``_force_remove_container`` delete boundary.
TODO(PR-2): extend the probe to the full ``create_container``/``delete_container``
facade over ``local_ssh_docker_endpoint`` (GPU-less spec seam + sshd-capable
target image needed), populating the redis call-order surface below.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from docker_oracle.harness import RecordingRedisService, make_create_request, make_docker_service
from payload_models.payloads import (
    ContainerCreateRequest,
    ContainerDeleteRequest,
    CustomOptions,
)
from services.docker_service import DockerService, _BoundLog
from services.rental_docker_sdk import GpuDockerConfig, RentalDockerSdkClient

# WHY: fixture functions imported into this module's namespace are registered by
# pytest for this module — the smallest reuse that leaves the source file untouched.
from test_rental_docker_sdk_local import (
    TARGET_IMAGE,
    local_docker_api_client,  # noqa: F401
    requires_local_docker,
)

# Same gate as test_rental_docker_sdk_local.py (decision D-B): default CI skips.
pytestmark = requires_local_docker

# Fixed identities keep every derived name (container, volume) and the golden
# snapshot byte-stable across runs; the trailing e7 marks probe-owned resources.
_PROBE_POD_ID = "00000000-0000-0000-0000-0000000000e7"
_PROBE_EXECUTOR_ID = "00000000-0000-0000-0000-00000000e701"
_PROBE_HOST_PORT = 20001

_SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "docker_oracle"
    / "e7_differential_probe_surface.json"
)

_SECRET_ENV_KEY_PATTERN = re.compile(r"(?i)secret|token|password|credential|_key")


def _make_probe_payload() -> ContainerCreateRequest:
    # trap TERM so the graceful-stop grace window (30s) resolves instantly,
    # mirroring the lifecycle command of test_rental_docker_sdk_local.py.
    return make_create_request(
        executor_id=_PROBE_EXECUTOR_ID,
        pod_id=_PROBE_POD_ID,
        docker_image=TARGET_IMAGE,
        available_ports=[],
        custom_options=CustomOptions(
            startup_commands="sh -c 'trap : TERM INT; sleep infinity & wait'",
            environment={
                "E7_SENTINEL": "sentinel-value",
                "E7_SECRET_TOKEN": "must-not-reach-the-snapshot",
            },
        ),
    )


def _mask_env(environment: dict[str, str]) -> dict[str, str]:
    # WHY: env values may carry secrets (registry passwords, jupyter tokens);
    # the differential compares the key set plus non-secret values only.
    return {
        key: "<masked>" if _SECRET_ENV_KEY_PATTERN.search(key) else value
        for key, value in environment.items()
    }


def _env_entries_to_dict(env_entries: list[str] | None) -> dict[str, str]:
    return dict(entry.partition("=")[::2] for entry in env_entries or [])


def _normalized_port_bindings(port_bindings: dict[str, Any] | None) -> dict[str, list[str]]:
    # WHY: inspect wraps each binding as {"HostIp": "", "HostPort": ...} and the
    # HostIp default shape drifts across daemon versions; the comparable fact is
    # container-port -> host-port.
    return {
        port: [binding["HostPort"] for binding in bindings or []]
        for port, bindings in (port_bindings or {}).items()
    }


@dataclasses.dataclass(frozen=True)
class DifferentialProbeResult:
    """Captured comparable surface plus the raw pieces the test asserts on."""

    surface: dict[str, Any]
    spec_argv: tuple[str, ...]
    inspected_cmd: list[str] | None


class DifferentialProbe:
    """Drives the monolith's live docker layers through a real daemon and
    captures the old-vs-new comparable surface (see module docstring for scope)."""

    def __init__(self, api_client: Any, service: DockerService, redis: RecordingRedisService):
        self._api_client = api_client
        self._service = service
        self._redis = redis

    async def run(self, payload: ContainerCreateRequest) -> DifferentialProbeResult:
        # run the create layer, inspect the real container, then run the delete
        # boundary; always leaves the daemon clean
        container_name = DockerService.get_container_name(payload)
        local_volume = f"volume_{payload.pod_id}"

        client = RentalDockerSdkClient(self._api_client)
        self._cleanup(container_name, local_volume)
        try:
            run_spec = self._build_run_spec(payload, container_name, local_volume)

            await client.pull(image=payload.docker_image)
            # WHY Mock ssh_client: the retry loop touches SSH only on the
            # vloopback stale-mount repair branch, unreachable on this happy path.
            await self._service._run_rental_docker_create_with_port_retry(
                docker_client=client,
                ssh_client=Mock(),
                run_spec=run_spec,
                container_name=container_name,
                default_extra={"probe": "e7"},
                local_volume=local_volume,
            )
            inspect = self._api_client.inspect_container(container_name)

            delete_observations = await self._run_delete_boundary(
                client, payload, container_name, local_volume
            )

            return DifferentialProbeResult(
                surface={
                    "schema": "e7-differential-probe/v1",
                    # Empty at PR-1a depth (the narrow layer never touches redis);
                    # kept in the surface so the PR-2 full-facade drive fills it
                    # without a snapshot schema change.
                    "redis_call_order": [name for name, _ in self._redis.calls],
                    "run_spec": self._serialize_run_spec(run_spec),
                    "daemon": self._capture_daemon_view(inspect),
                    "delete": delete_observations,
                },
                spec_argv=run_spec.command,
                inspected_cmd=inspect["Config"].get("Cmd"),
            )
        finally:
            self._cleanup(container_name, local_volume)
            await client.aclose()

    def _build_run_spec(
        self,
        payload: ContainerCreateRequest,
        container_name: str,
        local_volume: str,
    ):
        custom_options = CustomOptions.sanitize(payload.custom_options)
        return self._service._build_rental_container_run_spec(
            payload=payload,
            container_name=container_name,
            custom_options=custom_options,
            port_maps=[(22, _PROBE_HOST_PORT, _PROBE_HOST_PORT)],
            local_volume=local_volume,
            local_volume_path="/root",
            external_volume_name=None,
            # WHY empty GPU config: build_gpu_docker_config emits a device
            # request even for zero UUIDs, and a non-NVIDIA local daemon rejects
            # it; the GPU seam joins the probe in the PR-2 full-facade drive.
            gpu_devices=GpuDockerConfig(),
            # WHY None: --storage-opt size requires overlay-on-xfs with pquota,
            # unavailable on typical local daemons (Docker Desktop).
            effective_storage_limit_gb=None,
            cpu_count=payload.cpu_count,
        )

    async def _run_delete_boundary(
        self,
        client: RentalDockerSdkClient,
        payload: ContainerCreateRequest,
        container_name: str,
        local_volume: str,
    ) -> dict[str, Any]:
        # the daemon-reachable slice of delete_container: graceful stop (never
        # fatal) then the single fatal force-remove boundary
        delete_payload = ContainerDeleteRequest(
            miner_hotkey=payload.miner_hotkey,
            executor_id=payload.executor_id,
            pod_id=payload.pod_id,
            container_name=container_name,
            local_volume=local_volume,
        )
        log = _BoundLog({"probe": "e7", "pod_id": payload.pod_id})

        await self._service._stop_container_gracefully(client, delete_payload, log)
        running_after_stop = self._api_client.inspect_container(container_name)["State"]["Running"]
        await self._service._force_remove_container(client, delete_payload, log)
        return {
            "running_after_graceful_stop": running_after_stop,
            "removed_after_force_remove": not self._container_exists(container_name),
        }

    def _serialize_run_spec(self, run_spec: Any) -> dict[str, Any]:
        spec = dataclasses.asdict(run_spec)
        spec["environment"] = _mask_env(spec["environment"])
        return spec

    def _capture_daemon_view(self, inspect: dict[str, Any]) -> dict[str, Any]:
        # WHY this projection: only fields that are deterministic across local
        # daemons are captured — container id, timestamps and state are masked
        # out by never being selected.
        config = inspect["Config"]
        host_config = inspect["HostConfig"]
        environment = _env_entries_to_dict(config.get("Env"))
        return {
            "cmd": config.get("Cmd"),
            "entrypoint": config.get("Entrypoint"),
            "env_keys": sorted(environment),
            "env": _mask_env(environment),
            "labels": config.get("Labels") or {},
            "host_config": {
                "binds": host_config.get("Binds"),
                "port_bindings": _normalized_port_bindings(host_config.get("PortBindings")),
                "cap_add": host_config.get("CapAdd"),
                "restart_policy": host_config.get("RestartPolicy"),
                "sysctls": host_config.get("Sysctls"),
                "devices": host_config.get("Devices"),
                "nano_cpus": host_config.get("NanoCpus"),
                "memory": host_config.get("Memory"),
            },
        }

    def _container_exists(self, container_name: str) -> bool:
        import docker

        try:
            self._api_client.inspect_container(container_name)
            return True
        except docker.errors.NotFound:
            return False

    def _cleanup(self, container_name: str, volume_name: str) -> None:
        # idempotent pre/post cleanup so a crashed earlier run cannot poison
        # the fixed-name container/volume of the next one
        try:
            self._api_client.remove_container(container_name, force=True, v=True)
        except Exception:
            pass
        try:
            self._api_client.remove_volume(volume_name, force=True)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_e7_differential_probe_self_consistency(
    local_docker_api_client,  # noqa: F811
    update_snapshot,
):
    # Arrange — monolith service with the recording redis fake wired in, so the
    # captured surface already carries the redis call-order channel for PR-2.
    service = make_docker_service()
    redis = RecordingRedisService()
    service.redis_service = redis
    probe = DifferentialProbe(local_docker_api_client, service, redis)
    payload = _make_probe_payload()

    # Act
    result = await probe.run(payload)

    # Assert — the argv==Config.Cmd differential (same invariant as
    # test_rental_docker_sdk_local.py's breakout test): the command the service
    # built is exactly what the real daemon runs, with no host-shell hop.
    assert result.inspected_cmd == list(result.spec_argv)

    # Assert — byte-equal golden snapshot (repo's own --update-snapshot flow:
    # write-on-missing then fail asking for a re-run; byte-equal afterwards).
    serialized = json.dumps(result.surface, indent=2, sort_keys=True) + "\n"
    _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if update_snapshot or not _SNAPSHOT_PATH.exists():
        _SNAPSHOT_PATH.write_text(serialized, encoding="utf-8")
        if not update_snapshot:
            pytest.fail(
                f"E7 golden snapshot did not exist; wrote it. Re-run to assert. Path={_SNAPSHOT_PATH}"
            )
        return
    expected = _SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert serialized == expected, (
        "E7 differential surface drifted from the golden snapshot. If the change is an "
        "intended behavior change, re-record with --update-snapshot (make e7-record) and "
        f"review the diff as a docker_service behavior change. Path={_SNAPSHOT_PATH}"
    )
