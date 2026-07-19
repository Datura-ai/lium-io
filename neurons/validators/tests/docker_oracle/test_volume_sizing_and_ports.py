"""C-VOL oracle: volume-sizing branches (C1-C9) + port-planner lock failure (C20).

DRAFT sketches C.1/C.3; per DRAFT section 6 these branches are invisible at the
ContainerCreated boundary, so the asserts here target the finer unit boundary:
the returned ``VolumeSizingResult`` dataclass, host SSH-command presence/order,
and the warning log key.

Baseline reality (6be5649f): tests/test_docker_service.py:4281-4458 already pins
most of the C1-C8 table (shipped with DAH-2183; the DRAFT's [N] tags are stale).
This file groups the C-VOL contracts under the oracle and adds the finer pins the
sketches call for: the exact SSH command sequence per path, the byte-exact
VolumeMinSizeError message, the tie-break winner for ``capped_by``, the full
result-field set on fallback, and the parse formats the legacy test misses.

Real arithmetic pinned here (all sketch numbers verified against the code):
overhead 20 GB, headroom 10 GB, GB = 1024**3, pool = share * max(df + existing
- 20G, 0), request_cap = volume_limit_gb * 1.5 G, df_guard = max((df - 10G) *
1.5, 0), slice = min(candidates), volume = floor(slice * 2/3) and storage =
floor(slice * 1/3) each floored to >= 1 GB.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import redis.exceptions
from docker_oracle.harness import make_create_request, make_docker_service
from payload_models.payloads import PayloadPortMapping
from services.docker_service import VolumeMinSizeError, _parse_volume_size_to_bytes

_GB = 1024**3  # mirrors _FRESH_SIZING_GB_BYTES: sizing is 1024-based, not SI

_DF_HEADER = "Filesystem           1-blocks       Used Available Capacity Mounted on\n"


def _df_stdout(df_avail_bytes: int) -> str:
    # POSIX `df -P -B1` table; the code reads the 4th column of the data line
    return _DF_HEADER + f"/dev/vda1            1000 500 {df_avail_bytes}  80% /hostfs\n"


class RoutingSSHClient:
    """Local fake SSH client with per-command stdout routing.

    The shared ``RecordingSSHClient`` returns one fixed stdout for every
    command, but the fresh sizing path needs different text per command
    (docker info vs df vs volume ls vs volume inspect). Routes are
    ``(substring, stdout-or-exception)`` pairs, first match wins; every
    command is recorded in ``.commands`` before routing so failure tests
    still see how far the code got. An unrouted command fails the test.
    """

    def __init__(self, routes: list[tuple[str, str | Exception]]) -> None:
        self.commands: list[str] = []
        self._routes = routes

    async def run(self, command: str, *args: Any, **kwargs: Any) -> SimpleNamespace:
        self.commands.append(command)
        for needle, outcome in self._routes:
            if needle in command:
                if isinstance(outcome, Exception):
                    raise outcome
                return SimpleNamespace(stdout=outcome, stderr="", exit_status=0)
        raise AssertionError(f"unexpected ssh command: {command}")


def make_sizing_ssh(
    *,
    df_avail_bytes: int = 0,
    volume_ls_stdout: str = "",
    volume_inspect_stdout: str = "",
    df_error: Exception | None = None,
) -> RoutingSSHClient:
    # route table for the fresh-sizing measurement triplet behind get_docker_root_dir
    df_outcome: str | Exception = (
        df_error if df_error is not None else _df_stdout(df_avail_bytes)
    )
    return RoutingSSHClient(
        [
            ("docker info", "/var/lib/docker\n"),
            ("df -P -B1 /hostfs", df_outcome),
            ("volume ls", volume_ls_stdout),
            ("volume inspect", volume_inspect_stdout),
        ]
    )


# ---------------------------------------------------------------------------
# C.1 volume sizing (resolve_volume_sizing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sizing_pool_bound() -> None:
    # Arrange: share=0.5, df=900G, one 300G vloopback volume, no request cap
    # (volume_limit_gb=None drops the request_cap candidate entirely).
    svc = make_docker_service()
    payload = make_create_request(disk_share=0.5, volume_limit_gb=None, storage_limit_gb=1)
    ssh_client = make_sizing_ssh(
        df_avail_bytes=900 * _GB,
        volume_ls_stdout="volume_abc vloopback:latest\nscratch local\n",
        volume_inspect_stdout=f"{300 * _GB}|<no value>\n",
    )

    # Act
    result = await svc.resolve_volume_sizing(
        ssh_client=ssh_client, payload=payload, log_tag="tag", log_extra={}
    )

    # Assert
    # WHY: pins the DAH-2183 pool math — pool = df + existing - 20G overhead =
    # 1180G, slice = 0.5 * pool = 590G < df_guard (900-10)*1.5 = 1335G, then the
    # 2:1 volume:storage split with floor(): vol = floor(590*2/3) = 393,
    # storage = floor(590/3) = 196 (GB = 1024**3).
    assert result.path == "fresh"
    assert result.capped_by == "pool"
    assert result.volume_limit_gb == 393
    assert result.storage_limit_gb == 196
    assert result.df_avail_bytes == 900 * _GB
    assert result.existing_volumes_bytes == 300 * _GB
    # WHY: the fresh path measures over exactly four host commands in this
    # order: docker info (root dir), df inside the alpine helper container,
    # volume ls, volume inspect of the vloopback-driver volumes only.
    assert len(ssh_client.commands) == 4
    assert "docker info --format" in ssh_client.commands[0]
    assert "df -P -B1 /hostfs" in ssh_client.commands[1]
    assert "docker.io/library/alpine:3.19" in ssh_client.commands[1]
    assert "volume ls" in ssh_client.commands[2]
    assert "volume inspect volume_abc" in ssh_client.commands[3]
    assert "scratch" not in ssh_client.commands[3]


@pytest.mark.asyncio
async def test_sizing_df_guard_headroom_binds() -> None:
    # Arrange: tight disk — share=0.9, df=50G, 1000G of existing volumes
    # declared only via Options.size ("1000g"; Status size-max is "<no value>").
    svc = make_docker_service()
    payload = make_create_request(disk_share=0.9, volume_limit_gb=None, storage_limit_gb=1)
    ssh_client = make_sizing_ssh(
        df_avail_bytes=50 * _GB,
        volume_ls_stdout="volume_abc vloopback\n",
        volume_inspect_stdout="<no value>|1000g\n",
    )

    # Act
    result = await svc.resolve_volume_sizing(
        ssh_client=ssh_client, payload=payload, log_tag="tag", log_extra={}
    )

    # Assert
    # WHY: pins the df headroom guard — df_guard = max((df - 10G) * 1.5, 0) =
    # (50-10)*1.5 = 60G beats pool 0.9*(50+1000-20) = 927G; split 2:1 gives
    # vol = floor(60*2/3) = 40, storage = floor(60/3) = 20. The sketch's
    # (df-10G)*1.5 formula is CONFIRMED against the code.
    assert result.path == "fresh"
    assert result.capped_by == "df_guard"
    assert result.volume_limit_gb == 40
    assert result.storage_limit_gb == 20
    assert result.existing_volumes_bytes == 1000 * _GB


@pytest.mark.asyncio
async def test_sizing_request_cap_binds() -> None:
    # Arrange: share=0.5, df=900G, existing=300G, backend cap volume_limit_gb=100.
    svc = make_docker_service()
    payload = make_create_request(disk_share=0.5, volume_limit_gb=100, storage_limit_gb=50)
    ssh_client = make_sizing_ssh(
        df_avail_bytes=900 * _GB,
        volume_ls_stdout="volume_abc vloopback\n",
        volume_inspect_stdout=f"{300 * _GB}|<no value>\n",
    )

    # Act
    result = await svc.resolve_volume_sizing(
        ssh_client=ssh_client, payload=payload, log_tag="tag", log_extra={}
    )

    # Assert
    # WHY: pins the request cap — request_cap = volume_limit_gb * 1.5 G = 150G
    # beats pool 590G and df_guard 1335G (factor 1.5 CONFIRMED); the *1.5 then
    # 2/3 split lands the volume EXACTLY back on the requested 100 and storage
    # on 50, so a request-capped pod still gets its full requested volume.
    assert result.path == "fresh"
    assert result.capped_by == "request_cap"
    assert result.volume_limit_gb == 100
    assert result.storage_limit_gb == 50


@pytest.mark.asyncio
async def test_sizing_below_min_raises() -> None:
    # Arrange: share=0.5, df=30G, no existing volumes, min_volume_gb=10.
    svc = make_docker_service()
    payload = make_create_request(
        disk_share=0.5, volume_limit_gb=None, storage_limit_gb=1, min_volume_gb=10
    )
    ssh_client = make_sizing_ssh(df_avail_bytes=30 * _GB, volume_ls_stdout="")

    # Act / Assert
    # WHY: pins the reject threshold — pool = max(30-20, 0)*0.5 = 5G, vol =
    # floor(5*2/3) = 3 < min_volume_gb=10 raises; the check compares the
    # already-floored volume, and the message carries both numbers byte-exact.
    with pytest.raises(
        VolumeMinSizeError,
        match="Fresh vloopback sizing produced 3GB volume, below required minimum 10GB",
    ):
        await svc.resolve_volume_sizing(
            ssh_client=ssh_client, payload=payload, log_tag="tag", log_extra={}
        )
    # WHY: with zero vloopback volumes listed, the inspect command is skipped —
    # measurement is docker info + df + volume ls only.
    assert len(ssh_client.commands) == 3
    assert not any("volume inspect" in command for command in ssh_client.commands)


@pytest.mark.asyncio
async def test_sizing_low_free_floor_to_1gb() -> None:
    # Arrange: near-full disk — share=1.0, df=5G (below both the 20G overhead
    # and the 10G headroom), no existing volumes, no caps.
    svc = make_docker_service()
    payload = make_create_request(disk_share=1.0, volume_limit_gb=None, storage_limit_gb=1)
    ssh_client = make_sizing_ssh(df_avail_bytes=5 * _GB, volume_ls_stdout="")

    # Act
    result = await svc.resolve_volume_sizing(
        ssh_client=ssh_client, payload=payload, log_tag="tag", log_extra={}
    )

    # Assert
    # WHY: pins the negative clamp + 1 GB floor — pool max(5G-20G, 0) = 0 and
    # df_guard max((5G-10G)*1.5, 0) = 0 both clamp to 0, slice = 0, and
    # max(floor(0), 1) floors both limits to 1 GB instead of going negative;
    # the 0.0 tie resolves to the FIRST candidate in list order, so
    # capped_by == "pool" (candidate order: pool, request_cap, df_guard).
    assert result.path == "fresh"
    assert result.volume_limit_gb == 1
    assert result.storage_limit_gb == 1
    assert result.capped_by == "pool"


@pytest.mark.asyncio
async def test_sizing_measurement_failure_falls_back() -> None:
    # Arrange: the df measurement raises after docker info succeeded.
    svc = make_docker_service()
    payload = make_create_request(disk_share=0.5, volume_limit_gb=40, storage_limit_gb=20)
    ssh_client = make_sizing_ssh(df_error=Exception("df boom"))

    # Act
    with patch("services.docker_service.logger") as mock_logger:
        result = await svc.resolve_volume_sizing(
            ssh_client=ssh_client, payload=payload, log_tag="tag", log_extra={}
        )

    # Assert
    # WHY: measurement failure never breaks the rent — the payload limits are
    # echoed untouched under path="fresh_fallback" and every measurement field
    # stays None (VolumeSizingResult has NO warnings field; the fallback signal
    # lives only in the log).
    assert result.path == "fresh_fallback"
    assert result.volume_limit_gb == 40
    assert result.storage_limit_gb == 20
    assert result.capped_by is None
    assert result.df_avail_bytes is None
    assert result.existing_volumes_bytes is None
    # WHY: the exact log key "vloopback_fresh_sizing_fallback" is emitted via
    # logger.warning — the only observable trace of this branch (DRAFT C6).
    warning_keys = [call.args[0].message for call in mock_logger.warning.call_args_list]
    assert "vloopback_fresh_sizing_fallback" in warning_keys
    # WHY: measurement stops at the failing df — volume ls/inspect never run.
    assert len(ssh_client.commands) == 2
    assert not any("volume ls" in command for command in ssh_client.commands)


@pytest.mark.asyncio
async def test_sizing_storage_opt_unsupported() -> None:
    # Arrange: backend signals no --storage-opt support (storage_limit_gb=None)
    # while disk_share IS set — the gate must win over the fresh path.
    svc = make_docker_service()
    payload = make_create_request(disk_share=0.5, volume_limit_gb=7, storage_limit_gb=None)
    ssh_client = RoutingSSHClient([])

    # Act
    result = await svc.resolve_volume_sizing(
        ssh_client=ssh_client, payload=payload, log_tag="tag", log_extra={}
    )

    # Assert
    # WHY: storage_limit_gb is None is checked BEFORE disk_share, so even a
    # disk_share payload skips fresh derivation entirely (else dockerd rejects
    # --storage-opt on non-xfs/pquota hosts); limits pass through untouched,
    # storage stays None, and no SSH command is issued.
    assert result.path == "storage_opt_unsupported"
    assert result.volume_limit_gb == 7
    assert result.storage_limit_gb is None
    assert result.capped_by is None
    assert ssh_client.commands == []


@pytest.mark.asyncio
async def test_sizing_legacy_no_ssh() -> None:
    # Arrange: no disk_share, both backend limits set (pre-DAH-2183 contract).
    svc = make_docker_service()
    payload = make_create_request(disk_share=None, volume_limit_gb=40, storage_limit_gb=20)
    ssh_client = RoutingSSHClient([])

    # Act
    result = await svc.resolve_volume_sizing(
        ssh_client=ssh_client, payload=payload, log_tag="tag", log_extra={}
    )

    # Assert
    # WHY: legacy passthrough — backend limits are exact sizes echoed untouched,
    # every fresh-only field stays None, and no host measurement runs.
    assert result.path == "legacy"
    assert result.volume_limit_gb == 40
    assert result.storage_limit_gb == 20
    assert result.capped_by is None
    assert result.df_avail_bytes is None
    assert result.existing_volumes_bytes is None
    assert ssh_client.commands == []


def test_parse_volume_size_formats() -> None:
    # Arrange / Act / Assert
    # WHY: the sketch's base formats ("20401094656"/"19g"/"1t"/"<no value>"/""/
    # None, 1024-based multipliers) are already pinned at
    # tests/test_docker_service.py:4434; this extends only the shapes the regex
    # r"(\d+(?:\.\d+)?)\s*([kmgt]?)b?" accepts or rejects beyond that list.
    assert _parse_volume_size_to_bytes("1.5g") == 1610612736  # decimal fraction, 1.5*1024**3
    assert _parse_volume_size_to_bytes("19G") == 19 * 1024**3  # suffix is case-insensitive
    assert _parse_volume_size_to_bytes("20gb") == 20 * 1024**3  # trailing 'b' accepted
    assert _parse_volume_size_to_bytes("512m") == 512 * 1024**2
    assert _parse_volume_size_to_bytes("10k") == 10 * 1024
    assert _parse_volume_size_to_bytes("100b") == 100  # raw bytes with 'b'
    assert _parse_volume_size_to_bytes("19 g") == 19 * 1024**3  # inner whitespace allowed
    assert _parse_volume_size_to_bytes(" 19g ") == 19 * 1024**3  # outer whitespace stripped
    assert _parse_volume_size_to_bytes("1.5") == 1  # sub-byte fraction truncates via int()
    assert _parse_volume_size_to_bytes("12x") is None  # 'x' is not a size suffix
    assert _parse_volume_size_to_bytes("-1g") is None  # no sign accepted


# ---------------------------------------------------------------------------
# C.3 port planner (generate_portMappings) — lock-failure branch only;
# C18/C19 are covered at tests/test_docker_service.py:623/:673 (DRAFT 0.5).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lock_exc_cls",
    [redis.exceptions.LockError, redis.exceptions.LockNotOwnedError],
)
async def test_ports_lock_error_returns_empty(
    lock_exc_cls: type[redis.exceptions.LockError],
) -> None:
    # Arrange
    svc = make_docker_service()
    svc.redis_service.acquire_executor_lock = Mock(side_effect=lock_exc_cls("lock failed"))
    executor_id = str(uuid4())

    # Act
    result = await svc.generate_portMappings(
        miner_hotkey="miner",
        executor_id=executor_id,
        pod_id=uuid4(),
        available_ports_raw=[PayloadPortMapping(internal_port=20001, external_port=20001)],
        pod_mapping_raw=[],
    )

    # Assert
    # WHY: the planner catches exactly (redis.exceptions.LockError,
    # redis.exceptions.LockNotOwnedError) and degrades to ([], None), which
    # create_container later maps to NoPortMappings/failure_step="port_mapping"
    # (that boundary is pinned at tests/test_docker_service.py:623).
    assert result == ([], None)
    svc.redis_service.acquire_executor_lock.assert_called_once_with(executor_id)
