"""Read-only evidence for a two-phase filler removal (DAH-4978).

This module does not remove containers or authorize a rental. Capture before
submitting removal by full container ID; observe without Docker inspect while
Docker is blocked in filesystem cleanup. The caller still owns cleanup, GPU
health checks, disk admission and protection against a replacement filler.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

PROC_ROOT = Path('/proc')
CGROUP_ROOT = Path('/proc/1/root/sys/fs/cgroup')
DOCKER = '/usr/bin/docker'
CONTAINER_ID = re.compile(r'[0-9a-f]{64}')


class ReleaseEvidenceUnavailable(RuntimeError):
    """Missing or inconsistent evidence must never mean resources were released."""


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError as exc:
        raise ReleaseEvidenceUnavailable(f'Cannot read {path}') from exc


def _start_ticks(pid: int) -> int | None:
    try:
        stat = (PROC_ROOT / str(pid) / 'stat').read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReleaseEvidenceUnavailable('Cannot read process identity') from exc
    try:
        # comm may contain spaces and parentheses; starttime is field 22.
        return int(stat.rsplit(')', 1)[1].split()[19])
    except (IndexError, ValueError) as exc:
        raise ReleaseEvidenceUnavailable('Invalid process identity') from exc


def _command(argv: list[str]) -> str:
    try:
        return subprocess.run(
            argv, check=True, capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseEvidenceUnavailable(f'{argv[0]} did not return usable evidence') from exc


def _gpu_uuid(value: str) -> str:
    if not value.startswith('GPU-'):
        raise ValueError('Expected a physical GPU UUID')
    return 'GPU-' + str(UUID(value[4:]))


@dataclass(frozen=True)
class FillerReleaseSnapshot:
    container_id: str
    init_pid: int
    init_start_ticks: int
    boot_id: str
    cgroup_path: str
    cgroup_inode: int
    gpu_uuids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not CONTAINER_ID.fullmatch(self.container_id):
            raise ValueError('A full container ID is required')
        if self.init_pid <= 1 or self.init_start_ticks < 0 or self.cgroup_inode <= 0:
            raise ValueError('Invalid process or cgroup identity')
        parts = Path(self.cgroup_path).parts
        if not self.cgroup_path.startswith('/') or '..' in parts:
            raise ValueError('Invalid cgroup path')
        if parts[-1] not in {self.container_id, f'docker-{self.container_id}.scope'}:
            raise ValueError('The cgroup must be the root of the captured container')
        UUID(self.boot_id)
        normalized = tuple(_gpu_uuid(value) for value in self.gpu_uuids)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError('Distinct physical GPU UUIDs are required')
        object.__setattr__(self, 'gpu_uuids', normalized)


def _cgroup(snapshot: FillerReleaseSnapshot) -> Path:
    path = CGROUP_ROOT / snapshot.cgroup_path.lstrip('/')
    # Do not follow a stored path outside the host cgroup mount.
    if not path.resolve().is_relative_to(CGROUP_ROOT.resolve()):
        raise ReleaseEvidenceUnavailable('Cgroup path escaped the host mount')
    return path


def capture_filler_release(container_id: str, gpu_uuids: tuple[str, ...]) -> FillerReleaseSnapshot:
    """Capture before removal. Cgroup v1, missing PIDs and ambiguous identity are unsupported."""
    if not CONTAINER_ID.fullmatch(container_id):
        raise ValueError('A full container ID is required')
    try:
        [info] = json.loads(_command([DOCKER, 'container', 'inspect', container_id]))
        if info['Id'] != container_id or not info['Name'].lstrip('/').startswith('filler_'):
            raise ValueError('Expected the exact filler container')
        state = info['State']
        if not state['Running'] or state.get('Restarting', False):
            raise ValueError('Expected a running, non-restarting filler')
        pid = int(state['Pid'])
        start_ticks = _start_ticks(pid)
        if start_ticks is None:
            raise ValueError('Filler exited before capture')
        paths = [line[3:] for line in _read(PROC_ROOT / str(pid) / 'cgroup').splitlines()
                 if line.startswith('0::')]
        if len(paths) != 1:
            raise ValueError('A unified cgroup v2 hierarchy is required')
        # Sysbox/systemd may move init into a nested group. Observe the container
        # root, so sibling services cannot survive outside the sampled subtree.
        parts = Path(paths[0]).parts
        root_index = next(i for i, part in enumerate(parts)
                          if part in {container_id, f'docker-{container_id}.scope'})
        cgroup_path = str(Path(*parts[:root_index + 1]))
        snapshot = FillerReleaseSnapshot(
            container_id, pid, start_ticks, _read(PROC_ROOT / 'sys/kernel/random/boot_id'),
            cgroup_path, (CGROUP_ROOT / cgroup_path.lstrip('/')).stat().st_ino, gpu_uuids,
        )
        _cgroup(snapshot)
        # Capture must not combine the PID's old identity with a reused PID's cgroup.
        if _start_ticks(pid) != start_ticks:
            raise ValueError('Process identity changed during capture')
        return snapshot
    except (KeyError, TypeError, ValueError, OSError, StopIteration) as exc:
        raise ReleaseEvidenceUnavailable('Could not capture filler identity') from exc


def _processes_gone(snapshot: FillerReleaseSnapshot) -> bool:
    if _read(PROC_ROOT / 'sys/kernel/random/boot_id') != snapshot.boot_id:
        raise ReleaseEvidenceUnavailable('Host rebooted since capture')
    # A missing mount must not be confused with a removed container cgroup.
    _read(CGROUP_ROOT / 'cgroup.controllers')
    if _start_ticks(snapshot.init_pid) == snapshot.init_start_ticks:
        return False
    path = _cgroup(snapshot)
    try:
        inode = path.stat().st_ino
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise ReleaseEvidenceUnavailable('Cannot inspect captured cgroup') from exc
    if inode != snapshot.cgroup_inode:
        raise ReleaseEvidenceUnavailable('Captured cgroup was replaced')
    events = _read(path / 'cgroup.events').splitlines()
    populated = [line.split() for line in events if line.startswith('populated ')]
    if populated not in ([["populated", "0"]], [["populated", "1"]]):
        raise ReleaseEvidenceUnavailable('Missing or invalid cgroup populated field')
    # cgroup.events populated covers descendants, unlike just cgroup.procs.
    return populated == [["populated", "0"]]


@dataclass(frozen=True)
class FillerReleaseObservation:
    processes_gone: bool
    gpu_compute_empty: bool


def observe_filler_release(snapshot: FillerReleaseSnapshot) -> FillerReleaseObservation:
    """Observe process/cgroup death and compute clients, without touching Docker locks.

    Empty compute-client output is not a GPU health or memory-release guarantee;
    the existing wedged-GPU checks still apply. Unsupported/failed reads raise.
    """
    if not _processes_gone(snapshot):
        return FillerReleaseObservation(False, False)
    try:
        available = {
            _gpu_uuid(line.strip()) for line in _command([
                'nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits',
            ]).splitlines() if line.strip()
        }
        if not set(snapshot.gpu_uuids) <= available:
            raise ValueError('A captured GPU is missing')
        output = _command([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits',
        ])
        busy = set()
        for row in csv.reader(output.splitlines()):
            if not row:
                continue
            if len(row) != 2 or not row[1].strip().isdigit():
                raise ValueError('Invalid GPU process row')
            busy.add(_gpu_uuid(row[0].strip()))
    except ValueError as exc:
        raise ReleaseEvidenceUnavailable('Invalid GPU evidence') from exc
    # Check again after querying NVML, in case descendants appeared meanwhile.
    processes_gone = _processes_gone(snapshot)
    return FillerReleaseObservation(
        processes_gone, processes_gone and not (busy & set(snapshot.gpu_uuids)),
    )
