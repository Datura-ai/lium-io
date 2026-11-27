import json
from dataclasses import replace

import pytest

from services import filler_release as release

CID = 'a' * 64
GPU = 'GPU-11111111-1111-1111-1111-111111111111'
OTHER_GPU = 'GPU-22222222-2222-2222-2222-222222222222'
BOOT = '33333333-3333-3333-3333-333333333333'


def write_stat(proc, ticks):
    (proc / '123').mkdir(exist_ok=True)
    fields = ['0'] * 20
    fields[0], fields[19] = 'S', str(ticks)
    (proc / '123/stat').write_text('123 (worker (test)) ' + ' '.join(fields))


@pytest.fixture
def host(tmp_path, monkeypatch):
    proc, cg = tmp_path / 'proc', tmp_path / 'cgroup'
    (proc / 'sys/kernel/random').mkdir(parents=True)
    (proc / 'sys/kernel/random/boot_id').write_text(BOOT)
    group = cg / f'system.slice/docker-{CID}.scope'
    group.mkdir(parents=True)
    (cg / 'cgroup.controllers').write_text('cpu memory pids')
    (group / 'cgroup.events').write_text('populated 0\nfrozen 0\n')
    monkeypatch.setattr(release, 'PROC_ROOT', proc)
    monkeypatch.setattr(release, 'CGROUP_ROOT', cg)
    snapshot = release.FillerReleaseSnapshot(
        CID, 123, 100, BOOT, f'/system.slice/docker-{CID}.scope', group.stat().st_ino, (GPU,),
    )
    return proc, group, snapshot


@pytest.fixture
def nvml(monkeypatch):
    calls = []

    def command(argv):
        # Observation must not enter a Docker lock after removal started.
        assert argv[0] == 'nvidia-smi'
        calls.append(argv)
        return GPU + '\n' if '--query-gpu=uuid' in argv else ''

    monkeypatch.setattr(release, '_command', command)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize('state,released', [('empty', True), ('init_alive', False),
                                         ('descendants', False), ('pid_reused', True),
                                         ('cgroup_removed', True)])
async def test_observation_uses_full_process_identity_and_descendant_state(host, nvml, state, released):
    # Arrange
    proc, group, snapshot = host
    if state == 'init_alive':
        write_stat(proc, 100)
    elif state == 'descendants':
        (group / 'cgroup.events').write_text('populated 1\n')
        (group / 'cgroup.procs').write_text('')
    elif state == 'pid_reused':
        write_stat(proc, 200)
    elif state == 'cgroup_removed':
        (group / 'cgroup.events').unlink()
        group.rmdir()
    # Act
    result = release.observe_filler_release(snapshot)
    # Assert — PID reuse is harmless, but a live init or any descendants prevent handoff.
    assert result == release.FillerReleaseObservation(released, released)
    assert len(nvml) == (2 if released else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['mount', 'events', 'boot', 'inode'])
async def test_missing_or_replaced_host_evidence_raises(host, nvml, failure):
    # Arrange
    proc, group, snapshot = host
    if failure == 'mount':
        (release.CGROUP_ROOT / 'cgroup.controllers').unlink()
    elif failure == 'events':
        (group / 'cgroup.events').write_text('frozen 0\n')
    elif failure == 'boot':
        (proc / 'sys/kernel/random/boot_id').write_text('different-boot')
    else:
        snapshot = replace(snapshot, cgroup_inode=snapshot.cgroup_inode + 1)
    # Act / Assert — unknown evidence must not become a successful release.
    with pytest.raises(release.ReleaseEvidenceUnavailable):
        release.observe_filler_release(snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize('occupant,empty', [(GPU, False), (OTHER_GPU, True)])
async def test_compute_clients_are_checked_on_the_selected_gpu(host, monkeypatch, occupant, empty):
    # Arrange
    _, _, snapshot = host
    monkeypatch.setattr(release, '_command', lambda argv: (
        GPU + '\n' if '--query-gpu=uuid' in argv else f'{occupant}, 999\n'
    ))
    # Act
    result = release.observe_filler_release(snapshot)
    # Assert — a selected GPU with a compute client cannot be handed off.
    assert result.processes_gone
    assert result.gpu_compute_empty is empty


@pytest.mark.asyncio
@pytest.mark.parametrize('available,processes', [('', ''), (GPU, 'unavailable'), (GPU, f'{GPU}, N/A')])
async def test_missing_gpu_or_invalid_nvml_output_is_not_empty(host, monkeypatch, available, processes):
    # Arrange
    _, _, snapshot = host
    monkeypatch.setattr(release, '_command', lambda argv: (
        available if '--query-gpu=uuid' in argv else processes
    ))
    # Act / Assert — an unsupported or missing GPU is not an idle GPU.
    with pytest.raises(release.ReleaseEvidenceUnavailable):
        release.observe_filler_release(snapshot)


@pytest.mark.asyncio
async def test_descendants_appearing_during_nvml_query_revoke_observation(host, monkeypatch):
    # Arrange
    _, group, snapshot = host

    def command(argv):
        if '--query-gpu=uuid' in argv:
            return GPU
        (group / 'cgroup.events').write_text('populated 1\n')
        return ''

    monkeypatch.setattr(release, '_command', command)
    # Act
    result = release.observe_filler_release(snapshot)
    # Assert — cgroup emptiness is checked again after querying NVML.
    assert result == release.FillerReleaseObservation(False, False)


@pytest.mark.asyncio
async def test_capture_retains_identity_without_inspect_environment(host, monkeypatch):
    # Arrange
    proc, _, expected = host
    write_stat(proc, 100)
    (proc / '123/cgroup').write_text(f'0::{expected.cgroup_path}/system.slice/init.scope\n')
    info = {'Id': CID, 'Name': '/filler_lab', 'State': {'Running': True, 'Pid': 123},
            'Config': {'Env': ['SECRET=not-for-the-snapshot']}}
    monkeypatch.setattr(release, '_command', lambda argv: json.dumps([info]))
    # Act
    snapshot = release.capture_filler_release(CID, (GPU,))
    # Assert — snapshot serialization retains no inspect environment.
    assert snapshot == expected
    assert 'SECRET' not in repr(snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize('name,cgroup', [('/pod_customer', f'0::/docker/{CID}'),
                                        ('/filler_lab', '1:memory:/docker/legacy')])
async def test_capture_refuses_customer_containers_and_cgroup_v1(host, monkeypatch, name, cgroup):
    # Arrange
    proc, _, _ = host
    write_stat(proc, 100)
    (proc / '123/cgroup').write_text(cgroup)
    monkeypatch.setattr(release, '_command', lambda argv: json.dumps([
        {'Id': CID, 'Name': name, 'State': {'Running': True, 'Pid': 123}},
    ]))
    # Act / Assert — customer teardown is outside this experimental contract.
    with pytest.raises(release.ReleaseEvidenceUnavailable):
        release.capture_filler_release(CID, (GPU,))


@pytest.mark.asyncio
@pytest.mark.parametrize('path', [f'/../docker/{CID}', '/system.slice/unrelated.scope',
                                 f'/docker/{CID}/system.slice/init.scope'])
async def test_snapshot_rejects_paths_outside_its_container(host, path):
    # Arrange
    _, _, snapshot = host
    # Act / Assert — stored snapshots cannot select unrelated cgroups.
    with pytest.raises(ValueError):
        replace(snapshot, cgroup_path=path)
