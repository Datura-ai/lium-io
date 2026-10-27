"""DAH-5101: the monitor restarts an executor held down by its own stale containerd task."""

import shutil
import signal
from types import SimpleNamespace

from services.stale_task_recovery import recover_stale_executor_task

OWN = "a" * 64
OTHER = "b" * 64


def make_proc(tmp_path, processes, states=None, cgroups=None):
    """processes: pid -> (ppid, argv); states/cgroups: pid -> state letter / cgroup file."""
    for pid, (ppid, argv) in processes.items():
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")
        (directory / "stat").write_text(f"{pid} (some name) {(states or {}).get(pid, 'S')} {ppid} 1 1")
        (directory / "cgroup").write_text((cgroups or {}).get(pid, "0::/system.slice/containerd.service\n"))
    return str(tmp_path)


def killer(proc_root, killed, survivors=()):
    """A SIGKILL that ends every pid except the survivors."""

    def kill(pid, sig):
        killed.append((pid, sig))
        if pid not in survivors:
            shutil.rmtree(f"{proc_root}/{pid}")

    return kill


class FakeContainer:
    def __init__(self, container_id, status, error):
        self.id = container_id
        self.name = "executor-executor-1"
        self.attrs = {"State": {"Status": status, "Error": error}}
        self.started = False
        self.reloaded_attrs = None

    def reload(self):
        if self.reloaded_attrs is not None:
            self.attrs = self.reloaded_attrs

    def start(self):
        self.started = True


def client_with(container):
    return SimpleNamespace(containers=SimpleNamespace(list=lambda **_: [container]))


PROCESSES = {
    100: (
        1,
        [
            "/usr/bin/containerd-shim-runc-v2",
            "-namespace",
            "moby",
            "-id",
            OWN,
            "-address",
            "/run/containerd.sock",
        ],
    ),
    101: (100, ["/sbin/docker-init", "--", "/root/app/run.sh"]),
    102: (101, ["pdm", "run", "python", "src/executor.py"]),
    200: (1, ["/usr/bin/containerd-shim-runc-v2", "-namespace", "moby", "-id", OTHER]),
    201: (200, ["python", "train.py"]),
    300: (1, ["bash", "-c", f"echo -id {OWN}"]),
}


def test_kills_only_its_own_shim_and_children_then_starts_the_executor(tmp_path):
    container = FakeContainer(
        OWN,
        "exited",
        f"failed to create task for container: AlreadyExists: task {OWN}: already exists",
    )
    killed = []
    proc_root = make_proc(tmp_path, PROCESSES)

    done = recover_stale_executor_task(client_with(container), proc_root, kill=killer(proc_root, killed))

    assert sorted(killed) == [(100, signal.SIGKILL), (101, signal.SIGKILL), (102, signal.SIGKILL)]
    assert container.started
    assert done and "started executor-executor-1" in done


def test_another_containers_task_id_or_another_error_is_left_alone(tmp_path):
    proc_root = make_proc(tmp_path, PROCESSES)
    for container in (
        FakeContainer(OWN, "exited", f"AlreadyExists: task {OTHER}: already exists"),
        FakeContainer(OWN, "exited", "OCI runtime create failed"),
        FakeContainer(OWN, "running", f"AlreadyExists: task {OWN}: already exists"),
    ):
        killed = []

        done = recover_stale_executor_task(
            client_with(container), proc_root, kill=lambda pid, sig: killed.append(pid)
        )

        assert done is None and killed == [] and not container.started


def stale_container():
    return FakeContainer(OWN, "exited", f"AlreadyExists: task {OWN}: already exists")


def test_a_tree_in_uninterruptible_io_is_left_for_the_next_pass(tmp_path):
    container, killed = stale_container(), []
    proc_root = make_proc(tmp_path, PROCESSES, states={102: "D"})

    done = recover_stale_executor_task(client_with(container), proc_root, kill=killer(proc_root, killed))

    assert killed == [] and not container.started and "uninterruptible" in done


def test_a_process_that_survives_the_kill_keeps_the_executor_stopped(tmp_path):
    container, killed = stale_container(), []
    proc_root = make_proc(tmp_path, PROCESSES)

    done = recover_stale_executor_task(
        client_with(container), proc_root, kill=killer(proc_root, killed, survivors={102}), sleep=lambda _: None
    )

    assert not container.started and "survived" in done


def test_a_survivor_reparented_to_init_is_still_found_by_its_cgroup(tmp_path):
    # the next pass after a survivor: its shim is gone, the process now hangs off init
    container, killed = stale_container(), []
    proc_root = make_proc(
        tmp_path,
        {102: (1, ["pdm", "run", "python", "src/executor.py"]), 201: (200, ["python", "train.py"])},
        cgroups={102: f"0::/system.slice/docker-{OWN}.scope\n"},
    )

    recover_stale_executor_task(client_with(container), proc_root, kill=killer(proc_root, killed))

    assert killed == [(102, signal.SIGKILL)] and container.started


def test_a_restart_that_won_meanwhile_keeps_its_fresh_shim(tmp_path):
    container, killed = stale_container(), []
    container.reloaded_attrs = {"State": {"Status": "running", "Error": ""}}
    proc_root = make_proc(tmp_path, PROCESSES)

    done = recover_stale_executor_task(client_with(container), proc_root, kill=killer(proc_root, killed))

    assert done is None and killed == [] and not container.started
