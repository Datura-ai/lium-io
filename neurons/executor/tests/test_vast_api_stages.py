import subprocess
from unittest.mock import Mock

from vast_api.config import VastSettings
from vast_api.host_ops import HostOps
from vast_api.runs import RunStore
from vast_api.errors import ApiFailure, safe_error
from vast_api.stages import Stage, build_setup_ladder, run_ladder


def test_satisfied_stage_check_short_circuits_to_ok(tmp_path):
    store = RunStore(str(tmp_path))
    do = Mock()
    stages = [Stage("a", lambda: True, do)]
    run_id = store.create("setup", {}, ["a"])

    succeeded = run_ladder(stages, store, run_id)

    assert succeeded is True
    do.assert_not_called()
    assert store.get(run_id).stages[0].state == "ok"


def test_unsatisfied_stage_runs_do_and_verify(tmp_path):
    store = RunStore(str(tmp_path))
    do = Mock()
    verify = Mock()
    stages = [Stage("a", lambda: False, do, verify)]
    run_id = store.create("setup", {}, ["a"])

    succeeded = run_ladder(stages, store, run_id)

    assert succeeded is True
    do.assert_called_once()
    verify.assert_called_once()
    assert store.get(run_id).stages[0].state == "done"


def test_setup_ladder_stage_names_and_order(tmp_path):
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))

    stages, _ = build_setup_ladder(settings, Mock(), Mock(), Mock(), "mk-test")

    # kaalia before nested_daemon/gpu: the nvidia runtime is kaalia's shim
    assert [stage.name for stage in stages] == [
        "g0_gate", "image", "container", "data_root", "dmi_shim",
        "kaalia", "nested_daemon", "gpu", "register", "report",
    ]


def test_g0_failure_aborts_whole_ladder(tmp_path):
    store = RunStore(str(tmp_path))
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    docker_ops = Mock()
    docker_ops.executor_running.return_value = False
    stages, _ = build_setup_ladder(settings, Mock(), docker_ops, Mock(), "mk-test")
    run_id = store.create("setup", {}, [stage.name for stage in stages])

    succeeded = run_ladder(stages, store, run_id)

    assert succeeded is False
    doc = store.get(run_id)
    assert doc.state == "failed"
    assert doc.error["code"] == "g0_failed"
    assert doc.stages[0].state == "failed"
    assert all(stage.state == "skipped" for stage in doc.stages[1:])
    docker_ops.build_image.assert_not_called()


def _host_with_fake_run(tmp_path):
    # returns host whose run() is faked via `responses`: command name -> (rc, stdout);
    # commands not listed succeed silently
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    host = HostOps(settings)
    calls = []
    responses = {}

    def fake_run(args, check=True, timeout=300):
        calls.append(args)
        rc, stdout = responses.get(args[0], (0, ""))
        return subprocess.CompletedProcess(args, rc, stdout, "")

    host.run = fake_run
    return host, calls, settings, responses


def test_ensure_data_root_reuses_loop_and_skips_mkfs(tmp_path):
    host, calls, settings, responses = _host_with_fake_run(tmp_path)
    responses["mountpoint"] = (1, "")  # not mounted yet
    responses["test"] = (0, "")  # image file exists
    responses["blkid"] = (0, f'{settings.DATA_ROOT_IMG}: UUID="x" TYPE="xfs"')  # fs present
    responses["losetup"] = (0, f"/dev/loop7: []: ({settings.DATA_ROOT_IMG})\n")  # already attached

    host.ensure_data_root()

    command_names = [args[0] for args in calls]
    assert "mkfs.xfs" not in command_names
    assert ["losetup", "-f", "--show", settings.DATA_ROOT_IMG] not in calls
    mount_call = next(args for args in calls if args[0] == "mount")
    assert "/dev/loop7" in mount_call


def test_ensure_data_root_mkfs_when_image_has_no_filesystem(tmp_path):
    host, calls, settings, responses = _host_with_fake_run(tmp_path)
    responses["mountpoint"] = (1, "")
    responses["test"] = (0, "")  # image exists (e.g. crash between truncate and mkfs)
    responses["blkid"] = (2, "")  # but carries no filesystem
    responses["losetup"] = (0, "")  # no existing attachment

    host.ensure_data_root()

    assert "mkfs.xfs" in [args[0] for args in calls]


def test_dump_dmi_writes_content_in_place(tmp_path):
    host, calls, settings, responses = _host_with_fake_run(tmp_path)

    host.dump_dmi()

    tmp = f"{settings.DMI_BIN_HOST}.tmp"
    assert ["dmidecode", "--dump-bin", tmp] in calls  # dump to temp, never the target
    copy_call = next(args for args in calls if args[0] == "sh")
    assert f"cat {tmp} > {settings.DMI_BIN_HOST}" in copy_call[2]


def test_safe_error_keeps_curated_text_hides_internals():
    curated = safe_error(ApiFailure("rental_running", "live Vast contract container(s)"))
    leaky = safe_error(RuntimeError("nsenter -t 1 -m -n cat /var/lib/secret failed"))

    assert curated == "rental_running: live Vast contract container(s)"
    assert leaky == "RuntimeError"


def test_container_stage_state_mount_rail_aborts_without_remove(tmp_path):
    store = RunStore(str(tmp_path))
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    docker_ops = Mock()
    docker_ops.executor_running.return_value = True
    docker_ops.executor_network.return_value = "executor_default"
    docker_ops.image_exists.return_value = True
    container = Mock()
    docker_ops.get_vast_uns.return_value = container
    docker_ops.vast_uns_has_state_mount.return_value = False
    stages, _ = build_setup_ladder(settings, Mock(), docker_ops, Mock(), "mk-test")
    run_id = store.create("setup", {}, [stage.name for stage in stages])

    succeeded = run_ladder(stages, store, run_id)

    assert succeeded is False
    doc = store.get(run_id)
    assert doc.error["code"] == "state_mount_missing"
    container.remove.assert_not_called()


def _exec_responses(docker_ops, table):
    # exec_in_uns faked by first token of the command: token -> (rc, output)
    def fake_exec(cmd, env=None):
        key = cmd[0] if isinstance(cmd, list) else str(cmd)
        return table.get(key, (0, ""))

    docker_ops.exec_in_uns.side_effect = fake_exec


def test_register_check_adopts_persisted_numeric_id(tmp_path):
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    docker_ops = Mock()
    _exec_responses(docker_ops, {"test": (0, ""), "cat": (0, "147063\n")})
    stages, ctx = build_setup_ladder(settings, Mock(), docker_ops, Mock(), "mk-test")
    register = next(stage for stage in stages if stage.name == "register")

    satisfied = register.check()

    assert satisfied is True
    assert ctx["machine_id"] == 147063


def test_register_do_identifies_with_backend_minted_key(tmp_path):
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    docker_ops = Mock()
    _exec_responses(docker_ops, {"cat": (0, "abc123hex\n"), "systemctl": (0, "")})
    vast = Mock()
    vast.identify.return_value = {"success": False, "machine_id": 147200}
    stages, ctx = build_setup_ladder(settings, Mock(), docker_ops, vast, "mk-from-backend")
    register = next(stage for stage in stages if stage.name == "register")

    register.do()

    vast.identify.assert_called_once_with("mk-from-backend", "abc123hex")
    assert ctx["machine_id"] == 147200
    written = [c.args for c in docker_ops.write_file_in_uns.call_args_list]
    assert ("/var/lib/vastai_kaalia/numeric_machine_id", "147200") in written
