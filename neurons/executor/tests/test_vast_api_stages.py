import subprocess
from unittest.mock import Mock

import pytest

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


def test_existing_loop_losetup_failure_raises(tmp_path):
    # a failed losetup -j must not read as "no loop attached" — ensure_data_root
    # would attach a second loop to the same image
    host, calls, settings, responses = _host_with_fake_run(tmp_path)
    responses["losetup"] = (1, "")

    with pytest.raises(ApiFailure) as exc_info:
        host._existing_loop()

    assert exc_info.value.code == "host_command_failed"


def test_purge_data_root_umount_detach_rm_order(tmp_path):
    host, calls, settings, responses = _host_with_fake_run(tmp_path)
    responses["losetup"] = (0, f"/dev/loop7: []: ({settings.DATA_ROOT_IMG})\n")

    host.purge_data_root()

    assert calls == [
        ["umount", settings.DATA_ROOT_MOUNT],
        ["losetup", "-j", settings.DATA_ROOT_IMG],
        ["losetup", "-d", "/dev/loop7"],
        ["rm", "-f", settings.DATA_ROOT_IMG],
    ]


def test_purge_data_root_without_attached_loop_skips_detach(tmp_path):
    host, calls, settings, responses = _host_with_fake_run(tmp_path)
    responses["losetup"] = (0, "")

    host.purge_data_root()

    assert ["losetup", "-d", ""] not in calls
    assert not any(args[:2] == ["losetup", "-d"] for args in calls)
    assert ["rm", "-f", settings.DATA_ROOT_IMG] in calls


def test_reserve_publish_ports_merges_with_existing(tmp_path):
    host, calls, settings, responses = _host_with_fake_run(tmp_path)
    responses["sysctl"] = (0, "50000\n")  # pre-existing reservation must survive

    host.reserve_publish_ports()

    span = f"{settings.PORT_RANGE_START}-{settings.PORT_RANGE_END}"
    write_call = next(args for args in calls if args[0] == "sysctl" and args[1] == "-w")
    assert write_call[2] == f"net.ipv4.ip_local_reserved_ports=50000,{span}"
    persist_call = next(args for args in calls if args[0] == "sh")
    assert "/etc/sysctl.d/99-vast-uns-ports.conf" in persist_call[2]


def test_reserve_publish_ports_idempotent_when_already_reserved(tmp_path):
    host, calls, settings, responses = _host_with_fake_run(tmp_path)
    span = f"{settings.PORT_RANGE_START}-{settings.PORT_RANGE_END}"
    responses["sysctl"] = (0, f"{span}\n")

    host.reserve_publish_ports()

    assert not any(args[0] == "sysctl" and args[1] == "-w" for args in calls)


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


def _register_stage(tmp_path, docker_ops, vast, machine_id=None):
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    stages, ctx = build_setup_ladder(settings, Mock(), docker_ops, vast, "mk-test", machine_id)
    return next(stage for stage in stages if stage.name == "register"), ctx


def test_register_do_mismatched_machine_id_aborts_without_persist(tmp_path):
    # identify returned a different machine than the one asked to adopt — the wrong
    # identity must not be persisted and vastai must not be restarted with it
    docker_ops = Mock()
    _exec_responses(docker_ops, {"cat": (0, "abc123hex\n"), "systemctl": (0, "")})
    vast = Mock()
    vast.identify.return_value = {"machine_id": 147200}
    register, _ = _register_stage(tmp_path, docker_ops, vast, machine_id=147063)

    with pytest.raises(ApiFailure) as exc_info:
        register.do()

    assert exc_info.value.code == "identify_rejected"
    docker_ops.write_file_in_uns.assert_not_called()
    executed = [c.args[0] for c in docker_ops.exec_in_uns.call_args_list]
    assert ["systemctl", "restart", "vastai"] not in executed


def test_register_do_identify_without_machine_id_rejected(tmp_path):
    docker_ops = Mock()
    _exec_responses(docker_ops, {"cat": (0, "abc123hex\n")})
    vast = Mock()
    vast.identify.return_value = {"success": False, "error": "bad key"}
    register, _ = _register_stage(tmp_path, docker_ops, vast)

    with pytest.raises(ApiFailure) as exc_info:
        register.do()

    assert exc_info.value.code == "identify_rejected"
    docker_ops.write_file_in_uns.assert_not_called()


def test_register_do_vastai_restart_failure_raises(tmp_path):
    # a wedged restart leaves the pre-identity daemon running — never report success
    docker_ops = Mock()
    _exec_responses(docker_ops, {"cat": (0, "abc123hex\n"), "systemctl": (1, "restart wedged")})
    vast = Mock()
    vast.identify.return_value = {"machine_id": 147200}
    register, _ = _register_stage(tmp_path, docker_ops, vast)

    with pytest.raises(ApiFailure) as exc_info:
        register.do()

    assert exc_info.value.code == "host_command_failed"


def _gpu_stage(tmp_path, docker_ops):
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    stages, _ = build_setup_ladder(settings, Mock(), docker_ops, Mock(), "mk-test")
    return next(stage for stage in stages if stage.name == "gpu")


def test_gpu_check_daemon_reload_failure_fails_ladder(tmp_path):
    # a failed daemon-reload must not let the ×2 GPU gate pass vacuously
    store = RunStore(str(tmp_path))
    docker_ops = Mock()
    docker_ops.nested_gpu_ok.side_effect = [True, True]
    docker_ops.exec_in_uns.return_value = (1, "reload broken")
    gpu = _gpu_stage(tmp_path, docker_ops)
    run_id = store.create("setup", {}, ["gpu"])

    succeeded = run_ladder([gpu], store, run_id)

    assert succeeded is False
    doc = store.get(run_id)
    assert doc.error["code"] == "host_command_failed"
    assert docker_ops.nested_gpu_ok.call_count == 1  # second pass never reached


def test_gpu_lost_after_daemon_reload_is_gpu_broken(tmp_path):
    store = RunStore(str(tmp_path))
    docker_ops = Mock()
    docker_ops.nested_gpu_ok.side_effect = [True, False]
    docker_ops.exec_in_uns.return_value = (0, "")
    gpu = _gpu_stage(tmp_path, docker_ops)
    run_id = store.create("setup", {}, ["gpu"])

    succeeded = run_ladder([gpu], store, run_id)

    assert succeeded is False
    assert store.get(run_id).error["code"] == "gpu_broken"


def test_nested_daemon_check_requires_immutability(tmp_path):
    # a run interrupted before chattr +i leaves daemon.json mutable — the stage
    # must count as unsatisfied so the retry re-runs it
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    docker_ops = Mock()
    docker_ops.nested_daemon_matches_asset.return_value = True
    docker_ops.nested_docker_active.return_value = True
    docker_ops.nested_daemon_immutable.return_value = False
    stages, _ = build_setup_ladder(settings, Mock(), docker_ops, Mock(), "mk-test")
    nested = next(stage for stage in stages if stage.name == "nested_daemon")

    assert nested.check() is False
    docker_ops.nested_daemon_immutable.return_value = True
    assert nested.check() is True


def test_kaalia_do_key_travels_only_as_env(tmp_path):
    # an f-string regression would leak the key into process listings
    settings = VastSettings(RUNS_DIR=str(tmp_path / "runs"))
    docker_ops = Mock()
    recorded = []

    def fake_exec(cmd, env=None):
        recorded.append((cmd, env))
        return (0, "")

    docker_ops.exec_in_uns.side_effect = fake_exec
    stages, _ = build_setup_ladder(settings, Mock(), docker_ops, Mock(), "mk-secret-key")
    kaalia = next(stage for stage in stages if stage.name == "kaalia")

    kaalia.do()

    envs = [env for _, env in recorded if env]
    assert any(env.get("MACHINE_KEY") == "mk-secret-key" for env in envs)
    for cmd, _ in recorded:
        assert all("mk-secret-key" not in part for part in cmd)
