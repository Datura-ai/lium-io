import json
import subprocess
from unittest.mock import Mock

from vast_api.config import VastSettings
from vast_api.host_ops import HostOps
from vast_api.runs import RunStore
from vast_api.errors import ApiFailure
from vast_api.service import _parse_self_test, _safe_error
from vast_api.stages import SETUP_STAGE_NAMES, Stage, build_setup_ladder, run_ladder


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
    settings = VastSettings(VAST_API_TOKEN="t", RUNS_DIR=str(tmp_path / "runs"))

    stages, _ = build_setup_ladder(settings, Mock(), Mock(), Mock())

    assert [stage.name for stage in stages] == SETUP_STAGE_NAMES
    # kaalia before nested_daemon/gpu: the nvidia runtime is kaalia's shim
    assert SETUP_STAGE_NAMES == [
        "g0_gate", "image", "container", "data_root", "dmi_shim",
        "kaalia", "nested_daemon", "gpu", "register", "report",
    ]


def test_g0_failure_aborts_whole_ladder(tmp_path):
    store = RunStore(str(tmp_path))
    settings = VastSettings(VAST_API_TOKEN="t", RUNS_DIR=str(tmp_path / "runs"))
    docker_ops = Mock()
    docker_ops.executor_running.return_value = False
    stages, _ = build_setup_ladder(settings, Mock(), docker_ops, Mock())
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
    settings = VastSettings(VAST_API_TOKEN="t", RUNS_DIR=str(tmp_path / "runs"))
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


def test_parse_self_test_real_cli_shape():
    # trimmed from a live verdict on machine 147139 (2026-08-11)
    raw = json.dumps({
        "success": False,
        "failure_code": "preflight_requirements_failed",
        "phase": "preflight",
        "checks": [
            {"id": "cuda.version", "status": "pass", "summary": "CUDA version: actual 12.8 CUDA, required >= 11.8 CUDA"},
            {"id": "reliability", "status": "fail", "summary": "Reliability: actual 0.6684413 ratio, required > 0.9 ratio"},
            {"id": "network.direct_ports.recommended_max", "status": "info", "summary": "Recommended max ports"},
        ],
    })

    result = _parse_self_test("some CLI noise\n" + raw)

    assert result["passed"] is False
    assert result["failure_code"] == "preflight_requirements_failed"
    assert [(c["name"], c["status"]) for c in result["checks"]] == [
        ("cuda.version", "pass"),
        ("reliability", "fail"),
        ("network.direct_ports.recommended_max", "info"),
    ]


def test_parse_self_test_unparseable_output():
    result = _parse_self_test("no json here at all")

    assert result["passed"] is False
    assert result["checks"] == []
    assert "parse_error" in result


def test_safe_error_keeps_curated_text_hides_internals():
    curated = _safe_error(ApiFailure("no_machine", "no machine registered on this account"))
    leaky = _safe_error(RuntimeError("nsenter -t 1 -m -n cat /var/lib/secret failed"))

    assert curated == "no_machine: no machine registered on this account"
    assert leaky == "RuntimeError"
