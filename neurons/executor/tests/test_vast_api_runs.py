import pytest

from vast_api.errors import ApiFailure, ApiRefusal
from vast_api.runs import RunStore


def test_run_create_stream_succeed(tmp_path):
    store = RunStore(str(tmp_path))

    run_id = store.create("setup", {"machine_id": 1}, ["a", "b"])
    store.stage_started(run_id, "a")
    store.stage_finished(run_id, "a", "ok")
    store.stage_started(run_id, "b")
    store.stage_finished(run_id, "b", "done", "did it")
    store.finish(run_id, "succeeded", result={"machine_id": 1})

    doc = store.get(run_id)
    assert doc.state == "succeeded"
    assert doc.finished_at is not None
    assert [stage.state for stage in doc.stages] == ["ok", "done"]
    assert doc.stages[1].detail == "did it"
    assert doc.stages[1].took_s is not None
    assert doc.result == {"machine_id": 1}


def test_second_concurrent_run_refused(tmp_path):
    store = RunStore(str(tmp_path))
    store.create("setup", {}, ["a"])

    with pytest.raises(ApiRefusal) as exc_info:
        store.create("setup", {}, ["a"])

    assert exc_info.value.code == "run_in_progress"


def test_interrupted_run_marked_failed_on_init(tmp_path):
    store = RunStore(str(tmp_path))
    run_id = store.create("setup", {}, ["a", "b"])
    store.stage_started(run_id, "a")

    restarted_store = RunStore(str(tmp_path))  # simulates a manager restart mid-run

    doc = restarted_store.get(run_id)
    assert doc.state == "failed"
    assert doc.error == {"code": "interrupted", "detail": "manager restarted mid-run"}
    assert [stage.state for stage in doc.stages] == ["skipped", "skipped"]
    # a new run is accepted again after the interrupted one is failed
    assert restarted_store.create("setup", {}, ["a"]) is not None


def test_get_corrupt_run_doc_raises_api_failure(tmp_path):
    store = RunStore(str(tmp_path))
    (tmp_path / "setup-20260811-000000.json").write_text("{not json")

    with pytest.raises(ApiFailure) as exc_info:
        store.get("setup-20260811-000000")

    assert exc_info.value.code == "run_doc_corrupt"


def test_write_is_atomic_no_tmp_leftover(tmp_path):
    store = RunStore(str(tmp_path))

    run_id = store.create("setup", {}, ["a"])

    assert (tmp_path / f"{run_id}.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
