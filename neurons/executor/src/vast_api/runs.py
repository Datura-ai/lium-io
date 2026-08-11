import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from vast_api.errors import ApiFailure, ApiRefusal
from vast_api.schemas import RunDoc, StageDoc

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunStore:
    """Run docs as JSON files in RUNS_DIR; a run is an attempt journal, not a lock."""

    def __init__(self, runs_dir: str):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stage_started_at: dict[str, float] = {}
        self._mark_interrupted()

    def _mark_interrupted(self) -> None:
        # a run left `running` by a dead process can never finish — fail it on boot
        for doc in self.list_runs():
            if doc.state == "running":
                doc.state = "failed"
                doc.finished_at = _utcnow_iso()
                doc.error = {"code": "interrupted", "detail": "manager restarted mid-run"}
                for stage in doc.stages:
                    if stage.state in ("running", "pending"):
                        stage.state = "skipped"
                self._write(doc)
                logger.warning("marked run %s as interrupted", doc.run_id)

    def _path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def _write(self, doc: RunDoc) -> None:
        # atomic: temp file in the same dir + os.replace, so readers never see a torn doc
        path = self._path(doc.run_id)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(json.dumps(doc.model_dump(), indent=2))
        os.replace(tmp, path)

    def create(self, op: str, args: dict, stage_names: list[str]) -> str:
        # single-active-run enforcement lives here, at creation time
        with self._lock:
            for doc in self.list_runs():
                if doc.state == "running":
                    raise ApiRefusal("run_in_progress", f"run {doc.run_id} is still running")
            run_id = f"{op}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            doc = RunDoc(
                run_id=run_id,
                op=op,
                args=args,
                state="running",
                started_at=_utcnow_iso(),
                stages=[StageDoc(name=name) for name in stage_names],
            )
            self._write(doc)
            return run_id

    def get(self, run_id: str) -> RunDoc | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        try:
            return RunDoc.model_validate(json.loads(path.read_text()))
        except Exception:
            # a corrupt doc must surface as the contract envelope, not a bare 500
            raise ApiFailure("run_doc_corrupt", f"run {run_id} exists but is unreadable")

    def list_runs(self) -> list[RunDoc]:
        docs = []
        for path in self.runs_dir.glob("*.json"):
            try:
                docs.append(RunDoc.model_validate(json.loads(path.read_text())))
            except Exception:
                logger.warning("unreadable run doc %s", path, exc_info=True)
        docs.sort(key=lambda doc: doc.started_at, reverse=True)
        return docs

    def stage_started(self, run_id: str, name: str) -> None:
        self._stage_started_at[f"{run_id}:{name}"] = time.monotonic()
        self._set_stage(run_id, name, "running", None)

    def stage_finished(self, run_id: str, name: str, state: str, detail: str | None = None) -> None:
        started = self._stage_started_at.pop(f"{run_id}:{name}", None)
        took_s = round(time.monotonic() - started, 1) if started is not None else None
        self._set_stage(run_id, name, state, detail, took_s)

    def _set_stage(
        self, run_id: str, name: str, state: str, detail: str | None, took_s: float | None = None
    ) -> None:
        # stream one stage event into the run file
        with self._lock:
            doc = self.get(run_id)
            if doc is None:
                return
            for stage in doc.stages:
                if stage.name == name:
                    stage.state = state
                    if detail is not None:
                        stage.detail = detail
                    if took_s is not None:
                        stage.took_s = took_s
            self._write(doc)

    def finish(
        self,
        run_id: str,
        state: str,
        result: dict | None = None,
        error: dict | None = None,
    ) -> None:
        with self._lock:
            doc = self.get(run_id)
            if doc is None:
                return
            doc.state = state
            doc.finished_at = _utcnow_iso()
            doc.result = result
            doc.error = error
            if state == "failed":
                for stage in doc.stages:
                    if stage.state == "pending":
                        stage.state = "skipped"
            self._write(doc)
