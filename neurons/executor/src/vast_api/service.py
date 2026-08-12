import json
import logging
import threading

from core.config import settings as executor_settings
from vast_api.config import VastSettings
from vast_api.docker_ops import DockerOps
from vast_api.errors import ApiRefusal, safe_error
from vast_api.host_ops import HostOps
from vast_api.runs import RunStore
from vast_api.stages import build_setup_ladder, run_ladder
from vast_api.vast_client import VastClient

logger = logging.getLogger(__name__)


class VastManager:
    """Wires HostOps/DockerOps/VastClient/RunStore into the /vast operations.

    Local operations only: enrollment, local status, local teardown. Market
    operations and everything needing the account key live in the backend
    (plan-key-split).
    """

    def __init__(
        self,
        settings: VastSettings,
        host: HostOps,
        docker_ops: DockerOps,
        vast: VastClient,
        runs: RunStore,
    ):
        self.settings = settings
        self.host = host
        self.docker_ops = docker_ops
        self.vast = vast
        self.runs = runs

    # --- rails ---

    def _refuse_if_run_in_progress(self) -> None:
        # a live setup run mutates the same container/daemons — refuse without force
        for doc in self.runs.list_runs():
            if doc.state == "running":
                raise ApiRefusal("run_in_progress", "a setup run is active")

    def _refuse_if_rental_running(self) -> None:
        # a live Vast contract IS a C.<id> container in the nested dockerd
        # (plan-contract-events) — local detection, no account key needed.
        # No vast-uns at all → nothing can be rented; an unreachable nested
        # docker raises and refuses by failure, force is the override.
        if self.docker_ops.get_vast_uns() is None:
            return
        containers = self.docker_ops.vast_rental_containers()
        if containers:
            raise ApiRefusal("rental_running", f"live Vast contract container(s): {containers}")

    def _lium_renting_ports(self) -> list[int]:
        # external host ports the Lium rental path publishes; same parsing as the
        # validator's port_utils.get_all_ports (that module is not importable here)
        if executor_settings.RENTING_PORT_MAPPINGS:
            return [int(pair[1]) for pair in json.loads(executor_settings.RENTING_PORT_MAPPINGS)]
        port_range = executor_settings.RENTING_PORT_RANGE
        if not port_range:
            return []
        if "-" in port_range:
            start, end = (int(p.strip()) for p in port_range.split("-"))
            return list(range(start, end + 1))
        return [int(p.strip()) for p in port_range.split(",")]

    def _refuse_if_port_overlap(self) -> None:
        # vast-uns publishes PORT_RANGE_START..END on the host — a collision with the
        # Lium renting ports breaks pod creation (deploy_failed penalties)
        overlap = [
            port
            for port in self._lium_renting_ports()
            if self.settings.PORT_RANGE_START <= port <= self.settings.PORT_RANGE_END
        ]
        if overlap:
            raise ApiRefusal(
                "port_range_overlap",
                f"Lium renting ports overlap the vast-uns publish range "
                f"{self.settings.PORT_RANGE_START}-{self.settings.PORT_RANGE_END}: {overlap[:20]}",
            )

    # --- async setup run ---

    def setup(self, machine_key: str, machine_id: int | None = None, force: bool = False) -> str:
        # the overlap rail is not force-overridable: colliding ports break Lium
        # pod creation regardless of intent
        self._refuse_if_port_overlap()
        if not force:
            self._refuse_if_rental_running()
        # build first (pure closure construction) so the run doc's stage list is
        # derived from the ladder itself, never a second hand-kept name list
        stages, ctx = build_setup_ladder(
            self.settings, self.host, self.docker_ops, self.vast, machine_key, machine_id
        )
        # args must never include machine_key: run docs are readable via GET /vast/runs
        run_id = self.runs.create("setup", {"machine_id": machine_id}, [s.name for s in stages])

        def _execute() -> None:
            try:
                if run_ladder(stages, self.runs, run_id):
                    self.runs.finish(
                        run_id,
                        "succeeded",
                        result={"machine_id": ctx["machine_id"], "registered": True, "reporting": True},
                    )
            except Exception as exc:
                logger.exception("setup run %s crashed outside the ladder", run_id)
                try:
                    self.runs.finish(run_id, "failed", error={"code": "unexpected", "detail": safe_error(exc)})
                except Exception:
                    logger.critical("failed to persist terminal state for %s", run_id, exc_info=True)

        threading.Thread(target=_execute, name=run_id, daemon=True).start()
        return run_id

    # --- local teardown ---

    def delete(self, purge: bool = False, force: bool = False) -> dict:
        # local teardown only; the backend deletes the Vast account record FIRST,
        # then calls here (plan-key-split purge order)
        if not force:
            self._refuse_if_run_in_progress()
            self._refuse_if_rental_running()
        container_removed = self.docker_ops.delete_vast_uns()
        state_removed = False
        if purge:
            # the nested data-root must go too: old C.<contract_id> containers in it
            # would resurrect contracts of the deleted machine on the next setup
            self.host.purge_data_root()
            self.host.run(["rm", "-rf", self.settings.STATE_DIR_HOST])
            state_removed = True
        return {"container_removed": container_removed, "state_removed": state_removed}

    # --- status ---

    def status(self) -> dict:
        # local sections only, degrading independently; Vast-side data (listed,
        # reliability, offers) is merged by the backend, which holds the account key
        machine_id = None
        machine_id_error = None
        try:
            # check=False: file absent / unreadable rc is "not enrolled", not an error
            result = self.host.run(
                ["cat", f"{self.settings.STATE_DIR_HOST}/numeric_machine_id"], check=False
            )
            if result.stdout.strip().isdigit():
                machine_id = int(result.stdout.strip())
        except Exception as exc:
            # a broken read (e.g. nsenter timeout) must not 500 the whole status
            machine_id_error = safe_error(exc)

        rental: list | dict | None
        try:
            if self.docker_ops.get_vast_uns() is None:
                rental = None
            else:
                rental = self.docker_ops.vast_rental_containers()
        except Exception as exc:
            rental = {"error": safe_error(exc)}

        box: dict = {}
        try:
            box["executor_health"] = self.docker_ops.executor_health()
            box["vast_uns_up"] = self.docker_ops.get_vast_uns() is not None
            box["filler_running"] = self.docker_ops.filler_running()
        except Exception as exc:
            box["error"] = safe_error(exc)
        try:
            rc, active = self.docker_ops.exec_in_uns(["systemctl", "is-active", "vastai"])
            box["kaalia_active"] = rc == 0 and active.strip() == "active"
            _, last_log = self.docker_ops.exec_in_uns(
                ["journalctl", "-u", "vastai", "-n", "1", "--no-pager"]
            )
            box["kaalia_last_log"] = last_log.strip()[-300:]
        except Exception as exc:
            box["kaalia_error"] = safe_error(exc)
        try:
            box["gpus"] = self.host.gpu_stats()
        except Exception as exc:
            box["gpus_error"] = safe_error(exc)

        return {
            "machine_id": machine_id,
            "machine_id_error": machine_id_error,
            "rental_containers": rental,
            "box": box,
        }
