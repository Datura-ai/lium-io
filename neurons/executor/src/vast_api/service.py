import json
import logging
import threading
import time
from pathlib import Path

from vast_api.config import VastSettings
from vast_api.docker_ops import DockerOps
from vast_api.errors import ApiFailure, ApiRefusal, safe_error
from vast_api.host_ops import HostOps
from vast_api.runs import RunStore
from vast_api.stages import build_setup_ladder, run_ladder
from vast_api.vast_client import VastClient

logger = logging.getLogger(__name__)

GPU_BUSY_MEM_MIB = 1000


class VastManager:
    """Wires HostOps/DockerOps/VastClient/RunStore into the /vast operations."""

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

    # --- machine resolution ---

    def _resolve_machine(self) -> dict:
        # pick THIS box's machine on a (possibly shared) Vast account.
        # Codes are distinct on purpose: "no_machine" = account has zero machines,
        # "machine_unresolved" = machines exist but none matched this box by IP
        # (public_ipaddr can legitimately be empty on a fresh machine).
        machines = self.vast.get_machines()
        if not machines:
            raise ApiFailure("no_machine", "no machine registered on this account")
        # primary key: the numeric id persisted at register time — IP matching fails
        # on NAT'ed clouds where the host interface never carries the public address
        persisted = self.host.run(
            ["cat", f"{self.settings.STATE_DIR_HOST}/numeric_machine_id"], check=False
        ).stdout.strip()
        if persisted.isdigit():
            for m in machines:
                if m.get("id") == int(persisted):
                    return m
            # a stale persisted id must never fall through to guessing — on a shared
            # account the guess could be somebody else's machine
            raise ApiFailure(
                "machine_unresolved", f"persisted machine {persisted} is not on this account"
            )
        if len(machines) == 1:
            return machines[0]
        host_ips = self.host.local_ips()
        matches = [m for m in machines if m.get("public_ipaddr") in host_ips]
        if len(matches) == 1:
            return matches[0]
        raise ApiFailure(
            "machine_unresolved",
            f"cannot resolve this box among {len(machines)} account machines by public_ipaddr",
        )

    # --- async runs ---

    def _refuse_setup_if_rental_running(self) -> None:
        # the ladder restarts nested dockerd/vastai — forbidden during a live rental.
        # Only a fresh box (zero machines on the account) may proceed: any other
        # resolution failure cannot prove there is no rental.
        try:
            machine = self._resolve_machine()
        except ApiFailure as exc:
            if exc.code == "no_machine":
                logger.info("rental pre-flight skipped, fresh box: %s", exc.detail)
                return
            raise ApiRefusal(exc.code, f"cannot verify rental state: {exc.detail}")
        if machine.get("current_rentals_running", 0) > 0:
            raise ApiRefusal(
                "rental_running",
                f"machine {machine.get('id')} has {machine['current_rentals_running']} running rental(s)",
            )

    def setup(self, machine_id: int | None = None, force: bool = False) -> str:
        if not force:
            self._refuse_setup_if_rental_running()
        # build first (pure closure construction) so the run doc's stage list is
        # derived from the ladder itself, never a second hand-kept name list
        stages, ctx = build_setup_ladder(
            self.settings, self.host, self.docker_ops, self.vast, machine_id
        )
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

    def self_test(self) -> str:
        run_id = self.runs.create("selftest", {}, ["self_test"])

        def _fail(code: str, detail: str) -> None:
            self.runs.stage_finished(run_id, "self_test", "failed", detail)
            try:
                self.runs.finish(run_id, "failed", error={"code": code, "detail": detail})
            except Exception:
                logger.critical("failed to persist terminal state for %s", run_id, exc_info=True)

        def _execute() -> None:
            self.runs.stage_started(run_id, "self_test")
            try:
                machine = self._resolve_machine()
                raw_text = self.vast.self_test_machine(machine["id"])
                # raw/ subdir: RunStore globs RUNS_DIR/*.json, raw dumps must not look like run docs
                raw_dir = Path(self.settings.RUNS_DIR) / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / f"{run_id}-raw.json"
                raw_path.write_text(raw_text)
                result = _parse_self_test(raw_text)
                result["raw_path"] = str(raw_path)
                self.runs.stage_finished(run_id, "self_test", "done")
                self.runs.finish(run_id, "succeeded", result=result)
            except (ApiRefusal, ApiFailure) as exc:
                _fail(exc.code, exc.detail)
            except Exception as exc:
                logger.exception("self-test run %s crashed", run_id)
                _fail("unexpected", safe_error(exc))

        threading.Thread(target=_execute, name=run_id, daemon=True).start()
        return run_id

    # --- sync market ops ---

    def _refuse_if_run_in_progress(self) -> None:
        # a live setup/self-test run mutates the same container/daemons — refuse without force
        for doc in self.runs.list_runs():
            if doc.state == "running":
                raise ApiRefusal("run_in_progress", "a setup/self-test run is active")

    def _refuse_if_gpu_busy(self) -> None:
        # the Lium filler on a card blocks the offer — refuse without force
        busy = [gpu for gpu in self.host.gpu_stats() if gpu["mem_mib"] > GPU_BUSY_MEM_MIB]
        apps = self.host.gpu_compute_apps()
        if busy or apps:
            detail = f"busy GPUs {[gpu['idx'] for gpu in busy]}, {len(apps)} compute process(es)"
            raise ApiRefusal("gpu_busy", detail)

    def list_machine(self, price_gpu_usd: float, duration: str = "7 days", force: bool = False) -> dict:
        # always the proven unlist→list cycle — a repeated list does not rebuild a stuck listing
        if not force:
            self._refuse_if_run_in_progress()
        machine = self._resolve_machine()
        if not force:
            self._refuse_if_gpu_busy()
        self.vast.unlist_machine(machine["id"])
        self.vast.list_machine(machine["id"], price_gpu_usd, duration)
        # the CLI prints "offers created/updated" even when it listed NOTHING (seen live:
        # a restricted api key silently no-ops — listing needs the full account key), so
        # the only trustworthy signal is the offers actually appearing; they can lag ~1 min
        offers: list[dict] = []
        for _ in range(9):
            offers = self.vast.search_offers(machine["id"])
            if offers:
                break
            time.sleep(10)
        if not offers:
            raise ApiFailure(
                "listing_not_confirmed",
                "no offers appeared within 90s of unlist→list — stuck listing, "
                "a key without listing rights, or a busy GPU",
            )
        return {"listed": True, "offers": offers, "warnings": []}

    def unlist(self, force: bool = False) -> dict:
        if not force:
            self._refuse_if_run_in_progress()
        machine = self._resolve_machine()
        self.vast.unlist_machine(machine["id"])
        return {"listed": False, "offers_remaining": len(self.vast.search_offers(machine["id"]))}

    def price(self, price_gpu_usd: float) -> dict:
        # same as list until the in-place reprice experiment proves a plain re-list is safe
        return self.list_machine(price_gpu_usd)

    def delete(self, purge: bool = False, force: bool = False) -> dict:
        if not force:
            self._refuse_if_run_in_progress()
        machine = None
        try:
            machine = self._resolve_machine()
        except ApiFailure as exc:
            # only no_machine (zero records) proves there is no rental — any other
            # failure (unresolved id, Vast outage) must NOT silently bypass the
            # rental rail; refuse unless forced
            if exc.code != "no_machine" and not force:
                raise ApiRefusal(exc.code, exc.detail)
        if machine and machine.get("current_rentals_running", 0) > 0 and not force:
            raise ApiRefusal(
                "rental_running",
                f"machine {machine['id']} has {machine['current_rentals_running']} running rental(s)",
            )
        container_removed = self.docker_ops.delete_vast_uns()
        state_removed = False
        vast_record_deleted = False
        if purge:
            # account record first: a failed record delete must not leave an orphan
            # with the local identity already destroyed
            if machine:
                self.vast.delete_machine(machine["id"])
                vast_record_deleted = True
            self.host.run(["rm", "-rf", self.settings.STATE_DIR_HOST])
            state_removed = True
        return {
            "container_removed": container_removed,
            "state_removed": state_removed,
            "vast_record_deleted": vast_record_deleted,
        }

    # --- status ---

    def status(self) -> dict:
        # compose the doc; sections degrade independently
        machine: dict | None = None
        try:
            machine = self._resolve_machine()
            vast_section = {
                "machine_id": machine.get("id"),
                "listed": machine.get("listed"),
                "reliability2": machine.get("reliability2"),
                "verification": machine.get("verification"),
                "gpu_occupancy": machine.get("gpu_occupancy"),
                "rentals_running": machine.get("current_rentals_running"),
                "earn_hour": machine.get("earn_hour"),
                "inet_down": machine.get("inet_down"),
                "inet_up": machine.get("inet_up"),
            }
        except Exception as exc:
            vast_section = {"error": safe_error(exc)}

        try:
            offers = self.vast.search_offers(machine["id"]) if machine else []
        except Exception as exc:
            offers = {"error": safe_error(exc)}

        box_section: dict = {}
        try:
            box_section["executor_health"] = self.docker_ops.executor_health()
            box_section["vast_uns_up"] = self.docker_ops.get_vast_uns() is not None
            box_section["filler_running"] = self.docker_ops.filler_running()
        except Exception as exc:
            box_section["error"] = safe_error(exc)
        try:
            rc, active = self.docker_ops.exec_in_uns(["systemctl", "is-active", "vastai"])
            box_section["kaalia_active"] = rc == 0 and active.strip() == "active"
            _, last_log = self.docker_ops.exec_in_uns(
                ["journalctl", "-u", "vastai", "-n", "1", "--no-pager"]
            )
            box_section["kaalia_last_log"] = last_log.strip()[-300:]
        except Exception as exc:
            box_section["kaalia_error"] = safe_error(exc)
        try:
            box_section["gpus"] = self.host.gpu_stats()
        except Exception as exc:
            box_section["gpus_error"] = safe_error(exc)

        vast_verified = bool(machine and machine.get("verification") == "verified")
        return {
            "goal": {"lium_active": None, "vast_verified": vast_verified, "pair_ok": None},
            "vast": vast_section,
            "offers": offers,
            "box": box_section,
            "lium": {"active": None, "note": "not knowable from the box — enriched client-side"},
            "ttl": {"expires_at": None, "note": "Shadeform TTL — enriched client-side"},
        }


def _parse_self_test(raw_text: str) -> dict:
    # real CLI verdict (seen live on 147139): top-level success/failure_code/phase, checks
    # as a list of {id, status: pass|fail|info, summary}; tolerate CLI noise around the JSON
    try:
        raw = json.loads(raw_text)
    except ValueError:
        start = raw_text.find("{")
        raw = None
        if start >= 0:
            try:
                raw = json.loads(raw_text[start:])
            except ValueError:
                raw = None
        if raw is None:
            return {"passed": False, "checks": [], "parse_error": "no parseable JSON in self-test output"}
    checks = [
        {
            "name": c.get("id") or c.get("title", "?"),
            "status": c.get("status", "?"),
            "detail": c.get("summary"),
        }
        for c in raw.get("checks") or []
        if isinstance(c, dict)
    ]
    result = {"passed": bool(raw.get("success")), "checks": checks}
    for key in ("failure_code", "phase", "stage", "reason"):
        if raw.get(key):
            result[key] = raw[key]
    return result
