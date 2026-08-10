"""The launch path, in the order that makes it safe.

    guard -> resolve -> prepare -> MEASURE -> launch -> confirm

Nothing reaches QEMU until the artifacts written to disk hash to the triple the caller asked
for. Every refusal below is a refusal *before* a VM exists, which is what keeps a rejected
launch from leaving the node in a state someone has to clean up by hand.

Every method here is blocking. `create` runs `setup_instance`, spawns a process and then waits
minutes for a guest to boot — the routes call it on a worker thread so the daemon keeps
answering `/v1/state` while a launch is in flight, which is the only way the state transitions
are observable to anyone.
"""

import logging
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from cvmd import catalog
from cvmd.config import LaunchConfig
from cvmd.cvm import measure, ports, release, renter, sshkey, supervisor
from cvmd.cvm.instance import Instance, InstanceStore, PortReport, now_iso
from cvmd.cvm.switching import RELEASED, TIMED_OUT, SwitchRecord, SwitchStore
from cvmd.dstack.loader import DStackUnavailable, load_dstack
from cvmd.dstack.plan import setup_namespace
from cvmd.state.machine import NodeState, is_legal
from cvmd.state.store import StateStore

logger = logging.getLogger(__name__)

KIND_VALIDATION = "validation"
KIND_RENTER = "renter"

RUNNING_STATE = {
    KIND_VALIDATION: NodeState.VALIDATION_RUNNING,
    KIND_RENTER: NodeState.RENTER_RUNNING,
}

READINESS_POLL_SECONDS = 5

# Probing localhost, not the mapping's bind address: 0.0.0.0 is not a destination, and a
# forward bound to a public address is still reachable from the host it runs on.
PROBE_HOST = "127.0.0.1"


class LaunchFailure(Exception):
    """A refusal, carrying the HTTP status that describes *why* it was refused.

    Statuses are chosen so a caller can act on them without parsing prose: 503 means this host
    is not equipped for the request, 409 means the host's current state forbids it, 422 means
    the request named something this host will not run, 504 means the CVM was started but did
    not come up in time.
    """

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _describe(running: list[tuple[int, str]]) -> str:
    """How a refusal names the guests `supervisor.running_cvms()` found."""
    return ", ".join(f"pid {pid} ({name})" for pid, name in running)


@dataclass(frozen=True)
class Triple:
    qemu: str
    os_image_hash: str
    compose_hash: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.qemu, self.os_image_hash, self.compose_hash)


class CvmManager:
    def __init__(
        self,
        *,
        config: LaunchConfig,
        catalog_store: catalog.CatalogStore,
        store: StateStore,
        instances: InstanceStore,
        switches: SwitchStore,
    ) -> None:
        self._config = config
        self._catalog = catalog_store
        self._store = store
        self._instances = instances
        self._switches = switches

    # ------------------------------------------------------------------ reporting

    def last_switch(self) -> dict | None:
        """What `/v1/state` reports about the node's last mode change, if it has had one."""
        record = self._switches.current
        return record.report() if record is not None else None

    def describe(self) -> dict | None:
        """What `/v1/state` reports about the CVM, if there is one."""
        instance = self._instances.current
        if instance is None:
            return None
        return {
            **instance.report(),
            "supervisor_pid": instance.supervisor_pid,
            "supervisor_alive": supervisor.is_supervisor(
                instance.supervisor_pid, Path(instance.vm_dir)
            ),
            "created_at": instance.created_at,
        }

    # ------------------------------------------------------------------ reconcile

    @property
    def manages_cvms(self) -> bool:
        """Is this host configured to hold CVMs at all?

        A cvmd installed before its catalog and run directory exist is a legitimate state, and
        on such a host there is nothing to reconcile *against*. Re-deriving a state from facts
        it cannot observe would be worse than leaving the persisted one alone.
        """
        return self._config.run_dir is not None

    def reconcile(self) -> NodeState:
        """Derive the node's state from what is actually on the host.

        Runs at startup, which is the case that matters: a CVM outlives a cvmd restart by
        design, so the daemon must be able to come back under a running guest and land on the
        right state without pretending it launched it. `RECONCILING` doubles as the idle state
        — the machine has no separate IDLE, and every edge out of `RECONCILING` says so.
        """
        if not self.manages_cvms:
            return self._store.state

        self._sweep_staging()

        if self._instances.load_error:
            return self._fail(
                f"{self._instances.load_error}; refusing to assume this node is idle while a "
                f"CVM it forgot may still be running"
            )

        instance = self._instances.current
        if instance is not None:
            vm_dir = Path(instance.vm_dir)
            if supervisor.is_supervisor(instance.supervisor_pid, vm_dir):
                logger.info("adopted the running %s CVM %s", instance.kind, instance.instance_id)
                state = RUNNING_STATE.get(instance.kind)
                if state is None:
                    return self._fail(
                        f"CVM {instance.instance_id} has unknown kind {instance.kind!r}"
                    )
                return self._settle(state)
            return self._fail(
                f"the supervisor for CVM {instance.instance_id} (pid {instance.supervisor_pid}) "
                f"is gone but its directory {vm_dir} remains"
            )

        stray = self._stray_directories()
        if stray:
            return self._fail(
                f"{', '.join(str(path) for path in stray)} holds a CVM disk that cvmd has no "
                f"record of, so this node's resources may already be committed"
            )

        # No CVM here. If the persisted state claimed one was running, the node lost it while
        # cvmd was not looking — that is a failure, not an idle node, and saying so is the
        # difference between a node an operator investigates and one that silently drops work.
        if self._store.state in (
            NodeState.LAUNCHING,
            NodeState.VALIDATION_RUNNING,
            NodeState.RENTER_RUNNING,
        ):
            return self._fail(f"this node was {self._store.state} but no CVM is running on it")

        logger.info("no CVM on this node")
        return self._settle(NodeState.RECONCILING)

    def _stray_directories(self) -> list[Path]:
        run_dir = self._config.run_dir
        return supervisor.vm_directories(run_dir) if run_dir else []

    def _sweep_staging(self) -> None:
        """Remove every renter staging directory left on this host.

        Safe to do unconditionally, and only ever runs at reconciliation. A launch discards its
        own staging directory the moment the measurement passes, and `run_instance` never reads
        it — so anything still here belongs to a launch that was killed partway, and it is a
        customer's compose sitting in a directory with no CVM to justify it.
        """
        renter_dir = self._config.renter_dir
        if renter_dir is None or not renter_dir.is_dir():
            return
        for child in sorted(renter_dir.iterdir()):
            if child.is_dir():
                logger.info("removing the renter staging directory %s left by an earlier run", child)
                renter.discard(child)

    def _settle(self, state: NodeState) -> NodeState:
        """Move to `state`, clearing any stale error, going the long way if the edge is illegal.

        The machine's edges describe an orderly lifecycle; reconciliation runs exactly when the
        node has not had one. `RECONCILING` reaches every state, so it is the universal
        staging point — and reaching it from a running state means something discontinuous
        happened, which is what `FAILED` is for. Coercing the edge instead would report a state
        the host was never observed in.
        """
        current = self._store.state
        if current == state:
            if self._store.document.last_error is not None:
                self._store.record_error(None)
            return state

        if not is_legal(current, state):
            if not is_legal(current, NodeState.RECONCILING):
                self._store.transition(
                    NodeState.FAILED,
                    last_error=f"reconciliation found {state} while the node was {current}",
                )
            # From the staging point the edge is legal, so ask again from there. Recursing
            # rather than transitioning straight away is what keeps RECONCILING working as a
            # destination as well as a waypoint: the equality branch above is then the one that
            # answers, instead of this line requesting the RECONCILING -> RECONCILING self-edge
            # the machine does not have.
            self._store.transition(NodeState.RECONCILING)
            return self._settle(state)
        return self._store.transition(state).state

    def _fail(self, reason: str) -> NodeState:
        logger.error("reconciliation failed: %s", reason)
        if self._store.state == NodeState.FAILED:
            self._store.record_error(reason)
            return NodeState.FAILED
        return self._store.transition(NodeState.FAILED, last_error=reason).state

    # ------------------------------------------------------------------ create

    def create(self, *, kind: str, triple: Triple) -> dict:
        """Launch a catalog-pinned CVM and return its report. Raises LaunchFailure on refusal."""
        dstack, mappings = self._preflight()
        artifact = self._resolve(kind, triple)
        return self._launch(
            kind=kind,
            triple=triple,
            artifact=artifact,
            dstack=dstack,
            mappings=mappings,
            instance_id=str(uuid.uuid4()),
        )

    def create_renter(self, order: renter.RenterOrder) -> dict:
        """Launch a CVM for one customer order (DAH-2580).

        The same path as `create`, with the compose staged from the order instead of read out
        of the manifest — see `cvm/renter.py` for why that does not weaken what authorizes the
        launch. The catalog still decides the image and the QEMU build, and the measurement gate
        still refuses anything that does not hash to the `compose_hash` the request named.

        The staging directory is removed as soon as the measurement has passed, on every path.
        `setup_instance` is the only thing that ever reads it — `run_instance` works from
        `app-compose.json` — so from that moment the customer's compose has no reason to remain
        readable in a plain directory on a host they do not control.
        """
        if self._config.renter_dir is None:
            raise LaunchFailure(503, "this host has no launch configuration for renter_dir")

        dstack, mappings = self._preflight()
        base = self._resolve_base(order)
        return self._launch(
            kind=KIND_RENTER,
            triple=Triple(
                qemu=order.qemu,
                os_image_hash=order.os_image_hash,
                compose_hash=order.compose_hash,
            ),
            dstack=dstack,
            mappings=mappings,
            instance_id=str(uuid.uuid4()),
            order=order,
            base=base,
            rental_id=order.rental_id,
        )

    def _preflight(self):
        """Everything that must hold before a launch of any kind is attempted.

        Ordered so that the cheap, host-wide refusals come first: a node that is not configured,
        cannot run the launcher, or is already holding a guest should say so before an order is
        written anywhere.
        """
        missing = self._config.missing()
        if missing:
            raise LaunchFailure(
                503, f"this host has no launch configuration for {', '.join(missing)}"
            )

        dstack = self._load_dstack()
        mappings = self._parse_ports()
        self._assert_ssh_port_is_forwarded(mappings)
        self._assert_node_is_free()
        try:
            ports.assert_free(mappings)
        except ports.PortError as exc:
            raise LaunchFailure(409, str(exc)) from exc
        return dstack, mappings

    def _launch(
        self,
        *,
        kind: str,
        triple: Triple,
        dstack,
        mappings: list[ports.PortMapping],
        instance_id: str,
        artifact: catalog.Artifact | None = None,
        order: renter.RenterOrder | None = None,
        base: catalog.Artifact | None = None,
        rental_id: str | None = None,
    ) -> dict:
        """prepare -> MEASURE -> launch -> confirm.

        A renter order is staged **here**, after `_enter_launching`, and that ordering is
        load-bearing rather than tidy. `_enter_launching` can run reconciliation — a node left
        FAILED by a previous refusal has to be re-derived before it may launch again — and
        reconciliation sweeps stale staging directories. Staging before it therefore writes a
        compose that the very next line deletes, and the launch fails with the file it just
        wrote reported missing. Found on au11, not in a unit test: the sweep only fires when the
        node reaches `_enter_launching` in a state that needs reconciling.
        """
        self._enter_launching()

        staged_dir: Path | None = None
        if order is not None:
            staged_dir = self._config.renter_dir / instance_id
            try:
                artifact = renter.stage(order, base=base, directory=staged_dir)
            except catalog.CatalogError as exc:
                self._store.transition(
                    NodeState.FAILED, last_error=f"staging this order failed: {exc}"
                )
                raise LaunchFailure(500, f"staging this order failed: {exc}") from exc

        vm_dir = self._config.run_dir / instance_id
        try:
            self._prepare(dstack, artifact=artifact, vm_dir=vm_dir)
            measured = self._measure(dstack, artifact=artifact, vm_dir=vm_dir, triple=triple)
        except LaunchFailure:
            # The directory is the whole footprint of a launch that never started a process,
            # so removing it returns the node to exactly where it was.
            self._discard(vm_dir)
            raise
        finally:
            renter.discard(staged_dir)

        instance = self._start(
            dstack_scripts_dir=self._config.dstack_scripts_dir,
            artifact=artifact,
            instance_id=instance_id,
            vm_dir=vm_dir,
            mappings=mappings,
            measured=measured,
            kind=kind,
            rental_id=rental_id,
        )
        return self._await_ready(instance, mappings)

    def _load_dstack(self):
        try:
            return load_dstack(self._config.dstack_scripts_dir)
        except DStackUnavailable as exc:
            raise LaunchFailure(503, f"this host cannot run the dstack launcher: {exc}") from exc

    def _resolve(self, kind: str, triple: Triple) -> catalog.Artifact:
        """Match the request against the signed catalog, re-verified on this very call.

        503 and 422 say different things and a caller acts on both: 503 means this host has no
        catalog it can trust right now — no manifest, an expired one, a signature that no longer
        checks out — and retrying later may work. 422 means the catalog is fine and does not
        contain what was asked for, which retrying never fixes.
        """
        try:
            artifacts = self._catalog.artifacts()
        except catalog.CatalogError as exc:
            raise LaunchFailure(503, str(exc)) from exc
        try:
            return catalog.resolve(
                artifacts,
                kind=kind,
                qemu=triple.qemu,
                os_image_hash=triple.os_image_hash,
                compose_hash=triple.compose_hash,
            )
        except catalog.TripleNotFound as exc:
            raise LaunchFailure(422, str(exc)) from exc

    def _resolve_base(self, order: renter.RenterOrder) -> catalog.Artifact:
        """The catalog entry approving this order's OS image and QEMU build.

        Same two statuses and the same distinction as `_resolve`: 503 is "this host has no
        catalog it can trust right now", 422 is "the catalog is fine and does not approve that".
        """
        try:
            artifacts = self._catalog.artifacts()
        except catalog.CatalogError as exc:
            raise LaunchFailure(503, str(exc)) from exc
        try:
            return catalog.resolve_base(
                artifacts, qemu=order.qemu, os_image_hash=order.os_image_hash
            )
        except catalog.TripleNotFound as exc:
            raise LaunchFailure(422, str(exc)) from exc

    def _parse_ports(self) -> list[ports.PortMapping]:
        try:
            return ports.parse_all(self._config.ports)
        except ports.PortError as exc:
            raise LaunchFailure(503, f"this host's cvm_ports setting is unusable: {exc}") from exc

    def _assert_ssh_port_is_forwarded(self, mappings: list[ports.PortMapping]) -> None:
        """A configured SSH guest port that nothing forwards is a refusal, not a degradation.

        Readiness and the report's fingerprint both come from reading the guest's host key
        through that forward. With no mapping to read it through there is no readiness check
        left at all, and answering 201 VALIDATION_RUNNING milliseconds after spawning QEMU
        would tell the caller the CVM is up before the guest has booted. Refusing here names
        the setting that is wrong, before anything is prepared or started.

        Asked through `_readiness_port`, so this is the same lookup the readiness loop will
        make rather than a second copy of the matching rule.
        """
        if self._config.ssh_guest_port is None or self._readiness_port(mappings) is not None:
            return
        forwarded = ", ".join(str(mapping) for mapping in mappings) or "(none)"
        raise LaunchFailure(
            503,
            f"this host's cvm_ssh_guest_port is {self._config.ssh_guest_port}, which none of "
            f"its cvm_ports forwards ({forwarded}); readiness and the launch report's SSH "
            f"fingerprint are both read through that forward, so there is nothing to probe",
        )

    def _assert_node_is_free(self) -> None:
        """One CVM per node, checked against the host rather than against cvmd's own records.

        The `/proc` scan comes first deliberately: it is the only check that also sees a CVM
        started by `lium-cvm.sh`, and during the CVM v2 rollout that is the one most likely to
        be there.
        """
        running = supervisor.running_cvms()
        if running:
            raise LaunchFailure(
                409,
                f"a confidential guest is already running on this host ({_describe(running)}); "
                f"GPU passthrough is exclusive, so this node can hold only one",
            )

        instance = self._instances.current
        if instance is not None:
            raise LaunchFailure(
                409,
                f"this node already has CVM {instance.instance_id} ({instance.kind}); destroy "
                f"it before launching another",
            )

        stray = self._stray_directories()
        if stray:
            raise LaunchFailure(
                409,
                f"{', '.join(str(path) for path in stray)} still holds a CVM disk; destroy it "
                f"before launching another",
            )

    def _enter_launching(self) -> None:
        # Neither FAILED nor SWITCHING reaches LAUNCHING directly, and for the same reason:
        # recovery is deliberate, so the daemon re-derives the host's real state instead of
        # assuming the condition cleared. SWITCHING only survives a crash mid-teardown — every
        # other path leaves it before returning — but a launch into a node whose last CVM's
        # memory is still draining is precisely what FR-C6 forbids.
        if self._store.state in (NodeState.FAILED, NodeState.SWITCHING):
            blocked = self._store.state
            self.reconcile()
            if self._store.state in (NodeState.FAILED, NodeState.SWITCHING):
                raise LaunchFailure(
                    409,
                    f"this node is {blocked} and reconciliation did not clear it: "
                    f"{self._store.document.last_error}",
                )
        self._store.transition(NodeState.LAUNCHING)

    def _prepare(self, dstack, *, artifact: catalog.Artifact, vm_dir: Path) -> None:
        """Write the VM directory using dstack's own `setup_instance`. No CVM exists yet."""
        namespace = setup_namespace(artifact=artifact, launch=self._config, vm_dir=vm_dir)
        try:
            self._config.run_dir.mkdir(parents=True, exist_ok=True)
            dstack.DStackManager().setup_instance(namespace)
        except Exception as exc:  # noqa: BLE001 - setup_instance raises whatever it hits
            self._store.transition(NodeState.FAILED, last_error=f"preparing {vm_dir} failed: {exc}")
            raise LaunchFailure(500, f"preparing the CVM failed: {exc}") from exc

    def _measure(
        self, dstack, *, artifact: catalog.Artifact, vm_dir: Path, triple: Triple
    ) -> measure.Measurements:
        try:
            measured = measure.measure(
                dstack=dstack, vm_dir=vm_dir, image_path=artifact.os_image_path
            )
            measure.assert_matches(measured, triple.as_tuple())
        except measure.MeasurementError as exc:
            self._store.transition(NodeState.FAILED, last_error=str(exc))
            raise LaunchFailure(409, str(exc)) from exc
        logger.info("%s measures as the requested triple", vm_dir)
        return measured

    def _discard(self, vm_dir: Path) -> None:
        shutil.rmtree(vm_dir, ignore_errors=True)

    def _start(
        self,
        *,
        dstack_scripts_dir: Path,
        artifact: catalog.Artifact,
        instance_id: str,
        vm_dir: Path,
        mappings: list[ports.PortMapping],
        measured: measure.Measurements,
        kind: str,
        rental_id: str | None = None,
    ) -> Instance:
        try:
            pid = supervisor.spawn(
                scripts_dir=dstack_scripts_dir,
                vm_dir=vm_dir,
                kp_port=self._config.key_provider_port,
            )
        except supervisor.SupervisorError as exc:
            self._store.transition(NodeState.FAILED, last_error=str(exc))
            self._discard(vm_dir)
            raise LaunchFailure(500, str(exc)) from exc

        # Recorded immediately: from here on a process exists, and a process cvmd has no record
        # of is one it can neither report nor stop. The VM directory covers the gap between
        # spawn and this write — `reconcile` finds it and fails the node loudly.
        return self._instances.set(
            Instance(
                instance_id=instance_id,
                kind=kind,
                artifact_id=artifact.id,
                vm_dir=str(vm_dir),
                supervisor_pid=pid,
                created_at=now_iso(),
                qemu=measured.qemu,
                os_image_hash=measured.os_image_hash,
                compose_hash=measured.compose_hash,
                ports=[
                    PortReport(
                        protocol=m.protocol,
                        address=m.address,
                        host_port=m.host_port,
                        guest_port=m.guest_port,
                    )
                    for m in mappings
                ],
                rental_id=rental_id,
            )
        )

    # ------------------------------------------------------------------ readiness

    def _readiness_port(self, mappings: list[ports.PortMapping]) -> ports.PortMapping | None:
        guest_port = self._config.ssh_guest_port
        if guest_port is None:
            return mappings[0] if mappings else None
        for mapping in mappings:
            if mapping.guest_port == guest_port:
                return mapping
        return None

    def _await_ready(self, instance: Instance, mappings: list[ports.PortMapping]) -> dict:
        """Hold until the guest answers, then move to the running state.

        The SSH host key is both the report's fingerprint field and the readiness signal: it
        can only be read once sshd inside the guest is answering through the forward, which is
        a much later — and much more meaningful — moment than QEMU accepting a connection.
        """
        vm_dir = Path(instance.vm_dir)
        probe = self._readiness_port(mappings)
        deadline = time.monotonic() + self._config.launch_timeout_seconds
        want_fingerprint = self._config.ssh_guest_port is not None and probe is not None
        weak_note = "port accepts connections; no SSH probe configured"
        if want_fingerprint and not sshkey.keyscan_available():
            # Asked once, here, rather than discovered inside the loop. `read_host_key` returns
            # None when ssh-keyscan is absent, which is indistinguishable from "the guest is not
            # up yet" — so without this the loop would poll for the whole launch timeout, then
            # fail a node whose CVM had booted fine.
            logger.warning(
                "ssh-keyscan is not installed, so no SSH host-key fingerprint can be reported "
                "for this launch; readiness falls back to a TCP accept on port %d",
                probe.host_port,
            )
            want_fingerprint = False
            weak_note = "port accepts connections; ssh-keyscan is not installed on this host"

        while time.monotonic() < deadline:
            if not supervisor.is_supervisor(instance.supervisor_pid, vm_dir):
                reason = f"the CVM exited while booting; see {supervisor.console_log_path(vm_dir)}"
                self._store.transition(NodeState.FAILED, last_error=reason)
                raise LaunchFailure(500, reason)

            if probe is None:
                # Nothing is forwarded, so there is nothing to probe. The supervisor being
                # alive is the only readiness evidence available, and saying so is better than
                # inventing a check.
                return self._running(instance, note="no forwarded port to probe")

            if want_fingerprint:
                fingerprint = sshkey.read_host_key(PROBE_HOST, probe.host_port)
                if fingerprint:
                    instance = self._instances.update(ssh_fingerprint=fingerprint)
                    return self._running(instance)
            elif sshkey.accepts_connection(PROBE_HOST, probe.host_port):
                return self._running(instance, note=weak_note)

            time.sleep(READINESS_POLL_SECONDS)

        reason = (
            f"the CVM did not become reachable on port {probe.host_port} within "
            f"{self._config.launch_timeout_seconds}s"
        )
        # Deliberately left running. A slow guest is the common cause, and killing it here
        # would destroy the console log's only live counterpart while it may still come up.
        self._store.transition(NodeState.FAILED, last_error=reason)
        raise LaunchFailure(504, reason)

    def _running(self, instance: Instance, *, note: str | None = None) -> dict:
        state = RUNNING_STATE.get(instance.kind)
        if state is None:
            raise LaunchFailure(500, f"no running state is defined for kind {instance.kind!r}")
        self._store.transition(state)
        logger.info("CVM %s is %s", instance.instance_id, state)
        report = {**instance.report(), "state": str(state)}
        if note:
            report["note"] = note
        return report

    # ------------------------------------------------------------------ destroy

    def destroy(self) -> dict:
        """Stop the CVM and hold until this node's hardware is verifiably free.

        Returns only when all four conditions in `cvm/release.py` hold together, which is what
        makes a 200 here mean the node can take the next CVM. The fleet's standing bug is the
        opposite of that: `lium-cvm.sh stop` reports success when its wait loop gives up, and
        QEMU is still holding the guest's memory afterwards. A stopped process is the start of
        a teardown, not the end of one.

        Idempotent on purpose: a platform whose teardown timed out has to be able to repeat the
        call, and answering "there was nothing here" is more useful to it than an error it has
        to special-case. A repeat continues the switch already in flight rather than starting a
        new one — see `_begin_switch`.

        The VM directory is removed once the node is verified free, disk included. A CVM's disk
        is not durable state — the validation CVM must measure identically on every launch, and
        a renter's is gone with the rental — and leaving it behind would block the next launch.
        A teardown that did NOT verify keeps the directory, which is what stops the node
        accepting a launch it cannot honour.
        """
        instance = self._instances.current
        if instance is None:
            stray = self._stray_directories()
            if not stray:
                return {"torn_down": False, "reason": "this node is running no CVM"}

            # A directory cvmd has no record of may still have a guest under it — that is what
            # "no record" usually means. Removing the disk from a live QEMU would report a
            # teardown that did not happen and leave the node's GPUs and forwarded ports held
            # by a process nothing can find, since the record that named its pid is what went
            # missing. The `/proc` scan is the only evidence available here, so it decides.
            running = supervisor.running_cvms()
            if running:
                reason = (
                    f"{', '.join(str(path) for path in stray)} holds a CVM cvmd has no record "
                    f"of and a confidential guest is still running ({_describe(running)}); its "
                    f"disk was left in place and it must be stopped by hand before this node "
                    f"is free"
                )
                self._fail(reason)
                raise LaunchFailure(409, reason)

            for path in stray:
                self._discard(path)
            self._settle(NodeState.RECONCILING)
            return {
                "torn_down": True,
                "detail": f"removed {len(stray)} CVM director(y/ies) cvmd had no record of",
            }

        vm_dir = Path(instance.vm_dir)
        # Written before the first signal. cvmd is restartable by design, and the baseline the
        # memory condition compares against is only meaningful if it was read while the guest
        # was still running — a daemon that came back afterwards could never take it again.
        record = self._begin_switch(instance)

        if self._store.state == NodeState.FAILED:
            self._store.transition(NodeState.RECONCILING)
        if self._store.state != NodeState.TEARDOWN:
            self._store.transition(NodeState.TEARDOWN)

        detail = self._stop(instance, vm_dir)

        self._store.transition(NodeState.SWITCHING)
        report = release.verify_released(
            self._release_checks(record),
            timeout=self._config.teardown_verify_timeout_seconds,
        )

        if not report.complete:
            reason = (
                f"CVM {instance.instance_id} was stopped ({detail}) but this node's hardware is "
                f"still held {report.duration_seconds:.0f}s later — {report.why_incomplete()}"
            )
            # The directory stays. It is the fail-closed record that this node still owes
            # something to a CVM that is gone, and `_assert_node_is_free` reads it, so a launch
            # into half-released hardware is refused rather than merely discouraged.
            self._switches.finish(outcome=TIMED_OUT, detail=detail, release=report.to_json())
            self._store.transition(NodeState.FAILED, last_error=reason)
            raise LaunchFailure(504, reason)

        self._discard(vm_dir)
        self._instances.clear()
        finished = self._switches.finish(outcome=RELEASED, detail=detail, release=report.to_json())
        self._store.transition(NodeState.RECONCILING)
        logger.info(
            "CVM %s released this node in %.0fs (%s)",
            instance.instance_id,
            report.duration_seconds,
            detail,
        )
        return {
            "torn_down": True,
            "instance_id": instance.instance_id,
            "detail": detail,
            "switch": finished.report(),
        }

    def _stop(self, instance: Instance, vm_dir: Path) -> str:
        """Ask the CVM to stop, escalating as far as it takes. Reports; never raises."""
        if not supervisor.is_supervisor(instance.supervisor_pid, vm_dir):
            return "the supervisor was already gone"
        try:
            dstack = self._load_dstack()
        except LaunchFailure as exc:
            # Without dstack there is no graceful path, but the process group is still ours to
            # signal — refusing to stop would be worse than stopping abruptly.
            logger.warning("%s; falling back to signalling the process group", exc.reason)
            return supervisor.shutdown_by_signal(
                instance.supervisor_pid, timeout=self._config.teardown_timeout_seconds
            )
        return supervisor.shutdown(
            dstack,
            vm_dir,
            instance.supervisor_pid,
            timeout=self._config.teardown_timeout_seconds,
        )

    def _begin_switch(self, instance: Instance) -> SwitchRecord:
        """Start — or resume — the record of this node's crossing between modes.

        A repeat teardown continues the switch it finds in flight instead of opening a new one.
        Two reasons, and both are about honesty rather than tidiness: the memory baseline has to
        be a reading from while the guest was running, and the window FR-I3 budgets for began at
        the platform's first call, not at whichever retry happened to succeed.
        """
        unfinished = self._switches.unfinished
        if unfinished is not None and unfinished.instance_id == instance.instance_id:
            return unfinished
        return self._switches.begin(
            SwitchRecord(
                instance_id=instance.instance_id,
                kind=instance.kind,
                vm_dir=instance.vm_dir,
                supervisor_pid=instance.supervisor_pid,
                started_at=now_iso(),
                guest_memory_kib=self._guest_memory_kib(),
                baseline_mem_available_kib=self._mem_available(),
                ports=list(instance.ports),
            )
        )

    def _guest_memory_kib(self) -> int | None:
        """How much memory this host gives a guest, in KiB, or None if it cannot be read.

        Degrades rather than refuses. A teardown is the wrong moment to discover a malformed
        size — the launch that used it already succeeded — and the memory condition says plainly
        that it has nothing to compare against.
        """
        if self._config.memory is None:
            return None
        try:
            return release.memory_kib(self._config.memory)
        except release.MemoryUnreadable as exc:
            logger.warning("this teardown cannot check the guest's memory: %s", exc)
            return None

    def _mem_available(self) -> int | None:
        try:
            return release.mem_available_kib()
        except release.MemoryUnreadable as exc:
            logger.warning("this teardown has no memory baseline: %s", exc)
            return None

    def _release_checks(self, record: SwitchRecord) -> release.ReleaseChecks:
        """The four conditions, bound to the CVM being torn down.

        Ports come from the instance record rather than from the config: those are the mappings
        QEMU was actually started with, and a config edited since the launch would have this
        check probing addresses no guest ever held.
        """
        return release.ReleaseChecks(
            supervisor_pid=record.supervisor_pid,
            mappings=[
                ports.PortMapping(
                    protocol=port.protocol,
                    address=port.address,
                    host_port=port.host_port,
                    guest_port=port.guest_port,
                )
                for port in record.ports
            ],
            guest_memory_kib=record.guest_memory_kib,
            baseline_mem_available_kib=record.baseline_mem_available_kib,
            memory_tolerance=self._config.teardown_memory_tolerance,
            hugepages=self._config.hugepages,
        )
