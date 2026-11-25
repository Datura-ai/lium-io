"""Restart the executor container only when its own API does not answer (DAH-7125, DAH-5182 phase 2).

This replaces the Docker healthcheck plus willfarrell/autoheal. A Docker healthcheck runs through `docker exec`. When
a renter's writes saturate the host disk, exec cannot start ("timed out starting health check") while the node API
still answers in milliseconds. The container then went unhealthy and autoheal restarted it once Docker recovered,
killing sshd and the validator's SSH session.

The watcher runs in the monitor container. The probe is a plain HTTP request to the executor's address, with no exec,
no inspect and no write. Docker is asked only after a probe fails, to confirm which container and address to probe.
A Docker that does not answer is "unknown" and never counts toward a restart. A restart needs FAILURES_BEFORE_RESTART
counted failures past the container's start period, as the healthcheck did (~2.5 min). Only one restart runs at a time,
the next one waits out an exponential backoff, and the node is probed once more right before acting, so an API that
recovered meanwhile is never restarted.
"""

import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

EXECUTOR_SERVICE_LABEL = "com.docker.compose.service=executor"
PROJECT_LABEL = "com.docker.compose.project"

PROBE_INTERVAL_S = 30
PROBE_TIMEOUT_S = 10
FAILURES_BEFORE_RESTART = 5
START_PERIOD_S = 120
BACKOFF_FIRST_S = 300
BACKOFF_MAX_S = 3600
RESTART_STOP_TIMEOUT_S = 30

ALIVE = "alive"
DOWN = "down"


def probe_http(url: str, timeout: float = PROBE_TIMEOUT_S) -> str:
    """ALIVE when the API answers at all (an error status is still an answer), DOWN when it does not."""
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return ALIVE
    except urllib.error.HTTPError:
        return ALIVE
    except (urllib.error.URLError, OSError):
        return DOWN


@dataclass(frozen=True)
class Target:
    container_id: str
    name: str
    ip: str
    running: bool
    started_at: float  # epoch seconds of State.StartedAt


def _epoch(started_at: str) -> float:
    # Docker writes nanoseconds and a Z; fromisoformat takes at most microseconds
    head, _, fraction = started_at.rstrip("Z").partition(".")
    return datetime.fromisoformat(f"{head}.{(fraction or '0')[:6]}+00:00").timestamp()


def target_of(container, project: str | None) -> Target | None:
    attrs = container.attrs
    networks = attrs.get("NetworkSettings", {}).get("Networks") or {}
    # the stack's own network first: the monitor is on it too
    ordered = [networks[name] for name in networks if name == f"{project}_default"] + list(networks.values())
    ip = next((net.get("IPAddress") for net in ordered if net.get("IPAddress")), "")
    state = attrs.get("State", {})
    if not ip:
        return None
    return Target(
        container_id=container.id,
        name=container.name,
        ip=ip,
        running=state.get("Status") == "running",
        started_at=_epoch(state.get("StartedAt") or "1970-01-01T00:00:00Z"),
    )


class ExecutorApiWatcher:
    def __init__(
        self,
        client,
        port: int,
        probe: Callable[[str], str] = probe_http,
        wall_clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        spawn: Callable[[Callable[[], None]], None] = lambda run: threading.Thread(target=run, daemon=True).start(),
        report: Callable[[str, str], None] = lambda event, reason: None,
        own_container: str | None = None,
    ):
        self.client = client
        self.port = port
        self.probe = probe
        self.wall_clock = wall_clock
        self.monotonic = monotonic
        self.spawn = spawn
        self.report = report
        self.own_container = own_container or socket.gethostname()
        self.project: str | None = None
        self.target: Target | None = None
        self.failures = 0
        self.backoff = BACKOFF_FIRST_S
        self.next_restart_at = 0.0
        self.restarting = False

    def _url(self, target: Target) -> str:
        return f"http://{target.ip}:{self.port}/version"

    def _resolve(self) -> Target | None:
        """The executor of this monitor's own stack, from Docker; None when Docker does not answer or has none."""
        try:
            if self.project is None:
                self.project = self.client.containers.get(self.own_container).labels.get(PROJECT_LABEL)
            if not self.project:
                return None  # without its own stack it could pick a neighbour's executor
            labels = [EXECUTOR_SERVICE_LABEL, f"{PROJECT_LABEL}={self.project}"]
            for container in self.client.containers.list(all=True, filters={"label": labels}):
                target = target_of(container, self.project)
                if target is not None:
                    return target
        except Exception:
            return None
        return None

    def _healthy(self) -> None:
        self.failures = 0
        self.backoff = BACKOFF_FIRST_S
        self.next_restart_at = 0.0

    def tick(self) -> str | None:
        """One probe. Returns "restart" when it started a restart, else None."""
        if self.target is None:
            self.target = self._resolve()
            if self.target is None:
                return None  # unknown: nothing to probe yet
        if self.probe(self._url(self.target)) == ALIVE:
            self._healthy()
            return None
        # the address can be stale (a restart or recreate moves it): confirm it before counting a failure
        current = self._resolve()
        if current is None:
            return None  # Docker does not answer: unknown, not a failure of the API
        if (current.container_id, current.ip) != (self.target.container_id, self.target.ip):
            self.target = current
            self.failures = 0
            return None
        self.target = current
        if not current.running or self.wall_clock() - current.started_at < START_PERIOD_S:
            # a stopped container is the restart policy's (and the stale-task recovery's); a young one is starting
            self.failures = 0
            return None
        self.failures += 1
        if self.failures < FAILURES_BEFORE_RESTART or self.restarting or self.monotonic() < self.next_restart_at:
            return None
        # a last look right before acting: an API that recovered meanwhile is a healthy node
        if self.probe(self._url(current)) == ALIVE:
            self._healthy()
            return None
        self.restarting = True
        self.next_restart_at = self.monotonic() + self.backoff
        self.backoff = min(self.backoff * 2, BACKOFF_MAX_S)
        self.spawn(lambda: self._restart(current))
        return "restart"

    def _restart(self, target: Target) -> None:
        try:
            self.client.containers.get(target.container_id).restart(timeout=RESTART_STOP_TIMEOUT_S)
            self.report("executor_api_restart", f"{target.name}: API did not answer for {self.failures} probes")
        except Exception as ex:
            self.report("executor_api_restart_failed", f"{target.name}: {ex}")
        finally:
            self.restarting = False
            self.failures = 0
            self.target = None  # the restart can move the address


def watch_executor_api_forever(client, port: int, report: Callable[[str, str], None]) -> None:
    watcher = ExecutorApiWatcher(client, port, report=report)
    while True:
        try:
            watcher.tick()
        except Exception as ex:
            report("executor_api_watch_error", str(ex))
        time.sleep(PROBE_INTERVAL_S)
