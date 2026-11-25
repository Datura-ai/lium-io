"""DAH-7125: the monitor restarts the executor only when its API does not answer over HTTP, never because Docker
could not start an exec (the healthcheck + autoheal path restarted live nodes under host disk stalls)."""

import http.server
import threading
from datetime import datetime, timezone

from services import api_watcher
from services.api_watcher import ALIVE, DOWN, ExecutorApiWatcher, probe_http

STARTED = "2026-11-25T00:00:00.123456789Z"
STARTED_EPOCH = datetime(2026, 11, 25, 0, 0, 0, 123456, tzinfo=timezone.utc).timestamp()


class FakeExecutor:
    def __init__(self, ip="executor-first", status="running", started_at=STARTED, container_id="c1"):
        self.id = container_id
        self.name = "executor-executor-1"
        self.attrs = {
            "State": {"Status": status, "StartedAt": started_at},
            "NetworkSettings": {"Networks": {"executor_default": {"IPAddress": ip}}},
        }
        self.restarts = 0

    def restart(self, timeout):
        self.restarts += 1


class FakeContainers:
    def __init__(self, executor):
        self.executor = executor
        self.docker_up = True
        self.filters = []

    def get(self, name):
        if not self.docker_up:
            raise TimeoutError("docker does not answer")
        if name == "monitor":
            return type("Monitor", (), {"labels": {"com.docker.compose.project": "executor"}})()
        return self.executor

    def list(self, all, filters):
        if not self.docker_up:
            raise TimeoutError("docker does not answer")
        self.filters.append(filters)
        return [self.executor]


class Harness:
    def __init__(self, **executor):
        self.executor = FakeExecutor(**executor)
        self.containers = FakeContainers(self.executor)
        self.answers = []  # ALIVE/DOWN per probe, the last one repeats
        self.now = STARTED_EPOCH + 3600
        self.mono = 1000.0
        self.reports = []
        self.watcher = ExecutorApiWatcher(
            type("Client", (), {"containers": self.containers})(),
            8001,
            probe=self.probe,
            wall_clock=lambda: self.now,
            monotonic=lambda: self.mono,
            spawn=lambda run: run(),
            report=lambda event, reason: self.reports.append(event),
            own_container="monitor",
        )

    def probe(self, url):
        assert url == f"http://{self.executor.attrs['NetworkSettings']['Networks']['executor_default']['IPAddress']}:8001/version"
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

    def ticks(self, n):
        out = []
        for _ in range(n):
            out.append(self.watcher.tick())
            self.now += 30
            self.mono += 30
        return out


def test_a_live_api_is_never_restarted_while_docker_is_stalled():
    h = Harness()
    h.answers = [ALIVE]
    h.ticks(1)
    h.containers.docker_up = False  # exec and inspect hang: the old healthcheck went unhealthy here

    h.ticks(20)

    assert h.executor.restarts == 0


def test_a_hung_api_is_restarted_after_five_failed_probes_about_two_and_a_half_minutes():
    h = Harness()
    h.answers = [DOWN]

    results = h.ticks(5)

    assert results == [None, None, None, None, "restart"]
    assert h.executor.restarts == 1
    assert h.reports == ["executor_api_restart"]
    assert h.containers.filters[0] == {
        "label": ["com.docker.compose.service=executor", "com.docker.compose.project=executor"]
    }


def test_failures_while_docker_does_not_answer_are_unknown_not_unhealthy():
    h = Harness()
    h.answers = [DOWN]
    h.ticks(1)
    h.containers.docker_up = False

    h.ticks(10)

    assert h.executor.restarts == 0
    assert h.watcher.failures == 1


def test_an_api_that_recovers_before_the_restart_runs_is_left_alone():
    h = Harness()
    h.answers = [DOWN, DOWN, DOWN, DOWN, DOWN, ALIVE]  # the last look right before acting answers

    h.ticks(5)

    assert h.executor.restarts == 0
    assert h.watcher.failures == 0


def test_a_container_inside_its_start_period_is_not_counted():
    h = Harness()
    h.now = STARTED_EPOCH + 10
    h.answers = [DOWN]

    h.ticks(3)  # still under 120 s after its start

    assert h.watcher.failures == 0
    h.ticks(5)  # the first of these is still at 100 s
    assert h.executor.restarts == 0
    h.ticks(1)
    assert h.executor.restarts == 1


def test_restarts_back_off_and_reset_once_the_api_answers():
    h = Harness()
    h.answers = [DOWN]

    h.ticks(5)
    assert h.executor.restarts == 1
    h.ticks(9)  # 4.5 min: the 5 min backoff holds the second restart
    assert h.executor.restarts == 1
    h.ticks(2)
    assert h.executor.restarts == 2
    assert h.watcher.backoff == 1200  # 5, 10, then 20 minutes

    h.answers = [ALIVE]
    h.ticks(1)
    assert h.watcher.backoff == api_watcher.BACKOFF_FIRST_S


def test_one_restart_at_a_time():
    h = Harness()
    h.answers = [DOWN]
    pending = []
    h.watcher.spawn = pending.append  # the restart has not finished

    h.ticks(5)
    h.mono += 10_000
    h.ticks(5)

    assert len(pending) == 1


def test_a_moved_address_is_followed_without_counting_a_failure():
    h = Harness()
    h.answers = [ALIVE]
    h.ticks(1)
    h.executor.attrs["NetworkSettings"]["Networks"]["executor_default"]["IPAddress"] = "executor-moved"
    h.answers = [ALIVE]
    h.watcher.probe = lambda url: ALIVE if "executor-moved" in url else DOWN

    h.ticks(1)

    assert h.watcher.target.ip == "executor-moved"
    assert h.watcher.failures == 0


def test_a_monitor_outside_a_compose_stack_watches_nothing():
    h = Harness()
    h.answers = [DOWN]
    h.containers.get = lambda name: type("Monitor", (), {"labels": {}})()

    h.ticks(10)

    assert h.executor.restarts == 0


def test_probe_http_counts_any_answer_as_alive():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500 if self.path == "/broken" else 200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        assert probe_http(f"http://127.0.0.1:{port}/version") == ALIVE
        assert probe_http(f"http://127.0.0.1:{port}/broken") == ALIVE
    finally:
        server.shutdown()
        server.server_close()
    assert probe_http(f"http://127.0.0.1:{port}/version", timeout=1) == DOWN
