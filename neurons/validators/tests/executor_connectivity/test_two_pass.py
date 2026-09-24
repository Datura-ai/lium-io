"""The two-pass port check against main's, on a simulated host driven through the real verifiers.

Main's orchestrator is reproduced below (`_main_verify`) from origin/main at the time of this
change, and both run against the same fake host: the containers it starts, the ports each
container's listeners are probed on, and what answers.
"""

import asyncio
import bisect
import random
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT
from services.executor_connectivity import port_verifiers
from services.executor_connectivity.models import (
    SECOND_PASS_BATCH_FAILED,
    SECOND_PASS_DISCARDED_CONTAINER_FAILED,
    SECOND_PASS_NO_PORTS_LEFT,
    SECOND_PASS_NOT_NEEDED,
    SECOND_PASS_RAN,
    SECOND_PASS_SKIPPED_BATCH_FAILED,
    SECOND_PASS_SKIPPED_CONTAINER_FAILED,
    ContainerStartResult,
    DindProbeResult,
    PortPair,
    PortRangeResult,
)
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector
from services.executor_connectivity.port_verifiers import (
    BatchVerifier,
    FallbackVerifier,
    SemiBatchVerifier,
)
from services.port_utils import get_all_ports
from services.task.checks.port_connectivity import PortConnectivityCheck


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def sleep(_seconds):
        return None

    monkeypatch.setattr(
        port_verifiers,
        "asyncio",
        SimpleNamespace(
            sleep=sleep,
            wait_for=asyncio.wait_for,
            gather=asyncio.gather,
            TimeoutError=asyncio.TimeoutError,
        ),
    )


@dataclass
class Host:
    """A simulated executor host.

    `open_ports`: external ports whose forward reaches the host. `batch_start_fails`: every
    --network=host container fails to start from the `batch_fail_after`-th on. `batch_blocked`: a
    host firewall drops --network=host listeners while published (-p) ports still answer.
    """

    open_ports: set[int]
    batch_start_fails: bool = False
    batch_fail_after: int = 0
    batch_blocked: bool = False
    dind_ok: bool = True
    containers: list[tuple[str, int]] = field(default_factory=list)
    probes: list[tuple[str, int | None, tuple[int, ...]]] = field(default_factory=list)
    _scripts: list[tuple[str, str]] = field(default_factory=list)
    _host_starts: int = 0

    async def run(self, ssh_client, name, script, network_mode, timeout):
        self.containers.append(
            (network_mode.split(" ")[0] if network_mode != "host" else "host", timeout)
        )
        if network_mode == "host":
            self._host_starts += 1
            if self.batch_start_fails and self._host_starts > self.batch_fail_after:
                return ContainerStartResult(ok=False, container_id=None, status="start failed")
        self._scripts.append((script, network_mode))
        return ContainerStartResult(ok=True, container_id="c", status="Up")

    async def cleanup(self, ssh_client, name):
        return None

    def _answers(self, port: PortPair, token: str) -> bool:
        mode = next(mode for script, mode in reversed(self._scripts) if token in script)
        if mode == "host" and self.batch_blocked:
            return False
        return port.external in self.open_ports

    async def test_many(self, session, host, ports, token, log_ctx=None):
        mode = next(mode for script, mode in reversed(self._scripts) if token in script)
        self.probes.append(
            (
                "host" if mode == "host" else "publish",
                (log_ctx or {}).get("port_pass"),
                tuple(p.external for p in ports),
            )
        )
        answered = [p for p in ports if self._answers(p, token)]
        return answered, [p for p in ports if p not in answered]

    async def test_one(self, session, host, port, token):
        self.probes.append(("single", None, (port.external,)))
        return self._answers(port, token)

    async def verify(
        self, port, *, ssh_client, host, container_name_prefix, sysbox_runtime, log_ctx=None
    ):
        self.containers.append(("dind", 0))
        ok = self.dind_ok and port.external in self.open_ports
        return DindProbeResult(success=ok, sysbox_runtime=ok, port=port)


def _info(*, port_range=None, port_mappings=None, ssh_port=22):
    return ExecutorSSHInfo(
        uuid="executor-1",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=ssh_port,
        port_mappings=port_mappings,
        port_range=port_range,
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )


def _probe(host: Host) -> PortProbe:
    return PortProbe(
        BatchVerifier(host, host), SemiBatchVerifier(host, host), FallbackVerifier(host, host)
    )


async def _main_verify(host: Host, info: ExecutorSSHInfo, unavailable: set[int], seed: int):
    """origin/main's PortSelector.select + PortProbe.probe + ConnectivityOrchestrator.verify."""
    random.seed(seed)
    all_ports = get_all_ports(info.port_range, info.port_mappings, info.ssh_port)
    ports = [PortPair(i, e) for i, e in all_ports if e not in unavailable][
        :BATCH_PORT_VERIFICATION_SIZE
    ]
    if not ports:
        return SimpleNamespace(
            status="no_ports", selected=(), successful=(), failed=(), dind=None, pass_one=0
        )

    kw = {"ssh_client": None, "host": info.address, "log_ctx": {}}
    successful, failed = await BatchVerifier(host, host).verify(ports, **kw)
    if not successful:
        successful, failed = await SemiBatchVerifier(host, host).verify(ports, max_ports=50, **kw)
    if not successful:
        successful, failed = await FallbackVerifier(host, host).verify(ports, max_ports=10, **kw)
    successful, failed = list(successful), list(failed)
    pass_one = len(successful)

    dind_port = successful.pop(0) if successful else random.choice(ports)
    dind = await host.verify(
        dind_port,
        ssh_client=None,
        host=info.address,
        container_name_prefix="c",
        sysbox_runtime=False,
    )
    (successful if dind.success else failed).append(dind_port)
    return SimpleNamespace(
        status="ok" if successful else "no_working_ports",
        selected=tuple(ports),
        successful=tuple(successful),
        failed=tuple(failed),
        dind=dind_port,
        pass_one=pass_one,
    )


async def _new_verify(host: Host, info: ExecutorSSHInfo, unavailable: set[int], seed: int):
    random.seed(seed)
    return await ConnectivityOrchestrator(PortSelector(), _probe(host), host).verify(
        executor_info=info,
        miner_hotkey="miner",
        sysbox_runtime=False,
        unavailable_ports=sorted(unavailable),
        ssh_client=None,
    )


def _host_copy(host: Host) -> Host:
    return Host(
        open_ports=set(host.open_ports),
        batch_start_fails=host.batch_start_fails,
        batch_fail_after=host.batch_fail_after,
        batch_blocked=host.batch_blocked,
        dind_ok=host.dind_ok,
    )


async def _both(
    host: Host, info: ExecutorSSHInfo, unavailable: set[int] = frozenset(), seed: int = 0
):
    main_host, new_host = _host_copy(host), _host_copy(host)
    main = await _main_verify(main_host, info, set(unavailable), seed)
    new = await _new_verify(new_host, info, set(unavailable), seed)
    return main, main_host, new, new_host


def _declared_externals(info, unavailable):
    return [
        e
        for _, e in get_all_ports(info.port_range, info.port_mappings, info.ssh_port)
        if e not in unavailable
    ]


def _shapes(seed: int, count: int):
    rng = random.Random(seed)
    shapes = []
    for n in range(count):
        kind = rng.choice(["wide", "wide", "default", "range", "list", "mappings"])
        if kind == "wide":
            kwargs = {"port_range": rng.choice(["40000-65535", "30000-65535", "50000-59999"])}
        elif kind == "default":
            kwargs = {}
        elif kind == "range":
            lo = rng.randint(1024, 64000)
            kwargs = {
                "port_range": f"{lo}-{min(65535, lo + rng.choice([0, 3, 150, 299, 300, 301, 450, 1500]))}"
            }
        elif kind == "list":
            kwargs = {
                "port_range": ",".join(
                    map(str, rng.sample(range(1024, 65535), rng.randint(1, 700)))
                )
            }
        else:
            internals = rng.sample(range(1024, 65535), rng.randint(1, 700))
            kwargs = {"port_mappings": str([[i, rng.randint(1024, 65535)] for i in internals])}
        info = _info(**kwargs)
        declared = _declared_externals(info, set())
        rented = set(rng.sample(declared, min(len(declared), rng.choice([0, 0, 2, 40, 250]))))
        free = [e for e in declared if e not in rented]

        how = rng.choice(
            ["all", "none", "top", "top", "sparse", "dense", "few-low-plus-top", "middle"]
        )
        if how == "all":
            open_ports = set(free)
        elif how == "none":
            open_ports = set()
        elif how == "top":
            open_ports = set(free[-rng.choice([1, 2, 50, 169, 170, 400, 2000, 6000]) :])
        elif how == "sparse":
            open_ports = {e for e in free if rng.random() < 0.002}
        elif how == "dense":
            open_ports = {e for e in free if rng.random() < 0.3}
        elif how == "few-low-plus-top":
            open_ports = set(
                rng.sample(free[:300], min(len(free[:300]), rng.choice([1, 2])))
            ) | set(free[-3000:])
        else:
            mid = len(free) // 2
            open_ports = set(free[mid : mid + rng.choice([10, 300, 3000])])

        host = Host(
            open_ports=open_ports,
            batch_start_fails=rng.random() < 0.12,
            batch_fail_after=rng.choice([0, 0, 1, 2]),
            batch_blocked=rng.random() < 0.12,
            dind_ok=rng.random() < 0.8,
        )
        shapes.append(pytest.param(host, kwargs, rented, n, id=f"{n}-{kind}-{how}"))
    return shapes


SHAPES = _shapes(1463, 400)


@pytest.mark.parametrize("host, kwargs, rented, seed", SHAPES)
@pytest.mark.asyncio
async def test_two_pass_against_main(host, kwargs, rented, seed):
    info = _info(**kwargs)
    main, main_host, new, new_host = await _both(host, info, rented, seed)

    if main.status == "no_ports":
        assert new.status == "no_ports"
        assert new_host.containers == main_host.containers == []
        return

    # main's verified count is after the DinD probe has taken its port
    assert (len(main.successful) >= MIN_PORT_COUNT) == (new.second_pass == SECOND_PASS_NOT_NEEDED)
    one_probes = [p for p in new_host.probes if p[1] != 2]
    two_probes = [p for p in new_host.probes if p[1] == 2]
    extra = len(new_host.containers) - len(main_host.containers)

    if new.second_pass == SECOND_PASS_NOT_NEEDED:
        # pass one is main's check: same containers, same probes, same published result
        assert new_host.containers == main_host.containers
        assert new_host.probes == main_host.probes
        assert new.selected_ports == main.selected
        assert new.successful_ports == main.successful
        assert new.failed_ports == main.failed
        assert new.dind_port == main.dind
        assert new.status == main.status
        return

    # below 3 on pass one: pass one is still main's probes, then at most one batch container more
    assert one_probes == main_host.probes
    assert extra in (0, 1)
    assert (extra == 1) == (
        new.second_pass
        in (SECOND_PASS_RAN, SECOND_PASS_BATCH_FAILED, SECOND_PASS_DISCARDED_CONTAINER_FAILED)
    )
    assert [c for c in new_host.containers if c[0] == "dind"] == [("dind", 0)]
    main_dind_ok = main.dind in main.successful
    if main.pass_one:
        # the container check ran first, on main's pass-one port, before any pass two
        assert new_host.containers[: len(main_host.containers)] == main_host.containers
        assert new.dind_port == main.dind
        assert new.dind_ok == main_dind_ok
        if extra:
            assert new_host.containers[-1][0] == "host"
    elif extra:
        assert new_host.containers[-2][0] == "host"
    if not new.dind_ok:
        # pass two never overrides a failed container check: main's result, never listed by pass two
        assert not new.sysbox_runtime
        assert len(new.successful_ports) <= len(main.successful)
        if not main_dind_ok:
            assert new.successful_ports == main.successful
            assert new.status == main.status
        if main.pass_one:
            assert new.second_pass == SECOND_PASS_SKIPPED_CONTAINER_FAILED
            assert extra == 0
            assert new_host.containers == main_host.containers
            assert new_host.probes == main_host.probes
            assert new.failed_ports == main.failed
        if new.second_pass == SECOND_PASS_DISCARDED_CONTAINER_FAILED:
            assert new.successful_ports == ()
            assert new.dind_port.external in {e for _, _, ports in two_probes for e in ports}
            assert all(not t.counted for t in new.port_ranges if t.pass_number == 2)
    else:
        assert len(new.successful_ports) >= len(main.successful)
    if new.second_pass == SECOND_PASS_SKIPPED_BATCH_FAILED:
        assert not any(mode == "host" for mode, _, _ in main_host.probes)
    tested_one = {e for _, _, ports in one_probes for e in ports}
    for _, _, ports in two_probes:
        assert not tested_one & set(ports)
        assert len(ports) <= BATCH_PORT_VERIFICATION_SIZE
        assert not set(ports) & rented


@pytest.mark.asyncio
async def test_the_shapes_cover_every_outcome():
    """The generated hosts reach every pass-two branch, so the property test above exercises them."""
    outcomes = set()
    for param in SHAPES:
        host, kwargs, rented, seed = param.values
        new = await _new_verify(_host_copy(host), _info(**kwargs), set(rented), seed)
        outcomes.add(new.second_pass)
    assert outcomes >= {
        SECOND_PASS_NOT_NEEDED,
        SECOND_PASS_SKIPPED_CONTAINER_FAILED,
        SECOND_PASS_SKIPPED_BATCH_FAILED,
        SECOND_PASS_NO_PORTS_LEFT,
        SECOND_PASS_RAN,
        SECOND_PASS_BATCH_FAILED,
        SECOND_PASS_DISCARDED_CONTAINER_FAILED,
    }


@pytest.mark.asyncio
async def test_wide_range_open_only_above_60000_goes_from_zero_to_three():
    info = _info(port_range="40000-65535")
    host = Host(open_ports=set(range(60001, 65536)))

    main, _, new, new_host = await _both(host, info)

    assert len(main.successful) == 0
    assert new.second_pass == SECOND_PASS_RAN
    assert len(new.successful_ports) >= MIN_PORT_COUNT
    assert all(p.external > 60000 for p in new.successful_ports)
    assert new.status == "ok"
    # the container check waited for pass two, passed on one of its answers, and gave it back
    assert new.dind_ok
    assert new.dind_port in new.successful_ports
    assert new.dind_port.external > 60000
    assert new_host.containers[-2:] == [("host", new_host.containers[-2][1]), ("dind", 0)]
    assert (await _published(new))[1:] == (True, True, True)
    pass_two = [t for t in new.port_ranges if t.pass_number == 2]
    assert [(t.first, t.last) for t in pass_two] == [
        (40000, 44999),
        (45000, 49999),
        (50000, 54999),
        (55000, 59999),
        (60000, 64999),
        (65000, 65535),
    ]
    assert sum(t.probed for t in pass_two) == BATCH_PORT_VERIFICATION_SIZE
    assert sum(t.answered for t in pass_two) == len(new.successful_ports)
    assert [t for t in new.port_ranges if t.pass_number == 1][0] == PortRangeResult(
        first=40000, last=44999, declared=5000, probed=300, answered=0
    )


@pytest.mark.asyncio
async def test_pass_two_never_reprobes_a_pass_one_port():
    info = _info(port_range="40000-65535")
    host = Host(open_ports={40001, 40002} | set(range(65000, 65536)))

    _, _, new, new_host = await _both(host, info)

    assert new.second_pass == SECOND_PASS_RAN
    (two,) = [ports for _, p, ports in new_host.probes if p == 2]
    one = {e for _, p, ports in new_host.probes if p != 2 for e in ports}
    assert one == set(range(40000, 40300))
    assert not one & set(two)
    assert two[-1] == 65535
    # pass one's two answers are kept and pass two's are added
    assert {40001, 40002} <= {p.external for p in new.successful_ports}
    # both passes' tallies see the combined answers, so each counts only answers among its own probes
    tallies = [r.as_dict() for r in new.port_ranges]
    assert all(t["answered"] <= t["probed"] <= t["declared"] for t in tallies)
    assert sum(t["answered"] for t in tallies if t["pass"] == 1) == 2
    assert [t["answered"] for t in tallies if t["pass"] == 1 and t["range"] == "65000-65535"] == [0]


@pytest.mark.asyncio
async def test_pass_two_is_deterministic_for_the_same_declaration_and_rental_set():
    info = _info(port_range="40000-65535")
    rented = {40010, 52000, 65535}
    runs = []
    for seed in range(4):
        host = Host(open_ports=set(range(64000, 65536)))
        await _new_verify(host, info, rented, seed)
        runs.append([ports for _, p, ports in host.probes if p == 2])
    assert all(r == runs[0] for r in runs)
    assert len(runs[0]) == 1

    other = Host(open_ports=set(range(64000, 65536)))
    await _new_verify(other, info, rented | {40020}, 0)
    assert [ports for _, p, ports in other.probes if p == 2] != runs[0]


@pytest.mark.asyncio
async def test_pass_two_is_skipped_when_pass_one_verifies_three():
    info = _info(port_range="40000-65535")
    host = Host(open_ports={40000, 40001, 40002} | set(range(60000, 65536)))

    main, main_host, new, new_host = await _both(host, info)

    assert new.second_pass == SECOND_PASS_NOT_NEEDED
    assert not [p for p in new_host.probes if p[1] == 2]
    assert new_host.containers == main_host.containers
    assert new.successful_ports == main.successful


async def _published(result, sysbox_runtime: bool = True):
    """What PortConnectivityCheck publishes for `result`: verified count, dind_ok, sysbox_runtime."""
    ctx = SimpleNamespace(
        state=SimpleNamespace(sysbox_runtime=sysbox_runtime),
        miner_hotkey="miner",
        executor=SimpleNamespace(uuid="executor-1"),
        services=SimpleNamespace(
            redis=SimpleNamespace(
                renting_in_progress=_async(False),
                record_dind_probe_miss=_async(False),
                clear_dind_probe_miss=_async(None),
            )
        ),
    )
    extra: dict[str, object] = {}
    kept = await PortConnectivityCheck._should_keep_last_known_sysbox(ctx, result, extra)
    return (
        len(result.successful_ports),
        result.dind_ok,
        sysbox_runtime if kept else result.sysbox_runtime,
        len(result.successful_ports) >= MIN_PORT_COUNT,
    )


def _async(value):
    async def call(*_args, **_kwargs):
        return value

    return call


@pytest.mark.parametrize("pass_one", [1, 2, 3])
@pytest.mark.asyncio
async def test_a_failed_container_check_on_a_pass_one_port_keeps_mains_result(pass_one):
    info = _info(port_range="40000-65535")
    # pass two would find a whole block at the top if it ran
    host = Host(
        open_ports=set(range(40000, 40000 + pass_one)) | set(range(60000, 65536)), dind_ok=False
    )

    main, main_host, new, new_host = await _both(host, info)

    assert main.pass_one == pass_one
    assert len(main.successful) == pass_one - 1
    assert new.second_pass == SECOND_PASS_SKIPPED_CONTAINER_FAILED
    # pass two did not run: main's containers, main's probes, main's result
    assert not [p for p in new_host.probes if p[1] == 2]
    assert new_host.containers == main_host.containers
    assert new_host.probes == main_host.probes
    assert new.selected_ports == main.selected
    assert new.successful_ports == main.successful
    assert new.failed_ports == main.failed
    assert new.dind_port == main.dind == PortPair(40000, 40000)
    assert new.status == main.status
    assert not new.dind_ok
    assert not new.sysbox_runtime
    assert not [t for t in new.port_ranges if t.pass_number == 2]
    # published exactly as main's failed-check result: pass_one - 1 verified, below the floor
    main_result = SimpleNamespace(
        successful_ports=main.successful, dind_ok=False, sysbox_runtime=False
    )
    assert await _published(new) == await _published(main_result)
    assert await _published(new) == (pass_one - 1, False, False, False)


@pytest.mark.asyncio
async def test_pass_two_does_not_run_when_the_dind_probe_keeps_all_three_answers():
    info = _info(port_range="40000-65535")
    host = Host(open_ports={40000, 40001, 40002} | set(range(60000, 65536)))

    main, main_host, new, new_host = await _both(host, info)

    assert new.second_pass == SECOND_PASS_NOT_NEEDED
    assert new.dind_ok
    assert new.dind_port == main.dind == PortPair(40000, 40000)
    assert not [p for p in new_host.probes if p[1] == 2]
    assert new_host.containers == main_host.containers
    assert new.successful_ports == main.successful
    assert len(new.successful_ports) == MIN_PORT_COUNT


@pytest.mark.parametrize("pass_one", [1, 2])
@pytest.mark.asyncio
async def test_a_passing_container_check_on_a_pass_one_port_lets_pass_two_count(pass_one):
    info = _info(port_range="40000-65535")
    host = Host(open_ports=set(range(40000, 40000 + pass_one)) | set(range(60000, 65536)))

    main, main_host, new, new_host = await _both(host, info)

    assert main.pass_one == len(main.successful) == pass_one
    assert new.second_pass == SECOND_PASS_RAN
    # the check ran first, on main's pass-one port, then pass two's one batch container
    assert new.dind_ok
    assert new.dind_port == main.dind == PortPair(40000, 40000)
    assert new_host.containers == main_host.containers + [new_host.containers[-1]]
    assert new_host.containers[-1][0] == "host"
    assert set(main.successful) <= set(new.successful_ports)
    assert len(new.successful_ports) >= MIN_PORT_COUNT
    assert new.status == "ok"
    assert all(t.counted for t in new.port_ranges)
    assert (await _published(new))[1:] == (True, True, True)


@pytest.mark.asyncio
async def test_no_pass_one_answer_and_a_failed_check_on_a_pass_two_port_is_mains_failed_result():
    info = _info(port_range="40000-65535")
    host = Host(open_ports=set(range(60000, 65536)), dind_ok=False)

    main, main_host, new, new_host = await _both(host, info)

    assert main.pass_one == 0
    assert main.successful == ()
    assert new.second_pass == SECOND_PASS_DISCARDED_CONTAINER_FAILED
    (two,) = [ports for _, p, ports in new_host.probes if p == 2]
    assert new.dind_port.external in two
    assert new.dind_port.external in host.open_ports
    assert not new.dind_ok
    assert [c for c in new_host.containers if c[0] == "dind"] == [("dind", 0)]
    # none of pass two's answers count: main's failed-check result, not listed
    assert new.successful_ports == main.successful == ()
    assert new.status == main.status == "no_working_ports"
    assert new.dind_port in new.failed_ports
    main_result = SimpleNamespace(successful_ports=(), dind_ok=False, sysbox_runtime=False)
    assert await _published(new) == await _published(main_result) == (0, False, False, False)
    # pass two's probes stay in the tally only, marked not counted
    pass_two = [t for t in new.port_ranges if t.pass_number == 2]
    assert sum(t.probed for t in pass_two) == BATCH_PORT_VERIFICATION_SIZE
    assert sum(t.answered for t in pass_two) > MIN_PORT_COUNT
    assert all(not t.counted and t.as_dict()["counted"] is False for t in pass_two)
    assert all(
        t.counted and "counted" not in t.as_dict() for t in new.port_ranges if t.pass_number == 1
    )
    assert all(t.answered <= t.probed <= t.declared for t in new.port_ranges)


@pytest.mark.asyncio
async def test_a_failed_container_check_is_the_skip_reason_even_when_pass_ones_batch_failed():
    info = _info(port_range="40000-65535")
    # the batch container never starts, the published tier finds exactly 3, and the DinD probe fails
    host = Host(
        open_ports={40000, 40001, 40002} | set(range(60000, 65536)),
        batch_start_fails=True,
        dind_ok=False,
    )

    main, main_host, new, new_host = await _both(host, info)

    assert main.pass_one == MIN_PORT_COUNT
    assert new.second_pass == SECOND_PASS_SKIPPED_CONTAINER_FAILED
    assert not [p for p in new_host.probes if p[1] == 2]
    assert new_host.containers == main_host.containers
    assert new.successful_ports == main.successful
    assert len(new.successful_ports) == MIN_PORT_COUNT - 1


@pytest.mark.asyncio
async def test_pass_two_is_skipped_when_pass_ones_batch_container_never_ran():
    info = _info(port_range="40000-65535")
    host = Host(open_ports=set(range(60000, 65536)), batch_start_fails=True)

    main, main_host, new, new_host = await _both(host, info)

    assert new.second_pass == SECOND_PASS_SKIPPED_BATCH_FAILED
    assert new_host.containers == main_host.containers
    assert not [p for p in new_host.probes if p[1] == 2]
    assert new.selected_ports == main.selected
    assert not [t for t in new.port_ranges if t.pass_number == 2]


@pytest.mark.asyncio
async def test_pass_two_start_failure_costs_one_container_and_says_so():
    info = _info(port_range="40000-65535")
    # pass one's two batch attempts start (nothing answers), pass two's one attempt does not
    host = Host(open_ports=set(range(60000, 65536)), batch_start_fails=True, batch_fail_after=2)

    main, main_host, new, new_host = await _both(host, info)

    assert new.second_pass == SECOND_PASS_BATCH_FAILED
    assert len(new_host.containers) == len(main_host.containers) + 1
    assert len(new.successful_ports) == len(main.successful) == 0


@pytest.mark.asyncio
async def test_no_ports_left_for_pass_two():
    info = _info(port_range="40000-40299")
    host = Host(open_ports={40000})

    main, main_host, new, new_host = await _both(host, info)

    assert new.second_pass == SECOND_PASS_NO_PORTS_LEFT
    assert new_host.containers == main_host.containers
    assert new.successful_ports == main.successful


def _top_block_count(port_range: str, width: int, top: int = 65535) -> int:
    info = _info(port_range=port_range)
    declared = [
        PortPair(i, e) for i, e in get_all_ports(info.port_range, info.port_mappings, info.ssh_port)
    ]
    selector = PortSelector()
    one = selector.select(info, BATCH_PORT_VERIFICATION_SIZE, set(), declared=declared)
    two = selector.select_spread(declared, BATCH_PORT_VERIFICATION_SIZE, set(), tested=one)
    return sum(p.external > top - width for p in two)


@pytest.mark.parametrize(
    "port_range, width, dind_ok, verified",
    [
        ("40000-65535", 169, True, 2),
        ("40000-65535", 170, True, 3),
        ("40000-65535", 255, False, 0),
        ("40000-65535", 2000, False, 0),
        (None, 303, True, 2),
        (None, 304, True, 3),
        (None, 455, False, 0),
    ],
)
@pytest.mark.asyncio
async def test_pass_two_limit_for_a_block_forwarded_at_the_top(
    port_range, width, dind_ok, verified
):
    info = _info(port_range=port_range)
    host = Host(open_ports=set(range(65536 - width, 65536)), dind_ok=dind_ok)

    _, _, new, _ = await _both(host, info)

    # a failed container check on a pass-two port counts none of pass two, however wide the block
    assert new.second_pass == (
        SECOND_PASS_RAN if dind_ok else SECOND_PASS_DISCARDED_CONTAINER_FAILED
    )
    assert len(new.successful_ports) == verified


def test_pass_two_stride_on_40000_65535():
    info = _info(port_range="40000-65535")
    declared = [
        PortPair(i, e) for i, e in get_all_ports(info.port_range, info.port_mappings, info.ssh_port)
    ]
    selector = PortSelector()
    one = selector.select(info, BATCH_PORT_VERIFICATION_SIZE, set(), declared=declared)
    two = [
        p.external
        for p in selector.select_spread(declared, BATCH_PORT_VERIFICATION_SIZE, set(), tested=one)
    ]

    assert (two[0], two[-1], len(two)) == (40300, 65535, 300)
    assert {b - a for a, b in zip(two, two[1:])} == {84, 85}
    assert _top_block_count("40000-65535", 170) == 3
    assert _top_block_count("40000-65535", 169) == 2


@pytest.mark.parametrize(
    "port_range, picks, width",
    [("40000-65535", 3, 254), ("40000-65535", 4, 338), (None, 3, 454), (None, 4, 606)],
)
def test_pass_two_block_anywhere_above_pass_one(port_range, picks, width):
    """The narrowest block above pass one's ports that pass two always probes `picks` times, wherever it sits."""
    info = _info(port_range=port_range)
    declared = [
        PortPair(i, e) for i, e in get_all_ports(info.port_range, info.port_mappings, info.ssh_port)
    ]
    selector = PortSelector()
    one = selector.select(info, BATCH_PORT_VERIFICATION_SIZE, set(), declared=declared)
    two = [
        p.external
        for p in selector.select_spread(declared, BATCH_PORT_VERIFICATION_SIZE, set(), tested=one)
    ]
    lo = max(p.external for p in one) + 1

    def fewest(w):
        return min(
            bisect.bisect_right(two, x + w - 1) - bisect.bisect_left(two, x)
            for x in range(lo, 65536 - w + 1)
        )

    assert fewest(width) == picks
    assert fewest(width - 1) == picks - 1


# A partly rented host: 8 GPUs, 6 rented, 10 declared ports, and the 6 rented GPUs' pods hold 8 of them.
PARTLY_RENTED_HELD = frozenset(range(20000, 20008))


@pytest.mark.asyncio
async def test_partly_rented_host_probes_only_its_free_ports_and_counts_no_held_port_as_failed():
    info = _info(port_range="20000-20009")
    host = Host(open_ports=set(range(20000, 20010)))
    main, _, new, new_host = await _both(host, info, PARTLY_RENTED_HELD)
    probed = {e for _, _, ports in new_host.probes for e in ports}
    assert probed == {20008, 20009}
    assert {p.external for p in new.selected_ports} == {20008, 20009}
    assert new.failed_ports == ()
    assert (
        {p.external for p in new.successful_ports}
        == {p.external for p in main.successful}
        == {20008, 20009}
    )
    assert new.second_pass == SECOND_PASS_NO_PORTS_LEFT
    assert [r.as_dict() for r in new.port_ranges] == [
        {"pass": 1, "range": "20000-20009", "declared": 10, "probed": 2, "answered": 2}
    ]


@pytest.mark.asyncio
async def test_partly_rented_wide_range_never_probes_a_held_port_in_either_pass():
    held = set(range(40000, 40300))
    info = _info(port_range="40000-65535")
    host = Host(open_ports=held | set(range(65000, 65536)))
    main, _, new, new_host = await _both(host, info, held)
    probed = {e for _, _, ports in new_host.probes for e in ports}
    assert not probed & held
    assert min(p.external for p in new.selected_ports) == 40300
    assert new.second_pass == SECOND_PASS_RAN
    assert len(main.successful) == 0
    assert len(new.successful_ports) >= MIN_PORT_COUNT
    assert not {p.external for p in new.successful_ports + new.failed_ports} & held
