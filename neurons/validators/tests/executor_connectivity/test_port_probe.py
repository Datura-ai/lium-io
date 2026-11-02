import pytest

from services.executor_connectivity.models import BatchResult, PortPair
from services.executor_connectivity.port_probe import PortProbe


@pytest.mark.asyncio
async def test_port_probe_uses_batch_when_successful(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult(ports, []))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock()
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock()

    probe = PortProbe(batch, semi, fallback)

    result = await probe.probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1")

    assert list(result.successful) == ports
    assert list(result.failed) == []
    batch.verify.assert_called_once()
    semi.verify.assert_not_called()
    fallback.verify.assert_not_called()


@pytest.mark.asyncio
async def test_port_probe_uses_semi_batch_when_batch_empty(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult([], ports))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=(ports, []))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock()

    probe = PortProbe(batch, semi, fallback)

    result = await probe.probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1")

    assert list(result.successful) == ports
    assert list(result.failed) == []
    batch.verify.assert_called_once()
    semi.verify.assert_called_once()
    fallback.verify.assert_not_called()


@pytest.mark.asyncio
async def test_port_probe_keeps_partial_semi_batch_result(mocker):
    # DAH-2527: with chunked publication a busy port leaves the tier partially successful — that
    # result must be carried forward, not thrown away for the 10-port sequential tier.
    ports = [PortPair(9000 + i, 9000 + i) for i in range(3)]

    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult([], ports))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=(ports[:2], ports[2:]))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock()

    probe = PortProbe(batch, semi, fallback)

    result = await probe.probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1")

    assert list(result.successful) == ports[:2]
    assert list(result.failed) == ports[2:]
    fallback.verify.assert_not_called()


@pytest.mark.asyncio
async def test_port_probe_falls_through_to_fallback_when_batch_and_semi_empty(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult([], ports))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=([], ports))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock(return_value=(ports[:1], ports[1:]))

    probe = PortProbe(batch, semi, fallback)

    result = await probe.probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1")

    assert list(result.successful) == ports[:1]
    assert list(result.failed) == ports[1:]
    batch.verify.assert_called_once()
    semi.verify.assert_called_once()
    fallback.verify.assert_called_once()


@pytest.mark.asyncio
async def test_port_probe_fallback_tests_every_port_a_split_node_needs(mocker):
    ports = [PortPair(9000 + i, 9000 + i) for i in range(30)]
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult([], ports))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=([], ports))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock(return_value=(ports[:24], []))

    await PortProbe(batch, semi, fallback).probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1", min_ports=24)

    assert batch.verify.call_args.kwargs["min_ports"] == 24
    assert fallback.verify.call_args.kwargs["max_ports"] == 24


@pytest.mark.asyncio
async def test_port_probe_runs_next_tier_while_a_split_node_is_short(mocker):
    # a split node needs 24: 3 host-network successes must not stop the published-port tiers
    ports = [PortPair(9000 + i, 9000 + i) for i in range(30)]
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult(ports[:3], ports[3:]))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=(ports[:10], ports[10:]))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock(return_value=(ports[:24], ports[24:]))

    result = await PortProbe(batch, semi, fallback).probe(
        ports, ssh_client=mocker.Mock(), host="127.0.0.1", min_ports=24
    )

    assert list(result.successful) == ports[:24]
    assert result.tier == "fallback"
    semi.verify.assert_called_once()


@pytest.mark.asyncio
async def test_port_probe_keeps_the_best_tier_for_a_short_split_node(mocker):
    ports = [PortPair(9000 + i, 9000 + i) for i in range(30)]
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult(ports[:5], ports[5:]))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=(ports[:2], ports[2:]))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock(return_value=([], ports))

    result = await PortProbe(batch, semi, fallback).probe(
        ports, ssh_client=mocker.Mock(), host="127.0.0.1", min_ports=24
    )

    assert list(result.successful) == ports[:5]
    assert result.tier == "batch"


@pytest.mark.asyncio
async def test_port_probe_stops_once_every_selected_port_answered(mocker):
    # fewer ports selected than a split node needs (a rental holds the rest): all 21 answered, so no later tier runs
    ports = [PortPair(9000 + i, 9000 + i) for i in range(21)]
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult(ports, []))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock()
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock()

    result = await PortProbe(batch, semi, fallback).probe(
        ports, ssh_client=mocker.Mock(), host="127.0.0.1", min_ports=24
    )

    assert list(result.successful) == ports
    assert result.tier == "batch"
    semi.verify.assert_not_called()
    fallback.verify.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("external", [None, "passed", "blocked"])
@pytest.mark.parametrize("batch_ok, expect_tier", [(23, "semi_batch"), (24, "batch")])
async def test_port_probe_split_node_outside_reading_table(mocker, external, batch_ok, expect_tier):
    # a short split node runs the later tiers whatever the outside reading; a full batch stops; the reading is kept
    ports = [PortPair(9000 + i, 9000 + i) for i in range(24)]
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult(ports[:batch_ok], ports[batch_ok:], external))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=(ports, []))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock()

    result = await PortProbe(batch, semi, fallback).probe(
        ports, ssh_client=mocker.Mock(), host="127.0.0.1", min_ports=24
    )

    assert result.tier == expect_tier
    assert len(result.successful) == 24
    assert result.external == external
    assert semi.verify.call_count == (batch_ok < 24)
    fallback.verify.assert_not_called()


@pytest.mark.asyncio
async def test_port_probe_blocked_reading_stops_an_unsplit_node(mocker):
    ports = [PortPair(9000 + i, 9000 + i) for i in range(6)]
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult(ports[:3], ports[3:], "blocked"))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock()
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock()

    result = await PortProbe(batch, semi, fallback).probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1")

    assert result.external == "blocked" and result.tier == "batch"
    semi.verify.assert_not_called()


def _span(lo, hi):
    return [PortPair(p, p) for p in range(lo, hi)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "semi_ok, fb_ok, expect_ok, expect_tier",
    [
        # busy 40000; tiers verify complementary-overlapping ranges: every distinct success counts
        (_span(40010, 40030), _span(40001, 40024), _span(40010, 40030) + _span(40001, 40010), "fallback"),
        # fully overlapping: no double count
        (_span(40001, 40021), _span(40001, 40011), _span(40001, 40021), "semi_batch"),
        # complementary halves
        (_span(40015, 40030), _span(40001, 40015), _span(40015, 40030) + _span(40001, 40015), "semi_batch"),
        # busy port in the middle stays failed
        (_span(40000, 40012), _span(40013, 40030), _span(40000, 40012) + _span(40013, 40030), "fallback"),
    ],
)
async def test_port_probe_merges_distinct_successes_across_tiers(mocker, semi_ok, fb_ok, expect_ok, expect_tier):
    ports = _span(40000, 40030)
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult([], ports, "blocked"))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=(semi_ok, [p for p in ports if p not in semi_ok]))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock(return_value=(fb_ok, [p for p in ports if p not in fb_ok]))

    result = await PortProbe(batch, semi, fallback).probe(
        ports, ssh_client=mocker.Mock(), host="127.0.0.1", min_ports=30
    )

    assert list(result.successful) == expect_ok
    assert set(result.failed) == set(ports) - set(expect_ok)
    assert result.tier == expect_tier
