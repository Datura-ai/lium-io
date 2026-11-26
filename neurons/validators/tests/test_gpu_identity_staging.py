"""Optional: run the real scrape check over specs captured on a staging node.

`tools` writes the cases file this test reads. Each case is one captured (or mutated)
machine-scrape document plus the reasons the identity check should find. The test runs
`MachineSpecScrapeCheck` itself in both modes: shadow must always pass and log the reasons;
enforce must pass only when there are none. Skipped unless `LIUMD_IDENTITY_SPECS` is set.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from services.task.checks import gpu_identity, machine_spec_scrape
from services.task.checks.machine_spec_scrape import MachineSpecScrapeCheck
from services.task.messages import MachineSpecMessages as Msg

from test_liumd_specs_scrape import (
    DummySSHCommandRunner,
    _answer,
    _ctx,
    _liumd_on,
    _start_reading,
    make_command_result,
)

CASES = os.environ.get("LIUMD_IDENTITY_SPECS")


@pytest.mark.skipif(not CASES, reason="set LIUMD_IDENTITY_SPECS to a captured cases file")
@pytest.mark.asyncio
async def test_captured_specs_in_both_modes(monkeypatch, caplog):
    cases = json.loads(Path(CASES).read_text())["cases"]
    assert cases

    for case in cases:
        expected = set(case["expected_reasons"])
        for mode in ("shadow", "enforce"):
            caplog.clear()
            _liumd_on(monkeypatch)
            monkeypatch.setattr(
                machine_spec_scrape.settings, "VALIDATOR_LIUMD_EVIDENCE_MODE", mode
            )
            _start_reading(monkeypatch, _answer(case["specs"]))
            runner = DummySSHCommandRunner(result=make_command_result(success=False))

            with caplog.at_level(logging.INFO, logger=gpu_identity.__name__):
                result = await MachineSpecScrapeCheck().run(_ctx(runner))

            if mode == "shadow":
                assert result.passed is True, (case["name"], result.event.what_we_saw)
                assert result.event.what_we_saw.get("delivery") == "liumd", (case["name"], result.event.what_we_saw)
                [line] = [
                    r
                    for r in caplog.records
                    if r.getMessage() == gpu_identity.GPU_IDENTITY_EVENT
                ]
                reasons = set(line.msg.extra["reasons"])
                assert line.msg.extra["mode"] == "shadow", case["name"]
                assert line.msg.extra["would_fail"] is bool(reasons), case["name"]
                assert expected <= reasons, (case["name"], expected, reasons)
            elif expected:
                assert result.passed is False, case["name"]
                assert result.event.reason_code == Msg.GPU_IDENTITY_MISMATCH.reason, case["name"]
                assert expected <= set(result.event.what_we_saw["reasons"]), case["name"]
            else:
                assert result.passed is True, case["name"]
