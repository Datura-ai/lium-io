"""A host whose lscpu prints the raw family/model ("06/cf") gets a readable CPU name in its specs, and the
validator's TDX host check still accepts it."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

from services.task.checks.tdx_host import TdxHostCheck

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="module")
def cpu_model_name():
    ns = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", {"INTEL_FAMILY6_SERVER_CPUS", "cpu_model_name"}, {"re": re}
    )
    return ns["cpu_model_name"]


@pytest.mark.parametrize(
    "lscpu, shown",
    [
        ("06/cf", "Intel Xeon 5th Gen Scalable (Emerald Rapids), CPUID 06/cf"),
        ("06/ad", "Intel Xeon 6 (Granite Rapids), CPUID 06/ad"),
        ("06/7f", "06/7f"),  # an id not in the table stays as lscpu printed it
        ("19/11", "19/11"),  # not Intel family 6
        ("Intel(R) Xeon(R) Platinum 8592+", "Intel(R) Xeon(R) Platinum 8592+"),
        ("", ""),
    ],
)
def test_a_raw_cpuid_gets_a_name_and_a_real_name_is_kept(cpu_model_name, lscpu, shown):
    assert cpu_model_name(lscpu) == shown


@pytest.mark.parametrize("model", ["06/cf", "Intel Xeon 5th Gen Scalable (Emerald Rapids), CPUID 06/cf"])
def test_the_tdx_host_check_still_accepts_the_named_and_the_raw_form(model):
    assert TdxHostCheck.is_tdx_capable(model)


def test_the_tdx_host_check_does_not_read_a_cpuid_inside_another_name():
    assert not TdxHostCheck.is_tdx_capable("Intel(R) Core(TM) i9 06/cf edition")
