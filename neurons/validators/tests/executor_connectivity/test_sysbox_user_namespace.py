"""G21: hello-world inside the sysbox probe container passes under privileged runc too; only sysbox gives the
container a user namespace, so an identity uid map is logged and refuses sysbox."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services.executor_connectivity import dind_probe
from services.executor_connectivity.dind_probe import DindVerifier

SYSBOX_MAP = "         0     165536      65536\n"
IDENTITY_MAP = "         0          0 4294967295\n"


class _InnerSSH:
    def __init__(self, uid_map: str | None, exit_status: int = 0):
        self.uid_map = uid_map
        self.exit_status = exit_status

    async def run(self, command):
        if command == "cat /proc/self/uid_map":
            return SimpleNamespace(stdout=self.uid_map, stderr="", exit_status=self.exit_status)
        return SimpleNamespace(stdout="Hello from Docker!", stderr="", exit_status=0)


def _identity_logs(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if "SYSBOX_IDENTITY_UID_MAP" in record.getMessage()]


@pytest.mark.asyncio
async def test_a_user_namespace_is_sysbox(caplog):
    with caplog.at_level(logging.WARNING, logger=dind_probe.logger.name):
        works = await DindVerifier(Mock())._sysbox_works(_InnerSSH(SYSBOX_MAP), {})

    assert works is True
    assert _identity_logs(caplog) == []


@pytest.mark.asyncio
async def test_an_identity_map_is_logged_and_not_sysbox(caplog):
    with caplog.at_level(logging.WARNING, logger=dind_probe.logger.name):
        works = await DindVerifier(Mock())._sysbox_works(_InnerSSH(IDENTITY_MAP), {})

    assert works is False
    assert len(_identity_logs(caplog)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("uid_map,exit_status", [("", 0), (None, 0), (IDENTITY_MAP, 1)])
async def test_an_unreadable_map_decides_nothing(uid_map, exit_status):
    works = await DindVerifier(Mock())._sysbox_works(_InnerSSH(uid_map, exit_status), {})

    assert works is True


def _unreadable_logs(caplog) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records if "SYSBOX_UID_MAP_UNREADABLE" in record.getMessage()
    ]


def _ok_uid_maps(caplog) -> list[str]:
    return [
        record.msg.extra["uid_map"]
        for record in caplog.records
        if record.getMessage() == "Sysbox check ok"
    ]


class _TimeoutSSH(_InnerSSH):
    async def run(self, command):
        if command == "cat /proc/self/uid_map":
            raise TimeoutError
        return await super().run(command)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inner,expected",
    [
        (_InnerSSH(SYSBOX_MAP), "user_namespace"),
        (_InnerSSH(IDENTITY_MAP), "identity"),
        (_InnerSSH("", 0), "unreadable"),
        (_InnerSSH(IDENTITY_MAP, 1), "unreadable"),
        (_TimeoutSSH(None), "unreadable"),
    ],
)
async def test_sysbox_ok_records_the_uid_map_result(caplog, inner, expected):
    with caplog.at_level(logging.INFO, logger=dind_probe.logger.name):
        await DindVerifier(Mock())._sysbox_works(inner, {})

    assert _ok_uid_maps(caplog) == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inner", [_InnerSSH("", 0), _InnerSSH(None, 0), _InnerSSH(IDENTITY_MAP, 1), _TimeoutSSH(None)]
)
async def test_an_unreadable_map_is_logged(caplog, inner):
    with caplog.at_level(logging.WARNING, logger=dind_probe.logger.name):
        works = await DindVerifier(Mock())._sysbox_works(inner, {})

    assert works is True
    assert len(_unreadable_logs(caplog)) == 1
    assert _identity_logs(caplog) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("uid_map", [SYSBOX_MAP, IDENTITY_MAP])
async def test_a_readable_map_is_not_logged_as_unreadable(caplog, uid_map):
    with caplog.at_level(logging.WARNING, logger=dind_probe.logger.name):
        await DindVerifier(Mock())._sysbox_works(_InnerSSH(uid_map), {})

    assert _unreadable_logs(caplog) == []
