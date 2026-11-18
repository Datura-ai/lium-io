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


class _HelloWorldHangs(_InnerSSH):
    async def run(self, command):
        if command == "docker run --rm hello-world":
            raise TimeoutError
        return await super().run(command)


@pytest.mark.asyncio
async def test_a_timed_out_sysbox_test_measured_nothing():
    """30 Oct, pod c0877f60: a busy host is not a host without Sysbox."""
    assert await DindVerifier(Mock())._sysbox_works(_HelloWorldHangs(SYSBOX_MAP), {}) is None


class _ProcSysSSH:
    """DAH-6637: the sysbox-fs read inside the probe container."""

    def __init__(self, stderr: str = "", exit_status: int = 0, hangs: bool = False):
        self.stderr, self.exit_status, self.hangs = stderr, exit_status, hangs

    async def run(self, command):
        assert command == f"cat {dind_probe.SYSBOX_FS_PATH}"
        if self.hangs:
            raise TimeoutError
        return SimpleNamespace(stdout="" if self.exit_status else "probe\n", stderr=self.stderr, exit_status=self.exit_status)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stderr",
    [
        "cat: /proc/sys/kernel/hostname: Transport endpoint is not connected",
        "cat: /proc/sys/kernel/hostname: Input/output error",
    ],
)
async def test_a_broken_sysbox_fs_is_named(stderr):
    cause = await DindVerifier._read_sysbox_fs(_ProcSysSSH(stderr, exit_status=1), {})

    assert cause.code == dind_probe.HOST_SYSBOX_FS_BROKEN
    assert stderr in cause.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inner",
    [
        _ProcSysSSH(),
        _ProcSysSSH("cat: /proc/sys/kernel/hostname: Permission denied", exit_status=1),
        _ProcSysSSH(hangs=True),
    ],
)
async def test_a_clean_or_unanswered_sysbox_fs_read_decides_nothing(inner):
    assert await DindVerifier._read_sysbox_fs(inner, {}) is None


class _BrokenFsSSH(_InnerSSH):
    async def run(self, command):
        if command == f"cat {dind_probe.SYSBOX_FS_PATH}":
            return SimpleNamespace(stdout="", stderr="Transport endpoint is not connected", exit_status=1)
        return await super().run(command)


@pytest.mark.asyncio
async def test_a_passed_probe_carries_the_broken_sysbox_fs(mocker):
    """Sysbox still counts as present; the cause rides on DindProbeResult.error."""
    from contextlib import asynccontextmanager

    from services.executor_connectivity.models import PortPair

    @asynccontextmanager
    async def answered():
        yield _BrokenFsSSH(SYSBOX_MAP)

    async def connect(*args, **kwargs):
        return answered()

    ssh_service = Mock(generate_keypair=Mock(return_value=("private", "public")))
    mocker.patch.object(dind_probe.asyncssh, "import_private_key")
    verifier = DindVerifier(ssh_service)
    mocker.patch.object(verifier, "_connect_retrying_until_sshd_answers", side_effect=connect)

    async def run(cmd):
        return SimpleNamespace(exit_status=0, stdout="", stderr="")

    result = await verifier.verify(
        PortPair(9000, 9000), ssh_client=Mock(run=run), host="192.0.2.1", container_name_prefix="c", sysbox=True
    )

    assert (result.success, result.sysbox_runtime) == (True, True)
    assert result.error.code == dind_probe.HOST_SYSBOX_FS_BROKEN
