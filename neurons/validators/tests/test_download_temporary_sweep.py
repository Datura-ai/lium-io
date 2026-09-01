"""DAH-2805: the periodic sweep of abandoned huggingface_hub download temporaries.

A filler's weight download that is killed mid-flight leaves a `<etag>.<8hex>.incomplete` file that
nothing will ever read again — huggingface_hub >= 1.18 names every attempt uniquely and only unlinks
it in a `finally`. One prod node reached 741 GB of them and fell out of the rental listing. Age is
the only signal available from the validator: the writer lives inside a container, and the volume is
shared by every filler container on the node.
"""

from types import SimpleNamespace

import pytest

from services.const import CACHE_SWEEP_CONTAINER_NAME

from services.container_cleanup import (
    DOWNLOAD_TEMPORARY_MAX_AGE_MINUTES,
    DOWNLOAD_TEMPORARY_SWEEP_TIMEOUT_SECONDS,
    ContainerCleanup,
)


class FakeSshClient:
    def __init__(
        self,
        volumes: list[str],
        find_returns_paths: list[str] | None = None,
        volume_ls_fails: bool = False,
    ):
        self._volumes = volumes
        self._find_returns_paths = find_returns_paths or []
        self._volume_ls_fails = volume_ls_fails
        self.commands: list[str] = []

    async def run(self, command: str, *args, **kwargs):
        self.commands.append(command)
        if "volume ls" in command:
            if self._volume_ls_fails:
                return SimpleNamespace(exit_status=1, stdout="", stderr="boom")
            return SimpleNamespace(exit_status=0, stdout="\n".join(self._volumes), stderr="")
        if " find " in command:
            return SimpleNamespace(exit_status=0, stdout="\n".join(self._find_returns_paths), stderr="")
        return SimpleNamespace(exit_status=0, stdout="", stderr="")


def _find_command(ssh_client: FakeSshClient) -> str:
    return next((command for command in ssh_client.commands if " find " in command), "")


@pytest.mark.asyncio
async def test_sweeps_both_filler_cache_volumes():
    ssh_client = FakeSshClient(
        volumes=["dphn_cache_hf_model", "engy_cache_ckpt_v3"],
        find_returns_paths=["/cache0/hub/models--x/blobs/abc.1234abcd.incomplete"],
    )

    removed = await ContainerCleanup().sweep_abandoned_download_temporaries(ssh_client, "exec-1")

    assert removed == 1
    find_command = _find_command(ssh_client)
    # Mounted by NAME into the helper container: the validator's SSH session lands inside the
    # executor container, where the volume's host path does not exist.
    assert "-v dphn_cache_hf_model:/cache0" in find_command
    assert "-v engy_cache_ckpt_v3:/cache1" in find_command
    assert "find /cache0 /cache1" in find_command
    # Named, or the provider-side load gate counts our housekeeping against the miner.
    assert f"--name {CACHE_SWEEP_CONTAINER_NAME}" in find_command
    # A leftover from a dockerd restart would hold the name for good and kill the sweep silently.
    assert f"docker rm -f {CACHE_SWEEP_CONTAINER_NAME}" in find_command
    # A wedged docker daemon must not eat the executor's cycle ahead of the fatal port checks.
    assert f"timeout {DOWNLOAD_TEMPORARY_SWEEP_TIMEOUT_SECONDS} " in find_command


@pytest.mark.asyncio
async def test_only_aged_incomplete_files_are_targeted():
    ssh_client = FakeSshClient(volumes=["dphn_cache_hf_model"])

    await ContainerCleanup().sweep_abandoned_download_temporaries(ssh_client, "exec-1")

    find_command = _find_command(ssh_client)
    assert "-name '*.incomplete'" in find_command
    assert f"-mmin +{DOWNLOAD_TEMPORARY_MAX_AGE_MINUTES}" in find_command
    # `-delete` turns on `-depth`, which would silently disable the prune next to it.
    assert "-delete" not in find_command
    # The worker runtimes and the xet chunk cache are 10^4-10^5 inodes that never hold one.
    assert "-name xet -o -name runtimes" in find_command


@pytest.mark.asyncio
async def test_a_renters_volume_is_never_a_candidate():
    ssh_client = FakeSshClient(volumes=["volume_customer_pod", "dphn_cache_hf_model"])

    await ContainerCleanup().sweep_abandoned_download_temporaries(ssh_client, "exec-1")

    assert "volume_customer_pod" not in _find_command(ssh_client)


@pytest.mark.asyncio
async def test_a_node_without_filler_caches_runs_no_find():
    ssh_client = FakeSshClient(volumes=["volume_customer_pod"])

    removed = await ContainerCleanup().sweep_abandoned_download_temporaries(ssh_client, "exec-1")

    assert removed == 0
    assert _find_command(ssh_client) == ""


@pytest.mark.asyncio
async def test_dry_run_removes_nothing():
    ssh_client = FakeSshClient(volumes=["dphn_cache_hf_model"], find_returns_paths=["/data/a.incomplete"])

    removed = await ContainerCleanup(dry_run=True).sweep_abandoned_download_temporaries(ssh_client, "exec-1")

    assert removed == 0
    assert _find_command(ssh_client) == ""


@pytest.mark.asyncio
async def test_a_failed_listing_never_raises():
    # The sweep runs inside a non-fatal check: a failure here must not change the executor's verdict.
    ssh_client = FakeSshClient(volumes=["dphn_cache_hf_model"], volume_ls_fails=True)

    removed = await ContainerCleanup().sweep_abandoned_download_temporaries(ssh_client, "exec-1")

    assert removed == 0
