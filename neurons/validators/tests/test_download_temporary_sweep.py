"""DAH-2805: the periodic sweep of abandoned huggingface_hub download temporaries.

A filler's weight download that is killed mid-flight leaves a `<etag>.<8hex>.incomplete` file that
nothing will ever read again — huggingface_hub >= 1.18 names every attempt uniquely and only unlinks
it in a `finally`. One prod node reached 741 GB of them and fell out of the rental listing. Age is
the only signal available from the validator: the writer lives inside a container, and the volume is
shared by every filler container on the node.
"""

from types import SimpleNamespace

import pytest

from services.container_cleanup import DOWNLOAD_TEMPORARY_MAX_AGE_MINUTES, ContainerCleanup


class FakeSshClient:
    def __init__(
        self,
        volumes: list[str],
        mount_points: dict[str, str] | None = None,
        found_paths: list[str] | None = None,
        volume_ls_fails: bool = False,
    ):
        self._volumes = volumes
        self._mount_points = mount_points or {name: f"/var/lib/docker/volumes/{name}/_data" for name in volumes}
        self._found_paths = found_paths or []
        self._volume_ls_fails = volume_ls_fails
        self.commands: list[str] = []

    async def run(self, command: str, *args, **kwargs):
        self.commands.append(command)
        if "volume ls" in command:
            if self._volume_ls_fails:
                return SimpleNamespace(exit_status=1, stdout="", stderr="boom")
            return SimpleNamespace(exit_status=0, stdout="\n".join(self._volumes), stderr="")
        if "volume inspect" in command:
            inspected = [name for name in self._mount_points if name in command]
            return SimpleNamespace(
                exit_status=0, stdout="\n".join(self._mount_points[name] for name in inspected), stderr=""
            )
        if command.startswith("find "):
            return SimpleNamespace(exit_status=0, stdout="\n".join(self._found_paths), stderr="")
        return SimpleNamespace(exit_status=0, stdout="", stderr="")


def _find_command(ssh_client: FakeSshClient) -> str:
    return next((command for command in ssh_client.commands if command.startswith("find ")), "")


@pytest.mark.asyncio
async def test_sweeps_both_filler_cache_volumes():
    ssh_client = FakeSshClient(
        volumes=["dphn_cache_hf_model", "engy_cache_ckpt_v3"],
        found_paths=["/var/lib/docker/volumes/dphn_cache_hf_model/_data/hub/blobs/abc.1234abcd.incomplete"],
    )

    removed = await ContainerCleanup().sweep_abandoned_download_temporaries(ssh_client, "exec-1")

    assert removed == 1
    find_command = _find_command(ssh_client)
    assert "/var/lib/docker/volumes/dphn_cache_hf_model/_data" in find_command
    assert "/var/lib/docker/volumes/engy_cache_ckpt_v3/_data" in find_command


@pytest.mark.asyncio
async def test_only_aged_incomplete_files_are_targeted():
    ssh_client = FakeSshClient(volumes=["dphn_cache_hf_model"])

    await ContainerCleanup().sweep_abandoned_download_temporaries(ssh_client, "exec-1")

    find_command = _find_command(ssh_client)
    assert "-name '*.incomplete'" in find_command
    assert f"-mmin +{DOWNLOAD_TEMPORARY_MAX_AGE_MINUTES}" in find_command
    # `-delete` turns on `-depth`, which would silently disable the xet prune next to it.
    assert "-delete" not in find_command


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
    ssh_client = FakeSshClient(volumes=["dphn_cache_hf_model"], found_paths=["/data/a.incomplete"])

    removed = await ContainerCleanup(dry_run=True).sweep_abandoned_download_temporaries(ssh_client, "exec-1")

    assert removed == 0
    assert _find_command(ssh_client) == ""


@pytest.mark.asyncio
async def test_a_failed_listing_never_raises():
    # The sweep runs inside a non-fatal check: a failure here must not change the executor's verdict.
    ssh_client = FakeSshClient(volumes=["dphn_cache_hf_model"], volume_ls_fails=True)

    removed = await ContainerCleanup().sweep_abandoned_download_temporaries(ssh_client, "exec-1")

    assert removed == 0
