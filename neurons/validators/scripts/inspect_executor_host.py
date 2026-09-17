"""One-off, read-only: inspect executor hosts through the validator's own SSH path (DAH-3540).

Why: 39 of 43 executors in two Massed Compute /24 blocks still run the pre-14-Sep executor
image with Watchtower running (reports/EXECUTOR_FLEET_STALE_20260910.md). Docker Hub answers
correctly from inside the blocks, so the lead is the host's docker registry configuration or
a stuck updater. Providers cannot be asked to read it for us; the validator already has a
signed key path onto every executor, so this script walks that same path and reads, nothing
more.

How the connect path is reused (no new protocol, no new key handling):

1. ``SSHService.generate_ssh_key`` mints the one-shot key pair exactly as a verification cycle does.
2. ``MinerService._make_rest_request`` posts the signed ``SSHPubKeySubmitRequest`` to
   ``/api/validator/ssh-pubkey-submit`` with ``MinerService._generate_auth_headers`` and
   ``MinerService._sign_validator_pubkey`` (the express-lane shape: one executor id per request).
3. The miner answers ``AcceptSSHKeyRequest`` with the executor's ``ExecutorSSHInfo``
   (``address``, ``ssh_port``, ``ssh_username``). That is how the validator learns where to
   connect; the backend is not asked.
4. ``services.ssh_connect_timing.connect_with_phase_timing(host, port, username,
   client_keys=[pkey], known_hosts=None)`` opens the session, the same call
   ``DockerService.wait_for_port_check_containers`` makes.
5. The read-only commands in ``COMMANDS`` run, each wrapped by ``capped()`` so the node cuts
   stdout and stderr at ``OUTPUT_CAP`` bytes each before they reach the validator, then
   ``MinerService._remove_ssh_key_via_rest`` takes the key back, in a ``finally`` that runs after
   every submit, refused or timed out included. The miner answers 200 once it accepted the remove
   request; it does not confirm the removal on the executor. The remove is tried
   ``KEY_REMOVE_ATTEMPTS`` times with the SAME public key (review, taiberium 17 Sep: a rerun mints a
   new key and cannot take back the one the first run left); when every attempt fails the node
   block prints that public key for exact manual removal, the node is an error and the run exits 1.
6. A session that drops after login makes every read fail. A node with no completed read is an
   error (``NO_READ_COMPLETED``) and the run exits 1, whatever the key remove answered (review,
   taiberium 17 Sep: every command failed and the script exited 0).

The sshd we reach runs inside the ``executor-executor-1`` container. That container mounts the
host's ``/var/run/docker.sock`` and ``/etc/docker/daemon.json`` (neurons/executor/docker-compose.app.yml),
so ``docker info`` reports the HOST daemon's registry configuration and ``cat /etc/docker/daemon.json``
reads the HOST file. The cgroup and ``/.dockerenv`` lines in each block say where the shell ran.

What the table compares: ``executor_vs_hub`` is the digest of the image the running executor
container(s) run against ``fetch_executor_image_digest()``, the registry digest of
``settings.EXECUTOR_IMAGE_REF`` that the validator scores against (DAH-2701); ``runner_vs_hub``
is the running runner's digest against Docker Hub's ``compute-subnet-executor-runner:latest``, the
image Watchtower updates. Running digests come from every running container's image (by the
image's repository, as ``machine_scrape`` finds the executor), so zero matches read ``none`` and
several containers on different digests are listed by name instead of one being picked.
``watchtower_log`` and ``runner_log`` are the last 50 lines of each container: Watchtower says
what it pulled and restarted; the runner (``executor-executor-runner-1``) says whether its
``docker compose up --wait`` came up healthy, which is where a stale executor's failed start shows.

Usage (from the validator checkout, with the validator's env loaded):

    cd neurons/validators && pdm run python scripts/inspect_executor_host.py --executor-ids nodes.txt

``nodes.txt`` holds one ``<executor uuid> <miner hotkey>`` per line (``#`` starts a comment). A
bare uuid needs ``--miner-hotkey`` to name the one miner it belongs to. ``--dry-run`` prints the
plan and opens nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import aiohttp
import asyncssh
import bittensor

# Runs from ``neurons/validators`` like ``src/cli.py`` does; the validator package lives in ``src``.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from core.config import settings  # noqa: E402
from datura.requests.miner_requests import AcceptSSHKeyRequest, ExecutorSSHInfo  # noqa: E402
from datura.requests.validator_requests import SSHPubKeySubmitRequest  # noqa: E402
from services.default_docker_image_digest_service import (  # noqa: E402
    fetch_executor_image_digest,
    fetch_registry_digest,
)
from services.miner_service import (  # noqa: E402
    REST_SSH_SUBMIT_TIMEOUT,
    MinerService,
    _parse_miner_response,
)
from services.ssh_connect_timing import connect_with_phase_timing  # noqa: E402
from services.ssh_service import SSHService  # noqa: E402

RUNNER_IMAGE = "daturaai/compute-subnet-executor-runner"
EXECUTOR_IMAGE = "daturaai/compute-subnet-executor"
WATCHTOWER_CONTAINER = "executor-watchtower-1"
# the compose service `executor-runner` (neurons/executor/docker-compose.yml): its entrypoint runs
# `docker compose up --pull always --detach --wait --force-recreate`, so a `--wait` that failed
# (the executor never reported healthy) is in THIS log, not in Watchtower's
RUNNER_CONTAINER = "executor-executor-runner-1"
NO_DAEMON_JSON = "NO_DAEMON_JSON"
DAEMON_JSON_IS_DIR = "DAEMON_JSON_IS_DIR"  # docker made a directory because the host file did not exist at first `up`

# Bytes per stream per command that the node may send; `head -c` on the node cuts the rest.
OUTPUT_CAP = 65536
# The wrapped command's exit status rides on the last stdout line, after a blank one.
RC_MARKER = "__inspect_rc="
TRUNCATED_NOTE = f"[truncated at {OUTPUT_CAP} bytes]"
# The key remove is retried with the same public key: a rerun cannot remove a key it did not mint.
KEY_REMOVE_ATTEMPTS = 3
KEY_REMOVE_RETRY_DELAY = 2.0  # seconds between attempts
NO_READ_COMPLETED = "no read completed: the SSH session dropped after login"

# Every command reads. ``test_every_command_is_read_only`` refuses a verb that writes.
COMMANDS: tuple[tuple[str, str], ...] = (
    ("registry_config", "docker info --format '{{json .RegistryConfig}}'"),
    (
        "daemon_json",
        f"test -d /etc/docker/daemon.json && echo {DAEMON_JSON_IS_DIR}"
        f" || cat /etc/docker/daemon.json 2>/dev/null || echo {NO_DAEMON_JSON}",
    ),
    ("pulled_runner_images", f"docker images --digests {RUNNER_IMAGE}"),
    ("executor_images", f"docker images --digests {EXECUTOR_IMAGE}"),
    ("running", "docker ps --format '{{.Names}} {{.Image}}'"),
    (
        # the image each running container RUNS (name → image id), not the newest one pulled: a
        # Watchtower that pulled and failed to restart leaves both on the host
        "running_containers",
        "docker inspect --format '{{.Name}} {{.Image}}' $(docker ps -q) 2>&1",
    ),
    (
        # the repository digests of those image ids; `running_digests_by_repo` joins the two reads
        "running_image_digests",
        "docker image inspect --format '{{.Id}} {{json .RepoDigests}}'"
        " $(docker inspect --format '{{.Image}}' $(docker ps -q)) 2>&1",
    ),
    (
        "watchtower_log",
        f"docker logs {WATCHTOWER_CONTAINER} --tail 50 2>&1"
        " || docker logs $(docker ps -qf name=watchtower) --tail 50 2>&1",
    ),
    (
        # `docker compose up --wait` failures are here (the runner's entrypoint, `set -e`, exits
        # on them and the container restarts, so the last attempt's lines are at the tail)
        "runner_log",
        f"docker logs {RUNNER_CONTAINER} --tail 50 2>&1"
        " || docker logs $(docker ps -qf name=executor-runner) --tail 50 2>&1",
    ),
    ("cgroup", "head -1 /proc/1/cgroup; head -1 /proc/self/cgroup"),
    ("dockerenv", "test -f /.dockerenv && echo IN_CONTAINER || echo NO_DOCKERENV; hostname"),
)

Resolver = Callable[[str], Awaitable[tuple[str, int]]]


def capped(command: str) -> str:
    """``command`` with stdout and stderr each cut at ``OUTPUT_CAP`` bytes on the node.

    POSIX sh only (dash included): the inner group's stdout is parked on fd 3 while its stderr
    goes through the first ``head``, then fd 3 comes back as stdout for the second ``head``. A
    command that keeps writing past the cap gets SIGPIPE and stops. The command's exit status is
    printed as the last stdout line (``RC_MARKER``); a stdout that ``head`` cut has no marker, so
    the client reads it as truncated and the exit status as unknown.
    """
    return (
        f"{{ {{ {command}; printf '\\n{RC_MARKER}%s\\n' \"$?\"; }} 2>&1 1>&3 3>&-"
        f" | head -c {OUTPUT_CAP} 1>&2 3>&-; }} 3>&1 | head -c {OUTPUT_CAP}"
    )


@dataclass(frozen=True)
class Target:
    executor_id: str
    miner_hotkey: str


@dataclass
class CommandOutput:
    exit_status: int | None
    stdout: str
    stderr: str


@dataclass(frozen=True)
class HubDigests:
    """What the running digests are compared with; ``None`` means not fetched or unreachable."""

    executor: str | None = None  # fetch_executor_image_digest(): settings.EXECUTOR_IMAGE_REF, what the validator scores
    runner: str | None = None  # Docker Hub RUNNER_IMAGE:latest, what Watchtower pulls


def split_exit_status(stdout: str) -> tuple[str, int | None]:
    """(stdout without the ``RC_MARKER`` line, exit status); ``None`` when the marker is missing."""
    body, marker, rest = stdout.rpartition("\n" + RC_MARKER)
    if not marker or not rest.endswith("\n"):  # no marker, or the cap fell inside its digits
        return stdout, None
    digits = rest.strip()
    return body, int(digits) if digits.isdigit() else None


def clip(text: str, note: str | None = None) -> str:
    """Never keep more than ``OUTPUT_CAP`` characters of a stream; a stream at the cap says so."""
    if len(text) >= OUTPUT_CAP:
        text, note = text[:OUTPUT_CAP], TRUNCATED_NOTE
    text = text.strip()
    return f"{text}\n{note}".strip() if note else text


def capped_output(result: asyncssh.SSHCompletedProcess) -> CommandOutput:
    """The ``CommandOutput`` for one ``capped()`` run: marker off, both streams clipped."""
    stdout, exit_status = split_exit_status(result.stdout or "")
    note = None if exit_status is not None else f"[no exit status: stdout cut at {OUTPUT_CAP} bytes or the shell stopped early]"
    return CommandOutput(exit_status=exit_status, stdout=clip(stdout, note), stderr=clip(result.stderr or ""))


@dataclass
class NodeReport:
    target: Target
    executor: ExecutorSSHInfo | None = None
    outputs: dict[str, CommandOutput] = field(default_factory=dict)
    error: str | None = None
    key_removed: bool | None = None
    # the public key this run submitted, kept only when no remove attempt was accepted: it is what
    # a human must delete from the executor's authorized_keys, a rerun cannot do it
    public_key_left: str | None = None


def parse_targets(text: str, default_hotkey: str | None) -> list[Target]:
    """One ``uuid [hotkey]`` per line, or comma-separated ids; a bare id takes ``default_hotkey``."""
    targets: list[Target] = []
    for raw in text.replace(",", "\n").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        executor_id = parts[0]
        hotkey = parts[1] if len(parts) > 1 else default_hotkey
        if not hotkey:
            raise SystemExit(f"{executor_id}: no miner hotkey on the line and no --miner-hotkey")
        targets.append(Target(executor_id=executor_id, miner_hotkey=hotkey))
    if not targets:
        raise SystemExit("no executor ids given")
    return targets


def read_ids_argument(value: str | None) -> str:
    if value is None:
        return sys.stdin.read()
    path = pathlib.Path(value)
    if path.is_file():
        return path.read_text()
    return value


def parse_axon_overrides(values: list[str]) -> dict[str, tuple[str, int]]:
    """``HOTKEY=IP:PORT`` pairs that stand in for the metagraph lookup."""
    overrides: dict[str, tuple[str, int]] = {}
    for value in values:
        hotkey, _, address = value.partition("=")
        host, _, port = address.rpartition(":")
        if not (hotkey and host and port.isdigit()):
            raise SystemExit(f"--axon expects HOTKEY=IP:PORT, got {value!r}")
        overrides[hotkey] = (host, int(port))
    return overrides


def make_miner_service() -> MinerService:
    """A ``MinerService`` for its REST key-submit helpers only.

    ``_make_rest_request``, ``_generate_auth_headers``, ``_serialize_request``,
    ``_sign_validator_pubkey`` and ``_remove_ssh_key_via_rest`` read none of the task, redis or
    attestation services, so those stay ``None`` here; nothing in this script schedules a task,
    touches Redis or issues an attestation nonce.
    """
    return MinerService(
        ssh_service=SSHService(),
        task_service=None,
        redis_service=None,
        attestation_service=None,
    )


async def resolve_axon_from_metagraph(hotkey: str) -> tuple[str, int]:
    """The validator's own lookup: ``SubtensorClient.get_miner`` (central-miner override included)."""
    from clients.subtensor_client import SubtensorClient

    client = await SubtensorClient.initialize()
    neuron = await client.get_miner(hotkey)
    return neuron.axon_info.ip, int(neuron.axon_info.port)


async def run_commands(
    ssh_client: asyncssh.SSHClientConnection, outputs: dict[str, CommandOutput], per_command_timeout: float
) -> int:
    """Fill ``outputs`` one command at a time, so a node budget that runs out keeps what was read.

    Each command runs through ``capped()``, so ``ssh_client.run()`` never buffers more than
    ``OUTPUT_CAP`` bytes per stream; ``errors="replace"`` keeps a multibyte character that
    ``head -c`` split from failing the read. Returns how many reads the node answered (a read
    that raised — the session dropped, the channel closed — is recorded and not counted), so the
    caller can tell a node that answered nothing from one whose reads it can judge.
    """
    completed = 0
    for label, command in COMMANDS:
        try:
            result = await asyncio.wait_for(ssh_client.run(capped(command), errors="replace"), timeout=per_command_timeout)
            outputs[label] = capped_output(result)
            completed += 1
        except Exception as exc:  # one failed read must not hide the others
            outputs[label] = CommandOutput(exit_status=None, stdout="", stderr=f"{type(exc).__name__}: {exc}")
    return completed


async def remove_key_with_retries(
    miner_service: MinerService,
    *,
    base_url: str,
    my_key: bittensor.Keypair,
    public_key: bytes,
    miner_hotkey: str,
    executor_id: str,
    log_extra: dict,
) -> bool:
    """``_remove_ssh_key_via_rest`` up to ``KEY_REMOVE_ATTEMPTS`` times with the same public key.

    The key the miner holds is this run's; a rerun mints another and cannot take this one back,
    so the retries happen here, now, with the key that was submitted. True on the first accepted
    remove (a 200 with an ``SSHKeyRemoved`` body); False when every attempt failed.
    """
    for attempt in range(1, KEY_REMOVE_ATTEMPTS + 1):
        if await miner_service._remove_ssh_key_via_rest(
            base_url=base_url,
            my_key=my_key,
            public_key=public_key,
            miner_hotkey=miner_hotkey,
            executor_id=executor_id,
            log_extra={**log_extra, "remove_attempt": attempt},
        ):
            return True
        if attempt < KEY_REMOVE_ATTEMPTS:
            await asyncio.sleep(KEY_REMOVE_RETRY_DELAY)
    return False


async def inspect_executor(
    *,
    target: Target,
    miner_service: MinerService,
    my_key: bittensor.Keypair,
    resolve_axon: Resolver,
    node_timeout: float,
) -> NodeReport:
    """Key submit → connect → read → key remove, for one executor. Raises only when the local key generation or signing fails."""
    report = NodeReport(target=target)
    log_extra = {"miner_hotkey": target.miner_hotkey, "executor_id": target.executor_id, "context": "inspect_executor_host"}
    try:
        host, port = await resolve_axon(target.miner_hotkey)
    except Exception as exc:
        report.error = f"miner axon lookup failed: {exc}"
        return report
    base_url = f"http://{host}:{port}"

    private_key, public_key = miner_service.ssh_service.generate_ssh_key(my_key.ss58_address)
    submit = SSHPubKeySubmitRequest(
        public_key=public_key,
        validator_signature=miner_service._sign_validator_pubkey(my_key, public_key),
        executor_id=target.executor_id,
        miner_hotkey=target.miner_hotkey,
    )
    headers = miner_service._generate_auth_headers(my_key, target.miner_hotkey)
    headers["Content-Type"] = "application/json"

    try:
        status, response_data = await miner_service._make_rest_request(
            method="POST",
            url=f"{base_url}/api/validator/ssh-pubkey-submit",
            json_data=miner_service._serialize_request(submit),
            headers=headers,
            timeout=REST_SSH_SUBMIT_TIMEOUT,
            log_extra=log_extra,
            operation_name="SSH key submit",
        )
        if status != 200 or response_data is None:
            report.error = f"miner refused the key submit: HTTP {status} {response_data}"
            return report
        msg = _parse_miner_response(response_data)
        if not isinstance(msg, AcceptSSHKeyRequest):
            report.error = f"miner answered {type(msg).__name__}: {getattr(msg, 'details', '')}"
            return report
        executor = next((e for e in msg.executors if e.uuid == target.executor_id), None)
        if executor is None:
            report.error = f"miner accepted the key but listed {len(msg.executors)} other executor(s), not this id"
            return report
        report.executor = executor

        pkey = asyncssh.import_private_key(
            miner_service.ssh_service.decrypt_payload(my_key.ss58_address, private_key.decode("utf-8"))
        )
        try:
            async with asyncio.timeout(node_timeout):
                async with connect_with_phase_timing(
                    log_extra=log_extra,
                    host=executor.address,
                    port=executor.ssh_port,
                    username=executor.ssh_username,
                    client_keys=[pkey],
                    known_hosts=None,
                ) as ssh_client:
                    # a quarter of the node budget per read: one hung `docker logs` costs its slice, not the node
                    completed = await run_commands(
                        ssh_client, report.outputs, per_command_timeout=max(5.0, node_timeout / 4)
                    )
            if completed == 0 and report.outputs:
                # login worked and then every read raised (the session dropped, sshd closed the
                # channel): there is nothing to judge, so the node is a failure, not a clean row
                first = next(iter(report.outputs.values())).stderr
                report.error = f"{NO_READ_COMPLETED} ({first})"
        except TimeoutError:  # the node budget only; the key submit's own 30 s timeout is reported below
            report.error = f"node budget of {node_timeout:.0f} s ran out after {len(report.outputs)} of {len(COMMANDS)} reads"
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    finally:
        # Every submitted key gets a remove, whatever the submit answered: a submit that timed out
        # after the miner already pushed the key must not leave it there. The remove is idempotent,
        # so a refused submit costs one harmless extra request. Retried here with this run's key.
        report.key_removed = await remove_key_with_retries(
            miner_service,
            base_url=base_url,
            my_key=my_key,
            public_key=public_key,
            miner_hotkey=target.miner_hotkey,
            executor_id=target.executor_id,
            log_extra=log_extra,
        )
        if report.key_removed is not True:
            # `_remove_ssh_key_via_rest` logs and returns False instead of raising, so without this
            # line the node would look clean with a key still installed. Appended, so an earlier
            # error (a refused submit, a read that hung) is kept next to it in the table. No advice
            # to rerun: a rerun mints a new key and cannot remove this one — the block prints it.
            report.public_key_left = public_key.decode("utf-8", errors="replace").strip()
            # the miner listed the executor only when it accepted the submit, so the key is on the host
            # for sure in that case; after a refused or timed-out submit it may or may not have landed
            state = "is still installed" if report.executor is not None else "may still be installed"
            note = (
                f"key remove not accepted by miner after {KEY_REMOVE_ATTEMPTS} attempts:"
                f" the key {state}, remove it by hand (public key in the node block)"
            )
            report.error = f"{report.error}; {note}" if report.error else note
    return report


def exit_code(reports: list[NodeReport]) -> int:
    """1 when any node reported an error or its key remove was not accepted; a key left behind is a failure."""
    return 1 if any(r.error or r.key_removed is not True for r in reports) else 0


def running_digests_by_repo(outputs: dict[str, CommandOutput]) -> dict[str, list[tuple[str, str]]] | None:
    """``repository -> [(container name, digest)]`` for every running container; ``None`` when a read failed.

    Joins ``running_containers`` (name, image id) with ``running_image_digests`` (image id,
    RepoDigests). A container is matched by its image's repository, the way ``machine_scrape``
    finds the executor container, never by its name.
    """
    containers = outputs.get("running_containers")
    images = outputs.get("running_image_digests")
    if containers is None or images is None or containers.exit_status != 0 or images.exit_status != 0:
        return None
    image_by_container: dict[str, str] = {}
    for line in containers.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("sha256:"):
            image_by_container[parts[0].lstrip("/")] = parts[1]
    repo_digests_by_image: dict[str, list[str]] = {}
    for line in images.stdout.splitlines():
        image_id, _, rest = line.partition(" ")
        if not image_id.startswith("sha256:"):
            continue
        try:
            repo_digests_by_image[image_id] = [d for d in (json.loads(rest) or []) if isinstance(d, str)]
        except (ValueError, TypeError):  # `null` for an image without RepoDigests, or a cut line
            continue
    if not set(image_by_container.values()) <= set(repo_digests_by_image):
        return None  # a container started or stopped between the two `docker ps -q`; say "?" rather than "none"
    by_repo: dict[str, list[tuple[str, str]]] = {}
    for name, image_id in sorted(image_by_container.items()):
        for entry in repo_digests_by_image[image_id]:
            repo, _, digest = entry.partition("@")
            if digest.startswith("sha256:"):
                by_repo.setdefault(repo, []).append((name, digest))
    return by_repo


def _short(digest: str) -> str:
    return digest[:19] if digest.startswith("sha256:") else digest


@dataclass(frozen=True)
class RunningImage:
    """One image's running containers: the table cell, and their image's digests when they all run the same image."""

    cell: str
    digests: frozenset[str] | None = None

    def vs_hub(self, hub_digest: str | None) -> str:
        if self.digests is None:
            return {"?": "?", "none": "none running"}.get(self.cell, "MIXED")
        if not hub_digest:
            return "?"
        return "current" if hub_digest in self.digests else "STALE"


def describe_running(matches: list[tuple[str, str]] | None) -> RunningImage:
    """``?`` when the reads failed, ``none`` when no running container runs the image, the digest
    (``×N`` for N containers on it) when they all run the same image, every container by name when
    they do not. One image can carry two RepoDigests of the same repository (pulled by tag and by
    digest); they are shown joined with ``+`` and either one counts as current."""
    if matches is None:
        return RunningImage("?")
    if not matches:
        return RunningImage("none")
    per_container: dict[str, set[str]] = {}
    for name, digest in matches:
        per_container.setdefault(name, set()).add(digest)
    distinct = {frozenset(digests) for digests in per_container.values()}
    if len(distinct) == 1:
        digests = next(iter(distinct))
        suffix = f" ×{len(per_container)}" if len(per_container) > 1 else ""
        return RunningImage("+".join(_short(d) for d in sorted(digests)) + suffix, digests)
    listed = ", ".join(f"{name}={'+'.join(_short(d) for d in sorted(digests))}" for name, digests in sorted(per_container.items()))
    return RunningImage(f"{len(distinct)} images in {len(per_container)} containers: {listed}")


def summarise(report: NodeReport, hub: HubDigests) -> dict[str, str]:
    """The one-line facts per node; every value is a string ready for the table."""
    out = report.outputs
    registry = out.get("registry_config")
    mirrors = "?"
    if registry and registry.stdout:
        try:
            cfg = json.loads(registry.stdout)
        except ValueError:
            cfg = None
        if isinstance(cfg, dict):  # `null` when the daemon behind the socket did not answer
            listed = cfg.get("Mirrors") or []
            mirrors = ", ".join(listed) if listed else "none"
        else:
            mirrors = "unparsable"
    daemon = out.get("daemon_json")
    if daemon is None or daemon.exit_status is None or not daemon.stdout:
        daemon_present = "?"  # the read never ran or failed; not "yes"
    elif daemon.stdout.strip() == NO_DAEMON_JSON:
        daemon_present = "no"
    elif daemon.stdout.strip() == DAEMON_JSON_IS_DIR:
        daemon_present = "DIR"
    else:
        daemon_present = "yes"

    pulled_digest = "?"
    images = out.get("pulled_runner_images")
    if images and images.stdout:
        for line in images.stdout.splitlines()[1:]:
            cols = line.split()
            if len(cols) >= 3 and cols[2].startswith("sha256:"):
                pulled_digest = cols[2]
                break

    by_repo = running_digests_by_repo(out)
    executor = describe_running(None if by_repo is None else by_repo.get(EXECUTOR_IMAGE, []))
    runner = describe_running(None if by_repo is None else by_repo.get(RUNNER_IMAGE, []))

    watchtower = out.get("watchtower_log")
    last_line = "?"
    if watchtower is not None:
        lines = [line for line in watchtower.stdout.splitlines() if line.strip()]
        last_line = lines[-1][:120] if lines else "(empty)"

    where = "?"
    dockerenv = out.get("dockerenv")
    if dockerenv is not None and "IN_CONTAINER" in dockerenv.stdout:
        where = "executor container"
    elif dockerenv is not None and "NO_DOCKERENV" in dockerenv.stdout:
        where = "host"

    return {
        "node": report.target.executor_id,
        "mirror": mirrors,
        "daemon_json": daemon_present,
        "executor_running": executor.cell,
        "executor_vs_hub": executor.vs_hub(hub.executor),
        "runner_running": runner.cell,
        "runner_pulled": _short(pulled_digest),
        "runner_vs_hub": runner.vs_hub(hub.runner),
        "shell_ran_in": where,
        "watchtower_last_line": last_line,
        "error": report.error or "",
    }


SUMMARY_COLUMNS = [
    "node", "mirror", "daemon_json", "executor_running", "executor_vs_hub", "runner_running", "runner_pulled",
    "runner_vs_hub", "shell_ran_in", "watchtower_last_line", "error",
]


def render_node_block(report: NodeReport) -> str:
    head = f"### {report.target.executor_id} (miner {report.target.miner_hotkey})"
    if report.executor is not None:
        head += f" — {report.executor.address}:{report.executor.ssh_port} as {report.executor.ssh_username}"
    lines = [head, "```", f"# each command below ran wrapped by capped(): stdout and stderr cut at {OUTPUT_CAP} bytes on the node"]
    if report.error:
        lines.append(f"ERROR: {report.error}")
    for label, command in COMMANDS:
        output = report.outputs.get(label)
        if output is None:
            continue
        lines.append(f"$ {command}")
        if output.stdout:
            lines.append(output.stdout)
        if output.stderr:
            lines.append(f"[stderr] {output.stderr}")
        if output.exit_status not in (0, None):
            lines.append(f"[exit {output.exit_status}]")
        lines.append("")
    # the miner answers 200 once it accepted the request; it does not confirm the removal on the executor
    lines.append(f"key remove request accepted by miner (not confirmed on executor): {report.key_removed}")
    if report.public_key_left:
        left = "LEFT ON THE EXECUTOR" if report.executor is not None else "POSSIBLY LEFT ON THE EXECUTOR (submit not accepted)"
        lines.append(
            f"PUBLIC KEY {left} — delete exactly this line from the executor container's"
            " ~/.ssh/authorized_keys if present (a rerun mints a new key and cannot remove this one):"
        )
        lines.append(report.public_key_left)
    lines.append("```")
    return "\n".join(lines)


def render_summary(rows: list[dict[str, str]], hub: HubDigests) -> str:
    columns = SUMMARY_COLUMNS
    lines = [
        f"Expected executor digest ({settings.EXECUTOR_IMAGE_REF}, fetch_executor_image_digest, what the validator"
        f" scores against): {hub.executor or 'not fetched'}",
        f"Docker Hub {RUNNER_IMAGE}:latest digest now: {hub.runner or 'not fetched'}",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[c].replace("|", "\\|") for c in columns) + " |")
    return "\n".join(lines)


def render_plan(targets: list[Target], concurrency: int, node_timeout: float) -> str:
    lines = [
        f"DRY RUN — {len(targets)} executor(s), concurrency {concurrency}, {node_timeout:.0f} s per node. Nothing opened.",
        "Per node: metagraph axon lookup → POST /api/validator/ssh-pubkey-submit (this executor id) →"
        " ssh <address>:<ssh_port> as <ssh_username> → the reads below → POST /api/validator/ssh-pubkey-remove.",
    ]
    for _, command in COMMANDS:
        lines.append(f"  $ {command}")
    lines.append(f"Each read runs wrapped, stdout and stderr cut at {OUTPUT_CAP} bytes on the node, e.g.:")
    lines.append(f"  $ {capped(COMMANDS[0][1])}")
    for target in targets:
        lines.append(f"- {target.executor_id}  miner {target.miner_hotkey}")
    return "\n".join(lines)


async def fetch_hub_digests() -> HubDigests:
    """The executor digest the validator scores against, and the runner digest Watchtower pulls."""
    executor = await fetch_executor_image_digest()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        runner = await fetch_registry_digest(session, f"{RUNNER_IMAGE}:latest")
    return HubDigests(executor=executor, runner=runner)


async def run(
    targets: list[Target],
    *,
    concurrency: int,
    node_timeout: float,
    resolve_axon: Resolver,
    miner_service: MinerService,
    my_key: bittensor.Keypair,
    hub: HubDigests,
    out=sys.stdout,
) -> list[NodeReport]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(target: Target) -> NodeReport:
        async with semaphore:
            return await inspect_executor(
                target=target,
                miner_service=miner_service,
                my_key=my_key,
                resolve_axon=resolve_axon,
                node_timeout=node_timeout,
            )

    reports = await asyncio.gather(*(one(t) for t in targets))
    for report in reports:
        print(render_node_block(report), file=out)
        print(file=out)
    print(render_summary([summarise(r, hub) for r in reports], hub), file=out)
    return list(reports)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--executor-ids", metavar="FILE|id,id", help="file of `uuid [miner_hotkey]` lines, or ids inline; default: stdin")
    parser.add_argument("--miner-hotkey", help="miner hotkey for every id that has none on its line")
    parser.add_argument("--axon", action="append", default=[], metavar="HOTKEY=IP:PORT", help="skip the metagraph lookup for this miner")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per node for connect + reads (default 60)")
    parser.add_argument("--no-hub-digest", action="store_true", help="do not ask Docker Hub for the expected executor and runner digests")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; open nothing")
    return parser


async def resolve_axons_once(hotkeys: set[str], overrides: dict[str, tuple[str, int]]) -> Resolver:
    """One metagraph lookup per miner, before any node task starts; a failed lookup fails its nodes only."""
    resolved: dict[str, tuple[str, int] | Exception] = dict(overrides)
    for hotkey in sorted(hotkeys - set(overrides)):
        try:
            resolved[hotkey] = await resolve_axon_from_metagraph(hotkey)
        except Exception as exc:
            resolved[hotkey] = exc

    async def resolve(hotkey: str) -> tuple[str, int]:
        value = resolved[hotkey]
        if isinstance(value, Exception):
            raise value
        return value

    return resolve


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    targets = parse_targets(read_ids_argument(args.executor_ids), args.miner_hotkey)
    overrides = parse_axon_overrides(args.axon)
    if args.dry_run:
        print(render_plan(targets, args.concurrency, args.timeout))
        return 0

    async def _main() -> int:
        hub = HubDigests() if args.no_hub_digest else await fetch_hub_digests()
        my_key = settings.get_bittensor_wallet().get_hotkey()
        resolve_axon = await resolve_axons_once({t.miner_hotkey for t in targets}, overrides)
        reports = await run(
            targets,
            concurrency=args.concurrency,
            node_timeout=args.timeout,
            resolve_axon=resolve_axon,
            miner_service=make_miner_service(),
            my_key=my_key,
            hub=hub,
        )
        return exit_code(reports)

    return asyncio.run(_main())


if __name__ == "__main__":
    sys.exit(main())
