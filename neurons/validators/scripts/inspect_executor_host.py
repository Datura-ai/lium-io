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
5. The read-only commands in ``COMMANDS`` run, then ``MinerService._remove_ssh_key_via_rest``
   takes the key back, in a ``finally``.

The sshd we reach runs inside the ``executor-executor-1`` container. That container mounts the
host's ``/var/run/docker.sock`` and ``/etc/docker/daemon.json`` (neurons/executor/docker-compose.app.yml),
so ``docker info`` reports the HOST daemon's registry configuration and ``cat /etc/docker/daemon.json``
reads the HOST file. The cgroup and ``/.dockerenv`` lines in each block say where the shell ran.

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
from services.default_docker_image_digest_service import fetch_registry_digest  # noqa: E402
from services.miner_service import (  # noqa: E402
    REST_SSH_SUBMIT_TIMEOUT,
    MinerService,
    _parse_miner_response,
)
from services.ssh_connect_timing import connect_with_phase_timing  # noqa: E402
from services.ssh_service import SSHService  # noqa: E402

RUNNER_IMAGE = "daturaai/compute-subnet-executor-runner"
EXECUTOR_IMAGE = "daturaai/compute-subnet-executor"
RUNNER_CONTAINER_FILTER = "executor-runner"
WATCHTOWER_CONTAINER = "executor-watchtower-1"
NO_DAEMON_JSON = "NO_DAEMON_JSON"
DAEMON_JSON_IS_DIR = "DAEMON_JSON_IS_DIR"  # docker made a directory because the host file did not exist at first `up`

# Every command reads. ``test_every_command_is_read_only`` refuses a verb that writes.
COMMANDS: tuple[tuple[str, str], ...] = (
    ("registry_config", "docker info --format '{{json .RegistryConfig}}'"),
    (
        "daemon_json",
        f"test -d /etc/docker/daemon.json && echo {DAEMON_JSON_IS_DIR}"
        f" || cat /etc/docker/daemon.json 2>/dev/null || echo {NO_DAEMON_JSON}",
    ),
    ("pulled_runner_images", f"docker images --digests {RUNNER_IMAGE}"),
    (
        # the image the runner container RUNS, not the newest one pulled: a Watchtower that pulled
        # and failed to restart leaves both on the host
        "running_runner_digest",
        "docker image inspect --format '{{index .RepoDigests 0}}'"
        f" $(docker inspect --format '{{{{.Image}}}}' $(docker ps -qf name={RUNNER_CONTAINER_FILTER})) 2>&1",
    ),
    ("executor_images", f"docker images --digests {EXECUTOR_IMAGE}"),
    ("running", "docker ps --format '{{.Names}} {{.Image}}'"),
    (
        "watchtower_log",
        f"docker logs {WATCHTOWER_CONTAINER} --tail 50 2>&1"
        " || docker logs $(docker ps -qf name=watchtower) --tail 50 2>&1",
    ),
    ("cgroup", "head -1 /proc/1/cgroup; head -1 /proc/self/cgroup"),
    ("dockerenv", "test -f /.dockerenv && echo IN_CONTAINER || echo NO_DOCKERENV; hostname"),
)

Resolver = Callable[[str], Awaitable[tuple[str, int]]]


@dataclass(frozen=True)
class Target:
    executor_id: str
    miner_hotkey: str


@dataclass
class CommandOutput:
    exit_status: int | None
    stdout: str
    stderr: str


@dataclass
class NodeReport:
    target: Target
    executor: ExecutorSSHInfo | None = None
    outputs: dict[str, CommandOutput] = field(default_factory=dict)
    error: str | None = None
    key_removed: bool | None = None


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


async def run_commands(ssh_client, outputs: dict[str, CommandOutput], per_command_timeout: float) -> None:
    """Fill ``outputs`` one command at a time, so a node budget that runs out keeps what was read."""
    for label, command in COMMANDS:
        try:
            result = await asyncio.wait_for(ssh_client.run(command), timeout=per_command_timeout)
            outputs[label] = CommandOutput(
                exit_status=result.exit_status,
                stdout=(result.stdout or "").strip(),
                stderr=(result.stderr or "").strip(),
            )
        except Exception as exc:  # one failed read must not hide the others
            outputs[label] = CommandOutput(exit_status=None, stdout="", stderr=f"{type(exc).__name__}: {exc}")


async def inspect_executor(
    *,
    target: Target,
    miner_service: MinerService,
    my_key: bittensor.Keypair,
    resolve_axon: Resolver,
    node_timeout: float,
) -> NodeReport:
    """Key submit → connect → read → key remove, for one executor. Never raises."""
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

    key_accepted = False
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
        key_accepted = True
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
                    await run_commands(ssh_client, report.outputs, per_command_timeout=max(5.0, node_timeout / 4))
        except TimeoutError:  # the node budget only; the key submit's own 30 s timeout is reported below
            report.error = f"node budget of {node_timeout:.0f} s ran out after {len(report.outputs)} of {len(COMMANDS)} reads"
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    finally:
        if key_accepted:
            report.key_removed = await miner_service._remove_ssh_key_via_rest(
                base_url=base_url,
                my_key=my_key,
                public_key=public_key,
                miner_hotkey=target.miner_hotkey,
                executor_id=target.executor_id,
                log_extra=log_extra,
            )
    return report


def summarise(report: NodeReport, hub_runner_digest: str | None) -> dict[str, str]:
    """The one-line facts per node; every value is a string ready for the table."""
    out = report.outputs
    registry = out.get("registry_config")
    mirrors = "?"
    if registry and registry.stdout:
        try:
            cfg = json.loads(registry.stdout)
            listed = cfg.get("Mirrors") or []
            mirrors = ", ".join(listed) if listed else "none"
        except ValueError:
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

    running_digest = "?"
    running = out.get("running_runner_digest")
    if running and running.exit_status == 0 and "@sha256:" in running.stdout:
        running_digest = "sha256:" + running.stdout.strip().rsplit("@sha256:", 1)[1]
    vs_hub = "?"
    if hub_runner_digest and running_digest.startswith("sha256:"):
        vs_hub = "current" if running_digest == hub_runner_digest else "STALE"

    def _short(digest: str) -> str:
        return digest[:19] if digest.startswith("sha256:") else digest

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
        "running_digest": _short(running_digest),
        "pulled_digest": _short(pulled_digest),
        "vs_hub": vs_hub,
        "shell_ran_in": where,
        "watchtower_last_line": last_line,
        "error": report.error or "",
    }


def render_node_block(report: NodeReport) -> str:
    head = f"### {report.target.executor_id} (miner {report.target.miner_hotkey})"
    if report.executor is not None:
        head += f" — {report.executor.address}:{report.executor.ssh_port} as {report.executor.ssh_username}"
    lines = [head, "```"]
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
    lines.append(f"key removed from miner: {report.key_removed}")
    lines.append("```")
    return "\n".join(lines)


def render_summary(rows: list[dict[str, str]], hub_runner_digest: str | None) -> str:
    columns = ["node", "mirror", "daemon_json", "running_digest", "pulled_digest", "vs_hub", "shell_ran_in", "watchtower_last_line", "error"]
    lines = [
        f"Docker Hub {RUNNER_IMAGE}:latest digest now: {hub_runner_digest or 'not fetched'}",
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
    for target in targets:
        lines.append(f"- {target.executor_id}  miner {target.miner_hotkey}")
    return "\n".join(lines)


async def fetch_hub_runner_digest() -> str | None:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        return await fetch_registry_digest(session, f"{RUNNER_IMAGE}:latest")


async def run(
    targets: list[Target],
    *,
    concurrency: int,
    node_timeout: float,
    resolve_axon: Resolver,
    miner_service: MinerService,
    my_key: bittensor.Keypair,
    hub_runner_digest: str | None,
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
    print(render_summary([summarise(r, hub_runner_digest) for r in reports], hub_runner_digest), file=out)
    return list(reports)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--executor-ids", metavar="FILE|id,id", help="file of `uuid [miner_hotkey]` lines, or ids inline; default: stdin")
    parser.add_argument("--miner-hotkey", help="miner hotkey for every id that has none on its line")
    parser.add_argument("--axon", action="append", default=[], metavar="HOTKEY=IP:PORT", help="skip the metagraph lookup for this miner")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per node for connect + reads (default 60)")
    parser.add_argument("--no-hub-digest", action="store_true", help="do not ask Docker Hub for the current runner digest")
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
        hub_digest = None if args.no_hub_digest else await fetch_hub_runner_digest()
        my_key = settings.get_bittensor_wallet().get_hotkey()
        resolve_axon = await resolve_axons_once({t.miner_hotkey for t in targets}, overrides)
        reports = await run(
            targets,
            concurrency=args.concurrency,
            node_timeout=args.timeout,
            resolve_axon=resolve_axon,
            miner_service=make_miner_service(),
            my_key=my_key,
            hub_runner_digest=hub_digest,
        )
        return 1 if any(r.error for r in reports) else 0

    return asyncio.run(_main())


if __name__ == "__main__":
    sys.exit(main())
