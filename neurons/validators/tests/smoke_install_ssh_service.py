#!/usr/bin/env python3
"""Smoke test for DockerService SSH bootstrap on multiple Linux images.

This script connects to a localhost executor via SSH, starts a test container for each
configured image, runs DockerService.install_open_ssh_server_and_start_ssh_service,
injects an ephemeral SSH key, and verifies SSH access to the container.

pdm run tests/smoke_install_ssh_service.py \
    --ssh-private-key /home/pyon/.ssh/id_ed25519
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import asyncssh

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.docker_service import DockerService
from services.ssh_service import SSHService

DEFAULT_IMAGES = [
    "ubuntu:22.04",
    "alpine:3.19",
    "fedora:40",
    "amazonlinux:2",
]


@dataclass
class SmokeResult:
    image: str
    container_name: str
    success: bool
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test install_open_ssh_server_and_start_ssh_service "
            "across distro images."
        ),
    )
    parser.add_argument(
        "--ssh-host",
        default="127.0.0.1",
        help="Executor SSH host (default: 127.0.0.1)",
    )
    parser.add_argument("--ssh-port", type=int, default=22, help="Executor SSH port (default: 22)")
    parser.add_argument(
        "--ssh-user",
        default=getpass.getuser(),
        help="Executor SSH username (default: current user)",
    )
    parser.add_argument(
        "--ssh-private-key",
        required=True,
        help="Path to SSH private key used for localhost executor login",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=DEFAULT_IMAGES,
        help="Linux images to test. Defaults to apt/apk/dnf/yum coverage set.",
    )
    return parser.parse_args()


def validate_private_key(private_key_path: Path) -> str:
    if not private_key_path.exists() or not private_key_path.is_file():
        raise FileNotFoundError(f"SSH private key not found: {private_key_path}")

    private_key = private_key_path.read_text()
    try:
        asyncssh.import_private_key(private_key)
    except Exception as exc:  # pragma: no cover - defensive validation
        raise ValueError(f"Invalid SSH private key at {private_key_path}: {exc}") from exc

    return private_key


def parse_published_ssh_port(docker_port_output: str) -> int:
    for line in docker_port_output.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.search(r":(\d+)$", line)
        if match:
            return int(match.group(1))

    raise ValueError(f"Unable to parse SSH port from docker output: {docker_port_output!r}")


async def run_command_or_raise(
    ssh_client: asyncssh.SSHClientConnection,
    command: str,
    context: str,
):
    result = await ssh_client.run(command)
    if result.exit_status != 0:
        raise RuntimeError(
            f"{context} failed (exit={result.exit_status})\n"
            f"command: {command}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return result


async def smoke_test_image(
    docker_service: DockerService,
    ssh_client: asyncssh.SSHClientConnection,
    image: str,
    verify_host: str,
) -> SmokeResult:
    image_slug = image.replace("/", "-").replace(":", "-")
    container_name = f"ssh-smoke-{image_slug}-{uuid4().hex[:8]}"
    container_name_quoted = shlex.quote(container_name)

    try:
        run_command = (
            f"/usr/bin/docker run -d --name {container_name_quoted} -p 0:22 "
            f"{shlex.quote(image)} sh -c 'while true; do sleep 3600; done'"
        )
        await run_command_or_raise(ssh_client, run_command, f"docker run for {image}")

        await docker_service.install_open_ssh_server_and_start_ssh_service(
            ssh_client=ssh_client,
            container_name=container_name,
            log_tag=f"smoke_{container_name}",
            log_extra={"image": image, "container_name": container_name},
        )
        
        ssh_service = SSHService()
        private_key, public_key = ssh_service.generate_keypair()
        public_key_quoted = shlex.quote(public_key.strip())

        add_key_command = (
            f"/usr/bin/docker exec -i {container_name_quoted} sh -c "
            f"\"mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
            f"echo {public_key_quoted} >> /root/.ssh/authorized_keys && "
            f"chmod 600 /root/.ssh/authorized_keys\""
        )
        await run_command_or_raise(ssh_client, add_key_command, f"add ssh key for {image}")

        port_result = await run_command_or_raise(
            ssh_client,
            f"/usr/bin/docker port {container_name_quoted} 22/tcp",
            f"resolve mapped port for {image}",
        )
        mapped_port = parse_published_ssh_port(port_result.stdout)

        container_pkey = asyncssh.import_private_key(private_key)
        async with asyncssh.connect(
            host=verify_host,
            port=mapped_port,
            username="root",
            client_keys=[container_pkey],
            known_hosts=None,
        ) as container_ssh:
            verification = await container_ssh.run("echo smoke-ok")
            if verification.exit_status != 0 or verification.stdout.strip() != "smoke-ok":
                raise RuntimeError(
                    f"SSH verification failed for {image}. "
                    f"stdout={verification.stdout!r}, stderr={verification.stderr!r}"
                )

        return SmokeResult(
            image=image,
            container_name=container_name,
            success=True,
            message=f"PASS (container={container_name}, port={mapped_port})",
        )
    except Exception as exc:
        return SmokeResult(
            image=image,
            container_name=container_name,
            success=False,
            message=f"FAIL ({exc})",
        )
    finally:
        await ssh_client.run(
            f"/usr/bin/docker rm -f {container_name_quoted} >/dev/null 2>&1 || true"
        )


async def run_smoke_tests(args: argparse.Namespace) -> int:
    private_key_path = Path(args.ssh_private_key).expanduser().resolve()
    private_key = validate_private_key(private_key_path)

    docker_service = DockerService(
        ssh_service=None,
        redis_service=None,
        port_mapping_dao=None,
        attestation_service=None,
    )

    results: list[SmokeResult] = []

    async with asyncssh.connect(
        host=args.ssh_host,
        port=args.ssh_port,
        username=args.ssh_user,
        client_keys=[asyncssh.import_private_key(private_key)],
        known_hosts=None,
    ) as ssh_client:
        for image in args.images:
            print(f"[RUN ] {image}")
            result = await smoke_test_image(
                docker_service=docker_service,
                ssh_client=ssh_client,
                image=image,
                verify_host=args.ssh_host,
            )
            status = "PASS" if result.success else "FAIL"
            print(f"[{status}] {image} -> {result.message}")
            results.append(result)

    passed = sum(1 for result in results if result.success)
    failed = len(results) - passed
    print("\nSummary")
    print(f"- total: {len(results)}")
    print(f"- passed: {passed}")
    print(f"- failed: {failed}")

    if failed:
        print("\nFailed images:")
        for result in results:
            if not result.success:
                print(f"- {result.image}: {result.message}")
        return 1

    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_smoke_tests(args))


if __name__ == "__main__":
    raise SystemExit(main())
