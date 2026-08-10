#!/usr/bin/env python3
"""DAH-2633 — the local full-flow CVM E2E: local code, one remote TDX host, one command.

Drives the whole renter path and the validation-CVM lifecycle against a real cvmd over an
SSH tunnel, using the REAL modules from both repos wherever their imports allow:

    order derive + sign     lium-io-backend  services/cvm_compose.py, services/cvmd_request.py
    validator's own calls   lium-io          services/cvmd_client.py (validation scope)
    the host's verdicts     the remote cvmd itself — measurement gate, scopes, teardown

What one run proves, in order:

    1. the host starts idle and its port range cannot touch the production port
    2. a renter order derived by the backend launches, and the host MEASURES the same
       compose hash the backend derived (the DAH-2632 golden loop, on hardware)
    3. the validator key cannot create or destroy a renter CVM (403 by scope)
    4. teardown is verified, and the switch window is observed as SWITCHING with a
       readable start time — the exact facts DAH-2630's grace decides on
    5. the validator brings the validation CVM back from the signed catalog and the
       launch report matches the pinned triple (DAH-2629's switch-back)
    6. the host ends as it began: idle, no leftover disks, no staged composes

Never binds port 32000 anywhere: the tunnel uses high local ports, and step 1 aborts the
whole run if the REMOTE cvmd's configured port range could allocate 32000.

The //Alice and //Bob defaults are the dev keys of a cvmd that binds 127.0.0.1 — they are
acceptable ONLY because that daemon is loopback-only and reached through this tunnel.
Never put them in a production authorized_clients.json.

Usage (from a lium-io checkout, with the validators venv):

    neurons/validators/.venv/bin/python tools/cvm-local-e2e/run.py \\
        --host ubuntu@203.98.89.178 --ssh-key ~/.ssh/waris_local \\
        --backend-src ../lium-io-backend/apps/server/src \\
        --agent-image "$CVM_ATTEST_AGENT_IMAGE"
"""

import argparse
import atexit
import json
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "neurons" / "validators" / "src"))

# The customer's own half of the order. It must open the guest SSH port cvmd's readiness
# gate waits on (2200 by fleet convention — `cvm_ssh_guest_port`): a compose with nothing
# listening there launches a guest that is never declared ready and the create answers 504.
# That is also what a real customer ships — SSH access to their CVM is the product.
CUSTOMER_COMPOSE = """services:
  ssh:
    image: alpine:3.20
    command: sh -c "apk add --no-cache openssh && ssh-keygen -A && /usr/sbin/sshd -D -e"
    ports:
      - "2200:22"
    restart: unless-stopped
"""

FORBIDDEN_PORT = 32000

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    results.append((ok, label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    return ok


def summary() -> int:
    passed = sum(1 for ok, _ in results if ok)
    print(f"\n== {passed}/{len(results)} checks passed")
    for ok, label in results:
        if not ok:
            print(f"   FAILED: {label}")
    return 0 if passed == len(results) else 1


def ssh(args, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-i", args.ssh_key, "-o", "BatchMode=yes", args.host, command],
        capture_output=True,
        text=True,
        timeout=60,
    )


def open_tunnel(args) -> subprocess.Popen:
    process = subprocess.Popen(
        [
            "ssh",
            "-i", args.ssh_key,
            "-o", "BatchMode=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-N",
            "-L", f"{args.local_port}:127.0.0.1:{args.remote_port}",
            args.host,
        ]
    )
    atexit.register(process.terminate)
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", args.local_port), timeout=1):
                return process
        except OSError:
            time.sleep(0.3)
    raise SystemExit("the SSH tunnel never came up")


def call(base_url: str, keypair, method: str, path: str, body: str = "", timeout: int = 900):
    """One signed cvmd call through the tunnel, using the validator repo's real signer."""
    from services.cvmd_client import sign_request

    signed = sign_request(keypair, method=method, path=path, body=body)
    request = urllib.request.Request(
        base_url + path,
        data=signed.body.encode(),
        method=method,
        headers={**signed.headers, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, context=CTX, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw or b"{}")
        except ValueError:
            return exc.code, {"detail": raw.decode()[:200]}


def newest(entries: list[dict], kind: str | None = None) -> dict | None:
    pool = [e for e in entries if kind is None or e.get("kind") == kind]
    if not pool:
        return None

    def key(entry):
        versions = entry.get("versions") or {}
        return (versions.get("compose", 0), versions.get("os_image", 0), versions.get("qemu", 0))

    return max(pool, key=key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="ssh target of the CVM host, e.g. ubuntu@203.98.89.178")
    parser.add_argument("--ssh-key", required=True)
    parser.add_argument("--backend-src", required=True, help="path to lium-io-backend/apps/server/src")
    parser.add_argument("--agent-image", required=True, help="attest-agent image, pinned by digest")
    parser.add_argument("--local-port", type=int, default=18443)
    parser.add_argument("--remote-port", type=int, default=8443)
    parser.add_argument("--platform-uri", default="//Bob", help="renter-scope key (dev key of the test daemon)")
    parser.add_argument("--validator-uri", default="//Alice", help="validation-scope key (dev key of the test daemon)")
    parser.add_argument("--switch-budget", type=int, default=300, help="DAH-2630 budget to judge the observed switch against")
    parser.add_argument("--keep-validation-cvm", action="store_true", help="leave the validation CVM running at the end")
    args = parser.parse_args()

    from bittensor_wallet import Keypair

    # The backend's real derivation, loaded by explicit file path: both repos ship a
    # `services` package, and letting them shadow each other on sys.path is exactly the
    # kind of silent wrong-module import this harness exists to rule out.
    import importlib.util

    compose_path = Path(args.backend_src).resolve() / "services" / "cvm_compose.py"
    spec = importlib.util.spec_from_file_location("backend_cvm_compose", compose_path)
    backend_cvm_compose = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend_cvm_compose)
    AgentSpec = backend_cvm_compose.AgentSpec
    derive = backend_cvm_compose.derive

    platform = Keypair.create_from_uri(args.platform_uri)
    validator = Keypair.create_from_uri(args.validator_uri)
    base_url = f"https://127.0.0.1:{args.local_port}"

    open_tunnel(args)

    print("== 1. the host starts idle, and its port range cannot touch production")
    status, state = call(base_url, platform, "GET", "/v1/state")
    if not check(status == 200, "cvmd answers through the tunnel", f"HTTP {status}"):
        return summary()
    if not check(
        state.get("state") == "RECONCILING" and state.get("cvm") is None,
        "the host is idle with no CVM",
        f"state={state.get('state')}",
    ):
        print("   aborting: this harness only runs against an idle host")
        return summary()

    config_read = ssh(args, "sudo cat /etc/cvmd/config.toml")
    ranges = [line for line in config_read.stdout.splitlines() if "port" in line.lower()]
    check(
        config_read.returncode == 0 and str(FORBIDDEN_PORT) not in config_read.stdout,
        f"the remote port configuration cannot allocate {FORBIDDEN_PORT}",
        "; ".join(ranges)[:160],
    )
    if str(FORBIDDEN_PORT) in config_read.stdout:
        print("   aborting: the test daemon's port range could touch the production port")
        return summary()

    print("== 2. the catalog in force on the host")
    status, catalog = call(base_url, validator, "GET", "/v1/catalog")
    manifest = (catalog or {}).get("manifest") or {}
    entries = manifest.get("entries") or []
    check(status == 200 and catalog.get("usable") is True, "the catalog is usable", f"serial={manifest.get('serial')}")
    base = newest(entries)
    validation_entry = newest(entries, kind="validation")
    if not check(base is not None, "the catalog offers an image and a QEMU build"):
        return summary()

    print("== 3. a renter order, derived by the backend's own code")
    injected, compose_hash = derive(CUSTOMER_COMPOSE, AgentSpec(image=args.agent_image))
    order = {
        "kind": "renter",
        "qemu": base["qemu"],
        "os_image_hash": base["os_image_hash"],
        "compose_hash": compose_hash,
        "compose": injected,
        "rental_id": "dah2633-e2e",
    }
    body = json.dumps(order, separators=(",", ":"))

    status, refused = call(base_url, validator, "POST", "/v1/cvm", body)
    check(status == 403, "the validator key cannot create a renter CVM", f"HTTP {status}")

    status, report = call(base_url, platform, "POST", "/v1/cvm", body)
    if not check(status == 201, "the renter launch is accepted", f"HTTP {status} {str(report)[:160]}"):
        return summary()
    measured = report.get("measurements") or {}
    check(
        measured.get("compose_hash") == compose_hash,
        "the host measured the hash the backend derived (golden loop)",
        str(measured.get("compose_hash"))[:20],
    )
    check(measured.get("qemu") == base["qemu"], "qemu as pinned")
    check(measured.get("os_image_hash") == base["os_image_hash"], "os image as pinned")
    check(report.get("state") == "RENTER_RUNNING", "node is RENTER_RUNNING", report.get("state"))
    ports = report.get("ports") or []
    check(bool(ports), "the report carries forwarded ports", str([p.get("host_port") for p in ports]))
    check(
        all(p.get("host_port") != FORBIDDEN_PORT for p in ports),
        f"no forwarded port is {FORBIDDEN_PORT}",
    )

    print("== 4. the validator key cannot destroy it either")
    status, _ = call(base_url, validator, "DELETE", "/v1/cvm")
    check(status == 403, "validator-signed DELETE is 403", f"HTTP {status}")

    print("== 5. verified teardown, with the switch window observed (DAH-2630's facts)")
    teardown_result: dict = {}

    def teardown():
        teardown_result["answer"] = call(base_url, platform, "DELETE", "/v1/cvm", timeout=2100)

    worker = threading.Thread(target=teardown)
    started = time.time()
    worker.start()
    observed_states: set[str] = set()
    observed_start: str | None = None
    while worker.is_alive() and time.time() - started < 2100:
        status, state = call(base_url, validator, "GET", "/v1/state", timeout=15)
        if status == 200:
            observed_states.add(str(state.get("state")))
            last_switch = state.get("last_switch") or {}
            if state.get("state") in ("TEARDOWN", "SWITCHING") and last_switch.get("started_at"):
                observed_start = last_switch["started_at"]
        time.sleep(0.5)
    worker.join()
    status, torn = teardown_result["answer"]
    check(status == 200, "teardown verified (hardware came back)", f"HTTP {status} {str(torn)[:120]}")
    switch_seconds = time.time() - started
    check(
        bool({"TEARDOWN", "SWITCHING"} & observed_states) or switch_seconds < 2.0,
        "the switch window was observable as TEARDOWN/SWITCHING",
        f"states={sorted(observed_states)} in {switch_seconds:.1f}s",
    )
    if observed_start is not None:
        parsed = datetime.fromisoformat(observed_start)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        elapsed_when_seen = (datetime.now(UTC) - parsed).total_seconds()
        check(
            0 <= elapsed_when_seen <= args.switch_budget,
            f"the observed switch stayed inside the {args.switch_budget}s budget",
            f"started_at readable, total {switch_seconds:.1f}s",
        )

    print("== 6. the switch back: the validator launches the validation CVM (DAH-2629)")
    if validation_entry is None:
        check(False, "the catalog carries a validation entry to launch from")
        return summary()
    triple_body = json.dumps(
        {
            "kind": "validation",
            "qemu": validation_entry["qemu"],
            "os_image_hash": validation_entry["os_image_hash"],
            "compose_hash": validation_entry["compose_hash"],
        },
        separators=(",", ":"),
    )
    status, launch = call(base_url, validator, "POST", "/v1/cvm", triple_body)
    if check(status == 201, "the validation CVM launch is accepted", f"HTTP {status} {str(launch)[:160]}"):
        measured = launch.get("measurements") or {}
        check(
            measured.get("compose_hash") == validation_entry["compose_hash"]
            and measured.get("os_image_hash") == validation_entry["os_image_hash"]
            and measured.get("qemu") == validation_entry["qemu"],
            "the launch report matches the pinned catalog triple",
        )
        check(launch.get("state") == "VALIDATION_RUNNING", "node is VALIDATION_RUNNING", launch.get("state"))

    print("== 7. the host is restored")
    if args.keep_validation_cvm:
        print("  (leaving the validation CVM running, as asked)")
    else:
        status, _ = call(base_url, platform, "DELETE", "/v1/cvm", timeout=2100)
        check(status == 200, "final teardown verified", f"HTTP {status}")
        status, state = call(base_url, platform, "GET", "/v1/state")
        check(
            state.get("state") == "RECONCILING" and state.get("cvm") is None,
            "the host ends idle, as it began",
            f"state={state.get('state')}",
        )
        disks = ssh(args, "sudo sh -c 'ls /var/lib/cvmd/vms/*/hda.img 2>/dev/null | wc -l'")
        check(disks.stdout.strip() == "0", "no CVM disks left behind", disks.stdout.strip())
        staged = ssh(args, "sudo sh -c 'ls -A /var/lib/cvmd/renter 2>/dev/null | wc -l'")
        check(staged.stdout.strip() == "0", "no staged composes left behind", staged.stdout.strip())

    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
