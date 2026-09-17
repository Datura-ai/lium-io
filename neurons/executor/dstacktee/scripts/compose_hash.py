#!/usr/bin/env python3
"""Expected compose hash of a CVM created by `lium-cvm.sh new` from this checkout.

The validator whitelists the sha256 of shared/app-compose.json (dstack's compose_hash, measured
into RTMR3). That file is built from three measured inputs plus the runner digest stamped into
the compose: app/docker-compose.yml, app/init_script.sh, app/pre_launch_script.sh. Editing any
of them moves the hash; a hash the validator does not know scores the CVM zero (DAH-3602).

    compose_hash.py                     # the prod hash for the approved runner digest
    compose_hash.py --env staging       # another compose file
    compose_hash.py --digest sha256:…   # a runner digest under consideration
    compose_hash.py --release-notes     # the markdown section every executor release carries
    compose_hash.py --check run/vms/<name>/shared/app-compose.json   # exit 1 when a created CVM differs
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dstack import app_compose_json, build_app_compose  # noqa: E402

DSTACKTEE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = DSTACKTEE_DIR / "app"

# The executor-runner release providers pin as EXECUTOR_RUNNER_IMAGE_DIGEST in .env. Rotating it
# changes the measured compose, so the new hash goes into the validator whitelist
# (neurons/validators/src/services/const.py, TDX_WHITELIST["COMPOSE_HASH"]) in the same PR —
# neurons/validators/tests/test_tdx_compose_hash_whitelist.py fails otherwise.
APPROVED_RUNNER_IMAGE_DIGEST = (
    "sha256:8c07d3a91f8900bd3f0e19025fb7a2c32f82577385550187b28c7769060d18b7"
)

COMPOSE_FILES = {
    "prod": "docker-compose.yml",
    "staging": "docker-compose.staging.yml",
    "local": "docker-compose.local.yml",
}
DIGEST_PLACEHOLDER = "${EXECUTOR_RUNNER_IMAGE_DIGEST}"
RELEASE_NOTES_HEADING = "## CVM attestation"


def measured_app_compose(env: str = "prod", digest: str = APPROVED_RUNNER_IMAGE_DIGEST) -> str:
    """app-compose.json as `lium-cvm.sh new` writes it with default flags (--local-key-provider,
    no --enable-logs, no --enable-sysinfo) after stamping `digest` into the compose file."""
    compose = (
        (APP_DIR / COMPOSE_FILES[env])
        .read_text(encoding="utf-8")
        .replace(DIGEST_PLACEHOLDER, digest)
    )
    return app_compose_json(
        build_app_compose(
            compose,
            local_key_provider=True,
            enable_logs=False,
            enable_sysinfo=False,
            init_script=(APP_DIR / "init_script.sh").read_text(encoding="utf-8"),
            pre_launch_script=(APP_DIR / "pre_launch_script.sh").read_text(encoding="utf-8"),
        )
    )


def compose_hash(env: str = "prod", digest: str = APPROVED_RUNNER_IMAGE_DIGEST) -> str:
    return hashlib.sha256(measured_app_compose(env, digest).encode()).hexdigest()


def release_notes_section(env: str = "prod", digest: str = APPROVED_RUNNER_IMAGE_DIGEST) -> str:
    """The section a provider copies from: the digest for .env and the hash to check after `new`."""
    return "\n".join(
        [
            RELEASE_NOTES_HEADING,
            "",
            f"- Approved executor-runner digest (`EXECUTOR_RUNNER_IMAGE_DIGEST` in `.env`): `{digest}`",
            f"- Expected compose hash of a CVM created from this release: `{compose_hash(env, digest)}`",
            "",
            "Check after `sudo ./lium-cvm.sh new <name>` and before `run`:",
            "",
            "```bash",
            "sha256sum run/vms/<name>/shared/app-compose.json",
            "```",
            "",
            "The output must equal the expected compose hash. Any other value means the measured files "
            "or the digest differ from the release; the validator rejects the CVM and it scores zero.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", choices=sorted(COMPOSE_FILES), default="prod")
    parser.add_argument(
        "--digest", default=APPROVED_RUNNER_IMAGE_DIGEST, help="runner digest, sha256:<64-hex>"
    )
    parser.add_argument(
        "--release-notes", action="store_true", help="print the release-notes section"
    )
    parser.add_argument(
        "--check", metavar="APP_COMPOSE_JSON", help="compare a created CVM's app-compose.json"
    )
    args = parser.parse_args(argv)
    if not args.digest.startswith("sha256:") or len(args.digest) != len("sha256:") + 64:
        parser.error("--digest must be sha256:<64-hex>")
    expected = compose_hash(args.env, args.digest)
    if args.check:
        actual = hashlib.sha256(Path(args.check).read_bytes()).hexdigest()
        print(f"expected {expected}\nactual   {actual}")
        if actual != expected:
            print("MISMATCH: this CVM will not pass the validator whitelist", file=sys.stderr)
            return 1
        print("OK")
        return 0
    if args.release_notes:
        print(release_notes_section(args.env, args.digest))
        return 0
    print(expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
