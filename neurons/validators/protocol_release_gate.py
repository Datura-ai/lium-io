"""Refuses a prod validator release whose wire protocol is newer than the prod backend's.

The backend rejects message types, codes and fields it does not know, so a validator on a newer minor
protocol than the backend in prod sends replies the backend drops. backend_protocol_in_prod holds the
PROTOCOL_VERSION that lium-platform's prod backend vendors; raise it only after that backend is live in prod.
validator_protocol holds the PROTOCOL_VERSION this validator speaks (a test keeps it equal to lium_protocol's), so the
check runs inside the image build, whose context is this directory only. The Dockerfile runs it on every build
unless ALLOW_PROTOCOL_AHEAD=1 is passed, which docker_build.sh does for non-latest tags and the staging workflow does.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = HERE / "backend_protocol_in_prod"
VALIDATOR_PROTOCOL = HERE / "validator_protocol"


def _minor(version: str) -> tuple[int, int]:
    major, minor = version.strip().split(".")[:2]
    return int(major), int(minor)


def refusal(validator_version: str, backend_version: str) -> str | None:
    """Why the release must wait, or None when the prod backend reads everything this validator sends."""
    if _minor(validator_version) > _minor(backend_version):
        return (
            f"validator speaks protocol {validator_version.strip()}, the prod backend {backend_version.strip()}: "
            f"release the backend first, then set {PIN.name} to its version"
        )
    return None


def main() -> int:
    reason = refusal(VALIDATOR_PROTOCOL.read_text(), PIN.read_text())
    if reason:
        print(f"protocol release gate: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
