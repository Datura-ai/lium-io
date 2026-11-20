"""Refuses a prod validator release whose wire protocol is newer than the prod backend's.

The backend rejects message types, codes and fields it does not know, so a validator on a newer minor
protocol than the backend in prod sends replies the backend drops. backend_protocol_in_prod holds the
PROTOCOL_VERSION that lium-platform's prod backend vendors; raise it only after that backend is live in prod.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = HERE / "backend_protocol_in_prod"
PROTOCOL_INIT = HERE.parents[1] / "lium_protocol" / "lium_protocol" / "__init__.py"


def _minor(version: str) -> tuple[int, int]:
    major, minor = version.strip().split(".")[:2]
    return int(major), int(minor)


def refusal(validator_version: str, backend_version: str) -> str | None:
    """Why the release must wait, or None when the prod backend reads everything this validator sends."""
    if _minor(validator_version) > _minor(backend_version):
        return (
            f"validator speaks protocol {validator_version}, the prod backend {backend_version.strip()}: "
            f"release the backend first, then set {PIN.name} to its version"
        )
    return None


def main() -> int:
    match = re.search(r'^PROTOCOL_VERSION = "([^"]+)"', PROTOCOL_INIT.read_text(), re.M)
    if not match:
        print(f"protocol release gate: PROTOCOL_VERSION not found in {PROTOCOL_INIT}", file=sys.stderr)
        return 1
    reason = refusal(match.group(1), PIN.read_text())
    if reason:
        print(f"protocol release gate: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
