"""A renter CVM's stack, staged from the order instead of read from the catalog (DAH-2580).

The validation CVM and the renter CVM differ in exactly one input. Both run an approved OS
image on an approved QEMU build; only the compose is different, and only the renter's is
*derived* — the customer's docker-compose with the attest-agent injected into it, built by the
backend per order (DAH-2579). A manifest signed before the order existed cannot carry it.

So this module writes that one input to disk and hands back an `Artifact`, which lets the whole
launch path — prepare, MEASURE, launch, confirm — stay the single path it already was. Nothing
downstream needs to know where the compose came from.

**What still authorizes the launch.** Three things, and none of them is the staging below:

  * the OS image and the QEMU build are resolved out of the signed catalog, so a host cannot
    pair the customer's compose with an image nobody approved;
  * the request is signed by the platform key — `Scope.RENTER` — so a validator, a miner, or
    anyone who has reached the host's network cannot ask for a renter CVM at all;
  * `cvm/measure.py` hashes the `app-compose.json` that `setup_instance` actually wrote and
    refuses to start QEMU unless it equals the `compose_hash` the request named. That is what
    makes the text below evidence rather than a suggestion: a staged compose that has been
    tampered with — in transit, or on this disk between staging and prepare — measures to
    something else and never runs.

The staging directory belongs to the launch, not to the host, so it is removed when the CVM is
torn down. Leaving it would leave a customer's compose readable on a node they no longer hold.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from cvmd.atomic import write_bytes_durable
from cvmd.catalog import Artifact, CatalogError

logger = logging.getLogger(__name__)

COMPOSE_FILE = "compose.yml"
INIT_SCRIPT_FILE = "init_script.sh"
PRE_LAUNCH_SCRIPT_FILE = "pre_launch_script.sh"

# Same modes as the catalog's own materialized artifacts: read by root, which runs dstack.py,
# and by nobody else. A renter's compose can name registry credentials and workload env.
_FILE_MODE = 0o640
_DIR_MODE = 0o750


@dataclass(frozen=True)
class RenterOrder:
    """One customer order, in the terms that decide what gets measured.

    Every field here lands in `shared/app-compose.json`, which is what `compose_hash` is taken
    over. Sizing does not appear for the same reason it does not appear in a validation
    request: vCPUs, memory, disk and GPUs are provider configuration read from this host's
    config.toml, never something a request may choose.
    """

    qemu: str
    os_image_hash: str
    compose_hash: str
    compose: str
    init_script: str | None = None
    pre_launch_script: str | None = None
    local_key_provider: bool = True
    enable_logs: bool = False
    enable_sysinfo: bool = False
    # Carried through to the instance record and the launch report so an operator can match a
    # running CVM back to the rental it belongs to. Not measured, and deliberately so: two
    # rentals of the same stack must measure identically or the attestation would be useless.
    rental_id: str | None = None


def stage(order: RenterOrder, *, base: Artifact, directory: Path) -> Artifact:
    """Write the order's compose and scripts under `directory`, and describe them as an Artifact.

    `base` is the catalog entry that approved this OS image and QEMU build; everything except
    the compose comes from it. The returned artifact's `compose_hash` is the value the *request*
    named — not one computed here — because the only comparison that decides anything is
    `measure.assert_matches`, between that request and the file dstack actually writes.
    Computing a second hash here would put dstack's byte rules in one more place to drift.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    except OSError as exc:
        raise CatalogError(f"cannot create the renter staging directory {directory}: {exc}")

    compose_path = _write(directory / COMPOSE_FILE, order.compose)
    init_script = _write_optional(directory / INIT_SCRIPT_FILE, order.init_script)
    pre_launch_script = _write_optional(directory / PRE_LAUNCH_SCRIPT_FILE, order.pre_launch_script)

    logger.info("staged the renter compose for this order at %s", compose_path)
    return Artifact(
        # Names the base entry it was built on, so a launch report says which approved image and
        # QEMU build the order ran on rather than only that it was a renter CVM.
        id=f"renter:{base.id}",
        kind="renter",
        qemu=base.qemu,
        os_image_hash=base.os_image_hash,
        compose_hash=order.compose_hash,
        os_image_path=base.os_image_path,
        compose_path=compose_path,
        init_script=init_script,
        pre_launch_script=pre_launch_script,
        local_key_provider=order.local_key_provider,
        enable_logs=order.enable_logs,
        enable_sysinfo=order.enable_sysinfo,
    )


def discard(directory: Path | None) -> None:
    """Remove a staging directory. Never raises — it runs on the teardown path."""
    if directory is None:
        return
    shutil.rmtree(directory, ignore_errors=True)


def _write(path: Path, content: str) -> Path:
    try:
        write_bytes_durable(path, content.encode(), mode=_FILE_MODE)
    except OSError as exc:
        raise CatalogError(f"cannot write {path} for this renter order: {exc}")
    return path


def _write_optional(path: Path, content: str | None) -> Path | None:
    if content is None:
        return None
    return _write(path, content)
