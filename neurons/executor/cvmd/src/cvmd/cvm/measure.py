"""Measure what was prepared, and refuse to launch anything else.

The task's requirement is that a cvmd launch and a `lium-cvm.sh` launch of the same triple
produce identical MRTD and RTMR0-3. Calling the same launcher code makes that *likely*; this
module makes it *checked*. Between `setup_instance` writing the VM directory and QEMU being
started, cvmd hashes the artifacts it just produced and compares them with the triple the
caller named. A mismatch means the host would have attested as something the caller did not
ask for, so the launch does not happen.

The three values are the whole input to the measurements:

  compose_hash   sha256 of `<vm_dir>/shared/app-compose.json` — the same bytes and the same
                 digest dstack's own `app_compose_hash()` takes (vmm/src/app/qemu.rs).
                 Covers the compose file, the init and pre-launch scripts, and the
                 local-key-provider / logs / sysinfo flags, because `setup_instance` folds all
                 of them into that one file.
  os_image_hash  `<image_path>/digest.txt`, which is the value `run_instance` itself stamps
                 into the guest config as `os_image_hash`.
  qemu           `get_qemu_version_string()`, which is what lands in `vm_config["qemu_version"]`
                 and what the verifier reconstructs the expected measurements against.

Reading them from the artifacts rather than from the configuration is the point: configuration
says what was *intended*, and only the files say what was *built*.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

APP_COMPOSE = Path("shared") / "app-compose.json"
DIGEST_FILE = "digest.txt"

_CHUNK = 1024 * 1024


class MeasurementError(Exception):
    """An artifact could not be measured, or measured to something other than the request."""


@dataclass(frozen=True)
class Measurements:
    qemu: str
    os_image_hash: str
    compose_hash: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def compose_hash(vm_dir: Path) -> str:
    """sha256 of the measured app-compose.json, as the guest will report it."""
    path = vm_dir / APP_COMPOSE
    if not path.is_file():
        raise MeasurementError(f"{path} was not written — the VM directory is not prepared")
    return _sha256(path)


def os_image_hash(image_path: Path) -> str:
    """The image's own digest, read from the file `run_instance` reads it from."""
    path = image_path / DIGEST_FILE
    try:
        value = path.read_text().strip()
    except OSError as exc:
        raise MeasurementError(f"cannot read the OS image digest at {path}: {exc}") from exc
    if not value:
        raise MeasurementError(f"{path} is empty, so the OS image is unidentified")
    return value


def qemu_version(dstack: ModuleType) -> str:
    """The QEMU build string that ends up in the attested vm_config."""
    version = dstack.get_qemu_version_string()
    if not version:
        raise MeasurementError(
            "the configured QEMU did not report a version, so the build cannot be pinned"
        )
    return str(version)


def measure(*, dstack: ModuleType, vm_dir: Path, image_path: Path) -> Measurements:
    return Measurements(
        qemu=qemu_version(dstack),
        os_image_hash=os_image_hash(image_path),
        compose_hash=compose_hash(vm_dir),
    )


def assert_matches(actual: Measurements, expected: tuple[str, str, str]) -> None:
    """Compare against the requested (qemu, os_image_hash, compose_hash).

    Every difference is reported, not just the first: an operator who fixes one pin, relaunches
    and hits the next one has built a CVM twice to learn something one message could have said.
    """
    wanted = Measurements(*expected)
    differences = [
        f"{field}: requested {getattr(wanted, field)}, prepared {getattr(actual, field)}"
        for field in ("qemu", "os_image_hash", "compose_hash")
        if getattr(wanted, field) != getattr(actual, field)
    ]
    if differences:
        raise MeasurementError(
            "the prepared CVM would not measure as the requested triple, so it was not "
            "launched — " + "; ".join(differences)
        )
