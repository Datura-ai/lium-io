"""The files `liumd run` reads on this host. The executor never runs liumd itself; the validator
does, over SSH, and an SSH exec session does not carry this container's environment, so every
setting liumd needs is a file.

    python src/liumd_host_files.py children <out>   # image build: the child manifest
    python src/liumd_host_files.py host             # container start: the host settings

`children` hashes the programs liumd may spawn at the paths it runs them from and writes the
`lium.children/1` manifest `/usr/local/bin/liumd` hands to the binary (`LIUMD_CHILD_MANIFEST_FILE`,
which only a keyless dev build honours). `host` writes:

- `/etc/liumd/validator_hotkeys`: the validator hotkeys this image trusts (`core.config`, fixed at
  build like the executor's own trust anchor), one SS58 per line;
- `/etc/liumd/miner_hotkey`: `MINER_HOTKEY_SS58_ADDRESS`;
- `/etc/liumd/ports.json`: `{"port_range", "port_mappings", "ssh_port"}` from `RENTING_PORT_RANGE`,
  `RENTING_PORT_MAPPINGS` and `SSH_PORT`;
- and makes sure `/var/lib/liumd/nonces` exists, writable by root only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ETC_DIR = Path("/etc/liumd")
NONCE_DIR = Path("/var/lib/liumd/nonces")
CHILDREN_SCHEMA = "lium.children/1"
# liumd's manifest names and the paths it spawns them from. `python` is the launcher of both
# wrappers, and each wrapper loads the `.so` liumd passes it with `--lib`.
CHILDREN: tuple[tuple[str, str], ...] = (
    ("verifyx", "/usr/lib/libverifyx.so"),
    ("inspector", "/usr/lib/libinspector.so"),
    ("matmul", "/usr/lib/libdmcompverify.so"),
    ("python", "/root/app/.venv/bin/python"),
    ("matmul_script", "/root/app/src/decrypt_challenge.py"),
    ("verifyx_script", "/root/app/src/verifyx_executor.py"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def children_manifest(root: Path = Path("/")) -> dict:
    """The manifest of `CHILDREN` as found under `root` (the image root; a test's tree). Paths are
    followed the way liumd opens them, so the venv's `python` symlink pins the interpreter it
    points at. A child that is missing fails the build: the image would ship a liumd that can run
    none of the steps that need it."""
    children = []
    for name, path in CHILDREN:
        found = root / path.lstrip("/")
        if not found.is_file():
            raise FileNotFoundError(f"liumd child {name!r}: no file at {path}")
        children.append({"name": name, "path": path, "sha256": sha256_file(found)})
    return {"schema": CHILDREN_SCHEMA, "children": children}


def validator_hotkeys_text(hotkeys: dict[str, str]) -> str:
    return "".join(f"{ss58}\n" for ss58 in hotkeys.values())


def miner_hotkey_text(miner_hotkey: str) -> str:
    return f"{miner_hotkey.strip()}\n"


def ports_json_text(port_range: str | None, port_mappings: str | None, ssh_port: int) -> str:
    return (
        json.dumps(
            {
                "port_range": port_range or None,
                "port_mappings": port_mappings or None,
                "ssh_port": ssh_port,
            }
        )
        + "\n"
    )


def write_atomic(path: Path, text: str, mode: int = 0o644) -> None:
    """Write through a temporary file in the same directory, so liumd never reads half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_host_files(
    *,
    etc_dir: Path,
    nonce_dir: Path,
    validator_hotkeys: dict[str, str],
    miner_hotkey: str,
    port_range: str | None,
    port_mappings: str | None,
    ssh_port: int,
) -> None:
    write_atomic(etc_dir / "validator_hotkeys", validator_hotkeys_text(validator_hotkeys))
    write_atomic(etc_dir / "miner_hotkey", miner_hotkey_text(miner_hotkey))
    write_atomic(etc_dir / "ports.json", ports_json_text(port_range, port_mappings, ssh_port))
    nonce_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(nonce_dir, 0o700)


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "children":
        write_atomic(Path(argv[1]), json.dumps(children_manifest(), indent=1) + "\n")
        return 0
    if argv == ["host"]:
        # Imported here: `children` runs at image build, where the settings' required
        # MINER_HOTKEY_SS58_ADDRESS is not set.
        from core.config import VALIDATOR_HOTKEYS_SS58, settings

        write_host_files(
            etc_dir=ETC_DIR,
            nonce_dir=NONCE_DIR,
            validator_hotkeys=VALIDATOR_HOTKEYS_SS58,
            miner_hotkey=settings.MINER_HOTKEY_SS58_ADDRESS,
            port_range=settings.RENTING_PORT_RANGE,
            port_mappings=settings.RENTING_PORT_MAPPINGS,
            ssh_port=settings.SSH_PORT,
        )
        return 0
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
