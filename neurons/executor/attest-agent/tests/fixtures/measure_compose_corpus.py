#!/usr/bin/env python3
"""Measure the backend's derived corpus with the REAL dstack.py, and emit the golden file.

The other half of the DAH-2579 loop. `lium-io-backend`'s `services/cvm_compose.py` claims
to reproduce dstack's `app-compose.json` byte for byte; nothing in that repo can confirm it,
because dstack lives here. This runs the actual `DStackManager.setup_instance` over every
case and records what it actually wrote.

    lium-io-backend  src/test/fixtures/generate_cvm_compose_corpus.py  ->  corpus.json
    lium-io          this script                                      ->  golden.json
    lium-io-backend  test_dah2579_compose_hash.py                     compares them

Deliberately calls `setup_instance` rather than re-implementing its JSON. A second
implementation here would agree with the backend's for exactly as long as both were wrong
in the same way — which is the failure mode the golden file exists to catch.

    python measure_compose_corpus.py <corpus.json> > cvm_compose_golden.json

The golden file is committed in **lium-io-backend**, next to the test that reads it — one
copy with one owner. A second copy here would be a second thing to keep in step, and the
one that went stale would be the one nobody was running.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "dstacktee" / "scripts"


def load_dstack():
    """Import dstack.py as a library — the same way cvmd does, for the same reason."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        import dstack
    except ImportError as exc:  # pragma: no cover - a missing checkout, not a code path
        raise SystemExit(f"cannot import dstack.py from {SCRIPTS}: {exc}") from exc
    return dstack


def measure(dstack, case: dict, workdir: Path) -> tuple[str, str]:
    """Write one VM directory with the real launcher and hash what it produced."""
    compose_path = workdir / "docker-compose.yml"
    compose_path.write_text(case["injected_compose"], newline="")

    flags = case["flags"]
    scripts = {}
    for name in ("init_script", "pre_launch_script"):
        if flags.get(name):
            path = workdir / f"{name}.sh"
            path.write_text(flags[name], newline="")
            scripts[name] = str(path)

    # An image directory has to exist because `setup_instance` writes its basename into
    # vm-manifest.json — but nothing about it reaches app-compose.json, which is the only
    # file this script measures. An empty directory is enough and keeps the corpus free of
    # a gigabyte-scale dependency.
    image_dir = workdir / "dstack-nvidia-0.5.11"
    image_dir.mkdir()
    (image_dir / "metadata.json").write_text(
        '{"bios": "", "kernel": "", "cmdline": "", "initrd": "", "rootfs": ""}'
    )
    (image_dir / "digest.txt").write_text("a" * 64 + "\n")

    vm_dir = workdir / "vm"
    namespace = argparse.Namespace(
        compose_file=str(compose_path),
        dir=str(vm_dir),
        image=str(image_dir),
        vcpus=2,
        memory="2G",
        disk="20G",
        gpu=[],
        port=[],
        local_key_provider=flags.get("local_key_provider", True),
        enable_logs=flags.get("enable_logs", False),
        enable_sysinfo=flags.get("enable_sysinfo", False),
        init_script=scripts.get("init_script"),
        pre_launch_script=scripts.get("pre_launch_script"),
        env_file=None,
        pin_numa=False,
        hugepages=False,
    )
    dstack.DStackManager().setup_instance(namespace)

    produced = (vm_dir / "shared" / "app-compose.json").read_bytes()
    return produced.decode(), hashlib.sha256(produced).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a compose corpus with real dstack.py")
    parser.add_argument("corpus", type=Path, help="cvm_compose_corpus.json from the backend")
    args = parser.parse_args()

    dstack = load_dstack()
    corpus = json.loads(args.corpus.read_text())

    golden = []
    for case in corpus["cases"]:
        with tempfile.TemporaryDirectory() as tmp:
            app_compose, digest = measure(dstack, case, Path(tmp))
        golden.append(
            {
                "id": case["id"],
                "app_compose": app_compose,
                "compose_hash": digest,
                "agrees_with_backend": digest == case["derived_compose_hash"],
            }
        )

    disagreed = [entry["id"] for entry in golden if not entry["agrees_with_backend"]]
    print(
        json.dumps(
            {
                "version": 1,
                "source": "dstacktee/scripts/dstack.py DStackManager.setup_instance",
                "agent_image": corpus["agent_image"],
                "cases": golden,
            },
            indent=2,
        )
    )
    if disagreed:
        # Loud on stderr, and the file is still written: the disagreement IS the finding,
        # and a reader needs both sides of it to see where the bytes diverged.
        print(f"DISAGREEMENT on: {', '.join(disagreed)}", file=sys.stderr)
        raise SystemExit(1)
    print(f"all {len(golden)} cases agree with the backend's derivation", file=sys.stderr)


if __name__ == "__main__":
    main()
