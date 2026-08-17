"""cvmd calls the real dstack.py, and calling it produces the same bytes the CLI produces.

This is the file that holds DAH-2576's central claim up: *import dstack.py as a library — no
behavior fork, zero measured bytes changed*. Two halves:

1. The import works against the file actually in this repo, with the entry points cvmd calls.
2. Preparing a VM directory through cvmd's plan and through `dstack.py new` — the same command
   line `lium-cvm.sh` builds — yields byte-identical results.

The second is the one that would catch a fork. `app-compose.json` **is** the compose hash, so
if these two paths ever diverge by a byte, the CVM measures as something the validator did not
whitelist, and it would fail at attestation time on a host rather than here.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from cvmd.catalog import Artifact
from cvmd.config import LaunchConfig
from cvmd.cvm import measure
from cvmd.dstack.loader import DStackUnavailable, load_dstack
from cvmd.dstack.plan import setup_namespace

VCPUS = 16
MEMORY = "64G"
DISK = "200G"
GPUS = ("19:00.0", "3b:00.0")
PORTS = ("tcp:0.0.0.0:12200:2200", "tcp:0.0.0.0:18001:8001")


class TestTheImport:
    def test_the_repo_s_dstack_imports_as_a_library(self, dstack):
        """It is a CLI script with a sibling import, not a package. It still has to import."""
        assert hasattr(dstack, "DStackManager")
        assert hasattr(dstack, "shutdown_instance")
        assert hasattr(dstack, "start_server")
        assert hasattr(dstack, "get_qemu_version_string")

    def test_the_sibling_module_resolves(self, dstack):
        """dstack.py does `import host_api`, which only works if its directory is importable."""
        assert dstack.host_api is not None

    def test_loading_twice_returns_one_module(self, dstack_scripts):
        """Two launches must not each re-execute the launcher into a separate module object."""
        assert load_dstack(dstack_scripts) is load_dstack(dstack_scripts)

    def test_a_directory_without_dstack_is_refused(self, tmp_path):
        with pytest.raises(DStackUnavailable, match="no dstack.py"):
            load_dstack(tmp_path)

    def test_dstack_without_its_sibling_is_refused_before_import(self, tmp_path):
        """Refused for the real reason, rather than as an opaque ImportError mid-launch."""
        (tmp_path / "dstack.py").write_text("import host_api\n")
        with pytest.raises(DStackUnavailable, match="host_api.py"):
            load_dstack(tmp_path)


def _artifact(compose: Path, image: Path, scripts: tuple[Path, Path]) -> Artifact:
    init, pre_launch = scripts
    return Artifact(
        id="validation-test",
        kind="validation",
        qemu="10.1.0",
        os_image_hash="d" * 64,
        compose_hash="0" * 64,  # replaced by the measured value in the tests that need it
        os_image_path=image,
        compose_path=compose,
        init_script=init,
        pre_launch_script=pre_launch,
        local_key_provider=True,
        enable_logs=False,
        enable_sysinfo=False,
    )


def _launch_config(env: Path) -> LaunchConfig:
    return LaunchConfig(vcpus=VCPUS, memory=MEMORY, disk=DISK, gpus=GPUS, ports=PORTS, env_file=env)


def _cli_arguments(artifact: Artifact, env_file: Path, vm_dir: Path) -> list[str]:
    """Exactly the flags `lium-cvm.sh new` passes, in its order (lium-cvm.sh:461-474)."""
    argv = [
        "new",
        str(artifact.compose_path),
        "--init-script",
        str(artifact.init_script),
        "--pre-launch-script",
        str(artifact.pre_launch_script),
        "--dir",
        str(vm_dir),
        "--image",
        str(artifact.os_image_path),
        "--vcpus",
        str(VCPUS),
        "--memory",
        MEMORY,
        "--disk",
        DISK,
        "--env-file",
        str(env_file),
    ]
    for gpu in GPUS:
        argv += ["--gpu", gpu]
    for port in PORTS:
        argv += ["--port", port]
    argv.append("--local-key-provider")
    return argv


class TestNoBehaviorFork:
    """Prepare the same CVM both ways and compare every file."""

    @pytest.fixture
    def prepared(
        self, dstack, dstack_scripts, tmp_path, compose_file, guest_scripts, env_file, image_dir
    ):
        artifact = _artifact(compose_file, image_dir, guest_scripts)

        via_library = tmp_path / "run-library" / "instance"
        via_library.parent.mkdir()
        namespace = setup_namespace(
            artifact=artifact, launch=_launch_config(env_file), vm_dir=via_library
        )
        dstack.DStackManager().setup_instance(namespace)

        via_cli = tmp_path / "run-cli" / "instance"
        via_cli.parent.mkdir()
        result = subprocess.run(
            [sys.executable, str(dstack_scripts / "dstack.py")]
            + _cli_arguments(artifact, env_file, via_cli),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return via_library, via_cli

    def test_app_compose_is_byte_identical(self, prepared):
        """The measured artifact. Equal bytes here is equal compose_hash, by definition."""
        library, cli = prepared
        assert (library / "shared" / "app-compose.json").read_bytes() == (
            cli / "shared" / "app-compose.json"
        ).read_bytes()

    def test_the_compose_hash_is_the_same(self, prepared):
        library, cli = prepared
        assert measure.compose_hash(library) == measure.compose_hash(cli)

    def test_every_prepared_file_matches(self, prepared):
        """Not only the measured one — the whole directory, so a divergence anywhere shows up.

        `vm-manifest.json` is exempt on one field only: it stamps `created_at_ms` from the
        clock, and the two runs happen milliseconds apart. Every other key is compared.
        """
        library, cli = prepared

        def relative(root):
            return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())

        assert relative(library) == relative(cli)

        for name in relative(library):
            left, right = library / name, cli / name
            if name.name == "vm-manifest.json":
                left_manifest = json.loads(left.read_text())
                right_manifest = json.loads(right.read_text())
                assert left_manifest.pop("created_at_ms") > 0
                assert right_manifest.pop("created_at_ms") > 0
                # The instance id is the directory name, which differs by construction.
                assert left_manifest.pop("id") == "instance"
                assert right_manifest.pop("id") == "instance"
                assert left_manifest == right_manifest
            else:
                assert left.read_bytes() == right.read_bytes(), f"{name} differs"

    def test_the_flags_that_shape_the_measurement_are_present(self, prepared):
        """Guards the pairing itself: a missing flag would make both sides equally wrong.

        Each assertion below is a value that only arrives if the corresponding flag was passed,
        so an omission in `plan.py` fails here rather than at attestation time on a host.
        """
        library, _ = prepared
        app_compose = json.loads((library / "shared" / "app-compose.json").read_text())

        assert app_compose["local_key_provider_enabled"] is True  # --local-key-provider
        assert app_compose["init_script"].startswith("#!/bin/sh")  # --init-script
        assert app_compose["pre_launch_script"].startswith("#!/bin/sh")  # --pre-launch-script
        assert "executor-runner" in app_compose["docker_compose_file"]  # the compose itself
        assert app_compose["public_logs"] is False  # --enable-logs absent
        assert "public_sysinfo" not in app_compose  # --enable-sysinfo absent

        manifest = json.loads((library / "vm-manifest.json").read_text())
        assert manifest["vcpu"] == VCPUS
        assert manifest["memory"] == 64 * 1024
        assert manifest["disk_size"] == 200
        assert [gpu["slot"] for gpu in manifest["gpus"]["gpus"]] == list(GPUS)
        assert [(p["from"], p["to"]) for p in manifest["port_map"]] == [
            (12200, 2200),
            (18001, 8001),
        ]

    def test_the_env_file_reaches_the_guest_but_not_the_measurement(self, prepared):
        """`--env-file` writes `.user-config`, which is outside app-compose.json.

        Worth pinning: it means host env can differ between two CVMs that must still measure
        identically, which is what lets sizing be provider configuration.
        """
        library, _ = prepared
        user_config = json.loads((library / "shared" / ".user-config").read_text())
        assert user_config["env_vars"]["SSH_PORT"] == "2200"

        app_compose = (library / "shared" / "app-compose.json").read_text()
        assert "SSH_PORT" not in app_compose
