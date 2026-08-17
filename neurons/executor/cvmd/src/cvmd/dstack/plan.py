"""Build the argument object `DStackManager.setup_instance` expects.

`setup_instance` takes an `argparse.Namespace` because its only caller today is dstack.py's own
CLI. Handing it a Namespace is what makes cvmd a *library caller* rather than a fork: the
alternative — refactoring dstack.py to take a dataclass — is a change to the file that shapes
attestation, and this task forbids that.

Every field below mirrors one flag in `lium-cvm.sh new`. The pairing is the contract: if the
shell path grows a flag that changes what gets measured, this module has to grow with it, and
the compose-hash gate in `cvm/measure.py` is what makes the omission fail loudly instead of
silently producing a differently-measured CVM.

    lium-cvm.sh new                       here
    ----------------------------------    -----------------------------------------
    <compose_file>                        artifact.compose_path
    --dir "$VMS_DIR/$cvm_name"            vm_dir
    --image "$IMAGE_DIR/$OS_IMAGE_NAME"   artifact.os_image_path
    --init-script "$INIT_SCRIPT"          artifact.init_script
    --pre-launch-script "$PRE..."         artifact.pre_launch_script
    --vcpus / --memory / --disk           launch config (provider-set, never requested)
    --gpu (repeated) / --port (repeated)  launch config
    --env-file "$env_file"                launch config
    --local-key-provider                  artifact.local_key_provider
    --enable-logs / --enable-sysinfo      artifact.enable_logs / enable_sysinfo
    (not passed)                          pin_numa / hugepages, both False by default —
                                          argparse's store_true default, which is what the
                                          shell path leaves them at
"""

import argparse
from pathlib import Path

from cvmd.catalog import Artifact
from cvmd.config import LaunchConfig


def _script_argument(path: Path | None) -> str | None:
    """dstack.py opens the script when the value is truthy, so absent must be None, not ''."""
    return str(path) if path is not None else None


def setup_namespace(
    *, artifact: Artifact, launch: LaunchConfig, vm_dir: Path
) -> argparse.Namespace:
    """The exact arguments `lium-cvm.sh new` would produce for this artifact and this host."""
    return argparse.Namespace(
        compose_file=str(artifact.compose_path),
        dir=str(vm_dir),
        image=str(artifact.os_image_path),
        vcpus=launch.vcpus,
        memory=launch.memory,
        disk=launch.disk,
        # dstack.py treats `--gpu all` as the string list ["all"]; anything else is a list of
        # PCI slots. An empty list is a legal configuration — a CVM with no passthrough.
        gpu=list(launch.gpus),
        port=list(launch.ports),
        local_key_provider=artifact.local_key_provider,
        enable_logs=artifact.enable_logs,
        enable_sysinfo=artifact.enable_sysinfo,
        init_script=_script_argument(artifact.init_script),
        pre_launch_script=_script_argument(artifact.pre_launch_script),
        env_file=str(launch.env_file) if launch.env_file is not None else None,
        pin_numa=launch.pin_numa,
        hugepages=launch.hugepages,
    )
