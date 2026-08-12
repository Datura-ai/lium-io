import logging
import secrets
from collections.abc import Callable

from vast_api.config import VastSettings
from vast_api.docker_ops import DockerOps
from vast_api.errors import ApiFailure, ApiRefusal, safe_error
from vast_api.host_ops import HostOps
from vast_api.runs import RunStore
from vast_api.vast_client import VastClient

logger = logging.getLogger(__name__)


class Stage:
    def __init__(
        self,
        name: str,
        check: Callable[[], bool] | None,
        do: Callable[[], None],
        verify: Callable[[], None] | None = None,
    ):
        self.name = name
        self.check = check
        self.do = do
        self.verify = verify


def run_ladder(stages: list[Stage], store: RunStore, run_id: str) -> bool:
    # execute the ladder in order, streaming stage events; first failure aborts
    for stage in stages:
        store.stage_started(run_id, stage.name)
        try:
            if stage.check is not None and stage.check():
                store.stage_finished(run_id, stage.name, "ok")
                continue
            stage.do()
            if stage.verify is not None:
                stage.verify()
            store.stage_finished(run_id, stage.name, "done")
        except (ApiRefusal, ApiFailure) as exc:
            store.stage_finished(run_id, stage.name, "failed", exc.detail)
            store.finish(run_id, "failed", error={"code": exc.code, "detail": exc.detail})
            return False
        except Exception as exc:
            logger.exception("stage %s crashed in run %s", stage.name, run_id)
            store.stage_finished(run_id, stage.name, "failed", safe_error(exc))
            store.finish(run_id, "failed", error={"code": "unexpected", "detail": safe_error(exc)})
            return False
    return True


def build_setup_ladder(
    settings: VastSettings,
    host: HostOps,
    docker_ops: DockerOps,
    vast: VastClient,
    machine_key: str,
    machine_id: int | None = None,
) -> tuple[list[Stage], dict]:
    # build the idempotent G0→G2 ladder; ctx carries network + machine_id between
    # stages. machine_key is backend-minted per setup call and rotates on every
    # mint (plan-key-split) — the box never holds the account key.
    ctx: dict = {"machine_id": machine_id, "machine_key": machine_key}

    # --- g0_gate: never build a standalone Vast host ---

    def g0_check() -> bool:
        if not docker_ops.executor_running():
            return False
        network = docker_ops.executor_network()
        if network is None:
            return False
        ctx["network"] = network
        return True

    def g0_do() -> None:
        raise ApiFailure(
            "g0_failed",
            f"{settings.EXECUTOR_CONTAINER_NAME} not running or executor network missing — "
            "Vast goes ON a Lium machine, never on a box of its own",
        )

    # --- image ---

    def image_check() -> bool:
        return docker_ops.image_exists()

    def image_verify() -> None:
        if not docker_ops.image_exists():
            raise ApiFailure("host_command_failed", f"{settings.VAST_UNS_IMAGE_TAG} missing after build")

    # --- container ---

    def container_check() -> bool:
        container = docker_ops.get_vast_uns()
        if container is None:
            return False
        if not docker_ops.vast_uns_has_state_mount(container):
            # identity lives only inside this container: recreating it would throw
            # away the registered machine — refuse instead of silently fixing
            raise ApiFailure(
                "state_mount_missing",
                "vast-uns runs without the /var/lib/vastai_kaalia state mount",
            )
        return container.status == "running"

    def container_do() -> None:
        container = docker_ops.get_vast_uns()
        if container is not None:
            container.remove(force=True)  # stopped container with the state mount — safe to recreate
        # mounts must exist before docker run: state dir, DMI dump, loop-XFS data-root
        host.run(["mkdir", "-p", settings.STATE_DIR_HOST])
        if not host.path_exists(settings.DMI_BIN_HOST):
            host.dump_dmi()
        host.ensure_data_root()
        host.reserve_publish_ports()
        docker_ops.run_vast_uns(ctx["network"])

    def container_verify() -> None:
        container = docker_ops.get_vast_uns()
        if container is None or container.status != "running":
            raise ApiFailure("host_command_failed", "vast-uns did not reach running state")

    # --- data_root ---

    def data_root_check() -> bool:
        return host.data_root_mounted()

    def data_root_verify() -> None:
        if not host.data_root_mounted():
            raise ApiFailure("host_command_failed", f"{settings.DATA_ROOT_MOUNT} is not a mountpoint")

    # --- nested_daemon ---

    def nested_daemon_check() -> bool:
        return (
            docker_ops.nested_daemon_matches_asset()
            and docker_ops.nested_docker_active()
            and docker_ops.nested_daemon_immutable()
        )

    def nested_daemon_verify() -> None:
        if not docker_ops.nested_docker_active():
            raise ApiFailure("host_command_failed", "nested docker is not active after install")
        # verify the file, not the exit code: nvidia-ctk without --in-place changes nothing
        rc, _ = docker_ops.exec_in_uns(
            ["grep", "-E", "no-cgroups *= *false", "/etc/nvidia-container-runtime/config.toml"]
        )
        if rc != 0:
            raise ApiFailure("host_command_failed", "no-cgroups=false not set in nvidia-container-runtime config")

    # --- gpu: the check IS the satisfaction test, twice around a daemon-reload ---

    def gpu_check() -> bool:
        if not docker_ops.nested_gpu_ok():
            return False
        rc, output = docker_ops.exec_in_uns(["systemctl", "daemon-reload"])
        if rc != 0:
            # a failed reload would make the ×2 gate pass vacuously
            raise ApiFailure("host_command_failed", f"daemon-reload failed in vast-uns: {output[:200]}")
        return docker_ops.nested_gpu_ok()

    def gpu_do() -> None:
        raise ApiFailure(
            "gpu_broken",
            "nested --runtime=nvidia container cannot see the GPU (or loses it after daemon-reload)",
        )

    # --- dmi_shim ---

    def dmi_shim_check() -> bool:
        rc, _ = docker_ops.exec_in_uns(["dmidecode", "-t", "system"])
        return rc == 0

    def dmi_shim_do() -> None:
        # dump_dmi rewrites the CONTENT in place (inode kept), so the ro bind-mount
        # inside the running vast-uns sees the fresh dump immediately
        host.dump_dmi()

    def dmi_shim_verify() -> None:
        rc, output = docker_ops.exec_in_uns(["dmidecode", "-t", "system"])
        if rc != 0:
            raise ApiFailure("host_command_failed", f"dmidecode shim failed in vast-uns: {output[:200]}")

    # --- kaalia ---

    def kaalia_check() -> bool:
        rc, _ = docker_ops.exec_in_uns(["test", "-d", "/var/lib/vastai_kaalia/latest"])
        return rc == 0

    def kaalia_do() -> None:
        # apt lists must be present or the installer rolls the whole install back
        rc, output = docker_ops.exec_in_uns(["apt-get", "update"])
        if rc != 0:
            raise ApiFailure("install_rollback", f"apt-get update failed in vast-uns: {output[:300]}")
        rc, output = docker_ops.exec_in_uns(
            # -nv (not -q) so a download failure carries wget's own error text into the run doc
            ["bash", "-c", "wget -nv https://console.vast.ai/install -O /root/vi2.py"]
        )
        if rc != 0:
            raise ApiFailure("install_rollback", f"kaalia installer download failed: {output[:300]}")
        # the key travels as an env var, never interpolated into the shell line
        rc, output = docker_ops.exec_in_uns(
            [
                "bash", "-c",
                f'cd /root && python3 vi2.py "$MACHINE_KEY" --no-libvirt --no-partitioning '
                f"--ports {settings.PORT_RANGE_START} {settings.PORT_RANGE_END}",
            ],
            env={"MACHINE_KEY": ctx["machine_key"]},
        )
        if rc != 0:
            raise ApiFailure("install_rollback", f"kaalia installer failed: {output[-500:]}")

    def kaalia_verify() -> None:
        rc, _ = docker_ops.exec_in_uns(["test", "-d", "/var/lib/vastai_kaalia/latest"])
        if rc != 0:
            raise ApiFailure("install_rollback", "latest/ never appeared — installer rolled back")

    # --- register ---

    def register_check() -> bool:
        # local-only: /machines/ visibility belongs to the backend (plan-key-split).
        # A typo'd requested machine_id also fails backend-side, before this run starts.
        rc, _ = docker_ops.exec_in_uns(["test", "-f", "/var/lib/vastai_kaalia/machine_id"])
        if rc != 0:
            return False
        rc, output = docker_ops.exec_in_uns(["cat", "/var/lib/vastai_kaalia/numeric_machine_id"])
        if rc != 0 or not output.strip().isdigit():
            return False
        persisted = int(output.strip())
        if ctx["machine_id"] is not None and persisted != ctx["machine_id"]:
            return False
        ctx["machine_id"] = persisted
        return True

    def register_do() -> None:
        rc, output = docker_ops.exec_in_uns(["cat", "/var/lib/vastai_kaalia/machine_id"])
        if rc == 0 and output.strip():
            machine_api_key = output.strip()
        else:
            machine_api_key = secrets.token_hex(32)
            docker_ops.write_file_in_uns("/var/lib/vastai_kaalia/machine_id", machine_api_key)
        response = vast.identify(ctx["machine_key"], machine_api_key)
        # the identify quirk: success:false while the machine id IS created
        numeric_id = response.get("machine_id")
        if numeric_id is None:
            raise ApiFailure("identify_rejected", f"identify returned no machine_id: {str(response)[:200]}")
        if ctx["machine_id"] is not None and numeric_id != ctx["machine_id"]:
            raise ApiFailure(
                "identify_rejected",
                f"identified as machine {numeric_id}, expected to adopt {ctx['machine_id']}",
            )
        ctx["machine_id"] = numeric_id
        # persist the numeric id in the state dir: public_ipaddr matching fails on
        # NAT'ed clouds (hyperstack), so this is the primary resolve key from now on
        docker_ops.write_file_in_uns("/var/lib/vastai_kaalia/numeric_machine_id", str(numeric_id))
        # restart so kaalia picks up the identity (installer-started daemon has none);
        # a wedged restart must not report success with the pre-identity daemon running
        rc, output = docker_ops.exec_in_uns(["systemctl", "restart", "vastai"])
        if rc != 0:
            raise ApiFailure("host_command_failed", f"vastai restart failed after identify: {output[:200]}")

    def register_verify() -> None:
        # appearance in /machines/ is verified by the backend after the run; locally
        # verify the identity landed and kaalia runs with it
        rc, output = docker_ops.exec_in_uns(["cat", "/var/lib/vastai_kaalia/numeric_machine_id"])
        if rc != 0 or output.strip() != str(ctx["machine_id"]):
            raise ApiFailure("identify_rejected", "numeric_machine_id not persisted after identify")
        rc, output = docker_ops.exec_in_uns(["systemctl", "is-active", "vastai"])
        if rc != 0:
            raise ApiFailure(
                "identify_rejected", f"vastai unit not active after identify: {output.strip()[:100]}"
            )

    # --- report: no check — re-reporting an already-reporting machine is harmless ---

    def report_do() -> None:
        # send_mach_info.py is unpacked from daemon.tar.gz by the freshly restarted
        # kaalia — on a first install it appears seconds to minutes after register,
        # so wait for it instead of failing the whole run (seen live on 147260)
        wait = (
            "for i in $(seq 1 24); do"
            " [ -f /var/lib/vastai_kaalia/send_mach_info.py ] && exit 0; sleep 5; done; exit 1"
        )
        rc, _ = docker_ops.exec_in_uns(["bash", "-c", wait])
        if rc != 0:
            raise ApiFailure("report_failed", "send_mach_info.py never unpacked within 120s")
        # a 403 ("Invalid or missing nonce") is a daemon state, not an API change:
        # restart vastai and give it time to check in and obtain a nonce before the
        # retry — an immediate retry still 403s (seen live on 147260).
        # the script lives in the kaalia root, NOT in latest/ (verified on a live install)
        cmd = ["bash", "-c", "cd /var/lib/vastai_kaalia && python3 send_mach_info.py"]
        output = ""
        for attempt in range(3):
            if attempt:
                docker_ops.exec_in_uns(["systemctl", "restart", "vastai"])
                docker_ops.exec_in_uns(["sleep", "30"])
            rc, output = docker_ops.exec_in_uns(cmd)
            if rc == 0 and "403" not in output:
                return
        raise ApiFailure("report_failed", f"send_mach_info.py failed 3 times: {output[-300:]}")

    # kaalia MUST precede nested_daemon/gpu: the nested daemon.json declares the
    # nvidia runtime through kaalia_docker_shim, which only exists after the kaalia
    # install (proven on the fresh-box exam 2026-08-11: gpu_broken by construction).
    stages = [
        Stage("g0_gate", g0_check, g0_do),
        Stage("image", image_check, docker_ops.build_image, image_verify),
        Stage("container", container_check, container_do, container_verify),
        Stage("data_root", data_root_check, host.ensure_data_root, data_root_verify),
        Stage("dmi_shim", dmi_shim_check, dmi_shim_do, dmi_shim_verify),
        Stage("kaalia", kaalia_check, kaalia_do, kaalia_verify),
        Stage("nested_daemon", nested_daemon_check, docker_ops.install_nested_daemon, nested_daemon_verify),
        Stage("gpu", gpu_check, gpu_do),
        Stage("register", register_check, register_do, register_verify),
        Stage("report", None, report_do),
    ]
    return stages, ctx
