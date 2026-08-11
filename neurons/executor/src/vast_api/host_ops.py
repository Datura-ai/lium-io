import logging
import subprocess

from vast_api.config import VastSettings
from vast_api.errors import ApiFailure

logger = logging.getLogger(__name__)

NSENTER = ["nsenter", "-t", "1", "-m", "-n", "--"]


class HostOps:
    """Host-namespace operations via nsenter (the shell runs privileged + pid=host).

    Deliberately has NO methods touching host /etc/docker/daemon.json or systemd —
    that is a hard safety rail, not an omission.
    """

    def __init__(self, settings: VastSettings):
        self.settings = settings

    def run(self, args: list[str], check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
        # run one command inside the host mount+net namespaces
        cmd = NSENTER + args
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ApiFailure("host_command_failed", f"{args[0]} timed out after {timeout}s")
        if check and result.returncode != 0:
            raise ApiFailure(
                "host_command_failed",
                f"{' '.join(args)} rc={result.returncode}: {result.stderr.strip()[:500]}",
            )
        return result

    def path_exists(self, path: str) -> bool:
        return self.run(["test", "-e", path], check=False).returncode == 0

    def data_root_mounted(self) -> bool:
        return self.run(["mountpoint", "-q", self.settings.DATA_ROOT_MOUNT], check=False).returncode == 0

    def _existing_loop(self) -> str | None:
        # already-attached loop device for the data-root image, if any
        result = self.run(["losetup", "-j", self.settings.DATA_ROOT_IMG], check=False)
        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        return first_line.split(":")[0] or None

    def _image_has_filesystem(self) -> bool:
        # blkid rc 2 is the documented "nothing found"; any other non-zero rc is a
        # broken blkid and must never green-light mkfs.xfs -f over live data
        rc = self.run(["blkid", self.settings.DATA_ROOT_IMG], check=False).returncode
        if rc == 0:
            return True
        if rc == 2:
            return False
        raise ApiFailure("host_command_failed", f"blkid failed rc={rc}")

    def ensure_data_root(self) -> None:
        # build the loop-XFS nested data-root per SUCCESS-PATH phase 1 step 1, idempotently
        s = self.settings
        if self.data_root_mounted():
            return
        if not self.path_exists(s.DATA_ROOT_IMG):
            self.run(["truncate", "-s", f"{s.DATA_ROOT_SIZE_GB}G", s.DATA_ROOT_IMG])
        # mkfs keyed on filesystem presence, not file existence: a crash between
        # truncate and mkfs must not wedge every later run
        if not self._image_has_filesystem():
            self.run(["mkfs.xfs", "-q", "-f", s.DATA_ROOT_IMG])
        loop = self._existing_loop()
        attached_fresh = loop is None
        if attached_fresh:
            loop = self.run(["losetup", "-f", "--show", s.DATA_ROOT_IMG]).stdout.strip()
        self.run(["mkdir", "-p", s.DATA_ROOT_MOUNT])
        try:
            self.run(["mount", "-o", "prjquota", loop, s.DATA_ROOT_MOUNT])
        except ApiFailure:
            if attached_fresh:
                self.run(["losetup", "-d", loop], check=False)  # don't leak the loop device
            raise

    def dump_dmi(self) -> None:
        # dmidecode refuses to overwrite an existing dump, and the target file is
        # bind-mounted (inode pinned) into a running vast-uns — so dump to a temp
        # path and copy the CONTENT in place, keeping the inode
        target = self.settings.DMI_BIN_HOST
        tmp = f"{target}.tmp"
        self.run(["rm", "-f", tmp])
        self.run(["dmidecode", "--dump-bin", tmp])
        self.run(["sh", "-c", f"cat {tmp} > {target} && rm -f {tmp}"])

    def local_ips(self) -> list[str]:
        # all host interface addresses — used to pick this box's machine on a shared account
        result = self.run(["hostname", "-I"], check=False)
        return result.stdout.split()

    def _query_nvidia_smi(self, query: str, fields: list[str]) -> list[dict]:
        # ghost GPUs print "[Unknown Error]" for numeric fields — refuse loudly, not a bare 500
        result = self.run(["nvidia-smi", query, "--format=csv,noheader,nounits"])
        rows = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                rows.append({name: int(part.strip()) for name, part in zip(fields, line.split(","))})
            except ValueError:
                raise ApiFailure(
                    "host_command_failed", f"nvidia-smi returned non-numeric fields: {line!r}"[:300]
                )
        return rows

    def gpu_stats(self) -> list[dict]:
        # per-GPU memory + utilization from the host driver
        return self._query_nvidia_smi(
            "--query-gpu=index,memory.used,utilization.gpu", ["idx", "mem_mib", "util_pct"]
        )

    def gpu_compute_apps(self) -> list[dict]:
        # running compute processes on any GPU (the Lium filler shows up here)
        return self._query_nvidia_smi("--query-compute-apps=pid,used_memory", ["pid", "mem_mib"])
