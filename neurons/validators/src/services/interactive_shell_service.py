import shlex
import asyncssh
import logging
import hashlib
from core.utils import _m, get_extra_info
from services.node_host import public_host_sock

logger = logging.getLogger(__name__)


class InteractiveShellService:
    ssh_client: asyncssh.SSHClientConnection

    host: str
    username: str
    private_key: str
    port: int

    def __init__(self, host: str, username: str, private_key: str, port: int, known_hosts: asyncssh.SSHKnownHosts | None = None):
        self.host = host
        self.username = username
        self.private_key = private_key
        self.port = port
        self.known_hosts = known_hosts
        self.log_extra = {
            "host": host,
            "username": username,
            "port": port,
        }

    async def connect_asyncssh(self):
        # a hostname is resolved afresh on every connect, so a node whose IP changed is the same
        # node; the dial goes to the address that was checked, not to a second lookup
        dial_to_checked_address = await public_host_sock(self.host, self.port)
        pkey = asyncssh.import_private_key(self.private_key)
        self.ssh_client = await asyncssh.connect(
            host=self.host,
            port=self.port,
            **dial_to_checked_address,
            username=self.username,
            client_keys=[pkey],
            known_hosts=self.known_hosts,
            keepalive_interval=15,
            keepalive_count_max=3,
        )

    async def __aenter__(self):
        await self.connect_asyncssh()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Close the SSH connection to prevent file descriptor leaks
        try:
            if self.ssh_client:
                self.ssh_client.close()
                await self.ssh_client.wait_closed()
        except Exception as e:
            logger.error(_m(
                "Error: closing SSH connection",
                extra=get_extra_info({
                    **self.log_extra,
                    "error": str(e),
                }),
            ))

    async def read_file_content_over_scp(self, file_path: str) -> bytes:
        async with self.ssh_client.start_sftp_client() as sftp_client:
            async with sftp_client.open(file_path, 'rb') as file:
                file_content = await file.read()

        return file_content

    def get_md5_checksum_from_file_content(self, file_content: bytes):
        md5_hash = hashlib.md5()
        md5_hash.update(file_content)
        return md5_hash.hexdigest()

    def get_sha256_checksum_from_file_content(self, file_content: bytes):
        sha256_hash = hashlib.sha256()
        sha256_hash.update(file_content)
        return sha256_hash.hexdigest()

    async def get_checksums_over_scp(self, file_path: str):
        file_content = await self.read_file_content_over_scp(file_path)
        return f"{self.get_md5_checksum_from_file_content(file_content)}:{self.get_sha256_checksum_from_file_content(file_content)}"

    async def get_sha256_checksum_by_path(self, file_path: str) -> str:
        result = await self.ssh_client.run(
            f"sha256sum {shlex.quote(file_path)}",
            check=False,
        )
        if result.exit_status != 0:
            return ""

        output = result.stdout or ""
        parts = output.split(maxsplit=1)
        if not parts:
            return ""

        digest = parts[0]
        if len(digest) != 64:
            return ""

        return digest
