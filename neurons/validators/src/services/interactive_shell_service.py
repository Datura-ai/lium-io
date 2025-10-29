import gc
import pexpect
import tempfile
import os
import re
import asyncssh
import asyncio
import logging
import hashlib
from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)


class InteractiveShellService:
    ssh_client: asyncssh.SSHClientConnection
    priv_key_path: str | None = None

    host: str
    username: str
    private_key: str
    port: int
    remote_dir: str | None = None

    def __init__(
        self,
        host: str,
        username: str,
        private_key: str,
        port: int,
        known_hosts: asyncssh.SSHKnownHosts | None = None,
    ):
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

    async def connect_interactive_shell(self):
        try:
            # self.i_shell = await self.loop.run_in_executor(None, self._spawn_interactive_shell)
            await asyncio.to_thread(self._spawn_interactive_shell)
        except Exception as e:
            logger.error(_m(
                "Error: connecting interactive shell",
                extra=get_extra_info({
                    **self.log_extra,
                    "error": str(e),
                }),
            ))

    def _spawn_interactive_shell(self):
        # Build the SSH command
        ssh_command = f"ssh -p {self.port}"

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(self.private_key.encode())
            temp_key_path = temp.name
            self.priv_key_path = temp_key_path

        ssh_command += f" -i {temp_key_path}"

        # Add options to match typical human behavior
        ssh_command += " -o BatchMode=no -o StrictHostKeyChecking=no"

        # Add the destination
        ssh_command += f" {self.username}@{self.host}"

        try:
            i_shell = pexpect.spawn(ssh_command, timeout=10)

            i = i_shell.expect(['password:', 'continue connecting', 'Last login', pexpect.EOF, pexpect.TIMEOUT], timeout=30)
            if i == 1:  # Host key verification
                i_shell.sendline('yes')

            # Wait for shell prompt
            i = i_shell.expect(['root@'], timeout=30)

            # close shell
            i_shell.sendline("exit")
            i_shell.expect(pexpect.EOF)
            i_shell.close()

            gc.collect()  # Run the garbage collector
        except pexpect.TIMEOUT:
            raise Exception("i-ssh connection Timeout")
        except pexpect.EOF:
            raise Exception("i-ssh connection EOF error")

    async def connect_asyncssh(self):
        pkey = asyncssh.import_private_key(self.private_key)
        self.ssh_client = await asyncssh.connect(
            host=self.host,
            port=self.port,
            username=self.username,
            client_keys=[pkey],
            known_hosts=self.known_hosts,
        )

    async def __aenter__(self):
        # await self.connect_interactive_shell()
        await self.connect_asyncssh()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self.clear_remote_directory()

            if self.priv_key_path and os.path.isfile(self.priv_key_path):
                os.remove(self.priv_key_path)
        except:
            pass

        # try:
        #     if self.i_shell:
        #         if await asyncio.to_thread(self.i_shell.isalive):
        #             await asyncio.to_thread(self.i_shell.terminate, force=True)

        #         await asyncio.to_thread(self.i_shell.close)
        # except Exception as e:
        #     logger.error(_m(
        #         "Error: close interactive shell",
        #         extra=get_extra_info({
        #             **self.log_extra,
        #             "error": str(e),
        #         }),
        #     ))

    async def upload_directory(
        self, local_dir: str, remote_dir: str
    ):
        if not self.ssh_client:
            return

        self.remote_dir = remote_dir

        await self.clear_remote_directory()

        await self.ssh_client.run(f"mkdir -p {remote_dir}")

        """Uploads a directory recursively to a remote server using AsyncSSH."""
        async with self.ssh_client.start_sftp_client() as sftp_client:
            for root, dirs, files in os.walk(local_dir):
                relative_dir = os.path.relpath(root, local_dir)
                remote_path = os.path.join(self.remote_dir, relative_dir)

                # Create remote directory if it doesn't exist
                result = await self.ssh_client.run(f"mkdir -p {remote_path}")
                if result.exit_status != 0:
                    raise Exception(f"Failed to create directory {remote_path}: {result.stderr}")

                # Upload files
                upload_tasks = []
                for file in files:
                    local_file = os.path.join(root, file)
                    remote_file = os.path.join(remote_path, file)
                    upload_tasks.append(sftp_client.put(local_file, remote_file))

                # Await all upload tasks for the current directory
                await asyncio.gather(*upload_tasks)

    async def clear_remote_directory(self):
        if not self.ssh_client or not self.remote_dir:
            return

        try:
            await self.ssh_client.run(f"rm -rf {self.remote_dir}", timeout=10)
            # await self.exec_shell_command(f"rm -rf {self.remote_dir}")
        except Exception as e:
            pass

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

    # async def get_checksums_by_path(self, file_path: str):
    #     md5_output = await self.exec_shell_command(f'md5sum {file_path}')
    #     sha256_output = await self.exec_shell_command(f'sha256sum {file_path}')

    #     # Extract the checksums from the command outputs
    #     md5_sum = md5_output.replace(file_path, '').strip() if md5_output else None
    #     sha256_sum = sha256_output.replace(file_path, '').strip() if sha256_output else None

    #     return f'{md5_sum}:{sha256_sum}'

    # async def exec_shell_command(self, command: str):
    #     # return await self.loop.run_in_executor(None, self._exec_shell_command, command)
    #     return await asyncio.to_thread(self._exec_shell_command, command)

    # def _exec_shell_command(self, command: str):
    #     try:
    #         self.i_shell.sendline(f"{command} && echo 'STOPPED'")
    #         self.i_shell.expect(['STOPPED'], timeout=30)
    #         self.i_shell.expect(['STOPPED'], timeout=30)
    #         output_lines = [line.strip() for line in re.split(r'[\r\n]', self.i_shell.before.decode('utf-8')) if line.strip()]
    #         return output_lines[-1]
    #     except pexpect.TIMEOUT:
    #         raise Exception("i-ssh connection Timeout")
    #     except pexpect.EOF:
    #         raise Exception("i-ssh connection EOF error")
