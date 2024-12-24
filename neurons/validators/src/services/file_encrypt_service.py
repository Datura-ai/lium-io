import os
import subprocess
from typing import Annotated
from pathlib import Path
import tempfile
import shutil
import PyInstaller.__main__
from fastapi import Depends

from services.ssh_service import SSHService

from payload_models.payloads import MinerJobEnryptedFiles
from core.config import settings


class FileEncryptService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
    ):
        self.ssh_service = ssh_service
        self.wallet = settings.get_bittensor_wallet()

    def make_obfuscated_file(self, tmp_directory: str, file_path: str):
        subprocess.run(
            ['pyarmor', 'gen', '-O', tmp_directory, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE ,
        )
        return os.path.basename(file_path)

    def make_binary_file(self, tmp_directory: str, file_path: str):
        file_name = os.path.basename(file_path)

        PyInstaller.__main__.run([
            file_path,
            '--onefile',
            '--noconsole',
            '--log-level=ERROR',
            '--distpath', tmp_directory,
            '--name', file_name,
        ])

        subprocess.run(['rm', '-rf', 'build', f'{file_name}.spec'])

        return file_name

    def ecrypt_miner_job_files(self):
        tmp_directory = Path(__file__).parent / "temp"
        if tmp_directory.exists() and tmp_directory.is_dir():
            shutil.rmtree(tmp_directory)

        encrypt_key = self.ssh_service.generate_random_string()

        machine_scrape_file_path = str(
            Path(__file__).parent / ".." / "miner_jobs/machine_scrape.py"
        )
        with open(machine_scrape_file_path, 'r') as file:
            content = file.read()
        modified_content = content.replace('encrypt_key', encrypt_key)

        with tempfile.NamedTemporaryFile(delete=True) as machine_scrape_file:
            machine_scrape_file.write(modified_content.encode('utf-8'))
            machine_scrape_file.flush()
            os.fsync(machine_scrape_file.fileno())
            machine_scrape_file_name = self.make_binary_file(str(tmp_directory), machine_scrape_file.name)

        # generate score_script file
        score_script_file_path = str(Path(__file__).parent / ".." / "miner_jobs/score.py")
        with open(score_script_file_path, 'r') as file:
            content = file.read()
        modified_content = content.replace('encrypt_key', encrypt_key)

        with tempfile.NamedTemporaryFile(delete=True, suffix='.py') as score_file:
            score_file.write(modified_content.encode('utf-8'))
            score_file.flush()
            os.fsync(score_file.fileno())
            score_file_name = self.make_obfuscated_file(str(tmp_directory), score_file.name)

        return MinerJobEnryptedFiles(
            encrypt_key=encrypt_key,
            tmp_directory=str(tmp_directory),
            machine_scrape_file_name=machine_scrape_file_name,
            score_file_name=score_file_name,
        )
        
    def ecrypt_generate_signature(self, message: str):
        tmp_directory = Path(__file__).parent / "temp"
        if tmp_directory.exists() and tmp_directory.is_dir():
            shutil.rmtree(tmp_directory)
        # generate signature
        private_key, public_key = self.ssh_service.generate_ssh_key(self.wallet.get_hotkey().ss58_address)
        check_monitoring_script_path = str(Path(__file__).parent / ".." / "miner_jobs/check_monitoring_script.py")
        with open(check_monitoring_script_path, 'r') as file:
            content = file.read()
        modified_content = content.replace('private_key', private_key)
        modified_content = modified_content.replace('message', message)
        with tempfile.NamedTemporaryFile(delete=True, suffix='.py') as check_monitoring_script:
            check_monitoring_script.write(modified_content.encode('utf-8'))
            check_monitoring_script.flush()
            os.fsync(check_monitoring_script.fileno())
            check_monitoring_script_name = self.make_binary_file(str(tmp_directory), check_monitoring_script.name)

        return check_monitoring_script_name
