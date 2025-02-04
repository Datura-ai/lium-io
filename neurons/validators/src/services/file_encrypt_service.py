import os
import random
import subprocess
from typing import Annotated
from pathlib import Path
import tempfile
import shutil
import PyInstaller.__main__
import sys
from fastapi import Depends

from services.ssh_service import SSHService

from payload_models.payloads import MinerJobEnryptedFiles


class FileEncryptService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
    ):
        self.ssh_service = ssh_service

    def make_obfuscated_file(self, tmp_directory: str, file_path: str):
        subprocess.run(
            ['pyarmor', 'gen', '-O', tmp_directory, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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

    def make_binary_file_with_nuitka(self, tmp_directory: str, file_path: str):
        file_name = os.path.basename(file_path)

        subprocess.run([
            'nuitka', '--standalone', '--onefile',
            f'--output-dir={tmp_directory}',
            '--remove-output', '--quiet', '--no-progress',
            f'--output-filename={file_name}',
            file_path
        ])

        return file_name

    def ecrypt_miner_job_files(self):
        tmp_directory = Path(__file__).parent / "temp"
        if tmp_directory.exists() and tmp_directory.is_dir():
            shutil.rmtree(tmp_directory)

        # first chracter of variable shouldn't be digit
        encrypt_key_name = self.ssh_service.generate_random_string(random.randint(10, 100), True)
        encrypt_key_value = self.ssh_service.generate_random_string(random.randint(10, 100), False)

        machine_scrape_file_path = str(
            Path(__file__).parent / ".." / "miner_jobs/machine_scrape.py"
        )
        obfuscator_machine_scrape_file_path = str(
            Path(__file__).parent / ".." / "miner_jobs/obfuscator.py"
        )
        obfuscated_machine_scrape_file_path = str(
            Path(__file__).parent / ".." / "miner_jobs/obfuscated_machine_scrape.py"
        )
        with open(machine_scrape_file_path, 'r') as file:
            content = file.read()
        modified_content = content.replace('encrypt_key_name', encrypt_key_name).replace('encrypt_key_value', encrypt_key_value)

        with tempfile.NamedTemporaryFile(delete=True) as machine_scrape_file:
            machine_scrape_file.write(modified_content.encode('utf-8'))
            machine_scrape_file.flush()
            os.fsync(machine_scrape_file.fileno())
            command = [sys.executable, obfuscator_machine_scrape_file_path, machine_scrape_file.name]
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if random.choice([True, False]):
                machine_scrape_file_name = self.make_binary_file_with_nuitka(str(tmp_directory), obfuscated_machine_scrape_file_path)
            else:
                machine_scrape_file_name = self.make_binary_file(str(tmp_directory), obfuscated_machine_scrape_file_path)

        # generate score_script file
        score_script_file_path = str(Path(__file__).parent / ".." / "miner_jobs/score.py")
        with open(score_script_file_path, 'r') as file:
            content = file.read()
        modified_content = content

        with tempfile.NamedTemporaryFile(delete=True, suffix='.py') as score_file:
            score_file.write(modified_content.encode('utf-8'))
            score_file.flush()
            os.fsync(score_file.fileno())
            score_file_name = self.make_obfuscated_file(str(tmp_directory), score_file.name)

        return MinerJobEnryptedFiles(
            encrypt_key=encrypt_key_value,
            tmp_directory=str(tmp_directory),
            machine_scrape_file_name=machine_scrape_file_name,
            score_file_name=score_file_name,
        )
