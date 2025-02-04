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
import string

from services.ssh_service import SSHService

from payload_models.payloads import MinerJobEnryptedFiles


class FileEncryptService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
    ):
        self.ssh_service = ssh_service
        self.all_keys = {}
        self.original_keys = {}

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

    def generate_random_name(self, length=10):
        return '_' + ''.join(random.choices(string.ascii_letters, k=length))
    
    def generate_key_mappings(self):
        keys_for_encryption_key_generation = [
            'gpu.name', 'gpu.uuid', 'gpu.capacity', 'gpu.cuda', 'gpu.power_limit', 
            'gpu.graphics_speed', 'gpu.memory_speed', 'gpu.pcie', 'gpu.speed_pcie', 'gpu.utilization', 
            'gpu.memory_utilization'
        ]

        all_keys = {
            "gpu.name": "",
            "gpu.uuid": "",
            "gpu.capacity": "",
            "gpu.cuda": "",
            "gpu.power_limit": "",
            "gpu.graphics_speed": "",
            "gpu.memory_speed": "",
            "gpu.pcie": "",
            "gpu.speed_pcie": "",
            "gpu.utilization": "",
            "gpu.memory_utilization": ""
        }
        
        original_keys = {
            "gpu.name": "name",
            "gpu.uuid": "uuid",
            "gpu.capacity": "capacity",
            "gpu.cuda": "cuda",
            "gpu.power_limit": "power_limit",
            "gpu.graphics_speed": "graphics_speed",
            "gpu.memory_speed": "memory_speed",
            "gpu.pcie": "pcie",
            "gpu.speed_pcie": "pcie_speed",
            "gpu.utilization": "gpu_utilization",
            "gpu.memory_utilization": "memory_utilization"
        }

        # Generate dictionary key mapping on validator side
        for key in keys_for_encryption_key_generation:
            all_keys[key] = self.generate_random_name()

        self.all_keys = all_keys
        self.original_keys = original_keys
        
        encryption_key = ":".join([all_keys[key] for key in keys_for_encryption_key_generation])
        return all_keys, encryption_key

    def get_original_key(self, key: str):
        return self.original_keys.get(key, key)
    
    def get_all_keys(self):
        return self.all_keys
    
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

        command = [sys.executable, obfuscator_machine_scrape_file_path, machine_scrape_file_path]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        with open(obfuscated_machine_scrape_file_path, 'r') as file:
            obfuscated_content = file.read()
            
        all_keys, encryption_key = self.generate_key_mappings()
        for key, value in all_keys.items():
            obfuscated_content = obfuscated_content.replace(key, value)

        with tempfile.NamedTemporaryFile(delete=True) as machine_scrape_file:
            machine_scrape_file.write(obfuscated_content.encode('utf-8'))
            machine_scrape_file.flush()
            os.fsync(machine_scrape_file.fileno())

            if random.choice([True, False]):
                machine_scrape_file_name = self.make_binary_file_with_nuitka(str(tmp_directory), machine_scrape_file.name)
            else:
                machine_scrape_file_name = self.make_binary_file(str(tmp_directory), machine_scrape_file.name)

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
            encrypt_key=encryption_key,
            tmp_directory=str(tmp_directory),
            machine_scrape_file_name=machine_scrape_file_name,
            score_file_name=score_file_name,
        )
