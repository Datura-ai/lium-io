import os
import random
import shutil
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import PyInstaller.__main__
from fastapi import Depends
from payload_models.payloads import MinerJobEnryptedFiles

from services.ssh_service import SSHService

KEYS_FOR_ENCRYPTION_KEY_GENERATION = [
    "gpu.name",
    "gpu.uuid",
    "gpu.capacity",
    "gpu.cuda",
    "gpu.power_limit",
    "gpu.graphics_speed",
    "gpu.memory_speed",
    "gpu.pcie",
    "gpu.speed_pcie",
    "gpu.utilization",
    "gpu.memory_utilization",
]

ORIGINAL_KEYS = {
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
    "gpu.memory_utilization": "memory_utilization",
    '<default>': "<default>",
    'c_nvmlMemory_t_total': "total",
    'c_nvmlMemory_t_free': "free",
    'c_nvmlMemory_t_used': "used",
    'c_nvmlMemory_v2_t_version': "version",
    'c_nvmlMemory_v2_t_total': "total",
    'c_nvmlMemory_v2_t_reserved': "reserved",
    'c_nvmlMemory_v2_t_free': "free",
    'c_nvmlMemory_v2_t_used': "used",
    'c_nvmlUtilization_t_gpu': "gpu",
    'c_nvmlUtilization_t_memory': "memory",
    'upload_speed': "upload_speed",
    'download_speed': "download_speed",
    'network_speed_error': "network_speed_error",
    'net_speed_error': "net_speed_error",
    'gpu_count': "count",
    'gpu_driver': "driver",
    'gpu_cuda_driver': "cuda_driver",
    'gpu_details': "details",
    'gpu_scrape_error': "gpu_scrape_error",
    "data_gpu": "gpu",
    "data_processes": "gpu_processes",
    "data_cpu": "cpu",
    "data_nvidia_cfg": "nvidia_cfg",
    "nvidia_cfg_scrape_error": "nvidia_cfg_scrape_error",
    "cpu_count": "count",
    "cpu_model": "model",
    "cpu_clocks": "clocks",
    "cpu_utilization": "utilization",
    "cpu_scrape_error": "cpu_scrape_error",
    "ram_total": "total",
    "ram_free": "free",
    "ram_used": "used",
    "ram_available": "available",
    "ram_utilization": "utilization",
    "ram_scrape_error": "ram_scrape_error",
    "processes_pid": "pid",
    "processes_info": "info",
    "processes_container_name": "container_name",
    'hard_disk_total': "total",
    'hard_disk_used': "used",
    'hard_disk_free': "free",
    'hard_disk_utilization': "utilization",
    'hard_disk_scrape_error': "hard_disk_scrape_error",
    'data_os': "os",
    'data_ram': "ram",
    'data_hard_disk': "hard_disk",
    "data_docker_cfg_scrape_error": "docker_cfg_scrape_error",
    "data_docker_cfg": "docker_cfg",
    'os_scrape_error': "os_scrape_error",
    'data_network': "network",
    'data_md5_checksums': "md5_checksums",
    'md5_checksums_nvidia_smi': "nvidia_smi",
    'md5_checksums_libnvidia_ml': "libnvidia_ml",
    'md5_checksums_docker': "docker",
    'c_nvmlProcessInfo_v2_t_pid': "pid",
    'c_nvmlProcessInfo_v2_t_usedGpuMemory': "usedGpuMemory",
    'c_nvmlProcessInfo_v2_t_gpuInstanceId': "gpuInstanceId",
    'c_nvmlProcessInfo_v2_t_computeInstanceId': "computeInstanceId",
    '_fmt_usedGpuMemory': "usedGpuMemory",
    'docker_version': "version",
    'docker_container_id': "container_id",
    'docker_containers': "containers",
    'data_docker': "docker",
    'each_container_id': "container_id",
    'each_digest': "digest",
    'each_name': "name",
    'data_sysbox_runtime': "sysbox_runtime",
    'data_sysbox_runtime_scrape_error': "sysbox_runtime_scrape_error",
}


class FileEncryptService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
    ):
        self.ssh_service = ssh_service

    def make_obfuscated_file(self, tmp_directory: str, file_path: str):
        subprocess.run(
            ["pyarmor", "gen", "-O", tmp_directory, file_path],
            capture_output=True,
        )
        return os.path.basename(file_path)

    def make_binary_file(self, tmp_directory: str, file_path: str):
        file_name = os.path.basename(file_path)

        PyInstaller.__main__.run(
            [
                file_path,
                "--onefile",
                "--noconsole",
                "--log-level=ERROR",
                "--distpath",
                tmp_directory,
                "--name",
                file_name,
            ]
        )

        subprocess.run(["rm", "-rf", "build", f"{file_name}.spec"])

        return file_name

    def make_binary_file_with_nuitka(self, tmp_directory: str, file_path: str):
        file_name = os.path.basename(file_path)

        subprocess.run(
            [
                "nuitka",
                "--standalone",
                "--onefile",
                f"--output-dir={tmp_directory}",
                "--remove-output",
                "--quiet",
                "--no-progress",
                f"--output-filename={file_name}",
                file_path,
            ]
        )

        return file_name

    def generate_random_name(self):
        length = random.randint(3, 15)
        return "_" + "".join(random.choices(string.ascii_letters, k=length))

    def generate_key_mappings(self):
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
            "gpu.memory_utilization": "",
            "gpu_count": "",
            "gpu_driver": "",
            "gpu_cuda_driver": "",
            "gpu_details": "",
            "data_gpu": "",
            "data_docker_cfg_scrape_error": "",
            "data_docker_cfg": "",
            "data_processes": "",
            "data_cpu": "",
            "cpu_count": "",
            "cpu_model": "",
            "cpu_clocks": "",
            "cpu_utilization": "",
            "cpu_scrape_error": "",
            "ram_total": "",
            "ram_free": "",
            "ram_used": "",
            "ram_available": "",
            "ram_utilization": "",
            "ram_scrape_error": "",
            '<default>': "",
            'c_nvmlMemory_t_total': "",
            'c_nvmlMemory_t_free': "",
            'c_nvmlMemory_t_used': "",
            'c_nvmlMemory_v2_t_version': "",
            'c_nvmlMemory_v2_t_total': "",
            'c_nvmlMemory_v2_t_reserved': "",
            'c_nvmlMemory_v2_t_free': "",
            'c_nvmlMemory_v2_t_used': "",
            'c_nvmlUtilization_t_gpu': "",
            'c_nvmlUtilization_t_memory': "",
            'upload_speed': "",
            'download_speed': "",
            'network_speed_error': "",
            'net_speed_error': "",
            'hard_disk_total': "",
            'hard_disk_used': "",
            'hard_disk_free': "",
            'hard_disk_utilization': "",
            'hard_disk_scrape_error': "",
            'data_os': "",
            'os_scrape_error': "",
            'data_network': "",
            'data_hard_disk': "",
            'data_nvidia_cfg': "",
            "processes_pid": "",
            "processes_info": "",
            "gpu_scrape_error": "",
            "processes_container_name": "",
            'nvidia_cfg_scrape_error': "",
            'data_md5_checksums': "",
            'md5_checksums_nvidia_smi': "",
            'md5_checksums_libnvidia_ml': "",
            'md5_checksums_docker': "",
            'c_nvmlProcessInfo_v2_t_pid': "",
            'c_nvmlProcessInfo_v2_t_usedGpuMemory': "",
            'c_nvmlProcessInfo_v2_t_gpuInstanceId': "",
            'c_nvmlProcessInfo_v2_t_computeInstanceId': "",
            '_fmt_usedGpuMemory': "",
            'docker_version': "",
            'docker_container_id': "",
            'docker_containers': "",
            'data_docker': "",
            'data_ram': "",
            'each_container_id': "",
            'each_digest': "",
            'each_name': "",
            'machine_specs': "",
            'data_sysbox_runtime_scrape_error': "",
            'data_sysbox_runtime': "",
        }

        # Generate dictionary key mapping on validator side
        for key, value in all_keys.items():
            all_keys[key] = self.generate_random_name()

        encryption_key = "".join([all_keys[key] for key in KEYS_FOR_ENCRYPTION_KEY_GENERATION])
        return all_keys, encryption_key

    def ecrypt_miner_job_files(self):
        """
        Encrypts and obfuscates miner job files for secure execution.

        This function performs the following steps:
        1. Clears any existing temporary directory used for storing encrypted files.
        2. Defines file paths for the machine scrape script and its obfuscator.
        3. Runs the obfuscator script to generate an obfuscated version of the machine scrape script.
        4. Replaces dictionary keys in the obfuscated script with randomly generated names.
        5. Compiles the obfuscated script into a binary using Nuitka or another method.
        6. Generates a score script file and obfuscates it for secure execution.

        Returns: MinerJobEnryptedFiles
        """
        tmp_directory = Path(__file__).parent / "temp"
        if tmp_directory.exists() and tmp_directory.is_dir():
            shutil.rmtree(tmp_directory)

        # file pathes
        machine_scrape_file_path = str(
            Path(__file__).parent / ".." / "miner_jobs/machine_scrape.py"
        )
        obfuscator_machine_scrape_file_path = str(
            Path(__file__).parent / ".." / "miner_jobs/obfuscator.py"
        )
        obfuscated_machine_scrape_file_path = str(
            Path(__file__).parent / ".." / "miner_jobs/obfuscated_machine_scrape.py"
        )

        # run obfuscator.py to generate obfuscated_machine_scrape.py
        command = [sys.executable, obfuscator_machine_scrape_file_path, machine_scrape_file_path]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa

        with open(obfuscated_machine_scrape_file_path) as file:
            obfuscated_content = file.read()

        # replace dictionary keys with random names
        all_keys, encryption_key = self.generate_key_mappings()
        for key, value in all_keys.items():
            obfuscated_content = obfuscated_content.replace(key, value)

        # build binary with nuitka
        with tempfile.NamedTemporaryFile(delete=True) as machine_scrape_file:
            machine_scrape_file.write(obfuscated_content.encode("utf-8"))
            machine_scrape_file.flush()
            os.fsync(machine_scrape_file.fileno())
            
            machine_scrape_file_name = self.make_binary_file(
                str(tmp_directory), machine_scrape_file.name
            )

            # if random.choice([True, False]):
            #     machine_scrape_file_name = self.make_binary_file_with_nuitka(
            #         str(tmp_directory), machine_scrape_file.name
            #     )
            # else:
            #     machine_scrape_file_name = self.make_binary_file(
            #         str(tmp_directory), machine_scrape_file.name
            #     )

        # # generate score_script file
        # score_script_file_path = str(Path(__file__).parent / ".." / "miner_jobs/score.py")
        # with open(score_script_file_path) as file:
        #     content = file.read()
        # modified_content = content

        # with tempfile.NamedTemporaryFile(delete=True, suffix=".py") as score_file:
        #     score_file.write(modified_content.encode("utf-8"))
        #     score_file.flush()
        #     os.fsync(score_file.fileno())
        #     score_file_name = self.make_obfuscated_file(str(tmp_directory), score_file.name)

        return MinerJobEnryptedFiles(
            encrypt_key=encryption_key,
            all_keys=all_keys,
            tmp_directory=str(tmp_directory),
            machine_scrape_file_name=machine_scrape_file_name,
        )
