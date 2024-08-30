from typing import Annotated
import io
import logging
import asyncio
import random

import bittensor
from fastapi import Depends
from pydantic import BaseModel, field_validator

from services.ssh_service import SSHService
from paramiko import SSHClient, AutoAddPolicy, Ed25519Key 

from datura.requests.miner_requests import AcceptSSHKeyRequest

logger = logging.getLogger(__name__)

class ResourceType(BaseModel):
    cpu: int
    gpu: int
    memory: str
    volume: str
    
    @field_validator('cpu', 'gpu')
    def validate_positive_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f'{v} should be a valid non-negative integer string.')
        return v

    @field_validator('memory', 'volume')
    def validate_memory_format(cls, v: str) -> str:
        if not v[:-2].isdigit() or v[-2:].upper() not in ['MB', 'GB']:
            raise ValueError(f'{v} is not a valid format.')
        return v

class DockerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
    ):
        self.ssh_service = ssh_service
        
    def generate_portMappings(self, start_external_port=40000):
        internal_ports = [22, 22140, 22141, 22142, 22143]
        
        mappings = []
        used_external_ports = set()
        
        for i in range(len(internal_ports)):
            while True:
                external_port = random.randint(start_external_port, start_external_port + 10000)
                if external_port not in used_external_ports:
                    used_external_ports.add(external_port)
                    break
                
            mappings.append((internal_ports[i], external_port))
            
        return mappings
        

    async def create_docker(
        self,
        miner_address: str,
        miner_hotkey: str,
        msg: AcceptSSHKeyRequest,
        keypair: bittensor.Keypair,
        private_key: str,
        docker_image: str,
        public_key: str,
        resources: ResourceType,
    ):
        logger.info(f"Create Docker Container -> miner_address: {miner_address}, miner_hotkey: {miner_hotkey}")

        logger.info("Connect ssh")
        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = Ed25519Key.from_private_key(io.StringIO(private_key))

        ssh_client = SSHClient()
        ssh_client.set_missing_host_key_policy(AutoAddPolicy())
        ssh_client.connect(hostname=miner_address, username=msg.ssh_username, look_for_keys=False, pkey=pkey, port=msg.ssh_port)
        
        await asyncio.to_thread(self._create_docker, ssh_client, docker_image, public_key, resources)
        
        ssh_client.close()
        
        
    def _create_docker(
        self,
        ssh_client: SSHClient,
        docker_image: str,
        public_key: str,
        resources: ResourceType,
    ):
        try:
            logger.info('Pulling docker image')
            ssh_client.exec_command(f"docker pull ${docker_image}")
            
            # generate port maps
            port_maps = self.generate_portMappings()
            port_flags = ' '.join([f'-p {external}:{internal}' for internal, external in port_maps])
            
            # creat docker container with the port map & resource
            logger.info('Create docker container')
            ssh_client.exec_command(f'docker run -d {port_flags} -e PUBLIC_KEY="{public_key}" {docker_image}')
            
            # return port maps
        except Exception as e:
            logger.error('ssh connection error: %s', str(e))
