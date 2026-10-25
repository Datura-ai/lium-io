import uuid
import logging
import bittensor as bt
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_utils import to_hex, keccak
from typing import Optional, Any
from core.config import settings
from core.db import get_db
from daos.executor import ExecutorDao
from services.executor_service import ExecutorService
from services.ssh_service import MinerSSHService
from models.executor import Executor
from core.utils import _m
from protocol.miner_portal_request import AddExecutorFailed

logging.basicConfig(level=logging.INFO)


def require_executor_dao(func):
    def wrapper(self, *args, **kwargs):
        if not self.executor_dao:
            self.logger.error("ExecutorDao is not initialized. Set with_executor_db=True when creating CliService.")
            return False
        return func(self, *args, **kwargs)
    return wrapper


class CliService:
    def __init__(self, private_key: Optional[str] = None, with_executor_db: bool = False):
        """
        Initialize the CLI service.
        :param private_key: Ethereum private key for signing (optional).
        :param with_executor_db: If True, initializes the executor DAO for DB operations.
        """
        self.wallet = settings.get_bittensor_wallet()
        self.netuid = settings.BITTENSOR_NETUID
        self.config = settings.get_bittensor_config()
        self.hotkey = self.wallet.get_hotkey().ss58_address
        self.private_key = private_key
        self.executor_dao = ExecutorDao(session=next(get_db())) if with_executor_db else None

        # Create executor_service if with_executor_db is True
        if with_executor_db and self.executor_dao is not None:
            ssh_service = MinerSSHService()
            self.executor_service = ExecutorService(executor_dao=self.executor_dao, ssh_service=ssh_service)
        else:
            self.executor_service = None

        self.logger = logging.getLogger()

        self.default_extra = {
            "hotkey": self.hotkey,
            "netuid": self.netuid,
            "network": settings.BITTENSOR_NETWORK,
        }

    def get_node(self):
        """
        Get a SubstrateInterface node connection using the current config.
        Tries each entry of the ordered endpoint list (our proxy first, the public
        node last) so a down proxy does not fail the command.
        :return: SubstrateInterface instance
        """
        endpoints = settings.get_chain_endpoints()
        for index, endpoint in enumerate(endpoints):
            try:
                self.subtensor = bt.Subtensor(network=endpoint.value, config=self.config)
                return self.subtensor.substrate
            except Exception as e:
                if index == len(endpoints) - 1:
                    raise
                next_endpoint = endpoints[index + 1]
                self.logger.warning(_m(
                    f"Subtensor endpoint switched from={endpoint.value} to={next_endpoint.value}",
                    extra={
                        **self.default_extra,
                        "from": endpoint.value,
                        "to": next_endpoint.value,
                        "reason": "connect failed",
                        "error": str(e),
                    },
                ))

    def print_extrinsic_receipt(self, receipt) -> dict:
        """
        Returns a summary of the extrinsic receipt as a dict.
        :param receipt: The extrinsic receipt object
        :return: Dictionary summary of the receipt
        """
        summary = {
            "extrinsic_hash": getattr(receipt, "extrinsic_hash", None),
            "block_hash": getattr(receipt, "block_hash", None),
            "is_success": getattr(receipt, "is_success", None),
            "error_message": getattr(receipt, "error_message", None),
            "events": []
        }
        if hasattr(receipt, "triggered_events") and receipt.triggered_events:
            for event in receipt.triggered_events:
                summary["events"].append({
                    "attributes": event['event']
                })
        return summary

    def make_associate_evm_key_extrinsic(self, node) -> Any:
        """
        Create an extrinsic to associate the EVM key with the hotkey.
        :param node: SubstrateInterface instance
        :return: Composed call object or None if private_key is missing
        """
        if not self.private_key:
            self.logger.error("No private key provided for EVM association.")
            return None
        # 1. Encode block number to bytes (SCALE encoded u64, which is little-endian)
        block_number = node.query("System", "Number").value
        block_number_bytes = block_number.to_bytes(8, 'little')
        block_number_hash = keccak(block_number_bytes)

        # 2. Decode hotkey ss58 address to its public key bytes
        hotkey_bytes = bytes.fromhex(node.ss58_decode(self.hotkey))

        # 3. Construct message
        message_to_sign_bytes = hotkey_bytes + block_number_hash

        # 4. Sign the message with the EVM private key
        signable_message = encode_defunct(primitive=message_to_sign_bytes)
        account = Account.from_key(self.private_key)
        signature = account.sign_message(signable_message)
        signature_hex = to_hex(signature.signature)
        print("\n--- Preparing associate_evm_key extrinsic ---")
        print(f"  Hotkey (ss58): {self.hotkey}")
        print(f"  EVM Address: {account.address}")
        print(f"  Netuid: {self.netuid}")
        print(f"  Block number: {block_number}")
        print(f"  Signature: {signature_hex}")
        print("---------------------------------------------")
        return node.compose_call(
            call_module="SubtensorModule",
            call_function="associate_evm_key",
            call_params={
                "netuid": self.netuid,
                "evm_key": account.address,
                "block_number": block_number,
                "signature": signature_hex
            }
        )

    def submit_extrinsic(self, node, call: Any) -> Any:
        """
        Submit a signed extrinsic to the chain.
        :param node: SubstrateInterface instance
        :param call: Composed call object
        :return: The extrinsic response
        """
        extrinsic = node.create_signed_extrinsic(
            call=call,
            keypair=self.wallet.hotkey,
        )
        response = node.submit_extrinsic(extrinsic, wait_for_inclusion=True, wait_for_finalization=True)
        return response

    def associate_ethereum_address(self) -> bool:
        """
        Associate the Ethereum address with the Bittensor hotkey on-chain.
        :return: True if successful, False otherwise
        """
        try:
            node = self.get_node()

            evm_call = self.make_associate_evm_key_extrinsic(
                node=node
            )
            if evm_call:
                response = self.submit_extrinsic(node, evm_call)
                summary = self.print_extrinsic_receipt(response)
                if response.is_success:
                    self.logger.info(_m(
                        "Associate EVM address successfully",
                        extra={**self.default_extra, "summary": summary}
                    ))
                    return True
                else:
                    self.logger.error(_m(
                        "❌ Failed to associate",
                        extra={**self.default_extra, "error": str(response.error_message)}
                    ))
                    return False
            self.logger.error(_m(
                "Failed to create EVM association call",
                extra=self.default_extra
            ))
            return False
        except Exception as e:
            self.logger.error(_m(
                "❌ Failed to associate",
                extra={**self.default_extra, "error": str(e)}
            ))
            return False

    def get_uid_for_hotkey(self, hotkey):
        metagraph = self.subtensor.metagraph(netuid=self.netuid)
        return metagraph.hotkeys.index(hotkey)

    def get_associated_evm_address(self) -> str:
        """
        Get the EVM address for the Bittensor hotkey.
        :return: EVM address
        """
        node = self.get_node()
        uid = self.get_uid_for_hotkey(self.hotkey)
        self.logger.info(f"UID for hotkey {self.hotkey}: {uid}")
        associated_evm = node.query(module="SubtensorModule", storage_function="AssociatedEvmAddress", params=[self.netuid, uid])
        # async-substrate-interface 2.x decodes the record: ("0x…", block_number)
        evm_address_hex = associated_evm.value[0]

        self.logger.info(_m(
            f"EVM address for hotkey {self.hotkey}: {evm_address_hex}",
            extra=self.default_extra
        ))

        return evm_address_hex

    @require_executor_dao
    async def add_executor(
        self,
        address: str,
        port: int,
        price_per_gpu: float,
        validator: str | None = None,
    ) -> bool:
        """
        Add an executor to the database.
        :param address: Executor IP address
        :param port: Executor port
        :param price_per_gpu: GPU price per hour in USD
        :param validator: Validator hotkey
        :return: True if successful, False otherwise
        """
        if validator is None:
            validator = settings.DEFAULT_VALIDATOR_HOTKEY

        result = await self.executor_service.create(
            Executor(
                uuid=uuid.uuid4(),
                address=address,
                port=port,
                validator=validator,
                price_per_gpu=price_per_gpu
            )
        )
        return not isinstance(result, AddExecutorFailed)

    @require_executor_dao
    async def show_executors(self):
        """
        Show all executors in the database.
        :return: True if successful, False otherwise
        """
        try:
            executors = self.executor_dao.get_all_executors()
            result = [
                {
                    "uuid": str(executor.uuid),
                    "address": executor.address,
                    "port": executor.port,
                    "validator": executor.validator,
                    "price_per_gpu": executor.price_per_gpu
                }
                for executor in executors
            ]
            for ex in result:
                self.logger.info(f"{ex['uuid']} {ex['address']}:{ex['port']} -> validator: {ex['validator']}, price_per_gpu (USD/gpu/h): {ex['price_per_gpu']}")
            return True
        except Exception as e:
            self.logger.error("Failed in showing an executor: %s", str(e))
            return False

    @require_executor_dao
    async def switch_validator(self, address: str, port: int, validator: str):
        """
        Switch validator for an executor by address and port.
        :param address: Executor IP address
        :param port: Executor port
        :param validator: Validator hotkey
        :return: True if successful, False otherwise
        """
        try:
            payload = {'validator': validator}
            self.executor_dao.update(address, port, payload)
            self.logger.info(f"✅ Successfully switched validator for executor {address}:{port} to {validator}")
            return True
        except Exception as e:
            self.logger.error("Failed in switching validator: %s", str(e))
            return False

    @require_executor_dao
    async def remove_executor(self, address: str, port: int):
        """
        Remove an executor from the database by address and port.
        :param address: Executor IP address
        :param port: Executor port
        :return: True if successful, False otherwise
        """
        try:
            executor = self.executor_dao.find_one(address, port)
            if not executor:
                self.logger.error("No executor at %s:%d", address, port)
                return False

            self.executor_dao.delete_by_address_port(address, port)
            self.logger.info("Removed an executor(%s:%d)", address, port)
            return True
        except Exception as e:
            self.logger.error("Failed in removing an executor: %s", str(e))
            return False

    @require_executor_dao
    async def update_executor_price(
        self,
        address: str,
        port: int,
        *,
        price_per_gpu: float,
    ) -> bool:
        """
        Update the price per hour for an executor by address and port.
        :param address: Executor IP address
        :param port: Executor port
        :param price_per_gpu: New price per GPU in USD
        :return: True if successful, False otherwise
        """
        try:
            payload = {'price_per_gpu': price_per_gpu}
            self.executor_dao.update(address, port, payload)
            self.logger.info(f"✅ Saved a price of {price_per_gpu} USD/hour for executor {address}:{port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update executor price: %s", str(e))
            return False
