import uuid
import logging
import json
import bittensor as bt
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_utils import to_hex, keccak
from typing import Optional, Any
from core.config import settings
from core.db import get_db
from daos.executor import ExecutorDao
from models.executor import Executor
from core.utils import get_collateral_contract, _m
from core.const import REQUIRED_DEPOSIT_AMOUNT
from celium_collateral_contracts.address_conversion import h160_to_ss58
from bittensor.utils.balance import Balance

logging.basicConfig(level=logging.INFO)


def require_executor_dao(func):
    def wrapper(self, *args, **kwargs):
        if not self.executor_dao:
            self.logger.error("ExecutorDao is not initialized. Set with_executor_db=True when creating CliService.")
            return False
        return func(self, *args, **kwargs)
    return wrapper


class CliService:
    def __init__(self, private_key: Optional[str] = None, with_executor_db: bool = False, version: str = "1.0.2"):
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
        self.collateral_contract = (
            get_collateral_contract(miner_key=private_key, version=version)
            if private_key else get_collateral_contract(version=version)
        )
        self.executor_dao = ExecutorDao(session=next(get_db())) if with_executor_db else None
        self.logger = logging.getLogger()

        self.default_extra = {
            "hotkey": self.hotkey,
            "netuid": self.netuid,
            "contract_address": settings.COLLATERAL_CONTRACT_ADDRESS,
            "network": settings.BITTENSOR_NETWORK,
            "rpc_url": settings.SUBTENSOR_EVM_RPC_URL,
        }

    def get_node(self):
        """
        Get a SubstrateInterface node connection using the current config.
        :return: SubstrateInterface instance
        """
        self.subtensor = bt.subtensor(config=self.config)
        return self.subtensor.substrate

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

    def get_eth_ss58_address(self) -> str:
        """
        Get the Ethereum SS58 address for the Bittensor hotkey.
        :return: Ethereum SS58 address
        """
        account = Account.from_key(self.private_key)
        ss58_address = h160_to_ss58(account.address)
        return ss58_address

    def transfer_tao_to_eth_address(self, amount: float) -> bool:
        """
        Transfer TAO to the Ethereum SS58 address.
        :param amount: Amount of TAO to transfer
        :return: True if successful, False otherwise
        """
        try:
            ss58_address = self.get_eth_ss58_address()
            self.get_node()
            print(f'Transferring {amount} TAO to Ethereum SS58 address {ss58_address}.')
            print('Please enter your bittensor wallet password:')
            self.subtensor.transfer(
                wallet=self.wallet,
                dest=ss58_address,
                amount=Balance.from_tao(amount, self.netuid),
                wait_for_inclusion=True,
                wait_for_finalization=True
            )
            self.logger.info(_m(
                "✅ Transferred TAO to Ethereum SS58 address successfully",
                extra={**self.default_extra, "amount": amount, "to_address": ss58_address}
            ))
        except Exception as e:
            self.logger.error(_m(
                "❌ Failed to transfer TAO to Ethereum SS58 address",
                extra={**self.default_extra, "amount": amount, "to_address": ss58_address, "error": str(e)}
            ))

    async def get_balance_of_eth_address(self) -> str:
        balance = await self.collateral_contract.get_balance(self.collateral_contract.miner_address)
        self.logger.info(f"Balance of Eth address: {balance} TAO")
        return balance

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
        address_bytes = associated_evm.value[0][0]
        evm_address_hex = "0x" + bytes(address_bytes).hex()

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
        validator: str,
        deposit_amount: float | None = None,
        gpu_type: str | None = None,
        gpu_count: int | None = None
    ) -> bool:
        """
        Add an executor to the database and deposit collateral.
        :param address: Executor IP address
        :param port: Executor port
        :param validator: Validator hotkey
        :param deposit_amount: Amount of TAO to deposit (optional)
        :param gpu_type: Type of GPU (optional)
        :param gpu_count: Number of GPUs (optional)
        :return: True if successful, False otherwise
        """
        executor_uuid = uuid.uuid4()
        try:
            executor = self.executor_dao.save(Executor(uuid=executor_uuid, address=address, port=port, validator=validator))
            self.logger.info("Added executor (id=%s)", str(executor.uuid))
        except Exception as e:
            self.logger.error("❌ Failed to add executor: %s", str(e))
            return False

        if deposit_amount is None and gpu_type is None and gpu_count is None:
            self.logger.info("No deposit amount provided, skipping deposit.")
            return True

        if deposit_amount is None:
            if gpu_type is None or gpu_count is None:
                self.logger.error("gpu_type and gpu_count must be specified if deposit_amount is not provided.")
                return False
            if gpu_type not in REQUIRED_DEPOSIT_AMOUNT:
                self.logger.error(f"Unknown GPU type: {gpu_type}. Please use one of: {list(REQUIRED_DEPOSIT_AMOUNT.keys())}")
                return False
            deposit_amount = self._get_required_deposit_amount(gpu_type, gpu_count)
            if deposit_amount < settings.REQUIRED_TAO_COLLATERAL:
                deposit_amount = settings.REQUIRED_TAO_COLLATERAL
            self.logger.info(f"Calculated deposit amount: {deposit_amount} TAO for {gpu_count}x {gpu_type}")

        if deposit_amount < settings.REQUIRED_TAO_COLLATERAL:
            self.logger.error("Error: Minimum deposit amount is %f TAO.", settings.REQUIRED_TAO_COLLATERAL)
            return False

        try:
            balance = await self.collateral_contract.get_balance(self.collateral_contract.miner_address)
            self.logger.info(f"Miner balance: {balance} TAO for miner hotkey {self.hotkey}")
            if balance < deposit_amount:
                self.logger.error("Error: Insufficient balance in miner's address.")
                return False
            self.logger.info(
                f"Deposit amount {deposit_amount} for this executor UUID: {executor_uuid} "
                f"since miner {self.hotkey} is adding this executor"
            )
            await self.collateral_contract.deposit_collateral(deposit_amount, str(executor_uuid))
            self.logger.info("✅ Deposited collateral successfully.")
            return True
        except Exception as e:
            self.logger.error(_m(
                "❌ Failed to deposit collateral",
                extra={**self.default_extra, "error": str(e)}
            ))
            return False

    @require_executor_dao
    async def deposit_collateral(self, address: str, port: int, deposit_amount: float | None = None, gpu_type: str | None = None, gpu_count: int | None = None):
        """
        Deposit collateral for an existing executor in the database.
        :param address: Executor IP address
        :param port: Executor port
        :param deposit_amount: Amount of TAO to deposit (optional)
        :param gpu_type: Type of GPU (optional)
        :param gpu_count: Number of GPUs (optional)
        :return: True if successful, False otherwise
        """
        if deposit_amount is None:
            if gpu_type is None or gpu_count is None:
                self.logger.error("gpu_type and gpu_count must be specified if deposit_amount is not provided.")
                return False
            if gpu_type not in REQUIRED_DEPOSIT_AMOUNT:
                self.logger.error(f"Unknown GPU type: {gpu_type}. Please use one of: {list(REQUIRED_DEPOSIT_AMOUNT.keys())}")
                return False
            deposit_amount = self._get_required_deposit_amount(gpu_type, gpu_count)
            if deposit_amount < settings.REQUIRED_TAO_COLLATERAL:
                deposit_amount = settings.REQUIRED_TAO_COLLATERAL
            self.logger.info(f"Calculated deposit amount: {deposit_amount} TAO for {gpu_count}x {gpu_type}")

        if deposit_amount < settings.REQUIRED_TAO_COLLATERAL:
            self.logger.error("Error: Minimum deposit amount is %f TAO.", settings.REQUIRED_TAO_COLLATERAL)
            return False
        try:
            executor = self.executor_dao.findOne(address, port)
            executor_uuid = executor.uuid
            balance = await self.collateral_contract.get_balance(self.collateral_contract.miner_address)
            self.logger.info(f"Miner balance: {balance} TAO for miner hotkey {self.hotkey}")
            if balance < deposit_amount:
                self.logger.error("Error: Insufficient balance in miner's address.")
                return False
            self.logger.info(
                f"Deposit amount {deposit_amount} for this executor UUID: {executor_uuid} "
                f"since miner {self.hotkey} is going to add this executor"
            )
            await self.collateral_contract.deposit_collateral(deposit_amount, str(executor_uuid))
            self.logger.info("✅ Deposited collateral successfully.")
            return True
        except Exception as e:
            self.logger.error(_m(
                "❌ Failed to deposit collateral",
                extra={**self.default_extra, "error": str(e)}
            ))
            return False

    async def reclaim_collateral(self, executor_uuid: str):
        """
        Reclaim collateral for a specific executor from the contract.
        :param executor_uuid: UUID of the executor
        :return: True if successful, False otherwise
        """
        try:
            balance = await self.collateral_contract.get_balance(self.collateral_contract.miner_address)
            self.logger.info("Miner balance: %f TAO", balance)
            reclaim_amount = await self.collateral_contract.get_executor_collateral(executor_uuid)
            self.logger.info(
                f"Executor {executor_uuid} is being removed by miner {self.hotkey}. "
                f"The total collateral of {reclaim_amount} TAO will be reclaimed from the collateral contract."
            )
            _, event = await self.collateral_contract.reclaim_collateral("Manual reclaim", executor_uuid)
            import binascii
            json_payload = {
                "reclaim_request_id": event['args']['reclaimRequestId'],
                "amount": event['args']['amount'],  # If you want to convert from wei to TAO, do it here
                "expiration_time": event['args']['expirationTime'],
                "url": event['args']['url'],
                "url_content_md5_checksum": binascii.hexlify(event['args']['urlContentMd5Checksum']).decode(),
                "block_number": event['blockNumber'],
                "executor_uuid": event['args']['executorId'].hex()  # or decode as needed
            }

            self.logger.info(_m(
                "✅ Reclaimed collateral successfully.",
                extra={**self.default_extra, "reclaim_info": json_payload}
            ))
            return True
        except Exception as e:
            self.logger.error(_m(
                "❌ Failed to reclaim collateral",
                extra={**self.default_extra, "error": str(e)}
            ))
            return False

    @require_executor_dao
    async def get_miner_collateral(self):
        """
        Get the total miner collateral by summing up collateral from all registered executors.
        :return: True if successful, False otherwise
        """
        try:
            executors = self.executor_dao.get_all_executors()
            total_collateral = 0.0
            for executor in executors:
                executor_uuid = str(executor.uuid)
                collateral = await self.collateral_contract.get_executor_collateral(executor_uuid)
                total_collateral += float(collateral)
                self.logger.info("Executor %s collateral: %f TAO", executor_uuid, collateral)
            self.logger.info("Total miner collateral from all executors: %f TAO", total_collateral)
            return True
        except Exception as e:
            self.logger.error(_m(
                "❌ Failed in getting miner collateral",
                extra={**self.default_extra, "error": str(e)}
            ))
            return False

    @require_executor_dao
    async def get_executor_collateral(self, address: str, port: int):
        """
        Get the collateral amount for a specific executor by address and port.
        :param address: Executor IP address
        :param port: Executor port
        :return: True if successful, False otherwise
        """
        try:
            executor = self.executor_dao.findOne(address, port)
            executor_uuid = str(executor.uuid)
        except Exception as e:
            self.logger.error("❌ Failed to find executor: %s", str(e))
            return False
        try:
            collateral = await self.collateral_contract.get_executor_collateral(executor_uuid)
            self.logger.info("Executor %s collateral: %f TAO from collateral contract", executor_uuid, collateral)
            return True
        except Exception as e:
            self.logger.error(_m(
                "❌ Failed to get executor collateral",
                extra={**self.default_extra, "error": str(e)}
            ))
            return False

    @require_executor_dao
    async def get_reclaim_requests(self):
        """
        Get reclaim requests for the current miner from the collateral contract.
        :return: True if successful, False otherwise
        """
        try:
            reclaim_requests = await self.collateral_contract.get_reclaim_events()
            if not reclaim_requests:
                self.logger.info(json.dumps([]))
                return True
            executors = self.executor_dao.get_all_executors()
            executor_uuids = set(str(executor.uuid) for executor in executors)

            def to_dict(obj):
                if hasattr(obj, "__dict__"):
                    return dict(obj.__dict__)
                elif hasattr(obj, "_asdict"):
                    return obj._asdict()
                else:
                    return dict(obj)
            filtered_requests = [
                req for req in reclaim_requests
                if getattr(req, "amount", 0) != 0 and str(getattr(req, "executor_uuid", "")) in executor_uuids
            ]
            if not filtered_requests:
                self.logger.info(_m(
                    "No reclaim requests found for your executors.",
                    extra=self.default_extra
                ))
                return True
            # Use rich to print as a table
            json_output = [to_dict(req) for req in filtered_requests]
            self.logger.info(json.dumps(json_output, indent=4))
            return True
        except Exception as e:
            self.logger.error(_m(
                "❌ Failed to get miner reclaim requests",
                extra={**self.default_extra, "error": str(e)}
            ))
            return False

    async def finalize_reclaim_request(self, reclaim_request_id: int):
        """
        Finalize a reclaim request by its ID.
        :param reclaim_request_id: The ID of the reclaim request
        :return: True if successful, False otherwise
        """
        try:
            result = await self.collateral_contract.finalize_reclaim(reclaim_request_id)
            self.logger.info(f"✅ Successfully finalized reclaim request: {reclaim_request_id}")
            self.logger.info(result)
            return True
        except Exception as e:
            self.logger.error(_m(
                "❌ Failed to finalize reclaim request",
                extra={**self.default_extra, "error": str(e)}
            ))
            return False

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
                    "validator": executor.validator
                }
                for executor in executors
            ]
            for ex in result:
                self.logger.info(f"{ex['uuid']} {ex['address']}:{ex['port']} -> {ex['validator']}")
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
            self.executor_dao.update(Executor(uuid=uuid.uuid4(), address=address, port=port, validator=validator))
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
            self.executor_dao.delete_by_address_port(address, port)
            self.logger.info("Removed an executor(%s:%d)", address, port)
            return True
        except Exception as e:
            self.logger.error("Failed in removing an executor: %s", str(e))
            return False

    def _get_required_deposit_amount(self, gpu_type: str, gpu_count: int) -> float:
        # Handle missing GPU model gracefully
        unit_tao_amount = REQUIRED_DEPOSIT_AMOUNT.get(gpu_type)
        if unit_tao_amount is None:
            raise ValueError(f"Unknown GPU type: {gpu_type}. Please use one of: {list(REQUIRED_DEPOSIT_AMOUNT.keys())}")

        required_deposit_amount = unit_tao_amount * gpu_count * settings.COLLATERAL_DAYS
        return round(required_deposit_amount, 6)
