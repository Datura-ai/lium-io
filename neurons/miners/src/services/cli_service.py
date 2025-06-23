import bittensor as bt
from substrateinterface import SubstrateInterface
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_utils import to_hex, keccak
from typing import Tuple, Optional, Any
from core.config import settings

class CliService:
    def __init__(self):
        self.wallet = settings.get_bittensor_wallet()
        self.netuid = settings.BITTENSOR_NETUID
        self.config = settings.get_bittensor_config()
        self.hotkey = self.wallet.get_hotkey().ss58_address

    def get_node(self) -> SubstrateInterface:
        subtensor = bt.subtensor(config=self.config)
        return subtensor.substrate

    def print_extrinsic_receipt(self, receipt) -> dict:
        """Returns a summary of the extrinsic receipt as a dict."""
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

    def try_associate_hotkey(self, node: SubstrateInterface) -> Any:
        return node.compose_call(
            call_module="SubtensorModule",
            call_function="try_associate_hotkey",
            call_params={"hotkey": self.hotkey}
        )

    def make_associate_evm_key_extrinsic(self, node: SubstrateInterface, evm_private_key: str) -> Any:
        block_number = node.query("System", "Number").value
        block_number_bytes = block_number.to_bytes(8, 'little')
        block_number_hash = keccak(block_number_bytes)
        hotkey_bytes = bytes.fromhex(node.ss58_decode(self.hotkey))
        message_to_sign_bytes = hotkey_bytes + block_number_hash
        signable_message = encode_defunct(primitive=message_to_sign_bytes)

        account = Account.from_key(evm_private_key)
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
                "hotkey": self.hotkey,
                "evm_key": account.address,
                "block_number": block_number,
                "signature": signature_hex
            }
        )

    def submit_extrinsic(self, node: SubstrateInterface, call: Any) -> Any:
        extrinsic = node.create_signed_extrinsic(
            call=call,
            keypair=self.wallet.coldkey,
        )
        response = node.submit_extrinsic(extrinsic, wait_for_inclusion=True, wait_for_finalization=True)
        return response

    def associate_miner_ethereum_address(self, eth_private_key: str) -> Tuple[bool, Optional[str], Optional[dict]]:
        try:
            node = self.get_node()

            hotkey_call = self.try_associate_hotkey(node)
            if hotkey_call:
                self.submit_extrinsic(node, hotkey_call)
            evm_call = self.make_associate_evm_key_extrinsic(
                node=node,
                evm_private_key=eth_private_key,
            )
            if evm_call:
                response = self.submit_extrinsic(node, evm_call)
                summary = self.print_extrinsic_receipt(response)
                return (response.is_success, None if response.is_success else response.error_message, summary)
            return (False, "Failed to create EVM association call", None)
        except Exception as e:
            return (False, str(e), None)
