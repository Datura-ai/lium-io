import bittensor as bt
from substrateinterface import SubstrateInterface
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_utils import to_hex, keccak
from typing import Tuple, Optional, Any
import bittensor
from core.config import settings

class CliService:
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
                    "event_id": event.value.get("event_id"),
                    "module_id": event.value.get("module_id")
                })
        return summary

    def try_associate_hotkey(self, node: SubstrateInterface, hotkey_ss58: str) -> Any:
        return node.compose_call(
            call_module="SubtensorModule",
            call_function="try_associate_hotkey",
            call_params={"hotkey": hotkey_ss58}
        )

    def make_associate_evm_key_extrinsic(self, node: SubstrateInterface, hotkey_ss58: str, evm_address: str, evm_private_key: str, netuid: int) -> Any:
        block_number = node.query("System", "Number").value
        block_number_bytes = block_number.to_bytes(8, 'little')
        block_number_hash = keccak(block_number_bytes)
        hotkey_bytes = bytes.fromhex(node.ss58_decode(hotkey_ss58))
        message_to_sign_bytes = hotkey_bytes + block_number_hash
        account = Account.from_key(evm_private_key)
        if account.address.lower() != evm_address.lower():
            raise ValueError("Provided EVM private key does not correspond to the provided EVM address.")
        signable_message = encode_defunct(primitive=message_to_sign_bytes)
        signature = account.sign_message(signable_message)
        signature_hex = to_hex(signature.signature)

        print("\n--- Preparing associate_evm_key extrinsic ---")
        print(f"  Hotkey (ss58): {hotkey_ss58}")
        print(f"  EVM Address: {evm_address}")
        print(f"  Netuid: {netuid}")
        print(f"  Block number: {block_number}")
        print(f"  Signature: {signature_hex}")
        print("---------------------------------------------")

        return node.compose_call(
            call_module="SubtensorModule",
            call_function="associate_evm_key",
            call_params={
                "netuid": netuid,
                "hotkey": hotkey_ss58,
                "evm_key": evm_address,
                "block_number": block_number,
                "signature": signature_hex
            }
        )

    def submit_extrinsic(self, node: SubstrateInterface, wallet: bt.wallet, call: Any) -> Any:
        extrinsic = node.create_signed_extrinsic(
            call=call,
            keypair=wallet.coldkey,
        )
        response = node.submit_extrinsic(extrinsic, wait_for_inclusion=True, wait_for_finalization=True)
        return response

    def associate_miner_ethereum_address(self, w3, eth_private_key: str) -> Tuple[bool, Optional[str], Optional[dict]]:
        try:
            network_url = w3.provider.endpoint_uri.lower()
            if "127.0.0.1" in network_url or "localhost" in network_url:
                netuid = 1
                subtensor_network = "ws://127.0.0.1:9944"
            elif "test" in network_url:
                netuid = 37
                subtensor_network = "wss://test.finney.opentensor.ai:443"
            else:
                netuid = 51
                subtensor_network = "wss://entrypoint-finney.opentensor.ai:443"
            node = SubstrateInterface(url=subtensor_network)
            wallet = settings.get_bittensor_wallet()
            hotkey: bittensor.Keypair = wallet.get_hotkey()
            account = Account.from_key(eth_private_key)
            evm_address = account.address
            hotkey_call = self.try_associate_hotkey(node, hotkey.ss58_address)
            if hotkey_call:
                self.submit_extrinsic(node, wallet, hotkey_call)
            evm_call = self.make_associate_evm_key_extrinsic(
                node=node,
                hotkey_ss58=hotkey.ss58_address,
                evm_address=evm_address,
                evm_private_key=eth_private_key,
                netuid=netuid
            )
            if evm_call:
                response = self.submit_extrinsic(node, wallet, evm_call)
                summary = self.print_extrinsic_receipt(response)
                return (response.is_success, None if response.is_success else response.error_message, summary)
            return (False, "Failed to create EVM association call", None)
        except Exception as e:
            return (False, str(e), None) 