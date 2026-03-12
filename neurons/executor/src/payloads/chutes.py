import re

from pydantic import BaseModel, field_validator
from substrateinterface.utils.ss58 import is_valid_ss58_address


_SEED_PATTERN = re.compile(r"^(?:0x)?([0-9a-fA-F]{64})$")
_NODE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,62}$")


class ChutesInstallPayload(BaseModel):
    validator_hotkey: str
    hotkey_ss58: str
    hotkey_seed: str
    node_name: str

    @field_validator("validator_hotkey", "hotkey_ss58")
    @classmethod
    def validate_ss58(cls, value: str) -> str:
        normalized = value.strip()
        if not is_valid_ss58_address(normalized):
            raise ValueError("must be a valid SS58 address")
        return normalized

    @field_validator("hotkey_seed")
    @classmethod
    def validate_hotkey_seed(cls, value: str) -> str:
        normalized = value.strip()
        match = _SEED_PATTERN.fullmatch(normalized)
        if not match:
            raise ValueError("must be 64 hex characters with optional 0x prefix")
        return match.group(1).lower()

    @field_validator("node_name")
    @classmethod
    def validate_node_name(cls, value: str) -> str:
        normalized = value.strip()
        if not _NODE_NAME_PATTERN.fullmatch(normalized):
            raise ValueError(
                "must match hostname pattern ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,62}$"
            )
        return normalized


class ChutesCommandPayload(BaseModel):
    pass
