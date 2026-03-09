from pydantic import BaseModel


class ChutesInstallPayload(BaseModel):
    validator_hotkey: str
    hotkey_ss58: str
    hotkey_seed: str


class ChutesCommandPayload(BaseModel):
    pass
