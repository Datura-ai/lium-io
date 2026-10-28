from pydantic import BaseModel


class MinerAuthPayload(BaseModel):
    data_to_sign: str
    signature: str


class UploadSShKeyPayload(MinerAuthPayload):
    public_key: str
    validator_signature: str
    # Validator-issued attestation challenge (hex-encoded 32 bytes), relayed by
    # the miner. When present it is covered by validator_signature and folded
    # into TDX report_data[32:64] and the GPU-evidence nonce.
    nonce: str | None = None
    # The miner hotkey the validator issued this key for, and its signature over
    # ssh_pubkey_signing_blob(public_key, nonce, miner_hotkey). Absent from older validators/miners.
    miner_hotkey: str | None = None
    validator_miner_signature: str | None = None


class GetPodLogsPaylod(MinerAuthPayload):
    container_name: str


class MinerIdentityResponse(BaseModel):
    miner_hotkey: str
    nonce: str
    account_claim: str | None = None
