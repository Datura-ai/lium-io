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


class GetPodLogsPaylod(MinerAuthPayload):
    container_name: str
