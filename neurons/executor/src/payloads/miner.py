from pydantic import BaseModel


class MinerAuthPayload(BaseModel):
    data_to_sign: str
    signature: str


class UploadSShKeyPayload(MinerAuthPayload):
    public_key: str
    validator_signature: str | None = None


class GetPodLogsPaylod(MinerAuthPayload):
    container_name: str
