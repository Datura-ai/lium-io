from pydantic import BaseModel


class MinerAuthPayload(BaseModel):
    public_key: str
    signature: str
