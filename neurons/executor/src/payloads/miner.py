from pydantic import BaseModel


class SSHkeyPayload(BaseModel):
    public_key: str
    signature: str
