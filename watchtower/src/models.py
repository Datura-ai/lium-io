from pydantic import BaseModel


class SignaturePayload(BaseModel):
    """Base payload class for requests that require signature verification"""
    signature: str


class WatchtowerDigestResponse(SignaturePayload):
    """Response from watchtower endpoint containing image digest and signature"""
    digest: str
    timestamp: int
