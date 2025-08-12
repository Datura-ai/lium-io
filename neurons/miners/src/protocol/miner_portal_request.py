from typing import Literal

from pydantic import BaseModel


class Error(BaseModel, extra="allow"):
    msg: str
    type: str
    help: str = ""


class Response(BaseModel, extra="forbid"):
    """Message sent from miner portal to miner in response to AuthenticateRequest"""

    status: Literal["error", "success"]
    errors: list[Error] = []


class AuthenticationError(Exception):
    def __init__(self, reason: str, errors: list[Error]):
        self.reason = reason
        self.errors = errors
