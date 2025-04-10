from typing import Literal

from pydantic import BaseModel


class Error(BaseModel, extra="allow"):
    msg: str
    type: str
    help: str = ""


class Response(BaseModel, extra="forbid"):
    """Message sent from compute app to validator in response to AuthenticateRequest"""

    status: Literal["error", "success"]
    errors: list[Error] = []


class RentedMachine(BaseModel):
    miner_hotkey: str
    executor_id: str
    executor_ip_address: str
    executor_ip_port: str
    container_name: str | None = None
    is_rented: bool = True
    uptime_in_minutes: int | None = None

class RentedMachineResponse(BaseModel):
    machines: list[RentedMachine]
