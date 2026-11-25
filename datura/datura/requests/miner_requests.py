import enum
import json
import re

import pydantic
from datura.requests.base import BaseRequest


class RequestType(enum.Enum):
    GenericError = "GenericError"
    AcceptJobRequest = "AcceptJobRequest"
    DeclineJobRequest = "DeclineJobRequest"
    AcceptSSHKeyRequest = "AcceptSSHKeyRequest"
    FailedRequest = "FailedRequest"
    UnAuthorizedRequest = "UnAuthorizedRequest"
    SSHKeyRemoved = "SSHKeyRemoved"
    PodLogsResponse = "PodLogsResponse"


class Executor(pydantic.BaseModel):
    uuid: str
    address: str
    port: int


class BaseMinerRequest(BaseRequest):
    message_type: RequestType


class GenericError(BaseMinerRequest):
    message_type: RequestType = RequestType.GenericError
    details: str | None = None


class AcceptJobRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.AcceptJobRequest
    executors: list[Executor]


class DeclineJobRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.DeclineJobRequest


# The formats of the fields a node returns about itself. They reach shell commands and SSH options, so anything that
# is not a plain user name, absolute path, port list or SSH public key is refused here, for the miner and the
# validator alike. Every value the prod fleet returned on 25 Nov fits them. "" is "not reported": the validator's own
# rows for a node that gave no answer carry it, and it has no meaning to a shell.
SSH_USERNAME_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9._-]{0,31})?")
ABSOLUTE_PATH_RE = re.compile(r"(?:/[A-Za-z0-9._+/-]{0,4095})?")
# only the characters a port range is written with: the validator reads the numbers itself and tells the provider
# when the range is malformed, which a node refused here would never hear
PORT_RANGE_RE = re.compile(r"(?!-)[0-9A-Za-z_ ,:-]{0,8192}")
SSH_PUBLIC_KEY_RE = re.compile(
    r"(?:ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(?:256|384|521)|sk-ssh-ed25519@openssh\.com"
    r"|sk-ecdsa-sha2-nistp256@openssh\.com) [A-Za-z0-9+/]{1,16384}={0,3}(?: [A-Za-z0-9@._:+=-]{1,255})?"
)


def _require(pattern: re.Pattern, value: str, field: str) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(f"{field} is not in the expected format")
    return value


class ExecutorSSHInfo(pydantic.BaseModel):
    uuid: str
    address: str
    port: int
    ssh_username: str
    ssh_port: int
    python_path: str
    root_dir: str
    port_range: str | None = None
    port_mappings: str | None = None
    price_per_gpu: float | None = None
    ssh_host_key: str | None = None
    tdx_quote: str | None = None
    # NVIDIA confidential-compute GPU evidence, JSON-encoded
    # {"nonce": <hex>, "arch": <str>, "evidence_list": [...]} — collected in-CVM
    # only and bound to the same nonce as the TDX quote when one was issued.
    nvidia_payload: str | None = None
    # DAH-5729: the node's SSH host key as the portal recorded it when the account proved control of the node, sent
    # by a central miner. The validator requires the host key presented in its SSH handshake to be this one for a
    # hostname-addressed node. None from a standalone or older miner, or a portal that does not record it.
    expected_ssh_host_key: str | None = None
    # DAH-6539: the hotkeys the listing account's nodes may be set up for (its own and its listing hotkey) and, for a
    # pool-hotkey account, the claim its nodes report (MINER_ACCOUNT_CLAIM), as the portal checked them at
    # registration. The validator requires the machine it reached to be set up for them. None from an older miner.
    expected_node_hotkeys: list[str] | None = None
    expected_account_claim: str | None = None

    @pydantic.field_validator("ssh_username")
    @classmethod
    def _plain_user_name(cls, value: str) -> str:
        return _require(SSH_USERNAME_RE, value, "ssh_username")

    @pydantic.field_validator("python_path", "root_dir")
    @classmethod
    def _absolute_path(cls, value: str, info: pydantic.ValidationInfo) -> str:
        return _require(ABSOLUTE_PATH_RE, value, info.field_name)

    @pydantic.field_validator("port_range")
    @classmethod
    def _port_list(cls, value: str | None) -> str | None:
        return value if value is None else _require(PORT_RANGE_RE, value, "port_range")

    @pydantic.field_validator("port_mappings")
    @classmethod
    def _port_pairs(cls, value: str | None) -> str | None:
        if not value:
            return value
        try:
            pairs = json.loads(value)
        except ValueError:
            pairs = None
        if not isinstance(pairs, list) or not all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(type(port) is int and 0 <= port <= 65535 for port in pair)
            for pair in pairs
        ):
            raise ValueError("port_mappings is not in the expected format")
        return value

    @pydantic.field_validator("ssh_host_key")
    @classmethod
    def _ssh_public_key(cls, value: str | None) -> str | None:
        # a key file's padding is not part of the key, and a blank one is no key
        value = value.strip() if value is not None else None
        return _require(SSH_PUBLIC_KEY_RE, value, "ssh_host_key") if value else None


class AcceptSSHKeyRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.AcceptSSHKeyRequest
    # The executors that accepted the validator's key.
    executors: list[ExecutorSSHInfo]
    # DAH-3338: every executor the miner lists for this validator (and executor_id filter), whether
    # or not it accepted the key. An id here but not in `executors` is a node the miner knows and
    # could not reach — ExecutorUnreachable on the validator, not InvalidExecutorId. None from a
    # miner that predates the field: the validator then keeps reading a missing executor as invalid.
    known_executor_ids: list[str] | None = None
    # The known executors that answered the key upload with a refusal, by id, each with the node's own answer
    # (its `detail`, cut short). Such a node is reachable: the validator names the refusal instead of "could not
    # reach". None from an older miner.
    refused_executors: dict[str, str] | None = None


class SSHKeyRemoved(BaseMinerRequest):
    message_type: RequestType = RequestType.SSHKeyRemoved


class FailedRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.FailedRequest
    details: str | None = None


class UnAuthorizedRequest(FailedRequest):
    message_type: RequestType = RequestType.UnAuthorizedRequest


class PodLog(pydantic.BaseModel):
    uuid: str
    container_name: str | None = None
    container_id: str | None = None
    event: str | None = None
    exit_code: int | None = None
    reason: str | None = None
    error: str | None = None
    created_at: str


class PodLogsResponse(BaseMinerRequest):
    message_type: RequestType = RequestType.PodLogsResponse
    logs: list[PodLog] = []
