import enum

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
    # DAH-6539 follow-up (19 Nov 2026): every claim a node of this account's owner may carry, the listing account's and
    # those of the owner's other accounts (same own-key coldkey or same confirmed e-mail). A node an owner set up under
    # one of their own older accounts is theirs, not a relay. None from an older miner: `expected_account_claim` only.
    expected_account_claims: list[str] | None = None
    # shared (pool) hotkey -> the claims a machine set up for it must report to serve this listing: the pool hotkeys of
    # the listing account's owner, whatever hotkey the node is listed under. A machine set up for one of them passes
    # only with one of these claims (an empty claim proves nothing either way, as for the listing hotkey).
    expected_claims_by_hotkey: dict[str, list[str]] | None = None
    # DAH-6539: until this ISO-8601 time the portal read this node came from vouches for it. Until then the validator
    # also accepts `owner_binding` (the owner-wide binding: the hotkeys and claims of every account of the listing's
    # owner, keyed by the four expected_* field names above); after it, or when it does not parse, only the fields
    # above, which always hold the listing account's own binding (all a validator that predates `owner_binding`
    # reads): a miner can serve a cached snapshot, and an upload can finish, after it. None from an older miner or
    # portal: the fields above as they are.
    snapshot_valid_until: str | None = None
    owner_binding: dict | None = None


class AcceptSSHKeyRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.AcceptSSHKeyRequest
    # The executors that accepted the validator's key.
    executors: list[ExecutorSSHInfo]
    # DAH-3338: every executor the miner lists for this validator (and executor_id filter), whether
    # or not it accepted the key. An id here but not in `executors` is a node the miner knows and
    # could not reach — ExecutorUnreachable on the validator, not InvalidExecutorId. None from a
    # miner that predates the field: the validator then keeps reading a missing executor as invalid.
    known_executor_ids: list[str] | None = None


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
