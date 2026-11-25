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
    # DAH-6740: the same signature for each other hotkey of the listing owner's accounts (hotkey -> signature),
    # so a node set up under one of the owner's own keys takes a key issued for another. Absent from older miners.
    owner_hotkey_signatures: dict[str, str] | None = None


class GetPodLogsPaylod(MinerAuthPayload):
    container_name: str


class MinerIdentityResponse(BaseModel):
    miner_hotkey: str
    nonce: str
    account_claim: str | None = None
    # the SSH port validators dial, sent only while its host keys survive recreation (portal records them)
    ssh_host_key_port: int | None = None
    # `<type> <base64>` host key line -> base64 signature by that key over nonce, hotkey and claim (routes/apis.py)
    host_key_proofs: dict[str, str] | None = None
    # the renting ports this machine declares, as configured (`RENTING_PORT_RANGE`, `RENTING_PORT_MAPPINGS`;
    # both empty means the validator's default range), and the SSH port it reports to validators (`ssh_port`), which
    # they never rent out. Sent whenever the caller asks (`declared_ports=1`), so its absence then marks an older
    # executor: an Edit Address onto this machine checks that it declares every port its pods publish.
    declared_ports: dict[str, str | int | None] | None = None
