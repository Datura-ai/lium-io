"""End to end: the executor image's own `liumd` binary
(`neurons/executor/liumd/liumd`, the pinned dev-static build) answering `LiumdExecClient` and the
shadow over a real SSH exec channel.

The SSH server is asyncssh's, standing in for the executor container's sshd: it runs each exec
request as `/bin/sh -c <command>` with a minimal environment plus whatever the client asked to
set, and records both, so "no environment" is checked on the wire. `/usr/local/bin/liumd` is
rewritten to a copy of the image's wrapper (`liumd/liumd.sh`) whose paths point into a temporary
image: the host files come from the executor's own generator (`liumd_host_files.py`), the children
are fakes as in the liumd repo's validator parity check (`/bin/sh` pinned as the launcher, shell
scripts as the wrappers, files as the `.so`s) and print what the validator's judges accept.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import platform
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import asyncssh
import bittensor
import pytest
import services.matrix_validation_service as mvs
import services.verifyx_validation_service as vvs
from datura.requests.miner_requests import ExecutorSSHInfo
from datura.requests.validator_requests import MatmulStep, VerifyXStep
from protocol.vc_protocol.validator_requests import ValidationEvent
from services.liumd_exec_client import LIUMD_COMMAND, LiumdExecClient, LiumdRefusal
from services.local_verify_client import (
    STEP_NAMES,
    LocalVerifyAnswer,
    LocalVerifyUnavailable,
    build_intent,
    executor_deadline_s,
)
from services.task.liumd_shadow import run_liumd_shadow

from tests.helpers import build_context_config, build_services, build_state, make_context

EXECUTOR = Path(__file__).resolve().parents[2] / "executor"
BINARY = EXECUTOR / "liumd" / "liumd"
WRAPPER = EXECUTOR / "liumd" / "liumd.sh"
MINER = "5E2eMinerHotkeyOfTheExecutor"
UUID = "exec-e2e"
SPECS = {"gpu": {"count": 1, "details": [{"uuid": "GPU-1", "name": "H100", "capacity": 81559}]}}

pytestmark = pytest.mark.skipif(
    not BINARY.is_file() or platform.system() != "Linux" or platform.machine() != "x86_64",
    reason="the committed liumd binary is x86_64 Linux",
)

# Both wrappers get liumd's `--lib <held fd>` last; each prints what the validator's judge accepts
# from the real script. The matmul's sealed blob is unsealed by the fake libdmcompverify below.
MATMUL_SCRIPT = """
while [ $# -gt 0 ]; do case "$1" in --lib) lib=$2; shift 2 ;; *) shift ;; esac; done
echo "lib:$(cat "$lib")"
echo "UUID:  GPU-1"
echo 'RESULT_JSON: {"uuid": "GPU-1", "metrics": {"tflops": 42.0}, "sealed": "cafe"}'
"""
VERIFYX_SCRIPT = """
while [ $# -gt 0 ]; do case "$1" in --cipher_text) ct=$2; shift 2 ;; --lib) shift 2 ;; *) shift ;; esac; done
echo "${ct}-ok"
"""
VERIFYX_LIB = b"verifyx-lib"


def _load_executor_host_files():
    spec = importlib.util.spec_from_file_location(
        "executor_liumd_host_files", EXECUTOR / "src" / "liumd_host_files.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Image:
    """A temporary executor image: fake children, fixture docker, host files, the wrapper."""

    def __init__(self, root: Path, validator_ss58: str, verifyx_script: str = VERIFYX_SCRIPT):
        self.root = root
        host_files = _load_executor_host_files()
        sys_root = root / "sys"

        def put(rel: str, data: bytes) -> Path:
            path = sys_root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return path

        python = put("bin/python", Path("/bin/sh").read_bytes())
        python.chmod(0o755)
        children = {
            "python": python,
            "matmul_script": put("src/decrypt_challenge.sh", MATMUL_SCRIPT.encode()),
            "matmul": put("lib/libdmcompverify.so", b"matmul-lib"),
            "verifyx_script": put("src/verifyx_executor.sh", verifyx_script.encode()),
            "verifyx": put("lib/libverifyx.so", VERIFYX_LIB),
            "inspector": put("lib/libinspector.so", b"inspector-lib"),
        }
        put(
            ".liumd-fixture/docker_info.json",
            json.dumps(
                {
                    "ServerVersion": "27.3.1",
                    "DockerRootDir": "/var/lib/docker",
                    "DefaultRuntime": "runc",
                    "Runtimes": {"runc": {}, "sysbox-runc": {}},
                }
            ).encode(),
        )
        put(".liumd-fixture/containers.json", b"[]")
        put(
            ".liumd-fixture/statvfs.json",
            json.dumps(
                {"/var/lib/docker": {"total_bytes": 100, "free_bytes": 40, "used_bytes": 55}}
            ).encode(),
        )
        etc = root / "etc/liumd"
        manifest = {
            "schema": host_files.CHILDREN_SCHEMA,
            "children": [
                {"name": n, "path": str(p), "sha256": host_files.sha256_file(p)}
                for n, p in children.items()
            ],
        }
        host_files.write_atomic(etc / "children.json", json.dumps(manifest))
        self.nonces = root / "var/lib/liumd/nonces"
        host_files.write_host_files(
            etc_dir=etc,
            nonce_dir=self.nonces,
            validator_hotkeys={"current": validator_ss58},
            miner_hotkey=MINER,
            port_range="20000-20003",
            port_mappings=None,
            ssh_port=20000,
        )
        binary = root / "usr/local/lib/liumd/liumd"
        binary.parent.mkdir(parents=True)
        shutil.copy(BINARY, binary)
        # The image's wrapper, with the image paths moved under `root`; the defaults the binary
        # reads from /etc/liumd and /var/lib/liumd are the same files, named explicitly.
        text = WRAPPER.read_text()
        text = text.replace("/etc/liumd/children.json", str(etc / "children.json"))
        text = text.replace("/usr/local/lib/liumd/liumd", str(binary))
        fixture_env = (
            f"export LIUMD_VALIDATOR_HOTKEYS_FILE={etc / 'validator_hotkeys'}\n"
            f"export LIUMD_MINER_HOTKEY_FILE={etc / 'miner_hotkey'}\n"
            f"export LIUMD_PORTS_FILE={etc / 'ports.json'}\n"
            f"export LIUMD_NONCE_DIR={self.nonces}\n"
            f"export LIUMD_SYS_ROOT={sys_root}\n"
        )
        self.wrapper = root / "usr/local/bin/liumd"
        self.wrapper.parent.mkdir(parents=True)
        self.wrapper.write_text(text.replace("exec ", fixture_env + "exec ", 1))
        self.wrapper.chmod(0o755)


class _Server(asyncssh.SSHServer):
    def __init__(self, sshd: FakeSshd):
        self.sshd = sshd

    def begin_auth(self, username: str) -> bool:
        return False

    def session_requested(self) -> bool:
        return self.sshd.allow_sessions


class FakeSshd:
    """sshd as the executor container runs it, for exec requests only."""

    def __init__(self, liumd_path: Path | None, *, allow_sessions: bool = True):
        self.liumd_path = liumd_path
        self.allow_sessions = allow_sessions
        self.requests: list[dict] = []
        self.server = None

    async def __aenter__(self):
        key = asyncssh.generate_private_key("ssh-ed25519")
        # asyncssh consults session_requested only when no process factory is given.
        factory = {"process_factory": self._handle} if self.allow_sessions else {}
        self.server = await asyncssh.create_server(
            lambda: _Server(self), "127.0.0.1", 0, server_host_keys=[key], encoding=None, **factory
        )
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    async def _handle(self, process: asyncssh.SSHServerProcess) -> None:
        requested_env = {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in dict(process.env).items()
        }
        self.requests.append(
            {"command": process.command, "env": requested_env, "pty": process.term_type}
        )
        command = process.command or ""
        target = str(self.liumd_path) if self.liumd_path else "/nonexistent/liumd"
        command = command.replace("/usr/local/bin/liumd", target, 1)
        data = await process.stdin.read()
        run = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin", **requested_env},
        )
        out, err = await run.communicate(data)
        process.stdout.write(out)
        process.stderr.write(err)
        process.exit(run.returncode)

    def connect(self):
        return asyncssh.connect(
            "127.0.0.1",
            self.port,
            username="root",
            known_hosts=None,
            client_keys=None,
            agent_path=None,
            config=None,
        )


@pytest.fixture
def keypair():
    return bittensor.Keypair.create_from_uri("//LiumdE2eValidator")


@pytest.fixture
def image(tmp_path, keypair):
    return Image(tmp_path / "image", keypair.ss58_address)


def _intent(**overrides):
    fields = dict(
        executor_uuid=UUID,
        miner_hotkey=MINER,
        matmul=MatmulStep(dim_n=1900, dim_k=2_000_000, seed=3, cipher_text="c0"),
        verifyx=VerifyXStep(seed=5, cipher_text="vx-challenge"),
        parallel_gpu=False,
        deadline_s=executor_deadline_s(90),
    )
    fields.update(overrides)
    return build_intent(**fields)


@pytest.mark.asyncio
async def test_the_binary_answers_a_verify_result_through_parse_answer(image, keypair):
    async with FakeSshd(image.wrapper) as sshd, sshd.connect() as ssh:
        answer = await LiumdExecClient(keypair, timeout_s=60).run(ssh, _intent())

    assert isinstance(answer, LocalVerifyAnswer)
    assert set(answer.steps) == set(STEP_NAMES)
    assert {n: s.status for n, s in answer.steps.items()} == {n: "ok" for n in STEP_NAMES}
    assert answer.executor_version.startswith("liumd/")
    assert "RESULT_JSON" in answer.steps["matmul"].stdout
    assert answer.steps["verifyx"].stdout.strip() == "vx-challenge-ok"
    assert answer.steps["verifyx"].data["lib_sha256"] == hashlib.sha256(VERIFYX_LIB).hexdigest()
    # The exec request carried the command alone: no variable, no pty.
    assert sshd.requests == [{"command": LIUMD_COMMAND, "env": {}, "pty": None}]


@pytest.mark.asyncio
async def test_a_signer_the_host_does_not_trust_is_exit_4(image):
    stranger = bittensor.Keypair.create_from_uri("//NotThePinnedValidator")
    async with FakeSshd(image.wrapper) as sshd, sshd.connect() as ssh:
        answer = await LiumdExecClient(stranger, timeout_s=60).run(ssh, _intent())

    assert isinstance(answer, LiumdRefusal)
    assert (answer.exit_status, answer.error, answer.echoed) == (4, "bad_signature", True)


@pytest.mark.asyncio
async def test_a_replayed_intent_is_exit_5(image, keypair):
    intent = _intent()
    client = LiumdExecClient(keypair, timeout_s=60)
    async with FakeSshd(image.wrapper) as sshd, sshd.connect() as ssh:
        first = await client.run(ssh, intent)
        again = await client.run(ssh, intent)

    assert isinstance(first, LocalVerifyAnswer)
    assert isinstance(again, LiumdRefusal)
    assert (again.exit_status, again.error) == (5, "nonce_replayed")


@pytest.mark.asyncio
async def test_an_intent_for_another_miner_is_exit_4(image, keypair):
    async with FakeSshd(image.wrapper) as sshd, sshd.connect() as ssh:
        answer = await LiumdExecClient(keypair, timeout_s=60).run(
            ssh, _intent(miner_hotkey="5SomeoneElsesMiner")
        )

    assert isinstance(answer, LiumdRefusal)
    assert (answer.exit_status, answer.error) == (4, "intent_refused")


@pytest.mark.asyncio
async def test_a_host_missing_its_hotkeys_file_is_exit_6(image, keypair):
    (image.root / "etc/liumd/validator_hotkeys").unlink()
    async with FakeSshd(image.wrapper) as sshd, sshd.connect() as ssh:
        answer = await LiumdExecClient(keypair, timeout_s=60).run(ssh, _intent())

    assert isinstance(answer, LiumdRefusal)
    assert (answer.exit_status, answer.error) == (6, "agent_error")


@pytest.mark.asyncio
async def test_a_host_without_liumd_is_not_supported(keypair, tmp_path):
    not_executable = tmp_path / "liumd"
    not_executable.write_text("#!/bin/sh\n")
    for path, status in ((None, 127), (not_executable, 126)):
        async with FakeSshd(path) as sshd, sshd.connect() as ssh:
            with pytest.raises(LocalVerifyUnavailable) as err:
                await LiumdExecClient(keypair, timeout_s=10).run(ssh, _intent())
        assert err.value.reason == "not_supported"
        assert f"exit {status}" in err.value.detail


@pytest.mark.asyncio
async def test_a_refused_exec_channel_is_not_supported(image, keypair):
    async with FakeSshd(image.wrapper, allow_sessions=False) as sshd, sshd.connect() as ssh:
        with pytest.raises(LocalVerifyUnavailable) as err:
            await LiumdExecClient(keypair, timeout_s=10).run(ssh, _intent())

    assert err.value.reason == "not_supported"


def _judging_services(monkeypatch):
    wrapper = MagicMock(name="DMCompVerifyWrapper")
    wrapper.DMCompVerify_new.return_value = "ptr"
    wrapper.getCipherText.return_value = "deadbeef"
    wrapper._has_sealed = True
    generated: list[str] = []
    wrapper.generateChallenge.side_effect = lambda ptr, seed, info, uuid: generated.append(uuid)
    wrapper.unsealResult.side_effect = lambda ptr, blob: json.dumps(
        {"uuid": generated[-1], "metrics": {"tflops": 42.0}}
    )
    monkeypatch.setattr(mvs, "DMCompVerifyWrapper", lambda *_a, **_kw: wrapper)

    class _VerifyX:
        def __init__(self, lib_name, seed):
            self.seed = seed

        def generate_challenge(self, challenge_input):
            return f"vx{self.seed}".ljust(vvs.MIN_CIPHER_LEN, "0")

        def verify_response(self, response):
            if not response.endswith("-ok"):
                raise RuntimeError("cipher rejected")
            return {"ok": True}

    monkeypatch.setattr(vvs, "VerifyXValidator", _VerifyX)
    monkeypatch.setattr(vvs, "sha256_from_path", lambda _p: hashlib.sha256(VERIFYX_LIB).hexdigest())
    monkeypatch.setattr(
        vvs,
        "_perform_verification_checks",
        lambda payload: {"success": True, "ram": {}, "network": {"download_speed": 900.0}},
    )
    return mvs.ValidationService(), vvs.VerifyXValidationService()


def _event(check_id: str, reason: str) -> ValidationEvent:
    return ValidationEvent(
        event="e",
        reason_code=reason,
        severity="info",
        impact="",
        check_id=check_id,
        when=datetime.now(UTC),
        context={"execution_time_ms": 1500},
    )


@pytest.mark.asyncio
async def test_the_shadow_compares_the_binarys_answer_with_todays_verdicts(
    image, keypair, monkeypatch
):
    validation, verifyx = _judging_services(monkeypatch)
    # The top of getrandbits(64)'s range.
    monkeypatch.setattr(vvs.random, "getrandbits", lambda bits: 2**64 - 1)
    events = [
        _event("gpu.validate.verifyx", "VERIFYX_OK"),
        _event("gpu.validate.capability", "GPU_VERIFY_OK"),
    ]
    async with FakeSshd(image.wrapper) as sshd, sshd.connect() as ssh:
        ctx = make_context(
            executor=ExecutorSSHInfo(
                uuid=UUID,
                address="127.0.0.1",
                port=8001,
                ssh_username="root",
                ssh_port=22,
                python_path="/usr/bin/python",
                root_dir="/root/app",
            ),
            miner_hotkey=MINER,
            ssh=ssh,
            services=build_services(
                validation=validation,
                verifyx=verifyx,
                redis=SimpleNamespace(renting_in_progress=AsyncMock(return_value=False)),
            ),
            config=build_context_config(
                validator_keypair=keypair, first_pass=True, verifyx_enabled=True
            ),
            state=build_state(specs=SPECS),
        )
        record = await run_liumd_shadow(
            ctx,
            ok=True,
            events=events,
            deadline_monotonic=time.monotonic() + 10_000,
        )

    assert record["outcome"] == "compared", record
    assert record["agree"] is True
    for step in ("matmul", "verifyx"):
        assert record["steps"][step]["today"] == "pass"
        assert record["steps"][step]["liumd_verdict"] == "pass", record["steps"][step]
        assert record["steps"][step]["agree"] is True
    assert {record["steps"][n]["liumd_status"] for n in ("docker", "ports", "inspector")} == {"ok"}
    assert record["executor_version"].startswith("liumd/")
    assert [r["env"] for r in sshd.requests] == [{}]


# The validator draws the seed with random.getrandbits(64).
@pytest.mark.parametrize("seed", [2**63 + 5, 2**64 - 1])
@pytest.mark.asyncio
async def test_a_verifyx_seed_above_i64_reaches_the_child_as_sent(tmp_path, keypair, seed):
    image = Image(tmp_path / "image", keypair.ss58_address, verifyx_script='echo "argv: $*"\n')
    async with FakeSshd(image.wrapper) as sshd, sshd.connect() as ssh:
        answer = await LiumdExecClient(keypair, timeout_s=60).run(
            ssh, _intent(verifyx=VerifyXStep(seed=seed, cipher_text="vx-challenge"))
        )

    assert isinstance(answer, LocalVerifyAnswer), answer
    assert answer.steps["verifyx"].stdout.split()[:3] == ["argv:", "--seed", str(seed)]
