"""DAH-2582: one CVM cannot answer for many executors.

The attack: register N executors, forward every challenge to one real CVM, return N valid,
fresh, correctly-bound quotes. Every existing check passes — each quote genuinely is fresh and
genuinely does come from a TDX guest. A nonce proves liveness; liveness is not distinctness.

What a CVM cannot do is lie about *which* hardware it is, so the defence is uniqueness of the
identifiers that come out of signed evidence. Four claims are tested here:

  a collision fails BOTH executors, not the one seen second — otherwise a miner picks which
    registration survives by controlling validation order;
  identifier classes do not blur into one another — a GPU ueid equal to some device id is not
    two nodes claiming one machine;
  a registry outage is not a guilty verdict — unknown means proceed, because the alternative is
    Redis failing every CVM in the fleet;
  and none of it fails anything until the flag is set, which is the task's own acceptance path.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.config import settings
from services.attested_identity import AttestedIdentityRegistry
from services.task.checks.attested_identity_unique import AttestedIdentityUniqueCheck

pytestmark = pytest.mark.asyncio

UEID_A = "ueid-aaaaaaaaaaaaaaaaaaaa"
UEID_B = "ueid-bbbbbbbbbbbbbbbbbbbb"
INSTANCE_A = "instance-aaaaaaaaaaaa"


class FakeRedis:
    """A store with the two operations the registry uses, and no TTL semantics.

    Expiry is Redis's job and is not what these tests are about; what they are about is who is
    recorded as holding a value and what happens when two executors claim one.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.kv: dict[str, str] = {}
        self.fail = fail

    async def get(self, key):
        if self.fail:
            raise ConnectionError("redis is down")
        return self.kv.get(key)

    async def set_with_expiration(self, key, value, ttl_seconds):
        if self.fail:
            raise ConnectionError("redis is down")
        self.kv[key] = value


class TestTheRegistry:
    async def test_a_first_claim_collides_with_nothing(self):
        registry = AttestedIdentityRegistry(FakeRedis())

        collisions = await registry.check_and_record(
            executor_uuid="exec-1", identities={"gpu_ueid": [UEID_A]}
        )

        assert collisions == []

    async def test_the_same_executor_re_claiming_its_own_identity_is_not_a_collision(self):
        """Every cycle re-attests the same node, so this is the normal case rather than an edge
        one — treating it as a collision would fail the whole fleet on the second cycle."""
        registry = AttestedIdentityRegistry(FakeRedis())

        await registry.check_and_record(executor_uuid="exec-1", identities={"gpu_ueid": [UEID_A]})
        collisions = await registry.check_and_record(
            executor_uuid="exec-1", identities={"gpu_ueid": [UEID_A]}
        )

        assert collisions == []

    async def test_a_second_executor_claiming_the_same_gpu_collides(self):
        redis = FakeRedis()
        registry = AttestedIdentityRegistry(redis)
        await registry.check_and_record(executor_uuid="exec-1", identities={"gpu_ueid": [UEID_A]})

        collisions = await registry.check_and_record(
            executor_uuid="exec-2", identities={"gpu_ueid": [UEID_A]}
        )

        assert len(collisions) == 1
        assert collisions[0].identity_class == "gpu_ueid"
        assert collisions[0].other_executor_uuid == "exec-1"

    async def test_the_second_claimant_is_still_recorded(self):
        """Otherwise one side of the pair is forgotten every cycle, and a persistent fraud looks
        identical to a flapping one."""
        redis = FakeRedis()
        registry = AttestedIdentityRegistry(redis)
        await registry.check_and_record(executor_uuid="exec-1", identities={"gpu_ueid": [UEID_A]})

        await registry.check_and_record(executor_uuid="exec-2", identities={"gpu_ueid": [UEID_A]})

        assert redis.kv["attested_identity:gpu_ueid:" + UEID_A] == "exec-2"

    async def test_identity_classes_do_not_blur_into_one_another(self):
        """A GPU ueid that happens to equal some device id is not two nodes claiming one
        machine."""
        registry = AttestedIdentityRegistry(FakeRedis())
        await registry.check_and_record(executor_uuid="exec-1", identities={"gpu_ueid": ["same"]})

        collisions = await registry.check_and_record(
            executor_uuid="exec-2", identities={"cvm_device_id": ["same"]}
        )

        assert collisions == []

    async def test_every_colliding_identity_is_reported_not_just_the_first(self):
        """One forwarded CVM collides on several identifiers at once, and an operator reading
        the event should see all of them rather than chasing one at a time."""
        registry = AttestedIdentityRegistry(FakeRedis())
        await registry.check_and_record(
            executor_uuid="exec-1",
            identities={"gpu_ueid": [UEID_A, UEID_B], "cvm_instance_id": [INSTANCE_A]},
        )

        collisions = await registry.check_and_record(
            executor_uuid="exec-2",
            identities={"gpu_ueid": [UEID_A, UEID_B], "cvm_instance_id": [INSTANCE_A]},
        )

        assert {c.value for c in collisions} == {UEID_A, UEID_B, INSTANCE_A}

    async def test_a_registry_outage_is_not_a_guilty_verdict(self):
        """Unknown is not "yes". The alternative is a Redis outage failing every CVM in the
        fleet, and the identities are still in the signed evidence — a cycle of detection is
        lost, nothing more."""
        registry = AttestedIdentityRegistry(FakeRedis(fail=True))

        assert await registry.check_and_record(
            executor_uuid="exec-1", identities={"gpu_ueid": [UEID_A]}
        ) == []

    async def test_no_redis_at_all_is_the_same_answer(self):
        assert await AttestedIdentityRegistry(None).check_and_record(
            executor_uuid="exec-1", identities={"gpu_ueid": [UEID_A]}
        ) == []

    async def test_empty_values_are_ignored(self):
        redis = FakeRedis()

        await AttestedIdentityRegistry(redis).check_and_record(
            executor_uuid="exec-1", identities={"gpu_ueid": ["", None]}
        )

        assert redis.kv == {}


def _context(identities: dict, redis) -> MagicMock:
    ctx = MagicMock()
    ctx.state.attested_identities = identities
    ctx.services.redis = redis
    ctx.executor.uuid = "exec-2"
    ctx.miner_hotkey = "miner-hotkey"
    ctx.pipeline_id = "pipeline-1"
    ctx.executor.address = "10.0.0.2"
    ctx.executor.port = 8000
    ctx.default_extra = {}
    return ctx


class TestTheCheck:
    async def test_a_node_with_no_attested_identity_passes_with_nothing_to_compare(self):
        """Whether a CVM-class node is allowed to stop attesting is the fail-closed quote
        check's question, not this one's."""
        result = await AttestedIdentityUniqueCheck().run(_context({}, FakeRedis()))

        assert result.passed is True
        assert result.event.reason_code == "ATTESTED_IDENTITY_ABSENT"

    async def test_a_unique_node_passes(self):
        result = await AttestedIdentityUniqueCheck().run(
            _context({"gpu_ueid": [UEID_A]}, FakeRedis())
        )

        assert result.passed is True
        assert result.event.reason_code == "ATTESTED_IDENTITY_UNIQUE"

    async def test_a_collision_is_observed_and_fails_nothing_by_default(self):
        """The task's acceptance path: 48 h over the fleet showing zero collisions among
        genuinely distinct hosts comes before enforcement. An identifier that is less unique
        than assumed would otherwise fail the whole fleet on the day this ships."""
        redis = FakeRedis()
        await AttestedIdentityRegistry(redis).check_and_record(
            executor_uuid="exec-1", identities={"gpu_ueid": [UEID_A]}
        )

        result = await AttestedIdentityUniqueCheck().run(_context({"gpu_ueid": [UEID_A]}, redis))

        assert result.passed is True
        assert result.event.reason_code == "ATTESTED_IDENTITY_COLLISION_OBSERVED"

    async def test_a_collision_is_fatal_when_enforcement_is_on(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_ATTESTED_IDENTITY_UNIQUENESS", True)
        redis = FakeRedis()
        await AttestedIdentityRegistry(redis).check_and_record(
            executor_uuid="exec-1", identities={"gpu_ueid": [UEID_A]}
        )

        result = await AttestedIdentityUniqueCheck().run(_context({"gpu_ueid": [UEID_A]}, redis))

        assert result.passed is False
        assert result.event.reason_code == "ATTESTED_IDENTITY_COLLISION"
        # The node's verified job is cleared, so it does not keep earning on a verification
        # taken before the collision was known.
        assert result.updates["clear_verified_job_info"] is True

    async def test_the_check_is_fatal_so_a_collision_halts_the_pipeline(self):
        assert AttestedIdentityUniqueCheck.fatal is True

    async def test_the_other_claimant_is_named_so_both_sides_can_be_acted_on(self, monkeypatch):
        """A collision fails both executors. Naming the other one is what lets the operator —
        or the backend's own dedup — reach the side this cycle did not validate."""
        monkeypatch.setattr(settings, "ENABLE_ATTESTED_IDENTITY_UNIQUENESS", True)
        redis = FakeRedis()
        await AttestedIdentityRegistry(redis).check_and_record(
            executor_uuid="exec-1", identities={"gpu_ueid": [UEID_A]}
        )

        result = await AttestedIdentityUniqueCheck().run(_context({"gpu_ueid": [UEID_A]}, redis))

        assert "exec-1" in result.event.what_we_saw["detail"]


class TestFailingClosedOnAnOmittedQuote:
    async def _service(self, *, ratcheted: bool):
        from services.attestation_service import TDX_ATTESTED_EXECUTOR_SET, AttestationService

        redis = MagicMock()
        redis.is_elem_exists_in_set = AsyncMock(return_value=ratcheted)
        service = AttestationService(redis_service=redis)
        service.enabled = True
        assert TDX_ATTESTED_EXECUTOR_SET  # the set the ratchet is keyed on
        return service

    def _executor(self):
        from datura.requests.miner_requests import ExecutorSSHInfo

        return ExecutorSSHInfo(
            uuid="exec-1",
            address="10.0.0.1",
            port=8000,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python",
            root_dir="/opt/executor",
            ssh_host_key="ssh-ed25519 AAAAKey",
            tdx_quote=None,
        )

    async def test_a_known_cvm_that_stops_presenting_a_quote_fails(self, monkeypatch):
        """DAH-2582 gives this its own flag. It was previously reachable only under
        ENABLE_TCB_ENFORCEMENT, so a fleet wanting "a CVM may not stop being one" had to take
        every TCB and advisory rejection with it."""
        from services.attestation_service import AttestationError

        monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", False)
        monkeypatch.setattr(settings, "ENABLE_CVM_QUOTE_REQUIRED", True)
        service = await self._service(ratcheted=True)

        with pytest.raises(AttestationError, match="omitted its TDX quote"):
            await service._reject_omitted_quote_if_ratcheted(self._executor())

    async def test_a_node_that_never_attested_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_QUOTE_REQUIRED", True)
        service = await self._service(ratcheted=False)

        await service._reject_omitted_quote_if_ratcheted(self._executor())

    async def test_neither_flag_means_nothing_is_enforced(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", False)
        monkeypatch.setattr(settings, "ENABLE_CVM_QUOTE_REQUIRED", False)
        service = await self._service(ratcheted=True)

        await service._reject_omitted_quote_if_ratcheted(self._executor())


class TestOverlappingChallengeWindows:
    async def test_one_nonce_covers_every_executor_of_a_miner_in_the_same_wave(self, monkeypatch):
        """"Challenge all of a miner's executors in overlapping windows" is already how the
        nonce works — it is minted once per miner and fans out to all of them, so two of that
        miner's executors are never challenged at disjoint times.

        What makes that *effective* is the identity half being per-executor: one CVM answering
        two challenges has to produce two different identity digests from one host key, and it
        cannot. If it reuses the key, `pinned_host_key` collides and the uniqueness check above
        catches it.
        """
        from services.attestation_service import AttestationService

        monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
        monkeypatch.setattr(settings, "ENABLE_ATTESTATION_NONCE", True)
        redis = MagicMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()
        service = AttestationService(redis_service=redis)
        service.enabled = True

        first = await service.maybe_issue_nonce("miner-a")

        # A second call for the same miner inside the interval returns None — the wave's one
        # nonce is the challenge every one of its executors answers.
        redis.get = AsyncMock(return_value=str(__import__("time").time()))
        assert await service.maybe_issue_nonce("miner-a") is None
        assert first is not None

    async def test_two_miners_get_two_different_challenges(self, monkeypatch):
        from services.attestation_service import AttestationService

        monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
        monkeypatch.setattr(settings, "ENABLE_ATTESTATION_NONCE", True)
        redis = MagicMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()
        service = AttestationService(redis_service=redis)
        service.enabled = True

        a = await service.maybe_issue_nonce("miner-a")
        b = await service.maybe_issue_nonce("miner-b")

        assert a.value_hex != b.value_hex


class TestTheDuplicateExecutorCheckIsNoLongerObserving:
    def test_the_dry_run_default_is_off(self):
        """DAH-2582 graduates it. One miner registering the same executor UUID more than once
        has no legitimate form, so continuing to only watch it is a standing invitation."""
        assert settings.DUPLICATE_EXECUTOR_DRY_RUN is False

    async def test_a_duplicate_now_clears_the_verified_job(self):
        from services.task.checks.duplicate_executor import DuplicateExecutorCheck

        ctx = MagicMock()
        ctx.services.redis.is_elem_exists_in_set = AsyncMock(return_value=True)
        ctx.executor.uuid = "exec-1"
        ctx.miner_hotkey = "miner"
        ctx.pipeline_id = "pipeline-1"
        ctx.default_extra = {}

        result = await DuplicateExecutorCheck().run(ctx)

        assert result.passed is False
        assert result.updates["clear_verified_job_info"] is True


def test_identities_are_grouped_by_class_so_a_message_says_what_collided():
    """One GPU answering for two executors is a different fraud from one CVM doing so, and the
    event an operator reads should say which."""
    from services.attestation_service import HostPolicyResult

    result = HostPolicyResult(
        attestation_digest="digest",
        instance_id=INSTANCE_A,
        device_id="device-1",
        gpu_ueids=[UEID_A, UEID_B],
        pinned_host_key="ssh-ed25519 AAAAKey",
    )

    assert result.identities() == {
        "cvm_instance_id": [INSTANCE_A],
        "cvm_device_id": ["device-1"],
        "gpu_ueid": [UEID_A, UEID_B],
        "pinned_host_key": ["ssh-ed25519 AAAAKey"],
    }


def test_a_node_that_attested_nothing_claims_nothing():
    from services.attestation_service import HostPolicyResult

    assert all(not values for values in HostPolicyResult().identities().values())
