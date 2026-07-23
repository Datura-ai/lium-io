from __future__ import annotations

import json
import shlex
from dataclasses import replace
from datetime import datetime

from core.config import settings

from ..messages import CachedTemplateMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

# DAH-2470 — the executor's cache-prefetch loop publishes its own state here, inside the
# executor container. The validator's shell lands in that same container (run.sh starts
# sshd there and compose maps ${SSH_PORT}:22 to it), so this is a plain read.
PREFETCH_STATE_PATH = "/var/lib/lium/cache_prefetch_state.json"
# The writer caps the document at 4 KB. Read a little more so a document from a newer
# executor is still parseable, and refuse anything past that rather than bloat the log.
_PREFETCH_READ_BYTES = 8192
_PREFETCH_MAX_BYTES = 6144
# Enough of a bad file to recognise it, not enough to matter in the log.
_PREFETCH_ERROR_CHARS = 200


def _repo_digest(stdout: str | None, repo: str) -> str | None:
    """Bare sha256 of the RepoDigests entry matching ``repo`` ("repo@sha256:…"), or None.

    Strict fail-open: empty/invalid JSON, or no entry whose repo equals ``repo``, returns
    None. Matching by repo (not ``[0]``) avoids a false match when the image carries several
    RepoDigests from different repos.
    """
    try:
        entries = json.loads((stdout or "").strip() or "null") or []
    except Exception:
        return None
    for entry in entries:
        name, _, sha = str(entry).partition("@")
        if name == repo and sha:
            return sha
    return None


class CachedTemplateVerificationCheck:
    """Verify the executor has the recommended default image pre-pulled (DAH-2265).

    The executor cache pre-pull (``cache_template_service.py``) keeps the recommended default
    template image warm, and the rental-time DAH-1524 pull-skip turns DOCKER_PULL into a no-op
    when the image is already present. This check resolves the recommended image from the
    backend (the same ``/executors/default-docker-image`` endpoint the executor uses) and
    probes ``docker image inspect`` on the executor. The expected manifest digest comes from
    the per-cycle Docker Hub digest snapshot fetched at job-cycle start (``ctx.config``),
    not the backend response.

    Two-phase by ``settings.CACHED_TEMPLATE_CUTOFF``:
      * Before the cutoff — advisory only: emits a structured event and publishes
        ``recommended_image_cached`` / ``recommended_image_digest_match`` into
        ``executor.specs`` via pipeline state. Never changes score.
      * On/after the cutoff — critical (``fatal``): the two bad signals (image not pre-pulled,
        or stale content under an unchanged tag) return ``passed=False`` so the pipeline halts
        early, the executor scores 0, and the failure reason reaches the provider via
        ``log_text``. This check sits after ``TenantEnforcementCheck`` (the rented
        short-circuit), so it only ever gates *unrented* executors — an active customer rental
        is never failed by it.

    Fails open on every uncertainty (unknown GPU/driver, backend unreachable, empty
    recommendation, SSH error) by recording a skip and leaving score untouched.

    DAH-2470: on the failure path only, the event also carries ``prefetch_state`` — the
    executor's own record of what its cache-prefetch loop was doing — so the reason a node
    holds a stale image is readable in Grafana without SSHing anywhere.
    """

    check_id = "executor.validate.cached_template"
    # The DAH-2380 digest rollout is done, so the gate is live: a `passed=False` result
    # (NOT_CACHED / DIGEST_MISMATCH on/after CACHED_TEMPLATE_CUTOFF) halts the pipeline and the
    # executor scores 0. Every uncertainty still fails open inside run() (passed=True).
    fatal = True

    async def _read_prefetch_state(self, ctx: Context) -> dict:
        """Read the executor's prefetch-state document (DAH-2470).

        Only ever called on the failure path. Returns a dict either way: every way the
        read can fall short is reported as a distinguishable ``unavailable`` value,
        because absence is itself a finding — an executor too old to write the file,
        a loop that never started, or ``COMPUTE_REST_API_URL`` left unset.

        Never raises, so it can never change ``passed``.
        """
        try:
            result = await ctx.ssh.run(
                f"head -c {_PREFETCH_READ_BYTES} {shlex.quote(PREFETCH_STATE_PATH)}",
                check=False,
            )
        except Exception as exc:
            return {"unavailable": "ssh_read_failed", "error": str(exc)[:_PREFETCH_ERROR_CHARS]}

        if getattr(result, "exit_status", 1) != 0:
            return {"unavailable": "missing"}

        raw = (getattr(result, "stdout", None) or "").strip()
        if not raw:
            return {"unavailable": "empty"}
        if len(raw.encode("utf-8", "replace")) > _PREFETCH_MAX_BYTES:
            return {"unavailable": "oversized", "bytes": len(raw.encode("utf-8", "replace"))}

        try:
            state = json.loads(raw)
        except Exception:
            return {"unavailable": "unparseable", "raw_prefix": raw[:_PREFETCH_ERROR_CHARS]}
        if not isinstance(state, dict):
            return {"unavailable": "unparseable", "raw_prefix": raw[:_PREFETCH_ERROR_CHARS]}
        return state

    async def run(self, ctx: Context) -> CheckResult:
        gpu_model = ctx.state.gpu_model
        driver_version = str((ctx.state.specs or {}).get("gpu", {}).get("driver") or "")

        if not gpu_model or not driver_version:
            event = render_message(
                Msg.SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "reason": "missing gpu_model or driver_version",
                    "gpu_model": gpu_model,
                    "driver_version": driver_version,
                },
            )
            return CheckResult(passed=True, event=event)

        # Backend client already fails open to None on any error/non-200; the try/except
        # is belt-and-suspenders so an unexpected raise can never break the pipeline.
        try:
            images = await ctx.services.backend.get_default_docker_image(gpu_model, driver_version)
            backend_error = None
        except Exception as exc:
            images = None
            backend_error = str(exc)

        if not images:
            event = render_message(
                Msg.SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "reason": "no recommended image from backend",
                    "gpu_model": gpu_model,
                    "driver_version": driver_version,
                    "backend_error": backend_error,
                },
            )
            return CheckResult(passed=True, event=event)

        # The top entry is the primary recommended image that default-template rentals
        # request — the same image the executor pre-pull warms first.
        image_ref = images[0].image_ref
        docker_image = images[0].docker_image
        # Bare manifest digest the validator fetched from Docker Hub for this image, if any.
        backend_digest = ctx.config.default_docker_image_digests.get(image_ref)

        try:
            inspect = await ctx.ssh.run(
                f'/usr/bin/docker image inspect --format "{{{{json .RepoDigests}}}}" {shlex.quote(image_ref)}',
                check=False,
            )
        except Exception as exc:
            event = render_message(
                Msg.SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "reason": "docker image inspect failed",
                    "recommended_image": image_ref,
                    "error": str(exc),
                },
            )
            return CheckResult(passed=True, event=event)

        cached = inspect.exit_status == 0
        # Local RepoDigest for THIS repo (strict fail-open: any parse/lookup miss → None).
        local_digest = _repo_digest(getattr(inspect, "stdout", None), docker_image) if cached else None

        digest_match: bool | None = None
        if cached and backend_digest and local_digest:
            digest_match = local_digest == backend_digest

        # One CheckResult event: surface the most specific signal. The digest message refines
        # the cached axis; recommended_image_cached is still published independently below.
        if not cached:
            template = Msg.NOT_CACHED
        elif digest_match is True:
            template = Msg.DIGEST_MATCH
        elif digest_match is False:
            template = Msg.DIGEST_MISMATCH
        elif backend_digest:
            # Cached, a backend digest exists, but the local RepoDigest was unreadable/unmatched.
            template = Msg.DIGEST_SKIPPED
        else:
            # Cached, but the backend published no digest to compare against.
            template = Msg.CACHED

        # DAH-2265: on/after the cutoff this check turns critical. The two "bad" signals —
        # image not pre-pulled (NOT_CACHED) or stale content under an unchanged tag
        # (DIGEST_MISMATCH) — fail verification early, so the executor scores 0 and the reason
        # reaches the provider via log_text (the pipeline surfaces the last event). Before the
        # cutoff (and on every fail-open None path above) the check stays advisory: passed=True,
        # score untouched. The cached/digest signals are published to specs either way.
        after_cutoff = datetime.utcnow() >= settings.CACHED_TEMPLATE_CUTOFF
        should_fail = after_cutoff and template in (Msg.NOT_CACHED, Msg.DIGEST_MISMATCH)

        what = {
            "recommended_image": image_ref,
            "cached": cached,
            "digest_match": digest_match,
            "backend_digest": backend_digest,
            "local_digest": local_digest,
            "gpu_model": gpu_model,
            "driver_version": driver_version,
            "after_cutoff": after_cutoff,
        }
        # DAH-2470: when we are about to zero this node, also carry what the executor's
        # own prefetch loop was doing at the time, so a reader can tell a provider-side
        # cause (registry rate-limits, disk, network) from ours (loop gave up, never
        # retried). Only on the failure path — a handful of nodes per cycle — so healthy
        # nodes add neither an SSH round trip nor log volume.
        if should_fail:
            what["prefetch_state"] = await self._read_prefetch_state(ctx)

        event = render_message(
            template,
            ctx=ctx,
            check_id=self.check_id,
            severity="error" if should_fail else None,
            impact=(
                "Score set to 0 — recommended default image must be pre-pulled and current"
                if should_fail
                else None
            ),
            what=what,
        )
        return CheckResult(
            passed=not should_fail,
            event=event,
            updates={
                "state": replace(
                    ctx.state,
                    recommended_image_cached=cached,
                    recommended_image_digest_match=digest_match,
                )
            },
        )
