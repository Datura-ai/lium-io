from __future__ import annotations

import json
import logging
import shlex
import time
from dataclasses import replace
from datetime import UTC, datetime

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import CachedTemplateMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

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
# How long the first sighting of a node without its image is remembered. Past it, a node that is
# still uncached reopens the grace only while its executor's first sweep is also incomplete.
_FIRST_UNCACHED_TTL_SECONDS = 30 * 24 * 3600
# The executor's own error, quoted to the provider; the writer already caps it at 500.
_QUOTED_ERROR_CHARS = 300
# A failing NOT_CACHED whose prefetch document names no cause.
_NOT_CACHED_NEXT_STEP = (
    "Run `docker pull {ref}` on the host to see why the executor's pre-pull could not fetch it; "
    "the executor retries on its own."
)

# First match wins, so the "No such image" of an executor that predates the stream-error capture
# is read before the "404"/"not found" it also contains.
_PULL_ERROR_NEXT_STEPS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("no such image",),
        "The pull ended without the image and the registry's own error was not recorded (older "
        "executor releases drop it): update the executor, then run `docker pull {ref}` on the "
        "host to see it.",
    ),
    (
        ("toomanyrequests", "rate limit"),
        "Docker Hub is rate-limiting this host: log its Docker daemon in to Docker Hub "
        "(`docker login`) or wait for the limit to reset.",
    ),
    (
        ("no space left", "disk quota"),
        "The Docker root is out of space: free disk there; the executor retries on its own.",
    ),
    (
        ("unauthorized", "denied", "forbidden", "403"),
        "The registry refused the pull: check the Docker login and any registry-mirrors entry in "
        "/etc/docker/daemon.json.",
    ),
    (
        ("manifest unknown", "not found", "404"),
        "The registry, or a registry mirror in /etc/docker/daemon.json, does not serve this "
        "digest: remove or fix the mirror.",
    ),
    (
        (
            "timeout",
            "timed out",
            "connection reset",
            "connection refused",
            "unexpected eof",
            "tls",
            "no such host",
            "network is unreachable",
        ),
        "The host could not download from the registry: check its outbound connection to "
        "registry-1.docker.io and production.cloudflare.docker.com.",
    ),
)


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


def _first_sweep_completed(state: dict | None) -> bool | None:
    """Whether the executor's prefetch loop has finished a sweep; None when its document can't say.

    ``first_sweep_ok_at`` is published by newer executors; older ones only have the ``sweep_ok``
    count, which the writer sheds when the document runs over its size cap.
    """
    if not isinstance(state, dict) or "unavailable" in state:
        return None
    counts = state.get("outcome_counts")
    if state.get("first_sweep_ok_at") or (isinstance(counts, dict) and counts.get("sweep_ok")):
        return True
    if "first_sweep_ok_at" in state or isinstance(counts, dict):
        return False
    return None


def _quote(value: object) -> str:
    text = " ".join(str(value).split())
    if len(text) > _QUOTED_ERROR_CHARS:
        text = text[:_QUOTED_ERROR_CHARS] + "…"
    return f'"{text}"'


def _pull_error_next_step(error: str, ref: str) -> str:
    lowered = error.lower()
    for needles, step in _PULL_ERROR_NEXT_STEPS:
        if any(needle in lowered for needle in needles):
            return step.format(ref=ref)
    return f"Run `docker pull {ref}` on the host to reproduce it."


def _gib(value: object) -> str:
    return f"{value / 1024**3:.0f} GiB" if isinstance(value, int | float) else "unknown"


def _remediation(state: dict | None, image_ref: str, pull_ref: str, cached: bool) -> str | None:
    """The provider's next step, from what the executor's own prefetch loop recorded.

    None when the document names no specific cause; the template's generic text is used then.
    """
    if not isinstance(state, dict):
        return None
    if "unavailable" in state:
        why = "the image on the host is stale" if cached else "the image is missing"
        return (
            f"The executor published no pre-pull state ({state['unavailable']}): update the "
            f"executor, then run `docker pull {pull_ref}` on the host to see why {why}."
        )
    record = (state.get("images") or {}).get(image_ref) or {}
    pull_error = record.get("last_pull_error")
    # A later sweep that found the image current does not clear the error of an earlier pull.
    if pull_error and record.get("last_outcome") not in ("pull_ok", "up_to_date"):
        return (
            f"The executor's last pull of {image_ref} failed: {_quote(pull_error)}. "
            f"{_pull_error_next_step(str(pull_error), pull_ref)}"
        )
    if record.get("last_outcome") == "insufficient_disk":
        return (
            f"The executor skipped the pull for lack of disk: it needs "
            f"{_gib(record.get('last_disk_required_bytes'))} free and has "
            f"{_gib(record.get('last_disk_available_bytes'))}. Free space on the Docker root; "
            "the next sweep pulls it."
        )
    loop_outcome = state.get("last_outcome")
    if loop_outcome == "prefetch_disabled_no_backend_url":
        return (
            "The executor's pre-pull is off because COMPUTE_REST_API_URL is not set in its "
            "environment: set it and restart the executor."
        )
    if loop_outcome == "docker_unavailable":
        return (
            f"The executor's pre-pull cannot reach Docker ({_quote(state.get('docker_error'))}): "
            "check that /var/run/docker.sock is mounted into the executor container."
        )
    if loop_outcome == "gpu_unknown":
        return (
            f"The executor's pre-pull cannot read the GPU ({_quote(state.get('gpu_error'))}), so "
            "it does not know which image to pull: check the NVIDIA driver and that the executor "
            "container sees the GPUs."
        )
    if loop_outcome == "backend_no_templates":
        return (
            f"The executor's pre-pull got no image from the backend "
            f"({_quote(state.get('last_backend_error'))}): check that the executor can reach "
            f"{state.get('backend_url') or 'the backend'}."
        )
    if loop_outcome == "loop_error" and state.get("last_loop_error"):
        return f"The executor's pre-pull loop failed: {_quote(state['last_loop_error'])}."
    if not cached and record.get("last_pull_ok_at") and record.get("last_outcome") in (
        "pull_ok",
        "up_to_date",
    ):
        return (
            f"The executor pulled {image_ref} (last at {record['last_pull_ok_at']}) but it is no "
            "longer on the host: something removed it, e.g. `docker image prune -a` or "
            "`docker system prune -a`. Stop pruning it; the executor re-pulls it on its next sweep."
        )
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
    holds a stale image is readable in Grafana without SSHing anywhere. The failure's remediation
    quotes that document: the executor's own pull error and the step it points to.

    With ``settings.CACHED_TEMPLATE_FRESH_NODE_GRACE_ENABLED``, a node this validator has only just
    found without the image is held as PENDING (passed, score untouched by this check) while its
    executor's first pre-pull sweep is still running, for at most
    ``settings.CACHED_TEMPLATE_FRESH_NODE_GRACE_SECONDS`` (see ``_fresh_node_grace``). With it off
    the node fails as before and the would-be hold is only logged.
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

    async def _fresh_node_grace(self, ctx: Context, prefetch_state: dict) -> dict | None:
        """Whether a node found without its image is still inside its fresh-node grace.

        The window opens at this validator's first sighting of the node without the image (kept
        in Redis, so the executor cannot move it) and closes CACHED_TEMPLATE_FRESH_NODE_GRACE_SECONDS
        later, or as soon as the executor reports that its first pre-pull sweep completed. An
        executor whose document can't say gets the time bound alone. Never raises: a Redis error
        means no grace, which is the verdict this check gave before the grace existed.
        """
        grace_seconds = settings.CACHED_TEMPLATE_FRESH_NODE_GRACE_SECONDS
        if grace_seconds <= 0:
            return None
        now = time.time()
        try:
            first_uncached = await ctx.services.redis.first_uncached_at(
                ctx.executor.uuid, now, _FIRST_UNCACHED_TTL_SECONDS
            )
        except Exception as exc:
            return {"pending": False, "error": str(exc)[:_PREFETCH_ERROR_CHARS]}
        first_sweep_completed = _first_sweep_completed(prefetch_state)
        elapsed = max(0.0, now - first_uncached)
        return {
            "pending": elapsed < grace_seconds and first_sweep_completed is not True,
            "first_uncached_at": datetime.fromtimestamp(first_uncached, UTC).isoformat(),
            "seconds_since_first_uncached": round(elapsed),
            "grace_seconds": grace_seconds,
            "first_sweep_completed": first_sweep_completed,
        }

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
        remediation: str | None = None
        if should_fail:
            prefetch_state = await self._read_prefetch_state(ctx)
            what["prefetch_state"] = prefetch_state
            if template == Msg.NOT_CACHED:
                grace = await self._fresh_node_grace(ctx, prefetch_state)
                if grace is not None and settings.CACHED_TEMPLATE_FRESH_NODE_GRACE_ENABLED:
                    what["fresh_node_grace"] = grace
                    if grace["pending"]:
                        template = Msg.PENDING
                        should_fail = False
                elif grace is not None and grace["pending"]:
                    logger.info(
                        _m(
                            "Fresh-node grace is off: this node would have been held as pending",
                            extra=get_extra_info({**ctx.default_extra, "fresh_node_grace": grace}),
                        )
                    )
            if should_fail:
                pull_ref = f"{docker_image}@{backend_digest}" if backend_digest else image_ref
                remediation = _remediation(prefetch_state, image_ref, pull_ref, cached)
                if remediation is None and template == Msg.NOT_CACHED:
                    remediation = _NOT_CACHED_NEXT_STEP.format(ref=pull_ref)

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
            remediation=remediation,
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
