"""DAH-2470 — structured state file for the cache-template prefetch loop.

The prefetch loop already narrates every branch it takes, but ``core/logger.py``
attaches only a ``StreamHandler``: that diagnosis is written on the provider's
machine and never leaves it. This module mirrors the same information into a small
JSON document inside the executor container, where the validator's cached-template
check can read it over the SSH connection it already holds (``run.sh`` starts sshd
in this very container) and attach it to the failure event.

Two rules govern everything here:

* **Never wedge the loop.** Every mutator swallows its own errors. A broken state
  file must not change a single pull decision.
* **Never grow without bound.** The document is capped, because the validator's log
  line already carries a large monitoring payload and this rides along with it.

The file is deliberately *not* durable. Watchtower recreates the executor container
on every update, which resets the counters. ``started_at`` and ``sweep_count`` are
therefore always published, so a reader can see the window the counters cover and
never mistakes a fresh reset for a healthy node.
"""

import copy
import functools
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)

# Bump when the document shape changes in a way a reader must notice.
SCHEMA_VERSION = 1

# Inside the executor container. The validator's shell lands in this same container,
# so no bind-mount is involved.
STATE_PATH = "/var/lib/lium/cache_prefetch_state.json"

# Per-error and whole-document limits. 4 KB comfortably holds one image's full
# history; the reduction ladder in `_fit` handles anything larger.
MAX_ERROR_CHARS = 500
MAX_PAYLOAD_BYTES = 4096


class Outcome:
    """Every way a sweep can end. Named so a reader can group and count them.

    Loop-level outcomes describe the sweep as a whole; image-level outcomes describe
    what happened to one template within it.
    """

    # Loop level. The top-level `last_outcome` always holds one of these, so a reader can
    # tell "the loop finished a sweep" from "the loop keeps quitting early" without
    # reading the images map.
    PREFETCH_DISABLED_NO_BACKEND_URL = "prefetch_disabled_no_backend_url"
    DOCKER_UNAVAILABLE = "docker_unavailable"
    GPU_UNKNOWN = "gpu_unknown"
    BACKEND_NO_TEMPLATES = "backend_no_templates"
    MALFORMED_TEMPLATE = "malformed_template"
    LOOP_ERROR = "loop_error"
    # The sweep ran to the end. Individual templates may still have had a bad outcome —
    # that is what the images map is for.
    SWEEP_OK = "sweep_ok"

    # Image level.
    REMOTE_DIGEST_UNREADABLE = "remote_digest_unreadable"
    UP_TO_DATE = "up_to_date"
    INSUFFICIENT_DISK = "insufficient_disk"
    LOCK_HELD = "lock_held"
    PULL_OK = "pull_ok"
    PULL_FAILED = "pull_failed"


def _utcnow() -> str:
    """Timestamp in the same shape the rest of the fleet's JSON uses."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clip(value: object | None, limit: int = MAX_ERROR_CHARS) -> str | None:
    """Stringify and bound one error message. ``None`` stays ``None``."""
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _executor_version() -> str:
    """Executor release from ``version.txt``, the same file the validator reads."""
    try:
        version_file = Path(__file__).resolve().parents[2] / "version.txt"
        return version_file.read_text().strip() or "unknown"
    except Exception:
        return "unknown"


def _never_raises(method):
    """Diagnostics must never change prefetch behaviour, so absorb every failure."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            logger.warning(f"cache prefetch state: {method.__name__} failed: {e}")
        return None

    return wrapper


def _size(payload: str) -> int:
    return len(payload.encode("utf-8"))


def _dump(doc: dict) -> str:
    return json.dumps(doc, separators=(",", ":"), sort_keys=False, default=str)


def _fit(doc: dict) -> str:
    """Serialise ``doc``, shedding detail until it fits ``MAX_PAYLOAD_BYTES``.

    Detail is dropped least-useful-first: extra images, then long error text, then
    the counters, then the images map entirely. ``truncated`` marks that this ran, so
    a reader never mistakes a trimmed document for the whole story.
    """
    payload = _dump(doc)
    if _size(payload) <= MAX_PAYLOAD_BYTES:
        return payload

    doc = copy.deepcopy(doc)
    doc["truncated"] = True
    for shed in (_keep_newest_image, _shorten_errors, _drop_counts, _drop_images):
        shed(doc)
        payload = _dump(doc)
        if _size(payload) <= MAX_PAYLOAD_BYTES:
            break
    return payload


def _keep_newest_image(doc: dict) -> None:
    images = doc.get("images") or {}
    if len(images) <= 1:
        return
    newest = max(images.items(), key=lambda kv: kv[1].get("last_outcome_at") or "")
    doc["images"] = {newest[0]: newest[1]}


def _shorten_errors(doc: dict) -> None:
    short = MAX_ERROR_CHARS // 5
    for key, value in doc.items():
        if key.endswith("_error") and isinstance(value, str):
            doc[key] = _clip(value, short)
    for record in (doc.get("images") or {}).values():
        for key, value in record.items():
            if (key.endswith("_error") or key == "last_error") and isinstance(value, str):
                record[key] = _clip(value, short)


def _drop_counts(doc: dict) -> None:
    doc.pop("outcome_counts", None)
    for record in (doc.get("images") or {}).values():
        record.pop("outcome_counts", None)


def _drop_images(doc: dict) -> None:
    doc["images"] = {}


class CachePrefetchState:
    """Accumulates what the prefetch loop knows, and publishes it as JSON.

    Pass ``path=None`` for a throwaway recorder that never writes — used when
    ``_ensure_template`` is exercised on its own.
    """

    def __init__(
        self,
        *,
        path: str | os.PathLike | None = STATE_PATH,
        backend_url: str | None = None,
        refresh_interval_seconds: int | None = None,
    ) -> None:
        self._path = Path(path) if path else None
        self._started_at = _utcnow()
        self._sweep_count = 0
        self._executor_version = _executor_version()
        self._backend_url = backend_url
        self._refresh_interval_seconds = refresh_interval_seconds

        self._gpu_model: str | None = None
        self._driver_version: str | None = None
        self._gpu_resolved_at: str | None = None
        self._gpu_error: str | None = None

        self._docker_available: bool | None = None
        self._docker_error: str | None = None

        self._last_backend_status: int | None = None
        self._last_backend_template_count: int | None = None
        self._last_backend_error: str | None = None

        self._last_loop_error: str | None = None
        self._last_loop_error_at: str | None = None

        self._last_malformed_template: str | None = None
        self._last_malformed_template_at: str | None = None

        self._last_outcome: str | None = None
        self._last_outcome_at: str | None = None
        self._last_error: str | None = None
        self._outcome_counts: dict[str, int] = {}

        self._images: dict[str, dict] = {}

    # -- loop-level facts -------------------------------------------------------

    @_never_raises
    def begin_sweep(self) -> None:
        self._sweep_count += 1

    @_never_raises
    def note_docker(self, available: bool, error: object | None = None) -> None:
        self._docker_available = available
        self._docker_error = _clip(error)

    @_never_raises
    def note_gpu(self, gpu_model: str, driver_version: str, error: object | None = None) -> None:
        self._gpu_model = gpu_model
        self._driver_version = driver_version
        self._gpu_error = _clip(error)
        if gpu_model and gpu_model != "unknown":
            self._gpu_resolved_at = _utcnow()

    @_never_raises
    def note_backend(
        self,
        status: int | None = None,
        template_count: int | None = None,
        error: object | None = None,
    ) -> None:
        self._last_backend_status = status
        self._last_backend_template_count = template_count
        self._last_backend_error = _clip(error)

    @_never_raises
    def note_malformed_template(self, entry: object) -> None:
        """A backend entry we could not turn into an image reference.

        Kept in its own field rather than the outcome slot: the sweep still finishes, so
        `last_outcome` moves on to `sweep_ok` and would otherwise bury this.
        """
        self._last_malformed_template = _clip(entry)
        self._last_malformed_template_at = _utcnow()
        counts = self._outcome_counts
        counts[Outcome.MALFORMED_TEMPLATE] = counts.get(Outcome.MALFORMED_TEMPLATE, 0) + 1

    @_never_raises
    def note_loop_error(self, error: object) -> None:
        self._last_loop_error = _clip(error)
        self._last_loop_error_at = _utcnow()

    @_never_raises
    def record_loop_outcome(self, outcome: str, error: object | None = None) -> None:
        self._last_outcome = outcome
        self._last_outcome_at = _utcnow()
        self._last_error = _clip(error)
        self._outcome_counts[outcome] = self._outcome_counts.get(outcome, 0) + 1

    # -- per-image facts --------------------------------------------------------

    def _record(self, image_ref: str) -> dict:
        return self._images.setdefault(
            image_ref,
            {
                "last_outcome": None,
                "last_outcome_at": None,
                "last_error": None,
                "local_digests": [],
                "local_digest_first_seen_at": None,
                "digest_changed_at": None,
                "last_local_error": None,
                "expected_digest": None,
                "remote_digest": None,
                "last_remote_read_ok_at": None,
                "last_remote_error": None,
                "last_pull_attempt_at": None,
                "last_pull_ok_at": None,
                "last_pull_error": None,
                "last_cleanup_error": None,
                "last_disk_required_bytes": None,
                "last_disk_available_bytes": None,
                "outcome_counts": {},
            },
        )

    @_never_raises
    def note_local_digests(
        self, image_ref: str, digests: list[str], error: object | None = None
    ) -> None:
        record = self._record(image_ref)
        previous = record["local_digests"]
        record["local_digests"] = list(digests)
        record["last_local_error"] = _clip(error)
        if not previous and digests:
            record["local_digest_first_seen_at"] = _utcnow()
        elif previous and digests and previous != list(digests):
            # The content under the tag moved. This is the single clearest signal
            # that a pull actually landed, as opposed to merely being attempted.
            record["digest_changed_at"] = _utcnow()

    @_never_raises
    def note_expected_digest(self, image_ref: str, digest: str | None) -> None:
        self._record(image_ref)["expected_digest"] = digest

    @_never_raises
    def note_remote_digest(
        self, image_ref: str, digest: str | None, error: object | None = None
    ) -> None:
        record = self._record(image_ref)
        record["remote_digest"] = digest
        record["last_remote_error"] = _clip(error)
        if digest:
            record["last_remote_read_ok_at"] = _utcnow()

    @_never_raises
    def note_disk(self, image_ref: str, required_bytes: int, available_bytes: int) -> None:
        record = self._record(image_ref)
        record["last_disk_required_bytes"] = required_bytes
        record["last_disk_available_bytes"] = available_bytes

    @_never_raises
    def note_pull_attempt(self, image_ref: str) -> None:
        self._record(image_ref)["last_pull_attempt_at"] = _utcnow()

    @_never_raises
    def note_pull_ok(self, image_ref: str) -> None:
        record = self._record(image_ref)
        record["last_pull_ok_at"] = _utcnow()
        record["last_pull_error"] = None

    @_never_raises
    def note_pull_error(self, image_ref: str, error: object) -> None:
        self._record(image_ref)["last_pull_error"] = _clip(error)

    @_never_raises
    def note_cleanup_error(self, image_ref: str, error: object | None) -> None:
        self._record(image_ref)["last_cleanup_error"] = _clip(error)

    @_never_raises
    def record_image_outcome(
        self, image_ref: str, outcome: str, error: object | None = None
    ) -> None:
        record = self._record(image_ref)
        record["last_outcome"] = outcome
        record["last_outcome_at"] = _utcnow()
        record["last_error"] = _clip(error)
        counts = record["outcome_counts"]
        counts[outcome] = counts.get(outcome, 0) + 1

    # -- publication ------------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "executor_version": self._executor_version,
            "started_at": self._started_at,
            "updated_at": _utcnow(),
            "sweep_count": self._sweep_count,
            "refresh_interval_seconds": self._refresh_interval_seconds,
            "gpu_model": self._gpu_model,
            "driver_version": self._driver_version,
            "gpu_resolved_at": self._gpu_resolved_at,
            "gpu_error": self._gpu_error,
            "backend_url": self._backend_url,
            "docker_available": self._docker_available,
            "docker_error": self._docker_error,
            "last_backend_status": self._last_backend_status,
            "last_backend_template_count": self._last_backend_template_count,
            "last_backend_error": self._last_backend_error,
            "last_loop_error": self._last_loop_error,
            "last_loop_error_at": self._last_loop_error_at,
            "last_malformed_template": self._last_malformed_template,
            "last_malformed_template_at": self._last_malformed_template_at,
            "last_outcome": self._last_outcome,
            "last_outcome_at": self._last_outcome_at,
            "last_error": self._last_error,
            "outcome_counts": dict(self._outcome_counts),
            "images": copy.deepcopy(self._images),
        }

    def render(self) -> str:
        return _fit(self.as_dict())

    @_never_raises
    def flush(self) -> None:
        """Publish the document. Atomic, so a reader never sees a half-written file."""
        if self._path is None:
            return
        payload = self.render()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self._path)
