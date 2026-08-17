"""The host's copy of the approved catalog: what is cached, and what it becomes on disk.

Two jobs, and the order between them is the point.

**Install.** `install()` takes bytes that arrived from somewhere — the backend, or a seed file
Ansible staged — verifies them completely, refuses a rollback, and only then replaces the cache.
A manifest that fails any check never touches the cache, so a host whose backend starts serving
garbage keeps launching from the last good catalog rather than from nothing.

**Read.** `artifacts()` re-verifies the cached bytes on every call and materializes the
measured inputs before handing back paths. Re-verifying looks redundant — the bytes were
verified when they were installed — but the cache is a file on a host, and the signature is the
only thing that makes it evidence rather than a suggestion. Materializing every time is what
makes the *files* match the manifest too: an operator who edits the compose to add a service
gets their edit overwritten at the next launch instead of launching a stack nobody approved.

The OS image is the one input that stays where day-zero Ansible put it. It is gigabytes, so it
cannot travel in a manifest; `os_image_name` selects one of the staged images and
`cvm/measure.py` refuses the launch if its `digest.txt` is not the hash the manifest pinned.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cvmd.atomic import write_bytes_durable
from cvmd.catalog.artifacts import Artifact, CatalogError, assert_unambiguous
from cvmd.catalog.manifest import Entry, Manifest, assert_not_rollback, parse_manifest

logger = logging.getLogger(__name__)

COMPOSE_FILE = "compose.yml"
INIT_SCRIPT_FILE = "init_script.sh"
PRE_LAUNCH_SCRIPT_FILE = "pre_launch_script.sh"

# The materialized scripts are handed to dstack.py, which reads them; the compose likewise.
# Nothing here is executed by cvmd, so these are read-only to root and to nobody else.
_FILE_MODE = 0o640
_DIR_MODE = 0o750


@dataclass(frozen=True)
class CatalogConfig:
    """Where the catalog lives on this host, and who is allowed to have signed it."""

    # Daemon-owned, under state_dir. Deliberately not under /etc: Ansible manages /etc, and a
    # file both Ansible and the daemon write is a file that flaps on every converge.
    cache_path: Path | None = None
    # Operator-staged, under /etc. Read-only to cvmd. Lets a host be provisioned with a catalog
    # before it can reach the backend, and lets an air-gapped host have one at all.
    seed_path: Path | None = None
    # The ss58 this host trusts. Without it no manifest is usable, which is the correct
    # fail-closed default: an unsigned catalog is not a catalog.
    signer: str | None = None
    # Where day-zero Ansible staged the approved OS images. `os_image_name` selects one.
    images_dir: Path | None = None
    materialize_dir: Path | None = None
    manifest_url: str | None = None
    refresh_seconds: int = 300
    fetch_timeout_seconds: int = 30

    def missing(self) -> list[str]:
        """The settings a launch needs and this host does not have."""
        return [
            name
            for name in ("cache_path", "signer", "images_dir", "materialize_dir")
            if not getattr(self, name)
        ]


class CatalogStore:
    def __init__(self, config: CatalogConfig) -> None:
        self._config = config

    @property
    def config(self) -> CatalogConfig:
        return self._config

    # ------------------------------------------------------------------ reading

    def _require_configured(self) -> None:
        missing = self._config.missing()
        if missing:
            raise CatalogError(
                f"this host has no catalog configuration for {', '.join(missing)}, so it can "
                f"launch nothing"
            )

    def _read_cache(self) -> bytes | None:
        try:
            return self._config.cache_path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise CatalogError(f"cannot read the catalog at {self._config.cache_path}: {exc}")

    def current(self, *, require_fresh: bool = True, now: datetime | None = None) -> Manifest:
        """The manifest in force, verified again from the cached bytes."""
        self._require_configured()
        raw = self._read_cache()
        if raw is None:
            raise CatalogError(
                f"this host holds no catalog manifest ({self._config.cache_path} does not "
                f"exist), so there is nothing it is approved to launch"
            )
        return parse_manifest(raw, signer=self._config.signer, now=now, require_fresh=require_fresh)

    def _unusable_reason(self) -> str:
        """Why `/v1/catalog` reports `usable: false` — re-derived, not quoted.

        `current()` already said why, precisely, and that message is logged by the only
        caller of this. It is not what goes on the wire: a reason taken from a caught
        exception carries that exception's traceback with it, which is what CodeQL's
        `py/stack-trace-exposure` flags, and a signature verifier's internals are not
        something a route should hand out even to an authorized key.

        So the three states are re-derived from the same conditions `current()` checks. The
        endpoint keeps the job its docstring claims — saying which state this host is in —
        and the detail that distinguishes a malformed manifest from a badly signed one stays
        in the log, on the host whoever is debugging it is already reading.
        """
        missing = self._config.missing()
        if missing:
            return (
                f"this host has no catalog configuration for {', '.join(missing)}, so it can "
                f"launch nothing"
            )
        if not self._config.cache_path.exists():
            return (
                f"this host holds no catalog manifest ({self._config.cache_path} does not "
                f"exist), so there is nothing it is approved to launch"
            )
        return (
            f"the catalog manifest cached at {self._config.cache_path} could not be read or "
            f"did not verify; the reason is in this host's cvmd log"
        )

    def describe(self) -> dict:
        """What `/v1/catalog` answers. Never raises — a broken catalog is a reportable state."""
        report: dict = {
            "manifest_url": self._config.manifest_url,
            "signer": self._config.signer,
            "cache_path": str(self._config.cache_path) if self._config.cache_path else None,
        }
        try:
            current = self.current(require_fresh=False)
        except CatalogError as exc:
            logger.warning("this host's catalog is not usable: %s", exc)
            return {**report, "usable": False, "error": self._unusable_reason()}
        expired = current.is_expired()
        return {
            **report,
            "usable": not expired,
            "expired": expired,
            "manifest": current.report(),
        }

    def artifacts(self, *, now: datetime | None = None) -> list[Artifact]:
        """The approved stacks, with every measured input written to disk and pinned to a path."""
        current = self.current(now=now)
        artifacts = [self._materialize(entry) for entry in current.entries]
        assert_unambiguous(artifacts, where=f"manifest serial {current.serial}")
        return artifacts

    def _materialize(self, entry: Entry) -> Artifact:
        directory = self._config.materialize_dir / entry.id
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        except OSError as exc:
            raise CatalogError(f"cannot create {directory} for artifact {entry.id}: {exc}")

        compose_path = self._write(directory / COMPOSE_FILE, entry.compose, entry.id)
        init_script = self._write_optional(
            directory / INIT_SCRIPT_FILE, entry.init_script, entry.id
        )
        pre_launch_script = self._write_optional(
            directory / PRE_LAUNCH_SCRIPT_FILE, entry.pre_launch_script, entry.id
        )

        image_path = self._config.images_dir / entry.os_image_name
        if not image_path.is_dir():
            raise CatalogError(
                f"artifact {entry.id} needs the OS image {entry.os_image_name!r}, which is not "
                f"staged on this host ({image_path} is not a directory) — day-zero Ansible "
                f"stages images; a manifest cannot carry one"
            )

        return Artifact(
            id=entry.id,
            kind=entry.kind,
            qemu=entry.qemu,
            os_image_hash=entry.os_image_hash,
            compose_hash=entry.compose_hash,
            os_image_path=image_path,
            compose_path=compose_path,
            init_script=init_script,
            pre_launch_script=pre_launch_script,
            local_key_provider=entry.local_key_provider,
            enable_logs=entry.enable_logs,
            enable_sysinfo=entry.enable_sysinfo,
        )

    def _write(self, path: Path, content: str, artifact_id: str) -> Path:
        """Write `content` to `path` if it is not already exactly that. Returns the path.

        Compared before writing so an unchanged catalog does not rewrite files on every launch,
        and so the log line below means what it says: something on this host had drifted from
        what the platform signed.
        """
        encoded = content.encode()
        try:
            if path.exists() and path.read_bytes() == encoded:
                return path
        except OSError as exc:
            raise CatalogError(f"cannot read {path} while checking artifact {artifact_id}: {exc}")

        try:
            write_bytes_durable(path, encoded, mode=_FILE_MODE)
        except OSError as exc:
            raise CatalogError(f"cannot write {path} for artifact {artifact_id}: {exc}")
        logger.info("wrote %s for artifact %s from the signed manifest", path, artifact_id)
        return path

    def _write_optional(self, path: Path, content: str | None, artifact_id: str) -> Path | None:
        if content is None:
            # A previous manifest may have had one. Leaving it behind would not be used — the
            # Artifact would carry None — but it would sit on the host looking approved.
            path.unlink(missing_ok=True)
            return None
        return self._write(path, content, artifact_id)

    # ------------------------------------------------------------------ writing

    def install(self, raw: bytes, *, source: str, now: datetime | None = None) -> Manifest:
        """Verify `raw` completely and make it this host's catalog. Raises on any refusal.

        Nothing partial: the cache is replaced by one atomic rename after every check has
        passed, so a host is never running from a half-written manifest.
        """
        self._require_configured()
        now = now or datetime.now(UTC)
        candidate = parse_manifest(raw, signer=self._config.signer, now=now)

        try:
            held = self.current(require_fresh=False, now=now)
        except CatalogError:
            # No usable manifest here yet, or the one here is no longer readable. Either way
            # there is nothing to roll back *from*, so the candidate stands on its own checks.
            held = None
        assert_not_rollback(candidate, held)

        if held is not None and candidate.serial == held.serial and candidate.raw == held.raw:
            return held

        try:
            write_bytes_durable(self._config.cache_path, raw, mode=_FILE_MODE)
        except OSError as exc:
            raise CatalogError(f"cannot write the catalog to {self._config.cache_path}: {exc}")

        logger.info(
            "catalog serial %d from %s is now in force: %d artifact(s), floors %s, expires %s",
            candidate.serial,
            source,
            len(candidate.entries),
            candidate.floors,
            candidate.expires_at.isoformat(),
        )
        return candidate

    def install_seed_if_newer(self) -> Manifest | None:
        """Adopt the operator-staged manifest when it is newer than the cache.

        Runs at startup. A seed is how a host gets a catalog before it can reach the backend,
        and how an operator rolls one forward by hand; `assert_not_rollback` inside `install`
        is what stops it being a way to roll one *back*.
        """
        seed = self._config.seed_path
        if seed is None or not seed.is_file():
            return None
        try:
            raw = seed.read_bytes()
        except OSError as exc:
            logger.warning("cannot read the seed manifest at %s: %s", seed, exc)
            return None
        try:
            return self.install(raw, source=f"the seed at {seed}")
        except CatalogError as exc:
            # Not fatal. A stale seed left behind after the backend moved on is the ordinary
            # case, and it reads exactly like a rollback attempt.
            logger.info("the seed manifest at %s was not adopted: %s", seed, exc)
            return None
