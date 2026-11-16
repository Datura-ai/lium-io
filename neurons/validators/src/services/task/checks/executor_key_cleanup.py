"""Remove known unauthorized keys from the executor container's authorized_keys.

A key the validator uploads carries a `lium-uploaded-at=` marker and the executor expires it. A line written into
the file some other way carries no marker, so nothing ever removes it. Lines holding one of the known key bodies
below are removed on every cycle. Other unmarked lines are only counted and logged by fingerprint: executors from
before the marker write their own validator keys unmarked, and a provider may keep a key of their own there.
Observation and cleanup only: never fails or halts the node.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import shlex

from core.utils import _m, get_extra_info

from ..messages import ExecutorKeyCleanupMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

EXECUTOR_FOREIGN_KEY_BODIES = (
    "AAAAC3NzaC1lZDI1NTE5AAAAIICJqs5+U0I7iLRQK8NxCkqnzkO32PBKTmpNRRyLbDFJ",
    "AAAAC3NzaC1lZDI1NTE5AAAAINvDY1UGnXkbCf4/dwHeHBSLRKS+5i9UWD02uZ8peaen",
    "AAAAC3NzaC1lZDI1NTE5AAAAIKiF/63kOMkI92Fbz1KNsXHsO3JdFZxhHtGKP1/90F1I",
)
UPLOADED_KEY_MARKER = "lium-uploaded-at="
AUTHORIZED_KEYS = "/root/.ssh/authorized_keys"
MARKER_DIR = "/root"
MARKER_FILES = (".vh_pwned", ".xstar8_pwned")
SSH_TIMEOUT_SECONDS = 10


def cleanup_command(keys: str = AUTHORIZED_KEYS, marker_dir: str = MARKER_DIR) -> str:
    """Prints `<vh> <xstar8> <matching lines> <1 if they were removed>`, then the file as it was read.

    Only constants go into the command. The file is read through one descriptor (fd 3) and replaced by a rename,
    as the executor's own purge does; an append that lands on the old file after it was read is copied over to
    the new one, and the rename is skipped when the path no longer names the file that was read.
    """
    patterns = " ".join(f"-e {shlex.quote(body)}" for body in EXECUTOR_FOREIGN_KEY_BODIES)
    flags = "; ".join(
        f'[ -e {shlex.quote(f"{marker_dir}/{name}")} ] && m="$m 1" || m="$m 0"'
        for name in MARKER_FILES
    )
    return (
        f"m=; {flags}; f={shlex.quote(keys)}; "
        '[ -f "$f" ] || { echo "$m 0 0"; exit 0; }; '
        'exec 3<"$f" && s=$(stat -L -c %s /dev/fd/3) || exit 1; '
        f'n=$(head -c "$s" /dev/fd/3 | grep -cF {patterns}); r=0; '
        'if [ "${n:-0}" -gt 0 ] && t=$(mktemp "$f.XXXXXX"); then '
        # grep exit 1 = every line went; anything else (a full disk) must not replace the file
        f'head -c "$s" /dev/fd/3 | grep -vF {patterns} > "$t"; [ $? -le 1 ] '
        '&& chown --reference="$f" "$t" && chmod --reference="$f" "$t" '
        '&& [ "$(stat -c %i "$f")" = "$(stat -L -c %i /dev/fd/3)" ] && mv -f "$t" "$f" '
        '&& { tail -c +$((s + 1)) /dev/fd/3 >> "$f"; r=1; }; rm -f "$t"; fi; '
        'echo "$m ${n:-0} $r"; head -c "$s" /dev/fd/3'
    )


def fingerprint(line: str) -> str:
    """ssh-keygen's `SHA256:` fingerprint of the line's key blob; the body itself is never logged."""
    for token in line.split():
        if token.startswith("AAAA"):
            try:
                blob = base64.b64decode(token, validate=True)
            except ValueError:
                continue
            return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    return "unparsed"


def unmarked_lines(content: str) -> list[str]:
    return [
        line
        for line in (raw.strip() for raw in content.split("\n"))
        if line
        and not line.startswith("#")
        and UPLOADED_KEY_MARKER not in line
        and not any(body in line for body in EXECUTOR_FOREIGN_KEY_BODIES)
    ]


class ExecutorKeyCleanupCheck:
    check_id = "executor.observe.key_cleanup"
    fatal = False

    def __init__(self, keys: str = AUTHORIZED_KEYS, marker_dir: str = MARKER_DIR):
        self.keys, self.marker_dir = keys, marker_dir

    async def run(self, ctx: Context) -> CheckResult:
        what = {"executor_uuid": ctx.executor.uuid}
        try:
            result = await asyncio.wait_for(
                ctx.ssh.run(cleanup_command(self.keys, self.marker_dir), check=False),
                timeout=SSH_TIMEOUT_SECONDS,
            )
            header, _, content = (result.stdout or "").partition("\n")
            vh, xstar8, matched, removed = (int(part) for part in header.split())
        except Exception as exc:  # noqa: BLE001 - never fatal; a timeout or an unreadable answer lands here too
            logger.warning(
                _m(
                    "Executor authorized_keys cleanup not run",
                    extra=get_extra_info({**ctx.default_extra, **what, "error": repr(exc)[:300]}),
                )
            )
            return CheckResult(
                passed=True,
                event=render_message(Msg.UNREAD, ctx=ctx, check_id=self.check_id, what=what),
            )

        unmarked = unmarked_lines(content)
        what = {
            **what,
            "foreign_keys_found": matched,
            "foreign_keys_removed": matched if removed else 0,
            "unmarked_keys": len(unmarked),
        }
        if matched or unmarked:
            log = logger.warning if matched else logger.info
            log(
                _m(
                    "Executor authorized_keys: foreign or unmarked keys",
                    extra=get_extra_info(
                        {
                            **ctx.default_extra,
                            **what,
                            "miner_hotkey": ctx.miner_hotkey,
                            "removal_failed": bool(matched and not removed),
                            "unmarked_fingerprints": [fingerprint(line) for line in unmarked],
                            "vh_pwned_marker": bool(vh),
                            "xstar8_pwned_marker": bool(xstar8),
                        }
                    ),
                )
            )
        return CheckResult(
            passed=True,
            event=render_message(Msg.CHECKED, ctx=ctx, check_id=self.check_id, what=what),
        )
