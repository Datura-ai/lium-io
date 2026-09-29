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
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from aiohttp.client_exceptions import NonHttpUrlClientError
from yarl import URL

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


# What `redact` returns at most. The text is cut to this length before any rule runs, and no rule
# looks past the cut, so every character published has been read by every rule. When the text is
# cut, its last run of non-whitespace is dropped as well: a URL or token the cut splits is never
# published in part.
MAX_REDACTED_CHARS = 4 * MAX_ERROR_CHARS

# Published in place of a URL yarl cannot parse cleanly; nothing of the raw text is kept.
UNPARSEABLE_URL = "<unparseable URL>"
# Shown in place of an error's class name when reading that name fails.
UNNAMED_ERROR = "error"

# Words that make a parameter or key name secret when they are one of its words (`access_token`,
# `apiKey`, `X-Amz-Signature`), so `monkey`, `design` and `author` are not.
_SECRET_WORDS = frozenset(
    {
        "auth",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "csrf",
        "hmac",
        "jwt",
        "key",
        "pass",
        "passphrase",
        "passwd",
        "password",
        "pw",
        "pwd",
        "secret",
        "session",
        "sessionid",
        "sig",
        "signature",
        "token",
    }
)
# Prefixes a secret word is often written flush against (`apikey`, `accesstoken`).
_SECRET_PREFIXES = ("access", "api", "auth", "client", "private", "refresh", "secret", "session")
# Secret words that stay secret at the end of a longer word (`mypassword`, `PGPASSWORD`,
# `csrftoken`). `key` and `pass` are left out, so `monkey` and `bypass` stay.
_SECRET_SUFFIXES = ("password", "passwd", "passphrase", "secret", "token")
_NAME_WORDS = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def _name_words(name: str) -> list[str]:
    return [word.lower() for word in _NAME_WORDS.findall(name)]


def _is_secret_name(name: str) -> bool:
    for word in _name_words(name):
        if word in _SECRET_WORDS or word.endswith(_SECRET_SUFFIXES):
            return True
        if any(
            word.startswith(prefix) and word[len(prefix) :] in _SECRET_WORDS
            for prefix in _SECRET_PREFIXES
        ):
            return True
    return False


# URLs, first. A URL starts at `scheme://` and runs to the next whitespace, quote or `>`, whatever
# its host, port or length. Everything between `://` and the span's last `@` is userinfo, except an
# `@` that starts an image digest (`repo@sha256:`). A password holding a whitespace, quote or `>`
# ends that span early, so a later `@` followed by a host on the same line, before the next URL,
# ends the userinfo too. The query and fragment are replaced whole, and so is what follows a `?` or
# `#` written as `%3F` or `%23`; a path segment after a secret-named one (`/token/<v>`) is masked.
_URL_START = re.compile(r"(?i)\b[a-z][a-z0-9+.-]{0,31}://")
_URL_END = re.compile(r"[\s'\">]")
_LINE_END = re.compile(r"[\r\n]")
_DIGEST_AT = re.compile(r"@sha(?:256|384|512):")
# An `@` followed by a host: a bracketed IPv6 address or dot-separated labels, then what can end a
# host (a port, a path, a query, the end of the URL).
_HOST_AT = re.compile(
    r"@(?:\[[0-9A-Za-z:.%]+\]|[^\W_][\w-]*(?:\.[\w-]+)*)(?=[:/?#\s'\">),;\]]|$)"
)
_QUERY_START = re.compile(r"[?#]|%3[Ff]|%23")
_SEGMENT_NAME = re.compile(r"[A-Za-z_][\w.-]{0,127}")


def _mask_path(path: str) -> str:
    """``host/path`` with every segment that follows a secret-named segment replaced by ``***``."""
    segments = path.split("/")
    for index in range(2, len(segments)):
        name = segments[index - 1]
        if segments[index] and _SEGMENT_NAME.fullmatch(name) and _is_secret_name(name):
            segments[index] = "***"
    return "/".join(segments)


def _mask_url_span(span: str) -> str:
    """One URL's text after ``://``, without its userinfo, query and fragment."""
    at = span.rfind("@")
    while at != -1 and _DIGEST_AT.match(span, at):
        at = span.rfind("@", 0, at)
    rest = span
    head = ""
    if at != -1:
        # A `?` or `#` before that `@` means the `@` may sit in the query: keep nothing.
        if "?" in span[:at] or "#" in span[:at]:
            return "***"
        head, rest = "***@", span[at + 1 :]
    query = _QUERY_START.search(rest)
    if query:
        marker = "?" if query[0] in "?#" else query[0]
        return f"{head}{_mask_path(rest[: query.start()])}{marker}***"
    return head + _mask_path(rest)


def _mask_url(url: str) -> str:
    start = url.find("://") + 3 if "://" in url else 0
    return url[:start] + _mask_url_span(url[start:])


def _userinfo_end(text: str, start: int, end: int, bound: int) -> int:
    """Where the userinfo of the URL at ``text[start:end]`` ends (its `@`), or -1.

    A host-shaped `@` in ``text[end:bound]`` wins over one inside the span.
    """
    at = bound
    while (at := text.rfind("@", end, at)) != -1:
        if not _DIGEST_AT.match(text, at) and _HOST_AT.match(text, at):
            return at
    at = end
    while (at := text.rfind("@", start, at)) != -1:
        if not _DIGEST_AT.match(text, at):
            return at
    return -1


def _redact_urls(text: str) -> str:
    parts: list[str] = []
    pos = 0
    line_end = -1
    while match := _URL_START.search(text, pos):
        start = match.end()
        stop = _URL_END.search(text, start)
        end = stop.start() if stop else len(text)
        if line_end < end:
            line = _LINE_END.search(text, end)
            line_end = line.start() if line else len(text)
        following = _URL_START.search(text, end, line_end)
        bound = following.start() if following else line_end
        at = _userinfo_end(text, start, end, bound)
        if at > end:
            stop = _URL_END.search(text, at, bound)
            end = stop.start() if stop else bound
        parts += (text[pos:start], _mask_url_span(text[start:end]))
        pos = end
    parts.append(text[pos:])
    return "".join(parts)


# Then userinfo without a scheme: a whitespace-free `user:pass@host` (a URL written without
# `https://`, as aiohttp quotes it), and `user@host` followed by a port or a path, masked up to its
# last `@`. An e-mail address (`user@example.com`) stays.
_RUN = re.compile(r"\S+")
_RUN_LEAD = re.compile(r"(?:[A-Za-z_][\w.-]*=)?['\"(<\[/]*")


def _mask_bare_userinfo(match: re.Match) -> str:
    run = match[0]
    if "@" not in run or "://" in run:
        return run
    at = len(run)
    while (at := run.rfind("@", 0, at)) != -1:
        if not _DIGEST_AT.match(run, at) and (host := _HOST_AT.match(run, at)):
            break
    else:
        return run
    lead = _RUN_LEAD.match(run).end()
    if lead >= at:
        return run
    urlish = (
        ":" in run[lead:at]
        or run.startswith((":", "/"), host.end())
        or run[:lead].endswith("//")
    )
    if not urlish:
        return run
    tail = len(run[host.end() :].rstrip("'\")>],;")) + host.end()
    return run[:lead] + _mask_url_span(run[lead:tail]) + run[tail:]


def _redact_bare_userinfo(text: str) -> str:
    return _RUN.sub(_mask_bare_userinfo, text)


# Then secret-named values: `name=value`, `name: value`, `--name=value`, `'name': 'value'` and
# headers such as `Authorization: Bearer v`, `Private-Token: v` and `Cookie: a=b; c=d`. Only the
# name and separator are matched here; the value is read only after the name is known to be
# secret, so a pair such as `url='...'` never hides what follows it.
_NAMED_VALUE = re.compile(
    r"(?<![\w.])(?P<quote>['\"]?)(?P<name>[A-Za-z_][\w.-]{0,127})(?P=quote)"
    r"(?P<index>(?:\[[^\]\s]{0,64}\])*)\]?\s*(?P<sep>[:=])\s*"
)
# A class name before its message (`KeyError: 'x'`, `TokenRefreshError: ...`) names no value.
_CLASS_NAME_WORDS = frozenset({"error", "exception", "warning"})
# After a bare `key:` or `token:`, a quoted identifier (`invalid key: 'gpu_model'`) or an error
# phrase (`Token: unexpected EOF`) is prose, not a secret.
_PROSE_NAMES = frozenset({"key", "token"})
_QUOTED_IDENTIFIER = re.compile(r"(['\"])[A-Za-z_]{1,32}[0-9]{0,3}\1")
_ERROR_PHRASE = re.compile(
    r"(?i)(?:unexpected|invalid|missing|expired|required|empty|malformed|unknown|not|none|null"
    r"|undefined|failed)\b"
)
_AUTH_SCHEME = re.compile(r"(?i)(?:bearer|basic|token|digest)\s{1,8}")
_PLAIN_VALUE = re.compile(r"[^\s'\",;&]+")
_COOKIE_VALUE = re.compile(r"[^\r\n'\"]+")


def _quoted_value_end(text: str, start: int) -> int:
    """Index of the quote closing the value opened at ``start``, or the end of ``text``."""
    quote = text[start]
    pos = start + 1
    while (pos := text.find(quote, pos)) != -1:
        if text[pos - 1] != "\\":
            return pos
        pos += 1
    return len(text)


def _mask_named_value(text: str, name: str, start: int) -> tuple[str, int]:
    """The masked value of secret ``name`` starting at ``start``, and where the value ends."""
    lead = start
    if text.startswith(("b'", 'b"'), start):
        lead += 1
    if lead < len(text) and text[lead] in "'\"":
        close = _quoted_value_end(text, lead)
        return f"{text[start : lead + 1]}***", close
    if "cookie" in _name_words(name):
        value = _COOKIE_VALUE.match(text, start)
        return ("***", value.end()) if value else ("", start)
    scheme = _AUTH_SCHEME.match(text, start)
    lead = scheme.end() if scheme else start
    value = _PLAIN_VALUE.match(text, lead)
    if not value:
        return "", start
    return f"{text[start:lead]}***", value.end()


def _names_a_secret(text: str, match: re.Match) -> bool:
    name = match["name"]
    words = _name_words(name)
    if words and words[-1] in _CLASS_NAME_WORDS:
        name = ""
    if not (_is_secret_name(name) or _is_secret_name(match["index"])):
        return False
    if match["sep"] == ":" and not match["quote"] and name.lower() in _PROSE_NAMES:
        after = match.end()
        return not (_QUOTED_IDENTIFIER.match(text, after) or _ERROR_PHRASE.match(text, after))
    return True


def _redact_named_values(text: str) -> str:
    parts: list[str] = []
    pos = 0
    while match := _NAMED_VALUE.search(text, pos):
        parts.append(text[pos : match.end()])
        pos = match.end()
        if _names_a_secret(text, match):
            masked, pos = _mask_named_value(text, match["name"], pos)
            parts.append(masked)
    parts.append(text[pos:])
    return "".join(parts)


# Last, credentials without a name: a bearer/basic/token value (only when it is token-shaped, so
# prose such as "basic checks failed" stays: 16 or more token characters, or 8 or more with a
# letter and a digit) and known token formats.
_AUTH_VALUE = re.compile(r"(?i)\b(bearer|basic|token)(\s{1,8})([A-Za-z0-9][A-Za-z0-9._~+/=-]*)")
_TOKEN_FORMATS = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|eyJ[A-Za-z0-9_-]{1,}\.[A-Za-z0-9_-]{1,}\.[A-Za-z0-9_-]{1,})"
)


def _mask_auth_value(match: re.Match) -> str:
    value = match[3]
    shaped = len(value) >= 16 or (
        len(value) >= 8 and any(c.isdigit() for c in value) and any(c.isalpha() for c in value)
    )
    return f"{match[1]}{match[2]}***" if shaped else match[0]


def _drop_split_tail(text: str, urls: tuple[str, ...]) -> str:
    """``text`` (already cut) without the run the cut split, nor a carried URL the cut split."""
    end = len(text)
    while end and not text[end - 1].isspace():
        end -= 1
    text = text[:end]
    for url in filter(None, urls):
        head = url[: url.find("://") + 3] if "://" in url else url[:1]
        low = max(0, len(text) - len(url) + 1)
        start = len(text)
        while (start := text.rfind(head, low, start + len(head) - 1)) != -1:
            if url.startswith(text[start:]):
                text = text[:start]
                break
    return text


def redact(text: str, urls: tuple[str, ...] = ()) -> str:
    """``text`` cut to ``MAX_REDACTED_CHARS``, with any credentials in it replaced by ``***``.

    ``urls`` are URLs the error carries as attributes: each is masked as one URL wherever it
    appears, even when its password holds a space or a quote that ends a URL in free text.
    """
    cut = len(text) > MAX_REDACTED_CHARS
    if cut:
        text = _drop_split_tail(text[:MAX_REDACTED_CHARS], urls)
    for url in urls:
        text = text.replace(url, _mask_url(url))
    text = _redact_named_values(_redact_bare_userinfo(_redact_urls(text)))
    text = _TOKEN_FORMATS.sub("***", _AUTH_VALUE.sub(_mask_auth_value, text))
    # A `***` can be longer than the value it replaces; cutting the redacted text only shortens it.
    if len(text) > MAX_REDACTED_CHARS:
        text, cut = text[:MAX_REDACTED_CHARS], True
    return text + "…" if cut else text


def _urls_of(error: object) -> tuple[str, ...]:
    """URLs an aiohttp error carries, whatever their scheme, longest first.

    ``InvalidURL.url``, ``request_info.url`` / ``real_url``, and the first argument of
    ``NonHttpUrlClientError`` / ``NonHttpUrlRedirectClientError``, which carry their URL (a ``URL``
    or the raw ``Location``) there and have no ``.url``.
    """
    urls = set()
    try:
        info = getattr(error, "request_info", None)
        values = [
            getattr(error, "url", None),
            getattr(info, "url", None),
            getattr(info, "real_url", None),
        ]
        if isinstance(error, NonHttpUrlClientError) and error.args:
            values.append(error.args[0])
        for value in values:
            if value is not None:
                url = str.__str__(str(value))
                if url:
                    urls.add(url)
    except Exception:
        pass
    return tuple(sorted(urls, key=len, reverse=True))


class _Described(str):
    """What ``describe_error`` returns: ``_clip`` publishes it as is, never describing it twice."""


def describe_error(error: object) -> str:
    """How an error is shown in the log and in the document: its class and its redacted text.

    Never raises: it runs inside the loop's except clauses. An error whose class name cannot be
    read is named ``UNNAMED_ERROR``; one whose ``str()`` raises is shown by its class alone.
    """
    try:
        name = str.__str__(type(error).__name__)
    except Exception:
        name = UNNAMED_ERROR
    try:
        # `str.__str__` makes an exact str of a str subclass, whose methods could raise.
        text = str.__str__(str(error))
    except Exception:
        return _Described(name)
    try:
        message = redact(text, _urls_of(error))
        if isinstance(error, BaseException):
            return _Described(f"{name}: {message}" if message else name)
        return _Described(message)
    except Exception:
        return _Described(name)


def _public_url(url: str) -> str:
    """``url`` as ``scheme://host[:port]/path`` from yarl's parse, else ``UNPARSEABLE_URL``.

    The userinfo, query and fragment are never published. A parse that leaves an `@` after the
    host (a password holding an unencoded `/`, `?` or `#`, read as host, port and path) is refused,
    as is a URL without a scheme and host.
    """
    try:
        parsed = URL(url)
        if not (parsed.absolute and parsed.scheme and parsed.raw_host):
            return UNPARSEABLE_URL
        if "@" in f"{parsed.raw_path}{parsed.raw_query_string}{parsed.raw_fragment}":
            return UNPARSEABLE_URL
        host = f"[{parsed.raw_host}]" if ":" in parsed.raw_host else parsed.raw_host
        port = f":{parsed.explicit_port}" if parsed.explicit_port is not None else ""
        return redact(f"{parsed.scheme}://{host}{port}{parsed.raw_path}")
    except Exception:
        return UNPARSEABLE_URL


def _clip(value: object | None, limit: int = MAX_ERROR_CHARS) -> str | None:
    """Describe and bound one error message. ``None`` stays ``None``."""
    if value is None:
        return None
    text = str.__str__(value) if type(value) is _Described else describe_error(value)
    return _cut(text, limit)


def _cut(text: str, limit: int) -> str:
    text = str.__str__(text)
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
            doc[key] = _cut(value, short)
    for record in (doc.get("images") or {}).values():
        for key, value in record.items():
            if (key.endswith("_error") or key == "last_error") and isinstance(value, str):
                record[key] = _cut(value, short)


def _drop_counts(doc: dict) -> None:
    doc.pop("outcome_counts", None)
    for record in (doc.get("images") or {}).values():
        record.pop("outcome_counts", None)


def _drop_images(doc: dict) -> None:
    doc["images"] = {}


@dataclass(slots=True)
class _ImageRecord:
    """Everything known about one template, as a type rather than a bare dict.

    ``slots=True`` is the whole point: a mistyped field name raises instead of
    quietly adding an attribute nobody publishes. That matters here more than
    usual, because every writer below runs under ``@_never_raises`` — with a dict,
    the typo would be swallowed and the document would ship missing the field.
    """

    last_outcome: str | None = None
    last_outcome_at: str | None = None
    last_error: str | None = None
    local_digests: list[str] = field(default_factory=list)
    local_digest_first_seen_at: str | None = None
    digest_changed_at: str | None = None
    last_local_error: str | None = None
    expected_digest: str | None = None
    remote_digest: str | None = None
    last_remote_read_ok_at: str | None = None
    last_remote_error: str | None = None
    last_pull_attempt_at: str | None = None
    last_pull_ok_at: str | None = None
    last_pull_error: str | None = None
    last_cleanup_error: str | None = None
    last_disk_required_bytes: int | None = None
    last_disk_available_bytes: int | None = None
    outcome_counts: dict[str, int] = field(default_factory=dict)


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
        self._backend_url = _public_url(backend_url) if backend_url else None
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
        # The validator holds a fresh node's cached-image verdict until this is set. Top-level
        # rather than read off outcome_counts, which `_fit` sheds under size pressure.
        self._first_sweep_ok_at: str | None = None

        self._images: dict[str, _ImageRecord] = {}

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
        if outcome == Outcome.SWEEP_OK and self._first_sweep_ok_at is None:
            self._first_sweep_ok_at = self._last_outcome_at

    # -- per-image facts --------------------------------------------------------

    def _record(self, image_ref: str) -> _ImageRecord:
        return self._images.setdefault(image_ref, _ImageRecord())

    @_never_raises
    def note_local_digests(
        self, image_ref: str, digests: list[str], error: object | None = None
    ) -> None:
        record = self._record(image_ref)
        previous = record.local_digests
        record.local_digests = list(digests)
        record.last_local_error = _clip(error)
        if not previous and digests:
            record.local_digest_first_seen_at = _utcnow()
        elif previous and digests and previous != list(digests):
            # The content under the tag moved. This is the single clearest signal
            # that a pull actually landed, as opposed to merely being attempted.
            record.digest_changed_at = _utcnow()

    @_never_raises
    def note_expected_digest(self, image_ref: str, digest: str | None) -> None:
        self._record(image_ref).expected_digest = digest

    @_never_raises
    def note_remote_digest(
        self, image_ref: str, digest: str | None, error: object | None = None
    ) -> None:
        record = self._record(image_ref)
        record.remote_digest = digest
        record.last_remote_error = _clip(error)
        if digest:
            record.last_remote_read_ok_at = _utcnow()

    @_never_raises
    def note_disk(self, image_ref: str, required_bytes: int, available_bytes: int) -> None:
        record = self._record(image_ref)
        record.last_disk_required_bytes = required_bytes
        record.last_disk_available_bytes = available_bytes

    @_never_raises
    def note_pull_attempt(self, image_ref: str) -> None:
        self._record(image_ref).last_pull_attempt_at = _utcnow()

    @_never_raises
    def note_pull_ok(self, image_ref: str) -> None:
        record = self._record(image_ref)
        record.last_pull_ok_at = _utcnow()
        record.last_pull_error = None

    @_never_raises
    def note_pull_error(self, image_ref: str, error: object) -> None:
        self._record(image_ref).last_pull_error = _clip(error)

    @_never_raises
    def note_cleanup_error(self, image_ref: str, error: object | None) -> None:
        self._record(image_ref).last_cleanup_error = _clip(error)

    @_never_raises
    def record_image_outcome(
        self, image_ref: str, outcome: str, error: object | None = None
    ) -> None:
        record = self._record(image_ref)
        record.last_outcome = outcome
        record.last_outcome_at = _utcnow()
        record.last_error = _clip(error)
        counts = record.outcome_counts
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
            "first_sweep_ok_at": self._first_sweep_ok_at,
            # `asdict` copies as it flattens, so `_fit` can trim its result freely.
            "images": {ref: asdict(record) for ref, record in self._images.items()},
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
