"""The approved artifact catalog — which software stacks this host is allowed to launch.

Four modules, split by what each one is allowed to trust:

    artifacts.py   what an approved stack is, and how a request is matched against one. Pure.
    manifest.py    the signed document the platform publishes, and every check that makes it
                   evidence: signer, freshness, serial, floors.
    store.py       this host's copy — verify on read, materialize the measured inputs, install
                   atomically.
    client.py      getting the bytes. Trusts nothing about them.

Until DAH-2578 this was one file reading a plain JSON catalog written by Ansible, whose
integrity rested entirely on its filesystem permissions. The backend is the source of truth
now: a host launches what the platform signed, and a revoked artifact stops being launchable
everywhere as soon as the next manifest arrives.
"""

from cvmd.catalog.artifacts import (
    HEX64,
    Artifact,
    CatalogError,
    TripleNotFound,
    assert_hex64,
    assert_unambiguous,
    resolve,
    resolve_base,
)
from cvmd.catalog.client import FetchError, fetch, refresh_once
from cvmd.catalog.manifest import Entry, Manifest, assert_not_rollback, parse_manifest
from cvmd.catalog.store import CatalogConfig, CatalogStore

__all__ = [
    "HEX64",
    "Artifact",
    "CatalogConfig",
    "CatalogError",
    "CatalogStore",
    "Entry",
    "FetchError",
    "Manifest",
    "TripleNotFound",
    "assert_hex64",
    "assert_not_rollback",
    "assert_unambiguous",
    "fetch",
    "parse_manifest",
    "refresh_once",
    "resolve",
    "resolve_base",
]
