"""Fetching a manifest, and the two routes that expose it.

A real HTTP server on loopback rather than a mocked `urlopen`: the whole point of the client is
what it does with bytes that arrived over a socket, and a mock would agree with whatever the
code did. The server here is the smallest thing that can serve, stall, or lie.
"""

import http.server
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import manifest_entry, manifest_payload, sign_manifest, signed_request
from cvmd.app import create_app
from cvmd.catalog import CatalogConfig, CatalogStore, FetchError, fetch, refresh_once
from cvmd.config import Config
from fastapi.testclient import TestClient


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
        body, status = self.server.answer
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Silence. The test's output is the assertions, not an access log."""


@pytest.fixture
def backend():
    """A stand-in for the backend's manifest endpoint whose answer a test can change."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    server.answer = (b"{}", 200)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def url_of(server) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/v1/cvm-catalog/manifest"


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    image = tmp_path / "images" / "dstack-nvidia-0.5.11"
    image.mkdir(parents=True)
    (image / "digest.txt").write_text("a" * 64 + "\n")
    return tmp_path / "images"


@pytest.fixture
def catalog_config(tmp_path: Path, images_dir: Path, catalog_signer, backend) -> CatalogConfig:
    return CatalogConfig(
        cache_path=tmp_path / "state" / "catalog" / "manifest.json",
        signer=catalog_signer.ss58_address,
        images_dir=images_dir,
        materialize_dir=tmp_path / "state" / "catalog" / "artifacts",
        manifest_url=url_of(backend),
        fetch_timeout_seconds=5,
    )


@pytest.fixture
def store(catalog_config: CatalogConfig) -> CatalogStore:
    return CatalogStore(catalog_config)


def serve(backend, catalog_signer, *entries, status: int = 200, **kwargs) -> bytes:
    raw = sign_manifest(manifest_payload(*entries, **kwargs), catalog_signer)
    backend.answer = (raw, status)
    return raw


class TestFetching:
    def test_it_returns_the_body(self, backend, catalog_signer, catalog_config):
        raw = serve(backend, catalog_signer, manifest_entry())

        assert fetch(catalog_config.manifest_url, timeout=5) == raw

    def test_an_error_status_is_a_fetch_error(self, backend, catalog_signer, catalog_config):
        serve(backend, catalog_signer, manifest_entry(), status=503)

        with pytest.raises(FetchError, match="503"):
            fetch(catalog_config.manifest_url, timeout=5)

    def test_an_unreachable_backend_is_a_fetch_error(self):
        with pytest.raises(FetchError, match="cannot reach"):
            fetch("http://127.0.0.1:1/manifest", timeout=1)

    def test_an_empty_body_is_a_fetch_error(self, backend, catalog_config):
        backend.answer = (b"", 200)

        with pytest.raises(FetchError, match="empty body"):
            fetch(catalog_config.manifest_url, timeout=5)

    def test_an_oversized_body_is_refused(self, backend, catalog_config, monkeypatch):
        monkeypatch.setattr("cvmd.catalog.client.MAX_MANIFEST_BYTES", 16)
        backend.answer = (b"x" * 64, 200)

        with pytest.raises(FetchError, match="more than 16 bytes"):
            fetch(catalog_config.manifest_url, timeout=5)

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.invalid/m.json"])
    def test_only_http_urls_are_fetched(self, url):
        """`urlopen` speaks file: and ftp: too, and a manifest URL is configuration — the point
        is that a mistake there cannot turn into reading an arbitrary local file."""
        with pytest.raises(FetchError, match="must be http or https"):
            fetch(url, timeout=1)


class TestRefreshing:
    def test_a_good_manifest_becomes_the_catalog(self, store, backend, catalog_signer):
        serve(backend, catalog_signer, manifest_entry(), serial=5)

        assert refresh_once(store).serial == 5
        assert store.current().serial == 5

    def test_an_unreachable_backend_leaves_the_catalog_alone(self, store, backend, catalog_signer):
        """A network failure must not be able to stop a host launching."""
        serve(backend, catalog_signer, manifest_entry(), serial=5)
        refresh_once(store)
        offline = CatalogStore(replace(store.config, manifest_url="http://127.0.0.1:1/manifest"))

        assert refresh_once(offline) is None
        assert offline.current().serial == 5

    def test_a_tampered_manifest_leaves_the_catalog_alone(self, store, backend, catalog_signer):
        """The task's tamper case, end to end: one flipped byte on the wire changes nothing."""
        serve(backend, catalog_signer, manifest_entry(), serial=5)
        refresh_once(store)

        envelope = json.loads(
            sign_manifest(manifest_payload(manifest_entry(), serial=6), catalog_signer)
        )
        envelope["payload"] = envelope["payload"].replace("validation-v3", "validation-v4")
        backend.answer = (json.dumps(envelope).encode(), 200)

        assert refresh_once(store) is None
        assert [entry.id for entry in store.current().entries] == ["validation-v3"]

    def test_a_manifest_from_another_signer_leaves_the_catalog_alone(
        self, store, backend, catalog_signer, other_signer
    ):
        serve(backend, catalog_signer, manifest_entry(), serial=5)
        refresh_once(store)
        backend.answer = (
            sign_manifest(manifest_payload(manifest_entry(), serial=6), other_signer),
            200,
        )

        assert refresh_once(store) is None
        assert store.current().serial == 5

    def test_a_newly_approved_image_hash_arrives_on_the_next_fetch(
        self, store, backend, catalog_signer
    ):
        """The task's end-to-end case, from the host's side."""
        serve(backend, catalog_signer, manifest_entry(), serial=1)
        refresh_once(store)
        assert [a.os_image_hash for a in store.artifacts()] == ["a" * 64]

        serve(
            backend,
            catalog_signer,
            manifest_entry(),
            manifest_entry(id="validation-v4", os_image_hash="c" * 64, compose_hash="d" * 64),
            serial=2,
        )
        refresh_once(store)

        assert sorted(a.os_image_hash for a in store.artifacts()) == ["a" * 64, "c" * 64]

    def test_a_host_with_no_manifest_url_never_fetches(self, tmp_path, images_dir, catalog_signer):
        store = CatalogStore(
            CatalogConfig(
                cache_path=tmp_path / "m.json",
                signer=catalog_signer.ss58_address,
                images_dir=images_dir,
                materialize_dir=tmp_path / "artifacts",
            )
        )
        assert refresh_once(store) is None

    def test_a_newly_staged_seed_is_adopted_without_a_restart(
        self, tmp_path, images_dir, catalog_signer, backend
    ):
        """An operator who stages a newer seed should not also have to restart the daemon."""
        seed = tmp_path / "seed.json"
        store = CatalogStore(
            CatalogConfig(
                cache_path=tmp_path / "m.json",
                seed_path=seed,
                signer=catalog_signer.ss58_address,
                images_dir=images_dir,
                materialize_dir=tmp_path / "artifacts",
                manifest_url=url_of(backend),
                fetch_timeout_seconds=5,
            )
        )
        serve(backend, catalog_signer, manifest_entry(), serial=2)
        assert refresh_once(store).serial == 2

        seed.write_bytes(sign_manifest(manifest_payload(manifest_entry(), serial=7), catalog_signer))
        refresh_once(store)

        assert store.current().serial == 7

    def test_an_unreachable_backend_still_adopts_a_newer_seed(
        self, tmp_path, images_dir, catalog_signer
    ):
        """The air-gapped case: no URL to fetch from, and the seed is the whole mechanism."""
        seed = tmp_path / "seed.json"
        seed.write_bytes(sign_manifest(manifest_payload(manifest_entry(), serial=4), catalog_signer))
        store = CatalogStore(
            CatalogConfig(
                cache_path=tmp_path / "m.json",
                seed_path=seed,
                signer=catalog_signer.ss58_address,
                images_dir=images_dir,
                materialize_dir=tmp_path / "artifacts",
            )
        )

        assert refresh_once(store).serial == 4


class TestOverHttp:
    @pytest.fixture
    def client(self, clients_file, state_dir, catalog_config):
        config = Config(
            authorized_clients=clients_file, state_dir=state_dir, catalog=catalog_config
        )
        with TestClient(create_app(config), raise_server_exceptions=False) as test_client:
            yield test_client

    def test_the_catalog_route_reports_a_host_with_nothing(self, client, validator_key):
        response = signed_request(client, validator_key, "GET", "/v1/catalog")

        assert response.status_code == 200
        assert response.json()["usable"] is False
        assert "holds no catalog manifest" in response.json()["error"]

    def test_either_authorized_key_may_read_it(self, client, platform_key):
        assert signed_request(client, platform_key, "GET", "/v1/catalog").status_code == 200

    def test_an_unauthorized_key_may_not(self, client, stranger_key):
        assert signed_request(client, stranger_key, "GET", "/v1/catalog").status_code == 401

    def test_refresh_fetches_and_reports_the_new_catalog(
        self, client, platform_key, backend, catalog_signer
    ):
        serve(backend, catalog_signer, manifest_entry(), serial=9)

        response = signed_request(client, platform_key, "POST", "/v1/catalog/refresh")

        assert response.status_code == 200
        assert response.json()["fetched"] is True
        assert response.json()["catalog"]["manifest"]["serial"] == 9

    def test_refresh_reports_no_change_when_the_backend_is_unusable(
        self, client, platform_key, backend
    ):
        backend.answer = (b"not json", 200)

        response = signed_request(client, platform_key, "POST", "/v1/catalog/refresh")

        assert response.status_code == 200
        assert response.json()["fetched"] is False

    def test_the_catalog_route_then_shows_what_was_fetched(
        self, client, validator_key, platform_key, backend, catalog_signer
    ):
        serve(backend, catalog_signer, manifest_entry(), serial=9)
        signed_request(client, platform_key, "POST", "/v1/catalog/refresh")

        described = signed_request(client, validator_key, "GET", "/v1/catalog").json()

        assert described["usable"] is True
        assert described["manifest"]["entries"][0]["id"] == "validation-v3"
        assert described["signer"] == catalog_signer.ss58_address
