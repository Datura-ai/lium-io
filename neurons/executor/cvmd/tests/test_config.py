"""Fail-closed startup: config and the authorized-clients file."""

import json
from pathlib import Path

import pytest
from cvmd.app import create_app
from cvmd.auth.clients import AuthorizedClientsError, Scope, load_authorized_clients
from cvmd.config import Config, ConfigError, load_config

VALID_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
SECOND_HOTKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


def _write(path, payload):
    path.write_text(json.dumps(payload))
    return path


class TestAuthorizedClients:
    def test_valid_file_loads(self, tmp_path):
        path = _write(
            tmp_path / "clients.json",
            [
                {"hotkey": VALID_HOTKEY, "scope": "validation"},
                {"hotkey": SECOND_HOTKEY, "scope": "renter"},
            ],
        )
        clients = load_authorized_clients(path)
        assert clients.scope_for(VALID_HOTKEY) is Scope.VALIDATION
        assert clients.scope_for(SECOND_HOTKEY) is Scope.RENTER
        assert clients.scope_for("nobody") is None

    def test_missing_file_refuses(self, tmp_path):
        with pytest.raises(AuthorizedClientsError):
            load_authorized_clients(tmp_path / "absent.json")

    def test_malformed_json_refuses(self, tmp_path):
        path = tmp_path / "clients.json"
        path.write_text("{ not json")
        with pytest.raises(AuthorizedClientsError):
            load_authorized_clients(path)

    def test_broken_ss58_checksum_refuses(self, tmp_path):
        """A valid-shaped address with one character changed — the checksum catches it.

        This is the ss58 gate bittensor gives for free; cvmd has no hand-written base58 decode.
        """
        broken = VALID_HOTKEY[:-1] + ("A" if VALID_HOTKEY[-1] != "A" else "B")
        path = _write(tmp_path / "clients.json", [{"hotkey": broken, "scope": "validation"}])

        with pytest.raises(AuthorizedClientsError, match="ss58"):
            load_authorized_clients(path)

    def test_garbage_hotkey_refuses(self, tmp_path):
        path = _write(tmp_path / "clients.json", [{"hotkey": "not-an-address", "scope": "renter"}])
        with pytest.raises(AuthorizedClientsError):
            load_authorized_clients(path)

    def test_unknown_scope_refuses(self, tmp_path):
        """An unrecognised scope must not silently authorize nothing — or everything."""
        path = _write(tmp_path / "clients.json", [{"hotkey": VALID_HOTKEY, "scope": "admin"}])
        with pytest.raises(AuthorizedClientsError, match="unknown scope"):
            load_authorized_clients(path)

    def test_duplicate_hotkey_refuses(self, tmp_path):
        """Two scopes for one key: one would silently win."""
        path = _write(
            tmp_path / "clients.json",
            [
                {"hotkey": VALID_HOTKEY, "scope": "validation"},
                {"hotkey": VALID_HOTKEY, "scope": "renter"},
            ],
        )
        with pytest.raises(AuthorizedClientsError, match="more than once"):
            load_authorized_clients(path)

    def test_empty_list_refuses(self, tmp_path):
        path = _write(tmp_path / "clients.json", [])
        with pytest.raises(AuthorizedClientsError):
            load_authorized_clients(path)

    def test_missing_scope_field_refuses(self, tmp_path):
        path = _write(tmp_path / "clients.json", [{"hotkey": VALID_HOTKEY}])
        with pytest.raises(AuthorizedClientsError):
            load_authorized_clients(path)

    def test_flat_list_of_strings_refuses(self, tmp_path):
        """The DAH-2544 interface currently ships a flat list; it must not load as scopeless."""
        path = _write(tmp_path / "clients.json", [VALID_HOTKEY, SECOND_HOTKEY])
        with pytest.raises(AuthorizedClientsError):
            load_authorized_clients(path)


class TestAppStartup:
    def test_bad_clients_file_refuses_to_build_the_app(self, tmp_path, state_dir):
        path = _write(tmp_path / "clients.json", [{"hotkey": "garbage", "scope": "renter"}])
        config = Config(authorized_clients=path, state_dir=state_dir)

        with pytest.raises(AuthorizedClientsError):
            create_app(config)


class TestConfig:
    def test_loads_a_toml_file(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            "\n".join(
                [
                    'authorized_clients = "/etc/cvmd/authorized_clients.json"',
                    'state_dir = "/var/lib/cvmd"',
                    "port = 9443",
                    "max_body_bytes = 4096",
                ]
            )
        )
        config = load_config(path)
        assert config.port == 9443
        assert config.max_body_bytes == 4096
        assert config.skew_seconds == 60
        assert config.tls_enabled is False

    def test_the_renter_staging_directory_defaults_off_state_dir(self, tmp_path):
        """DAH-2580: a daemon-owned working directory, so naming `state_dir` is enough.

        Deliberately not under `run_dir` — that directory is scanned for stray CVM disks, and
        a staging directory appearing there would read as a guest with no record.
        """
        path = tmp_path / "config.toml"
        path.write_text(
            'authorized_clients = "/etc/cvmd/authorized_clients.json"\n'
            'state_dir = "/var/lib/cvmd"\n'
            'run_dir = "/var/lib/cvmd/vms"\n'
        )
        config = load_config(path)
        assert config.launch.renter_dir == Path("/var/lib/cvmd/renter")

    def test_the_renter_staging_directory_can_be_overridden(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            'authorized_clients = "/etc/cvmd/authorized_clients.json"\n'
            'state_dir = "/var/lib/cvmd"\n'
            'renter_dir = "/srv/cvmd-renter"\n'
        )
        assert load_config(path).launch.renter_dir == Path("/srv/cvmd-renter")

    def test_missing_required_key_refuses(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('state_dir = "/var/lib/cvmd"')
        with pytest.raises(ConfigError, match="authorized_clients"):
            load_config(path)

    def test_half_configured_tls_refuses(self, tmp_path):
        """A cert without a key would silently fall back to plaintext on a public port."""
        path = tmp_path / "config.toml"
        path.write_text('authorized_clients = "/a.json"\nstate_dir = "/b"\ntls_cert = "/c.pem"\n')
        with pytest.raises(ConfigError, match="together"):
            load_config(path)

    def test_non_integer_port_refuses(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('authorized_clients = "/a.json"\nstate_dir = "/b"\nport = "https"\n')
        with pytest.raises(ConfigError):
            load_config(path)

    def test_zero_body_cap_refuses(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('authorized_clients = "/a.json"\nstate_dir = "/b"\nmax_body_bytes = 0\n')
        with pytest.raises(ConfigError, match="positive"):
            load_config(path)

    @pytest.mark.parametrize(
        "mapping",
        [
            "tcp:12200",  # no guest side
            "tcp:0.0.0.0:0:22",  # port 0 — well-formed TOML, refused by the parser
            "tcp:0.0.0.0:70000:22",  # above 65535
            "sftp:0.0.0.0:2200:2200:2200",  # too many fields
        ],
    )
    def test_an_unusable_port_mapping_refuses(self, tmp_path, mapping):
        """Malformed is fatal at startup — absent and wrong are different problems.

        Left to the launch path, one of these makes a host that looks configured and refuses
        every launch at the first request, which is the state nobody discovers until a
        validator does.
        """
        path = tmp_path / "config.toml"
        path.write_text(
            f'authorized_clients = "/a.json"\nstate_dir = "/b"\ncvm_ports = ["{mapping}"]\n'
        )
        with pytest.raises(ConfigError, match="cvm_ports"):
            load_config(path)

    def test_a_usable_port_mapping_loads(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            'authorized_clients = "/a.json"\nstate_dir = "/b"\n'
            'cvm_ports = ["tcp:0.0.0.0:12200:2200", "udp:65535:1"]\n'
        )
        assert load_config(path).launch.ports == ("tcp:0.0.0.0:12200:2200", "udp:65535:1")

    def test_env_overrides_the_file(self, tmp_path, monkeypatch):
        path = tmp_path / "config.toml"
        path.write_text('authorized_clients = "/a.json"\nstate_dir = "/b"\nport = 9443\n')
        monkeypatch.setenv("CVMD_PORT", "8443")

        assert load_config(path).port == 8443
