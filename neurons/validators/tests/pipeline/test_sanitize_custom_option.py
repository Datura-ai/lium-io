"""Test suite for CustomOptions sanitization to prevent command injection."""

import pytest
from payload_models.payloads import CustomOptions


def test_sanitize_none_input():
    """Test sanitization with None input."""
    # Act
    result = CustomOptions.sanitize(None)

    # Assert - should return empty CustomOptions with all None fields
    assert isinstance(result, CustomOptions)
    assert result.volumes is None
    assert result.environment is None
    assert result.entrypoint is None
    assert result.internal_ports is None
    assert result.startup_commands is None
    assert result.shm_size is None
    assert result.initial_port_count is None


def test_sanitize_empty_custom_options():
    """Test sanitization with empty CustomOptions."""
    # Arrange
    empty_options = CustomOptions()

    # Act
    result = CustomOptions.sanitize(empty_options)

    # Assert - empty list/dict becomes None
    assert isinstance(result, CustomOptions)
    assert result.volumes is None
    assert result.environment is None
    assert result.entrypoint is None


def test_sanitize_volumes_malicious_injection():
    """Test volume sanitization against command injection attacks."""
    # Arrange
    malicious_options = CustomOptions(
        volumes=[
            "/root --mount type='bind',source=/,target=/host --privileged",
            "/var/run/docker.sock:/var/run/docker.sock",
            "/usr/bin/docker:/usr/bin/docker",
            "/etc/passwd:/etc/passwd",
            "/safe/path:/safe/container",  # This should pass
            "invalid_format",  # No colon - should be filtered
            "",  # Empty - should be filtered
            "   ",  # Whitespace only - should be filtered
        ]
    )

    # Act
    result = CustomOptions.sanitize(malicious_options)

    # Assert - should only keep the safe volume
    assert result.volumes == ["/safe/path:/safe/container"]


def test_sanitize_volumes_safe_paths():
    """Test volume sanitization with safe paths."""
    # Arrange
    safe_options = CustomOptions(
        volumes=[
            "/home/user/app:/app",
            "/data/storage:/data",
            "/tmp/cache:/tmp/cache",
            "/var/log/app:/var/log/app",
        ]
    )

    # Act
    result = CustomOptions.sanitize(safe_options)

    # Assert - all safe volumes should be preserved
    assert result.volumes == [
        "/home/user/app:/app",
        "/data/storage:/data",
        "/tmp/cache:/tmp/cache",
        "/var/log/app:/var/log/app",
    ]


def test_sanitize_environment_dangerous_keys():
    """Test environment sanitization against dangerous keys."""
    # Arrange
    dangerous_options = CustomOptions(
        environment={
            "PATH": "/malicious/path",
            "LD_LIBRARY_PATH": "/evil/lib",
            "LD_PRELOAD": "malicious.so",
            "PYTHONPATH": "/bad/python",
            "SAFE_VAR": "safe_value",
            "APP_CONFIG": "config_value",
            "": "empty_key",  # Should be filtered
            "   ": "whitespace_key",  # Should be filtered
        }
    )

    # Act
    result = CustomOptions.sanitize(dangerous_options)

    # Assert - only safe environment variables should remain
    assert result.environment == {
        "SAFE_VAR": "safe_value",
        "APP_CONFIG": "config_value",
    }


def test_sanitize_environment_safe_keys():
    """Test environment sanitization with safe keys."""
    # Arrange
    safe_options = CustomOptions(
        environment={
            "APP_NAME": "myapp",
            "DEBUG": "true",
            "PORT": "8080",
            "DATABASE_URL": "postgresql://localhost/db",
        }
    )

    # Act
    result = CustomOptions.sanitize(safe_options)

    # Assert - all safe environment variables should be preserved
    assert result.environment == {
        "APP_NAME": "myapp",
        "DEBUG": "true",
        "PORT": "8080",
        "DATABASE_URL": "postgresql://localhost/db",
    }


def test_sanitize_entrypoint_malicious():
    """Test entrypoint sanitization against malicious input."""
    # Arrange
    malicious_options = CustomOptions(
        entrypoint="--privileged --mount type=bind,source=/,target=/host"
    )

    # Act
    result = CustomOptions.sanitize(malicious_options)

    # Assert - only first part is kept (flags stripped by --entrypoint flag)
    assert result.entrypoint == "--privileged"


@pytest.mark.parametrize(
    "entrypoint",
    [
        "/usr/bin/python3",  # Absolute path
        "/bin/bash",
        "/app/main.py",
        "/usr/local/bin/myapp",
        "/abc/01.py",  # Numeric in path
        "/home/user/script_1.sh",
        "./script.sh",  # Relative path
        "abc/script.py",  # Relative path without ./
        "app/main.py",  # Relative path
        "myapp",  # Simple command
        "custom_app",  # Custom application
        "startup",  # Custom command
        "bash",  # Shell command (safe with --entrypoint flag)
        "python",  # Interpreter (safe with --entrypoint flag)
        "node",  # Interpreter (safe with --entrypoint flag)
    ],
)
def test_sanitize_entrypoint_safe_paths(entrypoint: str):
    """Test entrypoint sanitization with safe paths."""
    # Arrange
    options = CustomOptions(entrypoint=entrypoint)

    # Act
    result = CustomOptions.sanitize(options)

    # Assert - safe entrypoint should be preserved
    assert result.entrypoint == entrypoint


@pytest.mark.parametrize(
    "entrypoint",
    [
        "|rm -rf /",  # Command injection (starts with |)
        "$(whoami)",  # Command substitution (contains $)
        "rm; cat",  # Contains semicolon
        "test;",  # Contains semicolon
        "",  # Empty
        "   ",  # Whitespace only
    ],
)
def test_sanitize_entrypoint_invalid(entrypoint: str):
    """Test entrypoint sanitization with invalid entries."""
    # Arrange
    options = CustomOptions(entrypoint=entrypoint)

    # Act
    result = CustomOptions.sanitize(options)

    # Assert - invalid entrypoint should be filtered to None
    assert result.entrypoint is None


def test_sanitize_shm_size_malicious():
    """Test shm_size sanitization against malicious input."""
    # Arrange
    malicious_options = CustomOptions(shm_size="1g --privileged --mount type=bind")

    # Act
    result = CustomOptions.sanitize(malicious_options)

    # Assert - should only keep the valid part (1g) and filter out the malicious flags
    assert result.shm_size == "1g"


@pytest.mark.parametrize(
    "shm_size",
    [
        "1g",
        "512m",
        "1024",
        "2G",
        "256M",
    ],
)
def test_sanitize_shm_size_valid(shm_size: str):
    """Test shm_size sanitization with valid values."""
    # Arrange
    options = CustomOptions(shm_size=shm_size)

    # Act
    result = CustomOptions.sanitize(options)

    # Assert - valid shm_size should be preserved
    assert result.shm_size == shm_size


@pytest.mark.parametrize(
    "shm_size",
    [
        "invalid_size",
        "1x",  # Invalid unit
        "",  # Empty
        "   ",  # Whitespace only
    ],
)
def test_sanitize_shm_size_invalid(shm_size: str):
    """Test shm_size sanitization with invalid values."""
    # Arrange
    options = CustomOptions(shm_size=shm_size)

    # Act
    result = CustomOptions.sanitize(options)

    # Assert - invalid shm_size should be filtered to None
    assert result.shm_size is None


def test_sanitize_shm_size_extracts_valid_part():
    """Test that valid part is extracted from mixed shm_size input."""
    # Arrange
    mixed_options = CustomOptions(shm_size="1g --privileged")

    # Act
    result = CustomOptions.sanitize(mixed_options)

    # Assert - should extract valid part from malicious input
    assert result.shm_size == "1g"


def test_sanitize_preserves_safe_fields():
    """Test that safe fields are preserved during sanitization."""
    # Arrange
    original_options = CustomOptions(
        internal_ports=[8080, 9090],
        initial_port_count=5,
        startup_commands="echo 'Starting app'",
        volumes=["/safe/path:/safe/container"],
        environment={"SAFE_VAR": "safe_value"},
        entrypoint="/usr/bin/python3",
        shm_size="1g",
    )

    # Act
    result = CustomOptions.sanitize(original_options)

    # Assert - safe fields should be preserved
    assert result.internal_ports == [8080, 9090]
    assert result.initial_port_count == 5
    assert result.startup_commands == "echo 'Starting app'"
    assert result.volumes == ["/safe/path:/safe/container"]
    assert result.environment == {"SAFE_VAR": "safe_value"}
    assert result.entrypoint == "/usr/bin/python3"
    assert result.shm_size == "1g"


def test_sanitize_comprehensive_attack():
    """Test sanitization against a comprehensive attack scenario."""
    # Arrange - simulate a real attack attempt
    attack_options = CustomOptions(
        volumes=[
            "/root --mount type='bind',source=/,target=/host --privileged --mount type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock --mount type=bind,source=/usr/bin/docker,target=/usr/bin/docker",
            "/etc/passwd:/etc/passwd",
            "/var/run/docker.sock:/var/run/docker.sock",
        ],
        environment={
            "PATH": "/malicious/path",
            "LD_PRELOAD": "malicious.so",
            "SAFE_VAR": "safe_value",
        },
        entrypoint="bash --privileged --mount type=bind,source=/,target=/host",
        shm_size="1g --privileged",
    )

    # Act
    result = CustomOptions.sanitize(attack_options)

    # Assert - all malicious content should be filtered out
    assert result.volumes is None  # All volumes were dangerous (empty list becomes None)
    assert result.environment == {"SAFE_VAR": "safe_value"}  # Only safe env var
    assert result.entrypoint == "bash"  # Only first part allowed (flags stripped)
    assert result.shm_size == "1g"  # Valid part extracted from malicious input


def test_sanitize_whitespace_handling():
    """Test that whitespace is properly handled."""
    # Arrange
    options = CustomOptions(
        volumes=["  /safe/path:/safe/container  ", "   ", ""],
        environment={"  SAFE_VAR  ": "  safe_value  ", "": "empty"},
        entrypoint="  /usr/bin/python3  ",
        shm_size="  1g  ",
    )

    # Act
    result = CustomOptions.sanitize(options)

    # Assert - whitespace should be trimmed, empty values filtered
    assert result.volumes == ["/safe/path:/safe/container"]
    assert result.environment == {"SAFE_VAR": "safe_value"}
    assert result.entrypoint == "/usr/bin/python3"
    assert result.shm_size == "1g"
