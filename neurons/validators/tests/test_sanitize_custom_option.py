import pytest
from payload_models.payloads import CustomOptions


class TestCustomOptionsSanitization:
    """Test suite for CustomOptions sanitization to prevent command injection."""

    def test_sanitize_none_input(self):
        """Test sanitization with None input."""
        result = CustomOptions.sanitize(None)
        assert isinstance(result, CustomOptions)
        assert result.volumes is None
        assert result.environment is None
        assert result.entrypoint is None
        assert result.internal_ports is None
        assert result.startup_commands is None
        assert result.shm_size is None
        assert result.initial_port_count is None

    def test_sanitize_empty_custom_options(self):
        """Test sanitization with empty CustomOptions."""
        empty_options = CustomOptions()
        result = CustomOptions.sanitize(empty_options)
        assert isinstance(result, CustomOptions)
        assert result.volumes is None  # Empty list becomes None
        assert result.environment is None  # Empty dict becomes None
        assert result.entrypoint is None

    def test_sanitize_volumes_malicious_injection(self):
        """Test volume sanitization against command injection attacks."""
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
        
        result = CustomOptions.sanitize(malicious_options)
        
        # Should only keep the safe volume
        assert result.volumes == ["/safe/path:/safe/container"]

    def test_sanitize_volumes_safe_paths(self):
        """Test volume sanitization with safe paths."""
        safe_options = CustomOptions(
            volumes=[
                "/home/user/app:/app",
                "/data/storage:/data",
                "/tmp/cache:/tmp/cache",
                "/var/log/app:/var/log/app",
            ]
        )
        
        result = CustomOptions.sanitize(safe_options)
        
        # All safe volumes should be preserved
        assert result.volumes == [
            "/home/user/app:/app",
            "/data/storage:/data", 
            "/tmp/cache:/tmp/cache",
            "/var/log/app:/var/log/app",
        ]

    def test_sanitize_environment_dangerous_keys(self):
        """Test environment sanitization against dangerous keys."""
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
        
        result = CustomOptions.sanitize(dangerous_options)
        
        # Only safe environment variables should remain
        assert result.environment == {
            "SAFE_VAR": "safe_value",
            "APP_CONFIG": "config_value",
        }

    def test_sanitize_environment_safe_keys(self):
        """Test environment sanitization with safe keys."""
        safe_options = CustomOptions(
            environment={
                "APP_NAME": "myapp",
                "DEBUG": "true",
                "PORT": "8080",
                "DATABASE_URL": "postgresql://localhost/db",
            }
        )
        
        result = CustomOptions.sanitize(safe_options)
        
        # All safe environment variables should be preserved
        assert result.environment == {
            "APP_NAME": "myapp",
            "DEBUG": "true", 
            "PORT": "8080",
            "DATABASE_URL": "postgresql://localhost/db",
        }

    def test_sanitize_entrypoint_malicious(self):
        """Test entrypoint sanitization against malicious input."""
        malicious_options = CustomOptions(
            entrypoint="--privileged --mount type=bind,source=/,target=/host"
        )
        
        result = CustomOptions.sanitize(malicious_options)
        
        # Only first part is kept (flags stripped by --entrypoint flag)
        assert result.entrypoint == "--privileged"

    def test_sanitize_entrypoint_safe_paths(self):
        """Test entrypoint sanitization with safe paths."""
        test_cases = [
            "/usr/bin/python3",  # Absolute path
            "/bin/bash", 
            "/app/main.py",
            "/usr/local/bin/myapp",
            "/abc/01.py",  # Your example
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
        ]
        
        for entrypoint in test_cases:
            options = CustomOptions(entrypoint=entrypoint)
            result = CustomOptions.sanitize(options)
            assert result.entrypoint == entrypoint

    def test_sanitize_entrypoint_invalid(self):
        """Test entrypoint sanitization with invalid entries."""
        invalid_cases = [
            "|rm -rf /",  # Command injection (starts with |)
            "$(whoami)",  # Command substitution (contains $)
            "rm; cat",  # Contains semicolon
            "test;",  # Contains semicolon
            "",  # Empty
            "   ",  # Whitespace only
        ]
        
        for entrypoint in invalid_cases:
            options = CustomOptions(entrypoint=entrypoint)
            result = CustomOptions.sanitize(options)
            assert result.entrypoint is None

    def test_sanitize_shm_size_malicious(self):
        """Test shm_size sanitization against malicious input."""
        malicious_options = CustomOptions(
            shm_size="1g --privileged --mount type=bind"
        )
        
        result = CustomOptions.sanitize(malicious_options)
        
        # Should only keep the valid part (1g) and filter out the malicious flags
        assert result.shm_size == "1g"

    def test_sanitize_shm_size_valid(self):
        """Test shm_size sanitization with valid values."""
        valid_cases = [
            "1g",
            "512m", 
            "1024",
            "2G",
            "256M",
        ]
        
        for shm_size in valid_cases:
            options = CustomOptions(shm_size=shm_size)
            result = CustomOptions.sanitize(options)
            assert result.shm_size == shm_size

    def test_sanitize_shm_size_invalid(self):
        """Test shm_size sanitization with invalid values."""
        invalid_cases = [
            "invalid_size",
            "1x",  # Invalid unit
            "",  # Empty
            "   ",  # Whitespace only
        ]
        
        for shm_size in invalid_cases:
            options = CustomOptions(shm_size=shm_size)
            result = CustomOptions.sanitize(options)
            assert result.shm_size is None
        
        # Test that valid part is extracted from mixed input
        mixed_options = CustomOptions(shm_size="1g --privileged")
        result = CustomOptions.sanitize(mixed_options)
        assert result.shm_size == "1g"  # Should extract valid part

    def test_sanitize_preserves_safe_fields(self):
        """Test that safe fields are preserved during sanitization."""
        original_options = CustomOptions(
            internal_ports=[8080, 9090],
            initial_port_count=5,
            startup_commands="echo 'Starting app'",
            volumes=["/safe/path:/safe/container"],
            environment={"SAFE_VAR": "safe_value"},
            entrypoint="/usr/bin/python3",
            shm_size="1g"
        )
        
        result = CustomOptions.sanitize(original_options)
        
        # Safe fields should be preserved
        assert result.internal_ports == [8080, 9090]
        assert result.initial_port_count == 5
        assert result.startup_commands == "echo 'Starting app'"
        assert result.volumes == ["/safe/path:/safe/container"]
        assert result.environment == {"SAFE_VAR": "safe_value"}
        assert result.entrypoint == "/usr/bin/python3"
        assert result.shm_size == "1g"

    def test_sanitize_comprehensive_attack(self):
        """Test sanitization against a comprehensive attack scenario."""
        # Simulate a real attack attempt
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
            shm_size="1g --privileged"
        )
        
        result = CustomOptions.sanitize(attack_options)
        
        # All malicious content should be filtered out
        assert result.volumes is None  # All volumes were dangerous (empty list becomes None)
        assert result.environment == {"SAFE_VAR": "safe_value"}  # Only safe env var
        assert result.entrypoint == "bash"  # Only first part allowed (flags stripped)
        assert result.shm_size == "1g"  # Valid part extracted from malicious input

    def test_sanitize_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        options = CustomOptions(
            volumes=["  /safe/path:/safe/container  ", "   ", ""],
            environment={"  SAFE_VAR  ": "  safe_value  ", "": "empty"},
            entrypoint="  /usr/bin/python3  ",
            shm_size="  1g  "
        )
        
        result = CustomOptions.sanitize(options)
        
        # Whitespace should be trimmed, empty values filtered
        assert result.volumes == ["/safe/path:/safe/container"]
        assert result.environment == {"SAFE_VAR": "safe_value"}
        assert result.entrypoint == "/usr/bin/python3"
        assert result.shm_size == "1g"
