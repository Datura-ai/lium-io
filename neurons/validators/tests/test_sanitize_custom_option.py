from payload_models.payloads import CustomOptions


class TestCustomOptionsSanitization:
    """Test suite for CustomOptions sanitization to prevent command injection."""

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
