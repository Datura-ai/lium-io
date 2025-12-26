"""Tests for command_utils module."""
import pytest
from services.command_utils import prepare_shell_command


def test_prepare_shell_command_basic():
    """Test basic command preparation."""
    result = prepare_shell_command(
        "/usr/bin/docker exec {container} sh -c 'apt-get update'",
        container="my-container"
    )
    assert result == "/usr/bin/docker exec 'my-container' sh -c 'apt-get update'"


def test_prepare_shell_command_with_special_chars():
    """Test command preparation with special characters."""
    result = prepare_shell_command(
        "echo {message}",
        message="hello; rm -rf /"
    )
    # Should escape the semicolon and other special chars
    assert "rm -rf" not in result
    assert "'" in result  # Should be quoted


def test_prepare_shell_command_multiple_args():
    """Test command preparation with multiple arguments."""
    result = prepare_shell_command(
        "docker run -v {source}:{dest} {image}",
        source="/data/volume",
        dest="/mnt",
        image="ubuntu:latest"
    )
    assert "/data/volume" in result
    assert "/mnt" in result
    assert "ubuntu:latest" in result
    # All should be properly quoted
    assert result.count("'") >= 3


def test_prepare_shell_command_empty_values():
    """Test command preparation with empty values."""
    result = prepare_shell_command(
        "echo {value}",
        value=""
    )
    assert "''" in result or '""' in result


def test_prepare_shell_command_numeric_values():
    """Test command preparation with numeric values."""
    result = prepare_shell_command(
        "docker run -p {port}:8080 {image}",
        port=8080,
        image="nginx"
    )
    assert "8080" in result
    assert "nginx" in result

