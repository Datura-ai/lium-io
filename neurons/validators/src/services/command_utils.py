"""
Utility functions for safely constructing shell commands to prevent command injection.

This module provides helpers for building shell commands with proper escaping.
For subprocess calls with shell=False (list form), no escaping is needed.
For SSH commands or shell=True, use prepare_shell_command().
"""
import shlex
from typing import Any


def prepare_shell_command(template: str, **kwargs: Any) -> str:
    """
    Safely construct a shell command by escaping all user inputs.
    
    This function is for SSH commands or subprocess calls with shell=True.
    For subprocess calls with shell=False (list form), no escaping is needed.
    
    Args:
        template: Command template with {placeholder} syntax
        **kwargs: Values to substitute into placeholders (will be escaped)
    
    Returns:
        Safely escaped command string
    
    Example:
        >>> command = prepare_shell_command(
        ...     "/usr/bin/docker exec {container} sh -c 'apt-get update'",
        ...     container=container_name
        ... )
    """
    escaped = {k: shlex.quote(str(v)) for k, v in kwargs.items()}
    return template.format(**escaped)

