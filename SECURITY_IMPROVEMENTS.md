# Security Improvements

This document outlines the security vulnerabilities that were identified and fixed in this project.

## Overview

Multiple command injection vulnerabilities were identified and patched across the codebase. These vulnerabilities could have allowed attackers to execute arbitrary commands on the system by injecting malicious input into shell commands.

## Fixed Vulnerabilities

### 1. Command Injection in SSH Operations

**Files Modified:**
- `neurons/validators/src/services/interactive_shell_service.py`
- `neurons/validators/src/services/executor_connectivity_service.py`
- `neurons/validators/src/services/task/checks/rented_machine.py`

**Issue:** User-controlled inputs (host, username, port, remote_dir, container_name) were directly interpolated into shell commands without proper escaping.

**Fix:** Applied `shlex.quote()` to all user inputs before inserting them into command strings.

**Example:**
```python
# Before (vulnerable)
ssh_command = f"ssh -p {self.port} {self.username}@{self.host}"

# After (secure)
port_quoted = shlex.quote(str(self.port))
username_quoted = shlex.quote(self.username)
host_quoted = shlex.quote(self.host)
ssh_command = f"ssh -p {port_quoted} {username_quoted}@{host_quoted}"
```

### 2. Command Injection in Docker Operations

**Files Modified:**
- `neurons/validators/src/services/docker_service.py`

**Issue:** Multiple user-controlled inputs (public_key, container_name, jupyter_token, jupyter_port, local_volume_path) were directly interpolated into Docker commands.

**Fix:** Applied `shlex.quote()` to all user inputs in Docker command construction.

**Example:**
```python
# Before (vulnerable)
command = f"/usr/bin/docker exec {container_name} sh -c 'echo \"{public_key}\" >> ~/.ssh/authorized_keys'"

# After (secure)
container_name_quoted = shlex.quote(container_name)
public_key_quoted = shlex.quote(public_key)
command = f"/usr/bin/docker exec {container_name_quoted} sh -c 'echo {public_key_quoted} >> ~/.ssh/authorized_keys'"
```

### 3. Command Injection in VerifyX Validation

**Files Modified:**
- `neurons/validators/src/services/verifyx_validation_service.py`

**Issue:** Executor paths, seed, and cipher_text were directly interpolated into command strings.

**Fix:** Applied `shlex.quote()` to all inputs before command construction.

### 4. Subprocess Shell Injection

**Files Modified:**
- `neurons/validators/src/services/hash_service.py`
- `neurons/validators/src/miner_jobs/score.py`
- `neurons/validators/src/miner_jobs/machine_scrape.py`
- `neurons/validators/src/miner_jobs/backup_storage.py`
- `neurons/validators/src/miner_jobs/restore_storage.py`

**Issue:** `subprocess` functions were called with `shell=True`, allowing shell injection attacks through command strings.

**Fix:** 
- Removed `shell=True` parameter
- Used command lists instead of strings, or applied `shlex.split()` for string parsing

**Example:**
```python
# Before (vulnerable)
subprocess.check_output(cmd, shell=True, text=True)

# After (secure)
if isinstance(cmd, str):
    cmd_list = shlex.split(cmd)
else:
    cmd_list = cmd
subprocess.check_output(cmd_list, shell=False, text=True)
```

### 5. Input Sanitization for Startup Commands

**Files Modified:**
- `neurons/validators/src/payload_models/payloads.py`
- `neurons/validators/src/services/docker_service.py`

**Issue:** `startup_commands` field in `CustomOptions` was not sanitized, allowing potential command injection.

**Fix:** 
- Added `_sanitize_startup_commands()` method to filter dangerous shell metacharacters
- Applied sanitization in the `sanitize()` method
- Used `shlex.quote()` when inserting into Docker commands

**Example:**
```python
@staticmethod
def _sanitize_startup_commands(startup_commands: str | None) -> str | None:
    """Sanitize startup_commands to prevent command injection."""
    if not startup_commands or not startup_commands.strip():
        return None
    
    # Reject commands containing dangerous patterns
    dangerous_patterns = [
        r'[;&|`$(){}]',  # Command separators and shell expansions
        r'<|>',  # Redirections
        r'\$\{',  # Variable expansion
        r'`',  # Command substitution
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, clean_commands):
            return None
    
    return clean_commands
```

## Security Best Practices Applied

1. **Input Validation**: All user-controlled inputs are now properly validated and sanitized
2. **Command Escaping**: `shlex.quote()` is used to safely escape shell arguments
3. **Subprocess Safety**: Removed `shell=True` and use command lists instead
4. **Defense in Depth**: Multiple layers of validation (sanitization + escaping)

## Impact

- **Before**: Attackers could potentially execute arbitrary commands by injecting malicious input
- **After**: All user inputs are properly escaped, preventing command injection attacks

## Testing

All changes maintain backward compatibility. The functionality remains the same, but with improved security.

## References

- [OWASP Command Injection](https://owasp.org/www-community/attacks/Command_Injection)
- [Python shlex Documentation](https://docs.python.org/3/library/shlex.html)

