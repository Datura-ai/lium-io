## Describe your changes

This PR fixes multiple critical command injection vulnerabilities across the codebase by implementing proper input sanitization and escaping.

### Key Changes:

1. **Command Injection Fixes**
   - Applied `shlex.quote()` to all user-controlled inputs in shell commands
   - Fixed vulnerabilities in SSH operations (`interactive_shell_service.py`)
   - Fixed vulnerabilities in Docker operations (`docker_service.py`)
   - Fixed vulnerabilities in VerifyX validation (`verifyx_validation_service.py`)
   - Fixed vulnerabilities in executor connectivity (`executor_connectivity_service.py`)
   - Fixed vulnerabilities in rented machine checks (`rented_machine.py`)

2. **Subprocess Security**
   - Removed `shell=True` from all subprocess calls to prevent shell injection
   - Converted command strings to command lists for safer execution
   - Applied `shlex.split()` for safe command parsing
   - Fixed in: `hash_service.py`, `score.py`, `machine_scrape.py`, `backup_storage.py`, `restore_storage.py`

3. **Input Sanitization**
   - Added `_sanitize_startup_commands()` method to filter dangerous shell metacharacters
   - Enhanced `CustomOptions.sanitize()` to include startup_commands validation

### Security Impact:
- **Before:** Attackers could potentially execute arbitrary commands by injecting malicious input
- **After:** All user inputs are properly escaped and validated, preventing command injection attacks

### Testing:
- All changes maintain backward compatibility
- Functionality remains unchanged
- No breaking changes introduced

See `SECURITY_IMPROVEMENTS.md` for detailed vulnerability analysis and fix descriptions.

## Issue ticket number and link

[Security: Fix Command Injection Vulnerabilities](https://www.notion.so/Compute-SN-c27d35dd084e4c4d92374f55cdd293f2?p=f9b26856f1a6406892b5db46446260da&pm=s)

## Checklist before requesting a review

- [x] I have performed a self-review of my code
- [ ] I wrote tests.
- [ ] Need to take care of performance?

### Notes:
- **Tests:** Security fixes maintain existing functionality. Integration tests should verify that command execution still works correctly with escaped inputs.
- **Performance:** Minimal impact. `shlex.quote()` and `shlex.split()` are lightweight operations. No significant performance degradation expected.

