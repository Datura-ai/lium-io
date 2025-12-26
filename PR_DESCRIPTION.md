## Describe your changes

This PR addresses review feedback on the command injection security fixes by correcting double-quoting issues and improving code organization.

### Key Fixes:

1. **Fixed Double-Quoting in Subprocess Calls**
   - Removed `shlex.quote()` from list-based subprocess calls in `backup_storage.py` and `restore_storage.py`
   - List form is already safe; adding quotes inserts literal quote characters causing command failures

2. **Fixed Nested Quoting Issues**
   - Corrected double-quoting in `docker_service.py` where values were quoted multiple times
   - Properly escape shell commands without nested quotes

3. **Code Organization Improvements**
   - Moved `import shlex` from function bodies to file top (per project conventions)
   - Affected files: `machine_scrape.py`, `verifyx_validation_service.py`, `rented_machine.py`, `executor_connectivity_service.py`

4. **Added Centralized Helper**
   - Created `command_utils.py` with `prepare_shell_command()` function
   - Added tests in `test_command_utils.py` to prevent future mistakes
   - Provides a single, testable place for command construction logic

### Technical Details:

**Before (broken):**
```python
source_volume_quoted = shlex.quote(args.source_volume)
command = ["docker", "run", "-v", f"{source_volume_quoted}:{path_quoted}"]
# Docker receives: -v '/data/vol':'/mnt' — fails
```

**After (correct):**
```python
command = ["docker", "run", "-v", f"{args.source_volume}:{args.source_volume_path}"]
# List form is safe, no quoting needed
```

## Checklist before requesting a review

- [x] I have performed a self-review of my code
- [x] I wrote tests.
- [x] Need to take care of performance?

### Notes:
- **Tests:** Added tests for `command_utils.py` to ensure proper escaping behavior
- **Performance:** No performance impact; fixes prevent command failures

