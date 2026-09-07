def dict_diff(old: dict, new: dict, prefix: str = "") -> list[str]:
    """Return list of human-readable strings describing differences."""
    changes: list[str] = []
    all_keys = old.keys() | new.keys()
    for key in sorted(all_keys):
        path = f"{prefix}.{key}" if prefix else key
        old_val = old.get(key)
        new_val = new.get(key)
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            changes.extend(dict_diff(old_val, new_val, path))
        elif old_val != new_val:
            changes.append(f"[{path}]: {old_val} -> {new_val}")
    return changes
