from typing import Annotated, NamedTuple

from fastapi import Depends

from daos.executor import ExecutorDao
from daos.validator import ValidatorDao
from core.config import settings


class ValidatorRowMigration(NamedTuple):
    """What `migrate_validator_hotkey_rows` found and did: rows keyed to the old hotkey, rows moved."""

    found: int
    updated: int


def migrate_validator_hotkey_rows(
    executor_dao: ExecutorDao, old_hotkey: str, new_hotkey: str, *, dry_run: bool
) -> ValidatorRowMigration:
    """Re-key this miner's executor rows from one validator hotkey to another (the swap-day step for
    `executor.validator`; the central miner reads the portal and has no rows of its own).

    `dry_run` counts and changes nothing. Empty or equal hotkeys are refused before any query.
    """
    old_hotkey, new_hotkey = (old_hotkey or "").strip(), (new_hotkey or "").strip()
    if not old_hotkey or not new_hotkey:
        raise ValueError("both the old and the new validator hotkey are required")
    if old_hotkey == new_hotkey:
        raise ValueError(f"the old and the new validator hotkey are the same ({old_hotkey}); nothing to migrate")

    found = executor_dao.count_executors_for_validator(old_hotkey)
    if dry_run or not found:
        return ValidatorRowMigration(found=found, updated=0)

    moved = executor_dao.move_executors_to_validator(old_hotkey, new_hotkey)
    return ValidatorRowMigration(found=found, updated=moved)


class ValidatorService:
    def __init__(self, validator_dao: Annotated[ValidatorDao, Depends(ValidatorDao)]):
        self.validator_dao = validator_dao

    def is_valid_validator(self, validator_hotkey: str) -> bool:
        if settings.debug.SKIP_VALIDATOR_REGISTRATION_CHECK:
            return True

        return validator_hotkey in settings.accepted_validator_hotkeys
