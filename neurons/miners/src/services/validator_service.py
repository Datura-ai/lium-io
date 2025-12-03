from typing import Annotated

from fastapi import Depends

from daos.validator import ValidatorDao
from core.config import settings


class ValidatorService:
    def __init__(self, validator_dao: Annotated[ValidatorDao, Depends(ValidatorDao)]):
        self.validator_dao = validator_dao

    def is_valid_validator(self, validator_hotkey: str) -> bool:
        if settings.debug.SKIP_VALIDATOR_REGISTRATION_CHECK:
            return True
        
        return settings.DEFAULT_VALIDATOR_HOTKEY == validator_hotkey
