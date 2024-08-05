from daos.base import BaseDao

from models.validator import Validator


class ValidatorDao(BaseDao):
    def get_validator_by_hotkey(self, hotkey: str):
        return self.session.query(Validator).filter_by(validator_hotkey=hotkey).first()
