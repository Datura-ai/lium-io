from daos.base import BaseDao

from models.validator import Validator


class ValidatorDao(BaseDao):
    def save(self, validator: Validator) -> Validator:
        self.session.add(validator)
        self.session.commit()
        self.session.refresh(validator)
        return validator
        
    def get_validator_by_hotkey(self, hotkey: str):
        return self.session.query(Validator).filter_by(validator_hotkey=hotkey).first()
