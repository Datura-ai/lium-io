from typing import Optional
from daos.base import BaseDao
from models.executor import Executor


class ExecutorDao(BaseDao):
    def save(self, executor: Executor) -> Executor:
        self.session.add(executor)
        self.session.commit()
        self.session.refresh(executor)
        return executor

    def get_executors_for_validator(self, validator_key: str, executor_id: Optional[str] = None) -> list[Executor]:
        """Get executors that opened to valdiator

        Args:
            validator_key (str): validator hotkey string

        Return:
            List[Executor]: list of Executors
        """
        if executor_id:
            return list(self.session.query(Executor).filter_by(validator=validator_key, uuid=executor_id))

        return list(self.session.query(Executor).filter_by(validator=validator_key))
