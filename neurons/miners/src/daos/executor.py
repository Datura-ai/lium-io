from typing import Optional
from daos.base import BaseDao
from models.executor import Executor


class ExecutorDao(BaseDao):
    def save(self, executor: Executor) -> Executor:
        self.session.add(executor)
        self.session.commit()
        self.session.refresh(executor)
        return executor

    def findOne(self, address: str, port: int):
        executor = self.session.query(Executor).filter_by(
            address=address, port=port).first()
        if not executor:
            raise Exception('Not found executor')

        return executor

    def update(self, executor: Executor) -> Executor:
        existing_executor = self.findOne(executor.address, executor.port)

        existing_executor.address = executor.address
        existing_executor.port = executor.port
        existing_executor.validator = executor.validator

        self.session.commit()
        self.session.refresh(existing_executor)
        return existing_executor

    def delete_by_address_port(self, address: str, port: int) -> None:
        executor = self.findOne(address, port)

        self.session.delete(executor)
        self.session.commit()

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

    def get_all_executors(self) -> list[Executor]:
        return list(self.session.query(Executor).all())

    def find_by_uuid(self, uuid: str) -> Executor:
        return self.session.query(Executor).filter_by(uuid=uuid).first()

    def update_by_uuid(self, uuid: str, executor: Executor) -> Executor:
        existing_executor = self.find_by_uuid(uuid)
        existing_executor.validator = executor.validator
        existing_executor.address = executor.address
        existing_executor.port = executor.port
        self.session.commit()
        self.session.refresh(existing_executor)
        return existing_executor
