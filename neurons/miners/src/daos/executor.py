from typing import Optional
from daos.base import BaseDao
from models.executor import Executor


class ExecutorDao(BaseDao):
    def save(self, executor: Executor) -> Executor:
        self.session.add(executor)
        self.session.commit()
        self.session.refresh(executor)
        return executor

    def findOne(self, address: str, port: int) -> Executor:
        executor = self.session.query(Executor).filter_by(
            address=address, port=port).first()
        if not executor:
            raise Exception('Not found executor')

        return executor

    def update(self, address: str, port: int, payload: dict) -> Executor:
        """
        Update executor fields by address and port.
        :param address: Executor IP address
        :param port: Executor port
        :param payload: Dictionary of fields to update (e.g., {'validator': 'new_validator', 'price_per_hour': 0.5})
        :return: Updated Executor object
        """
        existing_executor = self.findOne(address, port)

        # Update only the fields provided in payload
        for field, value in payload.items():
            if hasattr(existing_executor, field) and field != 'uuid':
                setattr(existing_executor, field, value)

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
        existing_executor.price_per_hour = executor.price_per_hour
        self.session.commit()
        self.session.refresh(existing_executor)
        return existing_executor
