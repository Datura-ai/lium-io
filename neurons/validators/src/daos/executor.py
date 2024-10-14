import logging
from daos.base import BaseDao
from models.executor import Executor

logger = logging.getLogger(__name__)

class ExecutorDao(BaseDao):
    def upsert(self, executor: Executor) -> Executor:
        try:
            existing_executor = self.get_executor(
                miner_hotkey=executor.miner_hotkey, executor_id=executor.executor_id
            )

            if existing_executor:
                # Update the fields of the existing executor
                existing_executor.miner_address = executor.miner_address
                existing_executor.miner_port = executor.miner_port
                existing_executor.executor_ip_address = executor.executor_ip_address
                existing_executor.executor_ssh_username = executor.executor_ssh_username
                existing_executor.executor_ssh_port = executor.executor_ssh_port
                
                self.session.commit()
                self.session.refresh(existing_executor)
                return existing_executor
            else:
                # Insert the new executor
                self.session.add(executor)
                self.session.commit()
                self.session.refresh(executor)

                return executor
        except Exception as e:
            self.session.rollback()
            print(e)
            logger.error("Error upsert executor: %s", e)
            raise

    def rent(self, executor_id: str, miner_hotkey: str) -> Executor:
        try:
            executor = self.get_executor(executor_id, miner_hotkey)
            if executor:
                executor.rented = True
                self.session.commit()
                self.session.refresh(executor)

            return executor
        except Exception as e:
            self.session.rollback()
            print(e)
            logger.error("Error rent executor: %s", e)
            raise

    def unrent(self, executor_id: str, miner_hotkey: str) -> Executor:
        try:
            executor = self.get_executor(executor_id, miner_hotkey)
            if executor:
                executor.rented = False
                self.session.commit()
                self.session.refresh(executor)

            return executor
        except Exception as e:
            self.session.rollback()
            print(e)
            logger.error("Error unrent executor: %s", e)
            raise

    def get_executor(self, executor_id: str, miner_hotkey: str) -> Executor:
        try:
            return self.session.query(Executor).filter_by(miner_hotkey=miner_hotkey, executor_id=executor_id).first()
        except Exception as e:
            self.session.rollback()
            print(e)
            logger.error("Error get executor: %s", e)
            raise
