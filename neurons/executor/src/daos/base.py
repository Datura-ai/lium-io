from typing import Generic, TypeVar
from uuid import UUID

from sqlmodel import select, Session

from core.logger import get_logger

T = TypeVar("T")
logger = get_logger(name="app")


class BaseDao(Generic[T]):
    model: type[T] = None
    primary_key_field = "id"

    def get_primary_key_value(self, instance):
        return getattr(instance, self.primary_key_field)

    def find_by_id(self, session: Session, id: UUID | str) -> T | None:
        if self.model is None:
            raise NotImplementedError

        return session.exec(
            select(self.model).where(getattr(self.model, self.primary_key_field) == id)
        ).first()

    def save(self, session: Session, instance) -> T:
        try:
            session.add(instance)
            session.commit()
            session.refresh(instance)
            return instance
        except Exception as e:
            self.safe_rollback(session)
            logger.error("Error saving instance: %s", e, exc_info=True)
            raise

    def safe_rollback(self, session: Session):
        try:
            if session.in_transaction():
                session.rollback()
        except Exception as e:
            logger.error("Error rolling back transaction: %s", e, exc_info=True)

    def delete(self, session: Session, instance):
        try:
            session.delete(instance)
            session.commit()
        except Exception as e:
            self.safe_rollback(session)
            logger.error("Error deleting instance: %s", e, exc_info=True)
            raise

    def update(self, session: Session, id: UUID | str, payload: dict) -> T | None:
        """
        Update an instance by ID with the provided payload.

        :param id (UUID | str): The ID of the instance to update.
        :param payload (dict): The payload to update the instance with.
        """
        instance = self.find_by_id(session, id)
        if instance:
            try:
                for key, value in payload.items():
                    setattr(instance, key, value)
                session.commit()
                session.refresh(instance)
                return instance
            except Exception as e:
                self.safe_rollback(session)
                logger.error("Error deleting instance: %s", e, exc_info=True)
                raise
        return None
