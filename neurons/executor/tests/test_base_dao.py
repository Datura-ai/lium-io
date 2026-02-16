"""
Tests for BaseDao - generic database access object.

Covers:
- find_by_id with model not set
- save, update, delete error handling
- safe_rollback behavior
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from daos.base import BaseDao


class FakeModel:
    """Fake model for testing BaseDao."""

    def __init__(self, id=None, name=None):
        self.id = id or uuid4()
        self.name = name


class TestBaseDaoFindById(unittest.TestCase):
    """Tests for BaseDao.find_by_id."""

    def test_find_by_id_no_model_raises(self):
        """Should raise NotImplementedError when model is None."""
        dao = BaseDao()
        session = MagicMock()
        with self.assertRaises(NotImplementedError):
            dao.find_by_id(session, uuid4())

    def test_find_by_id_with_model(self):
        """Should execute query when model is set."""
        dao = BaseDao()
        dao.model = FakeModel
        session = MagicMock()
        mock_result = FakeModel(name="found")
        session.exec.return_value.first.return_value = mock_result

        result = dao.find_by_id(session, uuid4())
        session.exec.assert_called_once()
        self.assertEqual(result.name, "found")


class TestBaseDaoSave(unittest.TestCase):
    """Tests for BaseDao.save."""

    def test_save_success(self):
        """Successful save should add, commit, refresh, and return instance."""
        dao = BaseDao()
        session = MagicMock()
        instance = FakeModel(name="new")

        result = dao.save(session, instance)

        session.add.assert_called_once_with(instance)
        session.commit.assert_called_once()
        session.refresh.assert_called_once_with(instance)
        self.assertEqual(result, instance)

    def test_save_error_rollback(self):
        """Save failure should rollback and re-raise."""
        dao = BaseDao()
        session = MagicMock()
        session.commit.side_effect = RuntimeError("DB error")
        session.in_transaction.return_value = True

        with self.assertRaises(RuntimeError):
            dao.save(session, FakeModel())

        session.rollback.assert_called_once()


class TestBaseDaoDelete(unittest.TestCase):
    """Tests for BaseDao.delete."""

    def test_delete_success(self):
        """Successful delete should call session.delete and commit."""
        dao = BaseDao()
        session = MagicMock()
        instance = FakeModel()

        dao.delete(session, instance)

        session.delete.assert_called_once_with(instance)
        session.commit.assert_called_once()

    def test_delete_error_rollback(self):
        """Delete failure should rollback and re-raise."""
        dao = BaseDao()
        session = MagicMock()
        session.commit.side_effect = RuntimeError("DB error")
        session.in_transaction.return_value = True

        with self.assertRaises(RuntimeError):
            dao.delete(session, FakeModel())

        session.rollback.assert_called_once()


class TestBaseDaoUpdate(unittest.TestCase):
    """Tests for BaseDao.update."""

    def test_update_existing_instance(self):
        """Should update attributes and commit."""
        dao = BaseDao()
        dao.model = FakeModel
        session = MagicMock()
        existing = FakeModel(name="old")
        session.exec.return_value.first.return_value = existing

        result = dao.update(session, existing.id, {"name": "new"})

        self.assertEqual(result.name, "new")
        session.commit.assert_called_once()

    def test_update_nonexistent_returns_none(self):
        """Should return None when instance not found."""
        dao = BaseDao()
        dao.model = FakeModel
        session = MagicMock()
        session.exec.return_value.first.return_value = None

        result = dao.update(session, uuid4(), {"name": "new"})

        self.assertIsNone(result)
        session.commit.assert_not_called()

    def test_update_error_rollback(self):
        """Update failure should rollback and re-raise."""
        dao = BaseDao()
        dao.model = FakeModel
        session = MagicMock()
        existing = FakeModel(name="old")
        session.exec.return_value.first.return_value = existing
        session.commit.side_effect = RuntimeError("DB error")
        session.in_transaction.return_value = True

        with self.assertRaises(RuntimeError):
            dao.update(session, existing.id, {"name": "fail"})

        session.rollback.assert_called_once()


class TestBaseDaoSafeRollback(unittest.TestCase):
    """Tests for BaseDao.safe_rollback."""

    def test_rollback_when_in_transaction(self):
        """Should call rollback when session is in a transaction."""
        dao = BaseDao()
        session = MagicMock()
        session.in_transaction.return_value = True

        dao.safe_rollback(session)

        session.rollback.assert_called_once()

    def test_no_rollback_when_not_in_transaction(self):
        """Should not call rollback when no active transaction."""
        dao = BaseDao()
        session = MagicMock()
        session.in_transaction.return_value = False

        dao.safe_rollback(session)

        session.rollback.assert_not_called()

    def test_rollback_exception_handled(self):
        """Rollback errors should be caught and not propagated."""
        dao = BaseDao()
        session = MagicMock()
        session.in_transaction.return_value = True
        session.rollback.side_effect = RuntimeError("rollback failed")

        # Should not raise
        dao.safe_rollback(session)


if __name__ == "__main__":
    unittest.main(verbosity=2)
