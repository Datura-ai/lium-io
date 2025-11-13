import pytest

from tests.helpers import make_context


@pytest.fixture
def context_factory():
    def _factory(**overrides):
        return make_context(**overrides)

    return _factory
