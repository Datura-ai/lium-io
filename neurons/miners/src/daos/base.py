from core.db import SessionDep


class BaseDao:
    def __init__(self, session: SessionDep):
        self.session = session
