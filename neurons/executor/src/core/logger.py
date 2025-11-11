import logging
import json


def get_logger(name: str):
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "Name: %(name)s | Time: %(asctime)s | Level: %(levelname)s | File: %(filename)s | Function: %(funcName)s | Line: %(lineno)s | Process: %(process)d | Message: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class StructuredMessage:
    def __init__(self, message, extra: dict):
        self.message = message
        self.extra = extra

    def __str__(self):
        return "%s >>> %s" % (self.message, json.dumps(self.extra))  # noqa


_m = StructuredMessage
