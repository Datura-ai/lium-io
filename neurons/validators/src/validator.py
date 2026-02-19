import asyncio

from core.utils import configure_logs_of_other_modules, wait_for_services_sync
from core.validator import Validator

configure_logs_of_other_modules()
wait_for_services_sync()


async def run():
    """Run validator once in DRY_RUN mode without FastAPI server."""
    validator = Validator()
    await validator.start()
    await validator.stop()


if __name__ == "__main__":
    asyncio.run(run())
