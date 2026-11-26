from unittest.mock import AsyncMock, Mock


def create_process_answering_through(run):
    """`create_process` for an SSH double that answers through `run`: a bounded host command opens its channel with
    create_process and waits on it, as asyncssh's run() does inside."""

    async def create_process(command, *args, **kwargs):
        result = await run(command)
        return Mock(wait=AsyncMock(return_value=result), wait_closed=AsyncMock(), close=Mock())

    return create_process
