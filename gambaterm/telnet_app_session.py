"""
Provide an async context manager to create a prompt-toolkit app session
from a telnetlib3 writer.
"""
from __future__ import annotations

import os
import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, AsyncIterator, Iterator

from prompt_toolkit.data_structures import Size
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.application.current import create_app_session, AppSession

if TYPE_CHECKING:
    from telnetlib3.stream_writer import TelnetWriter


@asynccontextmanager
async def vt100_output_from_telnet(
    writer: TelnetWriter,
) -> AsyncIterator[Vt100_Output]:
    def get_size() -> Size:
        cols = writer.get_extra_info("cols") or 80
        rows = writer.get_extra_info("rows") or 24
        return Size(rows=rows, columns=cols)

    term = writer.get_extra_info("TERM") or "unknown"
    read_fd, write_fd = os.pipe()

    async def forward_output() -> None:
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        read_file = os.fdopen(read_fd, "rb")
        transport, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader),
            read_file,
        )
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError, EOFError):
            pass
        finally:
            transport.close()

    task = asyncio.create_task(forward_output())
    with open(write_fd, "w", newline="\r\n") as stdout:
        vt100_output = Vt100_Output(stdout, get_size, term=term)
        try:
            yield vt100_output
        finally:
            pass
    # stdout is closed, forward task will see EOF
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()


NAWS_DEBOUNCE_SECONDS = 0.1


@contextmanager
def bind_resize_telnet_to_app_session(
    writer: TelnetWriter, app_session: AppSession,
) -> Iterator[None]:
    protocol = writer.protocol
    original_on_naws = protocol.on_naws
    pending: list[asyncio.TimerHandle] = []

    def on_naws(rows: int, cols: int) -> None:
        original_on_naws(rows, cols)
        if pending:
            pending.pop().cancel()
        if app_session.app is not None:
            loop = asyncio.get_event_loop()
            pending.append(
                loop.call_later(NAWS_DEBOUNCE_SECONDS, app_session.app._on_resize)
            )

    try:
        protocol.on_naws = on_naws  # type: ignore[method-assign]
        yield
    finally:
        if pending:
            pending.pop().cancel()
        protocol.on_naws = original_on_naws  # type: ignore[method-assign]


@asynccontextmanager
async def telnet_to_app_session(
    writer: TelnetWriter,
) -> AsyncIterator[AppSession]:
    with create_pipe_input() as vt100_input:
        async with vt100_output_from_telnet(writer) as vt100_output:
            with create_app_session(
                input=vt100_input, output=vt100_output
            ) as app_session:
                with bind_resize_telnet_to_app_session(writer, app_session):
                    yield app_session
