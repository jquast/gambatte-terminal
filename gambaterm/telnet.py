from __future__ import annotations

import time
import hashlib
import asyncio
import argparse
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from concurrent.futures import ThreadPoolExecutor

from prompt_toolkit.application import AppSession

from .run import run
from .colors import ColorMode, detect_color_mode
from .file_input import console_input_from_file_context
from .main import add_base_arguments, add_optional_arguments
from .console import Console, InputGetter, GameboyColor
from .telnet_input import TelnetInputState, read_telnet_input

from .telnet_app_session import telnet_to_app_session


@contextmanager
def no_input_context(console: Console) -> Iterator[InputGetter]:
    """Provide a no-op input getter that returns no button presses."""
    yield lambda: set()


@contextmanager
def telnet_input_context(
    console: Console, state: TelnetInputState
) -> Iterator[InputGetter]:
    """Provide input from telnet keyboard state."""
    def get_input() -> set[Console.Input]:
        for event in state.pop_events():
            console.handle_event(event)
        return state.get_input()
    yield get_input


def _save_dir_name(username: str | None) -> str:
    """Hash the username into a safe directory name.

    :param username: telnet-negotiated username, or None if not available
    :returns: hex digest suitable for use as a directory name
    """
    if username is None:
        return "_anonymous"
    return hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]


def thread_target(
    app_session: AppSession,
    app_config: argparse.Namespace,
    username: str | None,
    color_mode: ColorMode,
    input_state: TelnetInputState | None = None,
) -> int:
    # Create save directory for user
    if app_config.input_file is None:
        save_directory = Path("telnet_save") / _save_dir_name(username)
        save_directory.mkdir(parents=True, exist_ok=True)
        app_config.save_directory = save_directory
    else:
        app_config.save_directory = None

    console: Console = app_config.console_cls(app_config)
    if app_config.input_file is not None:
        console_input_context = console_input_from_file_context(
            console, app_config.input_file, app_config.skip_inputs
        )
    elif input_state is not None:
        console_input_context = telnet_input_context(console, input_state)
    else:
        console_input_context = no_input_context(console)

    with console_input_context as get_console_input:
        try:
            # Prepare alternate screen
            app_session.output.enter_alternate_screen()
            app_session.output.erase_screen()
            app_session.output.hide_cursor()
            app_session.output.flush()

            # Run the emulator
            run(
                console,
                get_input=get_console_input,
                app_session=app_session,
                frame_advance=app_config.frame_advance,
                color_mode=color_mode,
                break_after=app_config.break_after,
                speed_factor=app_config.speed_factor,
                use_cpr_sync=True,
                sextant=app_config.sextant,
                cycle_color_on_ctrl_c=True,
            )
        except (KeyboardInterrupt, OSError):
            return 0
        else:
            return 0
        finally:
            # Wait for CPR
            time.sleep(0.1)
            # Clear alternate screen
            app_session.input.read_keys()
            app_session.output.erase_screen()
            app_session.output.quit_alternate_screen()
            app_session.output.show_cursor()
            # Flush if the connection is still active
            try:
                app_session.output.flush()
            except BrokenPipeError:
                pass


def make_telnet_shell(
    app_config: argparse.Namespace, executor: ThreadPoolExecutor
) -> object:
    """Create a telnet shell callback with app_config and executor bound."""

    async def telnet_shell(reader: object, writer: object) -> None:
        try:
            await _telnet_shell(reader, writer, app_config, executor)
        except KeyboardInterrupt:
            pass
        except SystemExit:
            pass
        except BaseException:
            traceback.print_exc()
        # Close the connection
        if not writer.is_closing():  # type: ignore[union-attr]
            writer.close()  # type: ignore[union-attr]

    return telnet_shell


def _get_protocol(writer: object) -> Any:
    """Get the telnetlib3 protocol from a writer, if accessible."""
    return getattr(writer, '_protocol', None)


async def _detect_true_color_telnet(
    reader: object, writer: object, timeout: float = 0.5
) -> bool:
    """Probe the telnet client for 24-bit color via DECRQSS.

    Must be called before the input reading task starts (exclusive reader access).

    :param reader: telnetlib3 reader
    :param writer: telnetlib3 writer
    :param timeout: seconds to wait for terminal response
    :returns: True if the terminal reports true color support
    """
    # Set an unlikely RGB value, query it with DECRQSS, then reset
    writer.write(b"\033[48:2:1:2:3m\033P$qm\033\\\033[m")  # type: ignore[union-attr]
    await writer.drain()  # type: ignore[union-attr]

    buf = b""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            chunk = await asyncio.wait_for(
                reader.read(256),  # type: ignore[union-attr]
                timeout=remaining,
            )
            if not chunk:
                break
            buf += chunk if isinstance(chunk, bytes) else chunk.encode("latin-1")
            if b"\033\\" in buf or b"\x9c" in buf:
                break
        except asyncio.TimeoutError:
            break

    text = buf.decode("latin-1", errors="replace")
    return "P1$r" in text and "48:2" in text and "1:2:3m" in text


def _fmt_idle(seconds: float) -> str:
    """Format idle duration as 'Xm' or 'X.Xs'."""
    if seconds >= 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds:.1f}s"


async def _log_connection_stats(
    writer: object,
    peer_host: str,
    peer_port: int,
    interval: float = 30.0,
    idle_timeout: float = 300.0,
) -> None:
    """Periodically log tx average Mbit/s and idle time for a connection.

    Closes the connection if the client is idle for longer than *idle_timeout* seconds.
    """
    protocol = _get_protocol(writer)
    if protocol is None:
        return

    start_time = time.monotonic()
    prev_time = start_time
    prev_tx: int = getattr(protocol, 'tx_bytes', 0)
    prev_rx: int = getattr(protocol, 'rx_bytes', 0)
    last_active_time = start_time

    try:
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            elapsed = now - start_time
            dt = now - prev_time

            rx: int = getattr(protocol, 'rx_bytes', 0)
            tx: int = getattr(protocol, 'tx_bytes', 0)

            if rx != prev_rx:
                last_active_time = now

            idle_duration = now - last_active_time

            # Average Mbit/s over the interval and total lifetime
            tx_mbps = (tx - prev_tx) * 8 / dt / 1_000_000 if dt > 0 else 0.0
            avg_tx_mbps = tx * 8 / elapsed / 1_000_000 if elapsed > 0 else 0.0

            minutes, secs = divmod(int(elapsed), 60)
            hours, minutes = divmod(minutes, 60)
            uptime = f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"

            idle_str = f" (idle {_fmt_idle(idle_duration)})" if idle_duration >= 1.0 else ""

            print(
                f"[Stats {peer_host}:{peer_port}] "
                f"up {uptime}, "
                f"tx {tx:,}B ({tx_mbps:.3f}/{avg_tx_mbps:.3f} Mbit/s)"
                f"{idle_str}"
            )

            if idle_duration >= idle_timeout:
                print(
                    f"[Stats {peer_host}:{peer_port}] "
                    f"kicking idle client after {_fmt_idle(idle_duration)}"
                )
                try:
                    await writer.drain()  # type: ignore[union-attr]
                    writer.write(  # type: ignore[union-attr]
                        b"\r\n\r\nConnection closed for idle client\r\n"
                    )
                    await writer.drain()  # type: ignore[union-attr]
                except (ConnectionError, OSError):
                    pass
                writer.close()  # type: ignore[union-attr]
                return

            prev_time = now
            prev_tx = tx
            prev_rx = rx
    except asyncio.CancelledError:
        # Log final stats on disconnect
        now = time.monotonic()
        elapsed = now - start_time
        tx = getattr(protocol, 'tx_bytes', 0)
        avg_tx_mbps = tx * 8 / elapsed / 1_000_000 if elapsed > 0 else 0.0
        print(
            f"[Stats {peer_host}:{peer_port}] "
            f"disconnected after {elapsed:.1f}s, "
            f"total tx {tx:,}B (avg {avg_tx_mbps:.3f} Mbit/s)"
        )


async def _telnet_shell(
    reader: object,
    writer: object,
    app_config: argparse.Namespace,
    executor: ThreadPoolExecutor,
) -> int:
    peername = writer.get_extra_info("peername")  # type: ignore[union-attr]
    peer_host = peername[0] if peername else "unknown"
    peer_port = peername[1] if peername else 0
    terminal_type = (
        writer.get_extra_info("TERM")  # type: ignore[union-attr]
        or "unknown"
    )
    username = (
        writer.get_extra_info("USER")  # type: ignore[union-attr]
        or None
    )
    print(
        f"> Telnet client connected ({peer_host}:{peer_port})"
        + (f" user={username}" if username else "")
    )

    # Check terminal
    if terminal_type == "unknown":
        print(
            "Warning: terminal type not negotiated, assuming xterm-256color."
        )
        terminal_type = "xterm-256color"

    # Determine color mode
    if app_config.color_mode is not None:
        color_mode = app_config.color_mode
    else:
        environment = {}
        environment["TERM"] = terminal_type
        colorterm = writer.get_extra_info("COLORTERM")  # type: ignore[union-attr]
        if colorterm:
            environment["COLORTERM"] = colorterm
        color_mode = detect_color_mode(environment)

    if color_mode == ColorMode.NO_COLOR:
        print(
            f"< Telnet client terminal `{terminal_type}` does not support colors"
        )
        writer.write(  # type: ignore[union-attr]
            f"Your terminal `{terminal_type}` doesn't seem to support colors.\r\n"
            .encode("utf-8")
        )
        await writer.drain()  # type: ignore[union-attr]
        return 1

    async with telnet_to_app_session(writer) as app_session:  # type: ignore[arg-type]
        loop = asyncio.get_event_loop()

        # Probe for true color if it wasn't already determined from env vars
        if app_config.color_mode is None and color_mode < ColorMode.HAS_24_BIT_COLOR:
            if await _detect_true_color_telnet(reader, writer):
                color_mode = ColorMode.HAS_24_BIT_COLOR

        height, width = app_session.output.get_size()
        print(
            "[Terminal Info] "
            f"{peer_host}: {terminal_type}, {color_mode}, {width}x{height}"
        )
        state = TelnetInputState()
        input_task = asyncio.create_task(
            read_telnet_input(reader, writer, state, app_session.input)
        )
        stats_task = asyncio.create_task(
            _log_connection_stats(writer, peer_host, peer_port)
        )
        try:
            return await loop.run_in_executor(
                executor,
                thread_target,
                app_session,
                argparse.Namespace(**vars(app_config)),
                username,
                color_mode,
                state,
            )
        finally:
            input_task.cancel()
            stats_task.cancel()
            try:
                await input_task
            except asyncio.CancelledError:
                pass
            try:
                await stats_task
            except asyncio.CancelledError:
                pass


async def run_server(
    app_config: argparse.Namespace, executor: ThreadPoolExecutor
) -> None:
    import telnetlib3  # noqa: E402

    shell = make_telnet_shell(app_config, executor)
    server = await telnetlib3.create_server(
        host=app_config.bind,
        port=app_config.port,
        shell=shell,
        encoding=False,
        force_binary=True,
        connect_maxwait=4.0,
        timeout=0,
    )
    bind, port = server.sockets[0].getsockname()[:2]
    print(f"Running telnet server on {bind}:{port}...", flush=True)
    # Sleep forever
    await asyncio.Future()


def main(
    parser_args: tuple[str, ...] | None = None,
    console_cls: type[Console] = GameboyColor,
) -> None:
    parser = argparse.ArgumentParser(
        description="Gambatte terminal front-end over telnet"
    )
    add_base_arguments(parser)
    add_optional_arguments(parser)
    console_cls.add_console_arguments(parser)
    parser.add_argument(
        "--bind",
        "-b",
        type=str,
        default="127.0.0.1",
        help="Bind address of the telnet server, "
        "use `0.0.0.0` for all interfaces (default is localhost)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8023,
        help="Port of the telnet server (default is 8023)",
    )

    # Parse arguments
    app_config = parser.parse_args(parser_args)
    app_config.console_cls = console_cls

    # Run an executor with no limit on the number of threads
    with ThreadPoolExecutor(max_workers=32) as executor:
        # Run the server in asyncio
        asyncio.run(run_server(app_config, executor))


if __name__ == "__main__":
    main()
