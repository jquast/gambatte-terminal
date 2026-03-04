from __future__ import annotations

import time
import hashlib
import asyncio
import argparse
import dataclasses
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


_COLOR_LABELS: dict[ColorMode, str] = {
    ColorMode.HAS_24_BIT_COLOR: "24bit",
    ColorMode.HAS_8_BIT_COLOR: "256",
    ColorMode.HAS_4_BIT_COLOR: "16",
    ColorMode.HAS_2_BIT_COLOR: "4",
    ColorMode.NO_COLOR: "none",
}


@dataclasses.dataclass
class _LiveStats:
    """Live per-connection stats shared between the emulator thread and the stats coroutine."""

    fps: float = 0.0
    rtt_ms: float = 0.0
    frames_per_sweep: int = 0
    fast_bw_mbps: float = 0.0
    slow_bw_mbps: float = 0.0
    frame_mean_bytes: int = 0
    frame_p95_bytes: int = 0
    sweep_tx_ms: float = 0.0   # last CPR transmission overhead (excl. RTT floor)
    nul_pad_last: int = 0


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
    rtt_floor: float = 0.0,
    bandwidth_bps: float = 0.0,
    live_stats: _LiveStats | None = None,
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
                cpr_rtt_floor=rtt_floor,
                cpr_bandwidth_bps=bandwidth_bps,
                max_bw_bps=getattr(app_config, 'max_connection_bw_mbps', 0.0) * 1_000_000,
                live_stats=live_stats,
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


async def _calibrate_connection(
    reader: object,
    writer: object,
    host: str = "unknown",
    rtt_probes: int = 5,
    bulk_sizes: tuple = (1_000, 10_000, 100_000, 500_000),
    timeout: float = 5.0,
) -> tuple[float, float]:
    """Measure RTT floor and downlink bandwidth via NUL bulk probes.

    Must be called before the input reading task starts (exclusive reader access).

    :param reader: telnetlib3 reader
    :param writer: telnetlib3 writer
    :param host: peer host for logging
    :param rtt_probes: number of CPR round-trips for RTT floor estimation
    :param bulk_sizes: byte counts to probe for bandwidth measurement
    :param timeout: total time budget in seconds
    :returns: (rtt_floor_s, bandwidth_bps)
    """
    # Minimum transmission time to trust a bandwidth sample; filters out
    # tiny probes that complete within measurement noise.
    _MIN_BW_OVERHEAD_S = 0.020

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout

    async def read_until_r() -> bool:
        """Read from reader until a CPR response terminator 'R' is seen.

        CPR (cursor position report) responses end with 'R' (ESC[row;colR).
        """
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return False
                chunk = await asyncio.wait_for(
                    reader.read(256),  # type: ignore[union-attr]
                    timeout=min(2.0, remaining),
                )
                if not chunk:
                    return False
                if isinstance(chunk, str):
                    chunk = chunk.encode("latin-1")
                if b"R" in chunk:
                    return True
        except asyncio.TimeoutError:
            return False

    # Phase 1: RTT floor — send ESC[6n (cursor position query) and time the echo.
    # Min of N samples removes OS scheduling jitter from the estimate.
    rtt_samples = []
    for _ in range(rtt_probes):
        if loop.time() >= deadline:
            break
        writer.write(b"\x1b[6n")  # type: ignore[union-attr]
        await writer.drain()  # type: ignore[union-attr]
        t0 = loop.time()
        if await read_until_r():
            rtt_samples.append(loop.time() - t0)

    if not rtt_samples:
        return 0.0, 0.0

    rtt_floor = min(rtt_samples)

    # Phase 2: bandwidth — prepend NUL bytes before the CPR query so the terminal
    # cannot reply until it has consumed the full probe payload.  The time beyond
    # rtt_floor is the transmission delay, giving us bytes-per-second.
    bw_samples = []
    for size in bulk_sizes:
        if loop.time() >= deadline:
            break
        writer.write(bytes(size) + b"\x1b[6n")  # type: ignore[union-attr]
        await writer.drain()  # type: ignore[union-attr]
        t0 = loop.time()
        if await read_until_r():
            overhead = (loop.time() - t0) - rtt_floor
            if overhead > _MIN_BW_OVERHEAD_S:
                bw_samples.append(size * 8 / overhead)

    if not bw_samples:
        bandwidth_bps = 0.0
    else:
        bw_samples.sort()
        bandwidth_bps = bw_samples[len(bw_samples) // 2]  # median

    if bandwidth_bps > 0 and rtt_floor > 0:
        # Half-RTT window in bytes / conservative initial frame size estimate
        sweep_budget = bandwidth_bps * rtt_floor / 2 / 8
        conservative_frame_bytes = 15_000  # ~15 KB; refined at runtime from history
        frames_per_sweep = max(1, int(sweep_budget / conservative_frame_bytes))
    else:
        frames_per_sweep = 1

    print(
        f"[Calibrate {host}] RTT floor: {rtt_floor * 1000:.1f}ms, "
        f"bandwidth: {bandwidth_bps / 1_000_000:.2f} Mbit/s, "
        f"frames_per_sweep: {frames_per_sweep}"
    )
    return rtt_floor, bandwidth_bps


def _fmt_idle(seconds: float) -> str:
    """Format idle duration as 'Xm' or 'X.Xs'."""
    if seconds >= 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds:.1f}s"


async def _log_connection_stats(
    writer: object,
    peer_host: str,
    peer_port: int,
    live_stats: _LiveStats | None = None,
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
            fps_str = f", {live_stats.fps:.1f} FPS" if live_stats and live_stats.fps > 0 else ""
            rtt_str = (
                f", RTT {live_stats.rtt_ms:.1f}ms" if live_stats and live_stats.rtt_ms > 0 else ""
            )

            print(
                f"[Stats {peer_host}:{peer_port}] "
                f"up {uptime}, "
                f"tx {tx:,}B ({tx_mbps:.3f}/{avg_tx_mbps:.3f} Mbit/s)"
                f"{fps_str}{rtt_str}{idle_str}"
            )

            if live_stats and live_stats.frames_per_sweep > 0:
                bw_limited = (
                    live_stats.sweep_tx_ms > live_stats.rtt_ms * 0.5
                    if live_stats.rtt_ms > 0 else False
                )
                regime = "bw-limited" if bw_limited else "rtt-limited"
                nul_str = f" nul-pad {live_stats.nul_pad_last}B," if live_stats.nul_pad_last > 0 else ""
                print(
                    f"[Sweep {peer_host}:{peer_port}] "
                    f"{live_stats.frames_per_sweep} fr/sweep,"
                    f"{nul_str} "
                    f"bw {live_stats.fast_bw_mbps:.3f}/{live_stats.slow_bw_mbps:.3f} Mbit/s, "
                    f"frame avg {live_stats.frame_mean_bytes // 1024}KB "
                    f"p95 {live_stats.frame_p95_bytes // 1024}KB, "
                    f"tx {live_stats.sweep_tx_ms:.0f}ms, "
                    f"{regime}"
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

        # Measure RTT floor and bandwidth for adaptive sweep-based flow control
        if getattr(app_config, 'no_calibrate', False):
            rtt_floor, bandwidth_bps = 0.0, 0.0
        else:
            import random
            from telnetlib3.accessories import PATIENCE_MESSAGES
            patience = random.choice(PATIENCE_MESSAGES)
            writer.write(f"{patience}...\r\n".encode("utf-8"))  # type: ignore[union-attr]
            await writer.drain()  # type: ignore[union-attr]
            rtt_floor, bandwidth_bps = await _calibrate_connection(
                reader, writer, host=peer_host
            )

        height, width = app_session.output.get_size()
        color_label = _COLOR_LABELS.get(color_mode, str(int(color_mode)))
        print(
            "[Terminal Info] "
            f"{peer_host}: {terminal_type}, {color_label}, {width}x{height}"
        )
        live_stats = _LiveStats(rtt_ms=rtt_floor * 1000)
        state = TelnetInputState()
        input_task = asyncio.create_task(
            read_telnet_input(reader, writer, state, app_session.input)
        )
        stats_task = asyncio.create_task(
            _log_connection_stats(writer, peer_host, peer_port, live_stats=live_stats)
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
                rtt_floor,
                bandwidth_bps,
                live_stats,
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

    robot_check = getattr(app_config, "robot_check", False)
    max_players = getattr(app_config, "max_players", 0)

    if robot_check or max_players > 0:
        from telnetlib3.guard_shells import ConnectionCounter, busy_shell
        from telnetlib3.guard_shells import robot_check as do_robot_check
        from telnetlib3.guard_shells import robot_shell

        counter = ConnectionCounter(max_players) if max_players > 0 else None
        inner_shell = shell

        async def guarded_shell(reader: object, writer: object) -> None:
            if counter is not None and not counter.try_acquire():
                try:
                    await busy_shell(reader, writer)  # type: ignore[arg-type]
                finally:
                    if not writer.is_closing():  # type: ignore[union-attr]
                        writer.close()  # type: ignore[union-attr]
                return
            try:
                if robot_check:
                    passed = await do_robot_check(reader, writer)  # type: ignore[arg-type]
                    if not passed:
                        await robot_shell(reader, writer)  # type: ignore[arg-type]
                        if not writer.is_closing():  # type: ignore[union-attr]
                            writer.close()  # type: ignore[union-attr]
                        return
                await inner_shell(reader, writer)
            finally:
                if counter is not None:
                    counter.release()

        shell = guarded_shell

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
        "--max-players",
        type=int,
        default=0,
        metavar="N",
        help="maximum concurrent players (0 = unlimited)",
    )
    parser.add_argument(
        "--robot-check",
        action="store_true",
        default=False,
        help="reject bots by checking if client responds to cursor position requests",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8023,
        help="Port of the telnet server (default is 8023)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        default=False,
        help="Skip RTT/bandwidth calibration on connect (disables adaptive sweep, legacy 1-frame-per-CPR behaviour)",
    )
    parser.add_argument(
        "--max-connection-bw-mbps",
        type=float,
        default=5.0,
        metavar="MBPS",
        help="Cap per-connection bandwidth used for NUL pacing in Mbit/s (default 5.0, 0 = unlimited)",
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
