#!/usr/bin/env python3
from __future__ import annotations

import re
import os
import sys
import time
import argparse
import contextlib

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input

from .run import run
from .console import GameboyColor, Console
from .audio import audio_player, no_audio
from .colors import detect_local_color_mode, ColorMode
from .keyboard_input import console_input_from_keyboard_context
from .controller_input import combine_console_input_from_controller_context
from .file_input import console_input_from_file_context, write_input_context
from .local_input import local_blessed_input_context

_KITTY_RESP_RE = re.compile(rb"\x1b\[\?\d+u")


def _probe_kitty_keyboard(timeout: float = 0.2) -> bool:
    """Probe the local terminal for kitty keyboard protocol support.

    Temporarily sets stdin to raw mode, sends a kitty keyboard capability
    query (``ESC [ ? u``), and checks for the expected response.

    :param timeout: Seconds to wait for a terminal response.
    :returns: ``True`` if the terminal responds to a kitty keyboard query.
    """
    if sys.platform == "win32":
        return False
    import tty
    import termios
    import select as _select
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except (termios.error, ValueError):
        return False
    try:
        tty.setraw(fd)
        sys.stdout.buffer.write(b"\x1b[?u")
        sys.stdout.buffer.flush()
        ready, _, _ = _select.select([fd], [], [], timeout)
        if not ready:
            return False
        data = os.read(fd, 64)
        return bool(_KITTY_RESP_RE.search(data))
    except OSError:
        return False
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except termios.error:
            pass


def add_base_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("romfile", metavar="ROM", type=str, help="Path to a rom file")
    parser.add_argument(
        "--input-file", "-i", default=None, help="Path to a bizhawk BK2 file"
    )


def add_optional_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--color-mode",
        "-c",
        type=int,
        default=None,
        help="Force a color mode "
        "(1: 4 greyscale colors, 2: 16 colors, 3: 256 colors, 4: 24-bit colors)",
    )
    parser.add_argument(
        "--frame-advance",
        "--fa",
        type=int,
        default=1,
        help="Number of frames to run before displaying the next one (default is 1)",
    )
    parser.add_argument(
        "--break-after",
        "--ba",
        type=int,
        default=None,
        help="Number of frames to run before forcing the emulator to stop "
        "(doesn't stop by default)",
    )
    parser.add_argument(
        "--speed-factor",
        "--sf",
        type=float,
        default=1.0,
        help="Speed factor to apply to the emulation "
        "(default is 1.0 corresponding to 60 FPS)",
    )
    parser.add_argument(
        "--skip-inputs",
        "--si",
        type=int,
        default=188,
        help="Number of frame inputs to skip in order to compensate "
        "for the lack of BIOS (default is 188)",
    )
    parser.add_argument(
        "--cpr-sync",
        "--cs",
        action="store_true",
        help="Use CPR synchronization to prevent video buffering",
    )
    parser.add_argument(
        "--enable-controller",
        "--ec",
        action="store_true",
        help="Enable game controller support",
    )
    parser.add_argument(
        "--write-input",
        "--wi",
        type=str,
        help="Enable game controller support",
    )
    parser.add_argument(
        "--sextant",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use sextant block rendering (auto-detected by default, "
        "use --no-sextant to disable)",
    )
    parser.add_argument(
        "--kitty",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use kitty keyboard protocol for input (auto-detected by default, "
        "use --no-kitty to disable)",
    )


def main(
    parser_args: tuple[str, ...] | None = None,
    console_cls: type[Console] = GameboyColor,
) -> None:
    parser = argparse.ArgumentParser(
        prog="gambaterm", description="Gambatte terminal front-end"
    )
    add_base_arguments(parser)
    add_optional_arguments(parser)
    console_cls.add_console_arguments(parser)
    parser.add_argument(
        "--disable-audio", "--da", action="store_true", help="Disable audio entirely"
    )
    args: argparse.Namespace = parser.parse_args(parser_args)
    console = console_cls(args)

    # Determine whether to use the blessed+kitty local input path
    if args.input_file is not None or sys.platform == "win32":
        use_kitty = False
    elif args.kitty is True:
        use_kitty = True
    elif args.kitty is False:
        use_kitty = False
    else:
        use_kitty = _probe_kitty_keyboard()

    if args.color_mode not in [None, 1, 2, 3, 4]:
        exit(
            f"Invalid color mode `{args.color_mode}`: the value must be between 1 and 4"
        )

    with contextlib.ExitStack() as stack:
        if use_kitty:
            pipe_input = stack.enter_context(create_pipe_input())
            app_session = stack.enter_context(create_app_session(input=pipe_input))
        else:
            app_session = stack.enter_context(create_app_session())

        # Build input context
        if args.input_file is not None:
            input_context = console_input_from_file_context(
                console, args.input_file, args.skip_inputs
            )
        elif use_kitty:
            input_context = local_blessed_input_context(console, pipe_input)
        else:
            input_context = console_input_from_keyboard_context(console)

        if args.input_file is None and args.enable_controller:
            input_context = combine_console_input_from_controller_context(
                console, input_context
            )

        if args.write_input:
            input_context = write_input_context(console, input_context, args.write_input)

        # Enter terminal raw mode
        with app_session.input.raw_mode():
            try:
                # Enter input context before color detection so the blessed thread
                # is running when detect_true_color_support sends its DECRQSS probe
                # (the terminal's response must be forwarded through pipe_input).
                with input_context as get_gb_input:
                    # Detect color mode
                    if args.color_mode is None:
                        args.color_mode = detect_local_color_mode(app_session)
                        if args.color_mode == ColorMode.NO_COLOR:
                            raise exit(
                                """\
The ANSI color support for your terminal could not be detected from your environment.
Try to force a color mode using the `--color-mode` option with a value between 1 and 4."""
                            )

                    # Prepare alternate screen
                    app_session.output.enter_alternate_screen()
                    app_session.output.erase_screen()
                    app_session.output.hide_cursor()
                    app_session.output.flush()

                    player = no_audio if args.disable_audio else audio_player
                    with player(console, args.speed_factor) as audio_out:
                        # Run the emulator
                        run(
                            console,
                            get_gb_input,
                            app_session=app_session,
                            audio_out=audio_out,
                            frame_advance=args.frame_advance,
                            color_mode=args.color_mode,
                            break_after=args.break_after,
                            speed_factor=args.speed_factor,
                            use_cpr_sync=args.cpr_sync,
                            sextant=args.sextant,
                        )

            # Deal with ctrl+c and ctrl+d exceptions
            except (KeyboardInterrupt, EOFError):
                pass

            # Exit normally
            else:
                exit()

            # Restore terminal to its initial state
            finally:
                # Wait for a possible CPR
                time.sleep(0.1)
                # Clear alternate screen
                app_session.input.read_keys()
                app_session.output.erase_screen()
                app_session.output.quit_alternate_screen()
                app_session.output.show_cursor()
                app_session.output.flush()


if __name__ == "__main__":
    main()
