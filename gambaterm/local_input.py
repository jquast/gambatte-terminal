"""Blessed + kitty keyboard input for local terminal sessions."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .console import Console, InputGetter
from .telnet_input import (
    TelnetInputState,
    _map_keystroke,
)


@contextmanager
def local_blessed_input_context(
    console: Console,
    pipe_input: Any,
) -> Iterator[InputGetter]:
    """Blessed + kitty keyboard input context for local terminal use.

    Uses ``blessed.Terminal.inkey()`` to read keystrokes inline each frame.
    Ctrl+C and Ctrl+D are raised as :exc:`KeyboardInterrupt` and
    :exc:`EOFError` respectively.  All other terminal events (CPR responses,
    etc.) are silently discarded; ``pipe_input`` is kept only to prevent
    ``app_session.input.read_keys()`` from competing on stdin.

    :param console: Console instance for input and event mapping.
    :param pipe_input: Prompt_toolkit PipeInput that owns the active
        AppSession; kept open so ``read_keys()`` reads from the pipe
        (always empty) rather than stdin.
    """
    from blessed import Terminal

    term = Terminal(force_styling=True)
    state = TelnetInputState()
    kitty_detected = False

    def get_input() -> set[Console.Input]:
        nonlocal kitty_detected
        while True:
            ks = term.inkey(timeout=0, esc_delay=0)
            if not ks:
                break
            ch = str(ks)
            if ch == '\x03':
                raise KeyboardInterrupt
            if ch == '\x04':
                raise EOFError
            kitty_detected = _map_keystroke(ks, state, kitty_detected)
        for event in state.pop_events():
            console.handle_event(event)
        return state.get_input()

    with term.raw():
        with term.enable_kitty_keyboard(disambiguate=True, report_events=True):
            yield get_input
