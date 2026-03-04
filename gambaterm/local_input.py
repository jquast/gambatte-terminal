"""Blessed + kitty keyboard input for local terminal sessions."""
from __future__ import annotations

import codecs
import os
import select
import sys
from contextlib import contextmanager
from typing import Any, Iterator

from .console import Console, InputGetter
from .telnet_input import (
    TelnetInputState,
    _CPR_PREFIX_RE,
    _CPR_RE,
    _build_blessed_maps,
    _map_keystroke,
    _resolve_keystroke,
)


@contextmanager
def local_blessed_input_context(
    console: Console,
    pipe_input: Any,
) -> Iterator[InputGetter]:
    """Blessed + kitty keyboard input context for local terminal use.

    Reads raw stdin bytes and parses them with blessed's sequence resolver each
    frame.  CPR responses (``ESC[row;colR``) are forwarded to ``pipe_input`` so
    that ``app_session.input.read_keys()`` delivers
    ``<cursor-position-response>`` events to the render loop, enabling CPR sync.
    Ctrl+C and Ctrl+D are raised as :exc:`KeyboardInterrupt` and
    :exc:`EOFError` respectively.

    :param console: Console instance for input and event mapping.
    :param pipe_input: Prompt_toolkit PipeInput; CPR responses are forwarded
        here so the render loop receives ``<cursor-position-response>`` events.
    """
    from blessed import Terminal

    term = Terminal(force_styling=True)
    state = TelnetInputState()
    mapper, codes, prefixes = _build_blessed_maps()
    dec_mode_cache: dict[int, int] = {}
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    kitty_detected = False
    buf = ""
    fd = sys.stdin.fileno()

    def get_input() -> set[Console.Input]:
        nonlocal kitty_detected, buf

        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if rlist:
            buf += decoder.decode(os.read(fd, 4096))

        while buf:
            m = _CPR_RE.match(buf)
            if m:
                pipe_input.send_bytes(buf[: m.end()].encode("latin-1"))
                buf = buf[m.end() :]
                continue
            if _CPR_PREFIX_RE.match(buf):
                break
            ks = _resolve_keystroke(buf, mapper, codes, prefixes, dec_mode_cache)
            consumed = len(ks) if len(ks) > 0 else 1
            if consumed == 0:
                break
            buf = buf[consumed:]
            ch = str(ks)
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x04":
                raise EOFError
            kitty_detected = _map_keystroke(ks, state, kitty_detected)

        for event in state.pop_events():
            console.handle_event(event)
        return state.get_input()

    with term.raw():
        with term.enable_kitty_keyboard(disambiguate=True, report_events=True):
            yield get_input
