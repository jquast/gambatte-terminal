from __future__ import annotations

import re
import time
import codecs
import asyncio
import threading
from typing import Any

# Matches a complete cursor-position response: ESC [ row ; col R
_CPR_RE = re.compile(r"\x1b\[\d+;\d+R")
# Matches a partial CPR prefix at end of buffer (wait for more data)
_CPR_PREFIX_RE = re.compile(r"\x1b\[\d*[;\d]*\Z")

from .console import Console

# Auto-release time for legacy (non-kitty) clients (~9 frames at 60fps)
HOLD_DURATION = 0.15
# Expected key repeat rate (~30 keys/sec)
EXPECTED_REPEAT_INTERVAL = 0.035
# Release if no repeat arrives — short window before kitty is confirmed
RELEASE_AFTER_NO_REPEAT = EXPECTED_REPEAT_INTERVAL * 4
# Once kitty is confirmed we have explicit release events, so hold for a long
# time and rely on those rather than the repeat timer.  OS keyboard repeat is
# suppressed for earlier-pressed keys while a second key is held, so a short
# timeout causes premature release of multi-key chords.
KITTY_CONFIRMED_HOLD = 5.0

# Blessed Keystroke name → Console.Input
KEY_INPUT_MAP: dict[str, Console.Input] = {
    "KEY_UP": Console.Input.UP,
    "KEY_DOWN": Console.Input.DOWN,
    "KEY_LEFT": Console.Input.LEFT,
    "KEY_RIGHT": Console.Input.RIGHT,
    "KEY_ENTER": Console.Input.START,
    "KEY_TAB": Console.Input.SELECT,
    "KEY_BACKSPACE": Console.Input.B,
    "KEY_DELETE": Console.Input.B,
}

# Character value → Console.Input
CHAR_INPUT_MAP: dict[str, Console.Input] = {
    "z": Console.Input.A,
    "a": Console.Input.A,
    "f": Console.Input.A,
    "v": Console.Input.A,
    " ": Console.Input.A,
    "x": Console.Input.B,
    "b": Console.Input.B,
    "s": Console.Input.B,
    "d": Console.Input.B,
    "c": Console.Input.B,
}

# Character value → Console.Event
CHAR_EVENT_MAP: dict[str, Console.Event] = {
    "0": Console.Event.SELECT_STATE_0,
    "1": Console.Event.SELECT_STATE_1,
    "2": Console.Event.SELECT_STATE_2,
    "3": Console.Event.SELECT_STATE_3,
    "4": Console.Event.SELECT_STATE_4,
    "5": Console.Event.SELECT_STATE_5,
    "6": Console.Event.SELECT_STATE_6,
    "7": Console.Event.SELECT_STATE_7,
    "8": Console.Event.SELECT_STATE_8,
    "9": Console.Event.SELECT_STATE_9,
    "l": Console.Event.LOAD_STATE,
    "k": Console.Event.SAVE_STATE,
}


class TelnetInputState:
    """Thread-safe input state shared between async reader and game thread."""

    def __init__(self) -> None:
        self._pressed: dict[Console.Input, float] = {}
        self._events: list[Console.Event] = []
        self._lock = threading.Lock()

    def press(self, button: Console.Input, hold_duration: float) -> None:
        """Register a button press with auto-release timestamp."""
        with self._lock:
            self._pressed[button] = time.monotonic() + hold_duration

    def release(self, button: Console.Input) -> None:
        """Release a button immediately."""
        with self._lock:
            self._pressed.pop(button, None)

    def get_input(self) -> set[Console.Input]:
        """Return currently pressed buttons, expiring old presses."""
        now = time.monotonic()
        with self._lock:
            expired = [b for b, t in self._pressed.items() if t <= now]
            for b in expired:
                del self._pressed[b]
            return set(self._pressed)

    def queue_event(self, event: Console.Event) -> None:
        """Queue an event for the game thread."""
        with self._lock:
            self._events.append(event)

    def pop_events(self) -> list[Console.Event]:
        """Drain and return queued events."""
        with self._lock:
            events = self._events
            self._events = []
            return events


def _build_blessed_maps() -> tuple[Any, dict[int, str], set[str]]:
    """Build blessed keyboard maps using a Terminal instance.

    :returns: (mapper, codes, prefixes) for use with resolve_sequence
    """
    from blessed import Terminal
    term = Terminal(kind="xterm", force_styling=True)
    return term._keymap, term._keycodes, term._keymap_prefixes


def _resolve_keystroke(
    text: str,
    mapper: Any,
    codes: dict[int, str],
    prefixes: set[str],
    dec_mode_cache: dict[int, int],
) -> Any:
    """Resolve a keystroke from text using blessed's standalone parser."""
    from blessed.keyboard import resolve_sequence
    return resolve_sequence(text, mapper, codes, prefixes, dec_mode_cache=dec_mode_cache)


def _map_keystroke(
    ks: Any,
    state: TelnetInputState,
    kitty_detected: bool,
    codes: dict[int, str],
) -> bool:
    """Map a blessed Keystroke to input state updates.

    Handles both legacy (non-kitty) terminals and kitty keyboard protocol.
    For kitty terminals, ``ks.uses_keyboard_protocol`` is True and we receive
    explicit press/repeat/release events; for legacy terminals we rely on
    auto-expiring hold timers refreshed by OS key-repeat events.

    :param ks: resolved blessed Keystroke
    :param state: shared input state to update
    :param kitty_detected: whether kitty protocol has been seen this session
    :param codes: blessed ``_keycodes`` mapping (int code → key name string)
    :returns: updated kitty_detected flag
    """
    # Kitty explicit release — immediately clear the button.
    if ks.uses_keyboard_protocol and ks.released:
        kitty_detected = True
        button = _get_button(ks, codes)
        if button is not None:
            state.release(button)
        return kitty_detected

    # Determine hold duration.
    # Kitty gives us explicit releases, so we hold for a long time and rely on
    # those.  OS key-repeat suppresses repeats for earlier keys while a second
    # key is held, so a short timeout causes premature release of multi-key
    # chords on legacy terminals.
    if ks.uses_keyboard_protocol:
        kitty_detected = True
        hold = KITTY_CONFIRMED_HOLD
    elif kitty_detected:
        # Kitty terminal sending a legacy-format press (report_events without
        # disambiguate): we still have explicit releases, so hold long.
        hold = KITTY_CONFIRMED_HOLD
    else:
        hold = HOLD_DURATION

    # Named key: arrows, enter, tab …
    # Kitty repeat events carry a _REPEATED suffix; strip it to find the base name.
    name = ks.name
    if name:
        base = name.removesuffix("_REPEATED")
        button = KEY_INPUT_MAP.get(base)
        if button is not None:
            state.press(button, hold)
            return kitty_detected

    # Kitty press with event_type=1 has name=None but _code identifies the key.
    if ks.uses_keyboard_protocol and ks._code is not None:
        base_name = codes.get(ks._code)
        if base_name:
            button = KEY_INPUT_MAP.get(base_name)
            if button is not None:
                state.press(button, hold)
                return kitty_detected

    # Character input.
    # Kitty encodes characters in ks.value (works for press and repeat);
    # legacy terminals expose the character via str(ks) when len == 1.
    char = ks.value if ks.uses_keyboard_protocol else str(ks)
    if len(char) == 1:
        event = CHAR_EVENT_MAP.get(char)
        if event is not None:
            state.queue_event(event)
            return kitty_detected
        button = CHAR_INPUT_MAP.get(char)
        if button is not None:
            state.press(button, hold)
            return kitty_detected

    return kitty_detected


def _get_button(ks: Any, codes: dict[int, str]) -> Console.Input | None:
    """Return the :class:`Console.Input` button for a keystroke, or ``None``.

    Handles both kitty-protocol and legacy keystrokes, including kitty's
    ``_REPEATED``/``_RELEASED`` name suffixes and character-key releases where
    ``ks.value`` is empty but ``ks._match.unicode_key`` carries the codepoint.

    :param ks: resolved blessed Keystroke
    :param codes: blessed ``_keycodes`` mapping (int code → key name string)
    """
    # Named key — strip kitty event-type suffixes before lookup.
    name = ks.name
    if name:
        for suffix in ("_REPEATED", "_RELEASED"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        button = KEY_INPUT_MAP.get(name)
        if button is not None:
            return button

    # Kitty press with event_type=1: name is None but _code identifies the key.
    if ks.uses_keyboard_protocol and ks._code is not None:
        base_name = codes.get(ks._code)
        if base_name:
            button = KEY_INPUT_MAP.get(base_name)
            if button is not None:
                return button

    # Character keys via kitty protocol.
    # ks.value is 'f' for press/repeat but '' for release; fall back to
    # _match.unicode_key (printable ASCII range) for release events.
    if ks.uses_keyboard_protocol:
        char = ks.value
        if not char:
            match = getattr(ks, "_match", None)
            unicode_key = getattr(match, "unicode_key", None)
            if unicode_key is not None and 32 <= unicode_key <= 126:
                char = chr(unicode_key)
        if char and len(char) == 1:
            return CHAR_INPUT_MAP.get(char)

    # Legacy plain character.
    char = str(ks)
    if len(char) == 1:
        return CHAR_INPUT_MAP.get(char)

    return None


async def read_telnet_input(
    reader: object,
    writer: object,
    state: TelnetInputState,
    pipe_input: Any = None,
) -> None:
    """Async reader loop: read bytes from telnet, parse keystrokes, update state.

    :param reader: telnetlib3 reader (has ``read`` coroutine)
    :param writer: telnetlib3 writer (has ``write`` method)
    :param state: shared input state
    :param pipe_input: optional prompt_toolkit pipe input to forward raw bytes into,
        enabling CPR sync and Ctrl+C/D handling in the render loop
    """
    mapper, codes, prefixes = _build_blessed_maps()
    dec_mode_cache: dict[int, int] = {}

    # Try to enable kitty keyboard protocol (disambiguate + report_events)
    writer.write(b"\x1b[=3u")  # type: ignore[union-attr]

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    kitty_detected = False
    buf = ""

    try:
        while True:
            data = await reader.read(4096)  # type: ignore[union-attr]
            if not data:
                break
            raw = data if isinstance(data, bytes) else data.encode("latin-1")
            if pipe_input is not None:
                pipe_input.send_bytes(raw)
            text = decoder.decode(raw)
            if not text:
                continue
            buf += text

            # Consume keystrokes from buffer
            while buf:
                # Bypass blessed for CPR sequences — it would split \x1b off as
                # KEY_ESCAPE and let the digits hit CHAR_EVENT_MAP as save-states.
                m = _CPR_RE.match(buf)
                if m:
                    buf = buf[m.end():]
                    continue
                # Hold a partial CPR prefix until the full sequence arrives.
                if _CPR_PREFIX_RE.match(buf):
                    break
                ks = _resolve_keystroke(buf, mapper, codes, prefixes, dec_mode_cache)
                consumed = len(ks) if len(ks) > 0 else 1
                if consumed == 0:
                    break
                buf = buf[consumed:]
                kitty_detected = _map_keystroke(ks, state, kitty_detected, codes)
    except (asyncio.CancelledError, ConnectionError, EOFError):
        pass
    finally:
        # Cleanup: disable kitty protocol
        try:
            writer.write(b"\x1b[=0u")  # type: ignore[union-attr]
        except (ConnectionError, OSError):
            pass
