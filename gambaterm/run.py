from __future__ import annotations

import os
import sys
import time
import contextlib
from itertools import count
from collections import deque
from typing import Callable, Deque, Iterator, cast

import numpy as np
import numpy.typing as npt
from blessed import Terminal

from .termblit import blit
from .sextant import blit_sextant
from .octant import blit_octant
from .audio import AudioOut
from .console import Console, InputGetter
from .keyboard_input import GameInputGetter
from .colors import ColorMode


@contextlib.contextmanager
def timing(deltas: Deque[float]) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        deltas.append(time.perf_counter() - start)


def get_ref(
    width: int,
    height: int,
    console: Console,
    sextant: bool = False,
    octant: bool = False,
) -> tuple[int, int]:
    if octant:
        rows = console.HEIGHT // 4
        cols = console.WIDTH // 2
    elif sextant:
        rows = console.HEIGHT // 3
        cols = console.WIDTH // 2
    else:
        rows = console.HEIGHT // 2
        cols = console.WIDTH
    margin_x = min(2, max(0, height - rows))
    margin_y = min(3, max(0, width - cols - 1))
    refx = margin_x + max(0, (height - margin_x - rows) // 2)
    refy = margin_y + max(0, (width - margin_y - cols) // 2)
    return refx, refy


_RESIZE_HINT = " !! Resize window and font size for higher resolution graphics !! "
_RESIZE_HINT_DURATION = 6.0


def _cycle_color(cycle: float) -> tuple[int, int, int]:
    """Map a 0–4 cycle position to an RGB color on the red→black→white→black ramp."""
    if cycle < 1.0:
        t = cycle
        return int(255 * t), 0, 0
    elif cycle < 2.0:
        t = 2.0 - cycle
        return int(255 * t), 0, 0
    elif cycle < 3.0:
        t = cycle - 2.0
        return int(255 * t), int(255 * t), int(255 * t)
    else:
        t = 4.0 - cycle
        return int(255 * t), int(255 * t), int(255 * t)


def _resize_hint_overlay(
    width: int,
    height: int,
    elapsed: float,
) -> bytes:
    """Render the resize hint text with a red/black/white/black flash."""
    text = _RESIZE_HINT
    max_len = width - 2
    if max_len <= 0 or height < 3:
        return b""
    if len(text) > max_len:
        text = text[:max_len]
    row = max(1, height - 2)
    col = max(1, (width - len(text)) // 2 + 1)
    # Linear bounce: 0→1 bright red, 1→2 fade to black,
    #                2→3 bright white, 3→4 fade to black, repeat.
    # Background is offset by 1 phase (90°) so we get:
    # red-on-black, black-on-red, white-on-black, black-on-white.
    cycle = (elapsed / 0.6) % 4.0
    ri, gi, bi = _cycle_color(cycle)
    bri, bgi, bbi = _cycle_color((cycle + 1.0) % 4.0)
    return (
        f"\033[{row};{col}H"
        f"\033[1;38;2;{ri};{gi};{bi};48;2;{bri};{bgi};{bbi}m{text}\033[0m"
    ).encode()


def write_bytes(term: Terminal, video_data: bytes) -> None:
    # Fix code page issue on windows:
    # `sys.stdout.buffer.raw` is a `WindowsConsoleIO` that always support UTF-8
    # regardless of the configured codepage
    if sys.platform == "win32" and term.stream.fileno() == sys.stdout.fileno():
        sys.stdout.buffer.write(video_data)
        sys.stdout.buffer.flush()
    else:
        os.write(term.stream.fileno(), video_data)


def run(
    console: Console,
    get_input: InputGetter,
    term: Terminal,
    audio_out: AudioOut | None = None,
    frame_advance: int = 1,
    color_mode: ColorMode = ColorMode.HAS_24_BIT_COLOR,
    break_after: int | None = None,
    speed_factor: float = 1.0,
    use_cpr_sync: bool = False,
    no_sextants: bool = False,
    no_octants: bool = False,
) -> None:
    assert color_mode > 0

    # Prepare buffers with invalid data
    video = np.full((console.HEIGHT, console.WIDTH), 0, np.uint32)
    audio = np.full((2 * console.TICKS_IN_FRAME, 2), -0x7FFF, np.int16)
    last_frame: npt.NDArray[np.uint32] | None = video.copy()

    # Determine rendering mode based on terminal size.
    # Pick the highest-resolution mode whose grid fits the terminal.
    def select_blit(w: int, h: int) -> Callable[..., bytes]:
        if w >= console.WIDTH and h >= console.HEIGHT // 2:
            return blit
        if not no_sextants and w >= console.WIDTH // 2 and h >= console.HEIGHT // 3:
            return cast(Callable[..., bytes], blit_sextant)
        if not no_octants:
            return cast(Callable[..., bytes], blit_octant)
        return blit

    # Print area (default to 24x80 if terminal reports zero)
    height = term.height or 24
    width = term.width or 80
    blit_fn = select_blit(width, height)
    refx, refy = get_ref(
        width,
        height,
        console,
        blit_fn is blit_sextant,
        blit_fn is blit_octant,
    )

    # Prepare reporting
    fps = console.FPS * speed_factor
    average_over = int(round(fps))  # frames
    ticks: Deque[float] = deque(maxlen=average_over)
    emu_deltas: Deque[float] = deque(maxlen=average_over)
    audio_deltas: Deque[float] = deque(maxlen=average_over)
    video_deltas: Deque[float] = deque(maxlen=average_over)
    sync_deltas: Deque[float] = deque(maxlen=average_over)
    total_deltas: Deque[float] = deque(maxlen=average_over)
    shifting: Deque[float] = deque(maxlen=average_over)
    shown_frames: Deque[int] = deque(maxlen=average_over)
    data_length: Deque[int] = deque(maxlen=average_over)
    start = time.time()

    # Create a 100 ms time shift to fill up audio buffer
    if audio_out:
        start -= 0.1

    # Resolve CPR state once (only GameInputGetter has it)
    cpr_state = get_input.cpr_state if isinstance(get_input, GameInputGetter) else None

    # Prepare state
    new_frame = False
    screen_ready = True
    frame_start_time = None

    # Resize hint: show for 6 seconds at startup if not in half-block mode
    hint_active = blit_fn in (blit_sextant, blit_octant)
    hint_start = time.perf_counter() if hint_active else 0.0

    # Loop over emulator frames
    for i in count():
        # Add total deltas
        if frame_start_time is not None:
            total_deltas.append(time.perf_counter() - frame_start_time)
        frame_start_time = time.perf_counter()

        # Break when frame limit is reach
        if break_after is not None and i >= break_after:
            return

        # Tick the emulator
        with timing(emu_deltas):
            console.set_input(get_input())
            offset, samples = console.advance_one_frame(video, audio)
            new_frame = new_frame or offset > 0
            ticks.append(samples)

        # Send audio
        with timing(audio_deltas):
            if audio_out:
                audio_out.send(audio[:samples, :])

        # Read keys for ctrl-c/ctrl-d detection.
        # We check for raw \x03 and \x04 bytes rather than physical key codes
        # so that detection is keyboard-layout agnostic: e.g. a Bépo user
        # presses Ctrl+C (physically Ctrl+H on US layout) and the terminal
        # translates it to \x03 (ETX) before putting it on stdin.
        # For blessed backend: keystrokes were collected during get_input().
        # With kitty protocol, ctrl+c arrives as an enhanced CSI sequence
        # (e.g. \x1b[99;5u) rather than raw \x03, so we also check blessed's
        # decoded key_name attribute.
        if isinstance(get_input, GameInputGetter) and get_input.cpr_state.keystrokes:
            for key in get_input.cpr_state.keystrokes:
                if str(key) == "\x03" or key.key_name == "KEY_CTRL_C":
                    raise KeyboardInterrupt
                if str(key) == "\x04" or key.key_name == "KEY_CTRL_D":
                    raise OSError
        # For X11/pynput backends: game input comes from X11 events / OS key
        # hooks (not stdin), so we read stdin here solely for ctrl-c/ctrl-d.
        # This replaces prompt-toolkit's read_keys().
        else:
            while True:
                key = term.inkey(timeout=0)
                if not key:
                    break
                if str(key) == "\x03":
                    raise KeyboardInterrupt
                if str(key) == "\x04":
                    raise OSError

        # Check for CPR response (set by keyboard handler during get_input)
        if use_cpr_sync and cpr_state is not None and cpr_state.cpr_received:
            screen_ready = True

        # Render video
        with timing(video_deltas):
            # Send the frame
            shift = shifting and shifting[-1] > 1 / fps
            if i % frame_advance == 0 and new_frame and screen_ready and not shift:
                new_frame = False
                # Check terminal size
                new_height = term.height or 24
                new_width = term.width or 80
                maybe_clear_seq = b""
                if (new_height, new_width) != (height, width):
                    maybe_clear_seq = b"\033[H\033[2J"
                    height, width = new_height, new_width
                    blit_fn = select_blit(width, height)
                    refx, refy = get_ref(
                        width,
                        height,
                        console,
                        blit_fn is blit_sextant,
                        blit_fn is blit_octant,
                    )
                    # Screen cleared — pass None to force full redraw
                    # (clears sextant/octant _display_cache, and Cython
                    # blit skips its last-frame comparison when last=None)
                    last_frame = None
                # Render frame with synchronized output mode (DEC 2026) to prevent flickering
                # when the screen is cleared, or an artificial CRT-like "rolling band" side-effects
                # from fast "sprite blinking" meant to cause "transparency" effect on original HW,
                # https://zladx.github.io/posts/links-awakening-partial-translucency
                video_data = (
                    b"\033[?2026h"
                    + maybe_clear_seq
                    + blit_fn(
                        video,
                        last_frame,
                        refx,
                        refy,
                        width,
                        height,
                        color_mode,
                    )
                    + b"\033[?2026l"
                )
                last_frame = video.copy()

                # Overlay resize hint during first 6 seconds
                if hint_active and blit_fn not in (blit_sextant, blit_octant):
                    hint_active = False
                elif hint_active:
                    elapsed = time.perf_counter() - hint_start
                    if elapsed < _RESIZE_HINT_DURATION:
                        video_data += _resize_hint_overlay(
                            width,
                            height,
                            elapsed,
                        )
                    else:
                        hint_active = False
                        last_frame = None
                        row = max(1, height - 2)
                        video_data += (f"\033[{row};1H\033[0m\033[K").encode()

                # Update reporting
                data_length.append(len(video_data))
                shown_frames.append(True)
            # Ignore this video frame
            else:
                video_data = None
                data_length.append(0)
                shown_frames.append(False)

        with timing(sync_deltas):
            # Video sync
            if video_data:
                write_bytes(term, video_data)
                # Send CPR request
                if use_cpr_sync:
                    term.stream.write("\033[6n")
                    term.stream.flush()
                    screen_ready = False
            # Timing sync
            increment = samples / console.TICKS_IN_FRAME
            deadline = start + increment / fps
            current = time.time()
            if current < deadline - 1e-3:
                time.sleep(deadline - current)
            # Use deadline as new reference to prevent shifting
            shifting.append(time.time() - deadline)
            start = deadline

        # Reporting
        if i % average_over == 1:
            tps = fps * console.TICKS_IN_FRAME
            emu_fps = tps * len(ticks) / sum(ticks)
            video_fps = emu_fps * sum(shown_frames) / len(shown_frames)
            total_fps = len(total_deltas) / sum(total_deltas)
            emu_percent = sum(emu_deltas) / len(emu_deltas) * total_fps * 100
            audio_percent = sum(audio_deltas) / len(audio_deltas) * total_fps * 100
            video_percent = sum(video_deltas) / len(video_deltas) * total_fps * 100
            data_rate = sum(data_length) / len(data_length) * total_fps / 1000
            title = f"Gambaterm - {total_fps:.0f} FPS | "
            title += f"{os.path.basename(console.romfile)} | "
            title += f"Emu: {emu_fps:.0f} FPS - {emu_percent:.0f}% CPU | "
            title += f"Video: {video_fps:.0f} FPS - {video_percent:.0f}% CPU - "
            title += f"{data_rate:.0f} KB/s | "
            title += f"Audio: {audio_percent:.0f}% CPU"
            term.stream.write(term.set_window_title(title))
            term.stream.flush()
