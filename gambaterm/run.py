from __future__ import annotations

import os
import sys
import time
import statistics
import contextlib
from itertools import count
from collections import deque
from typing import Any, Deque, Iterator

import numpy as np
from prompt_toolkit.application import AppSession

from .termblit import blit
from .sextant import blit_sextant
from .audio import AudioOut
from .console import Console, InputGetter
from .colors import ColorMode


@contextlib.contextmanager
def timing(deltas: Deque[float]) -> Iterator[None]:
    try:
        start = time.perf_counter()
        yield
    finally:
        deltas.append(time.perf_counter() - start)


def get_ref(
    width: int, height: int, console: Console, sextant: bool = False,
) -> tuple[int, int]:
    if sextant:
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


def write_bytes(app_session: AppSession, video_data: bytes) -> None:
    # Fix code page issue on windows:
    # `sys.stdout.buffer.raw` is a `WindowsConsoleIO` that always support UTF-8
    # regardless of the configured codepage
    if sys.platform == "win32" and app_session.output.fileno() == sys.stdout.fileno():
        sys.stdout.buffer.write(video_data)
        sys.stdout.buffer.flush()
    else:
        os.write(app_session.output.fileno(), video_data)


_COLOR_CYCLE = [
    ColorMode.HAS_4_BIT_COLOR,
    ColorMode.HAS_8_BIT_COLOR,
    ColorMode.HAS_24_BIT_COLOR,
]


def _next_color_mode(mode: ColorMode) -> ColorMode:
    try:
        idx = _COLOR_CYCLE.index(mode)
    except ValueError:
        idx = -1
    return _COLOR_CYCLE[(idx + 1) % len(_COLOR_CYCLE)]


def run(
    console: Console,
    get_input: InputGetter,
    app_session: AppSession,
    audio_out: AudioOut | None = None,
    frame_advance: int = 1,
    color_mode: ColorMode = ColorMode.HAS_24_BIT_COLOR,
    break_after: int | None = None,
    speed_factor: float = 1.0,
    use_cpr_sync: bool = False,
    sextant: bool | None = None,
    cycle_color_on_ctrl_c: bool = False,
    cpr_rtt_floor: float = 0.0,
    cpr_bandwidth_bps: float = 0.0,
    live_stats: Any = None,
) -> None:
    assert color_mode > 0

    # Prepare buffers with invalid data
    video = np.full((console.HEIGHT, console.WIDTH), 0xFFFFFFFF, np.uint32)
    audio = np.full((2 * console.TICKS_IN_FRAME, 2), -0x7FFF, np.int16)
    last_frame = video.copy()

    # Determine sextant mode
    height, width = app_session.output.get_size()
    if sextant is None:
        use_sextant = width < console.WIDTH + 6
    else:
        use_sextant = sextant
    blit_fn = blit_sextant if use_sextant else blit
    refx, refy = get_ref(width, height, console, use_sextant)

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

    # Prepare state
    new_frame = False
    screen_ready = True
    frame_start_time = None
    seen_size = (height, width)
    last_resize_time: float | None = None
    _RESIZE_DEBOUNCE = 0.05

    # Sweep-based CPR flow control state.
    # Instead of one CPR per frame, we pipeline `frames_per_sweep` frames before
    # asking for a CPR echo.  This fills roughly half an RTT worth of bandwidth.
    _FAST_EMA_ALPHA = 0.25   # reacts quickly to bandwidth drops
    _SLOW_EMA_ALPHA = 0.05   # tracks sustained throughput
    _SAFETY_FACTOR = 0.85    # conservative headroom to avoid overfilling buffers
    _SWEEP_WINDOW = 0.5      # fraction of RTT to fill per sweep (half-RTT)
    _MIN_OVERHEAD_S = 0.010  # ignore CPR cycles shorter than 10 ms (measurement noise)
    _FRAME_SIZE_P95_SIGMAS = 2   # mean + N*sigma ≈ p95 of frame size distribution
    _FRAME_HISTORY_LEN = 300     # ~30 s at ~10 fps; long window for stable VBR estimate
    _CONSERVATIVE_FRAME_BYTES = 15_000  # initial frame-size estimate before history fills
    # Opportunistic ramp: if RTT overhead is small relative to transmission time
    # (bandwidth-limited regime), try pipelining an extra frame to amortize RTT wait.
    # Back off if sweep time grows beyond this multiple of the RTT floor.
    _RAMP_MAX_FRAMES = 8         # never pipeline more than this many frames
    _RAMP_BLOAT_FACTOR = 3.0     # back off ramp if sweep_tx > rtt_floor * this
    ramp_frames: int = 0         # extra frames added by ramp (starts conservative)
    fast_bw_ema = cpr_bandwidth_bps
    slow_bw_ema = cpr_bandwidth_bps
    frames_per_sweep = 1
    frames_since_cpr = 0
    cpr_sent_at: float | None = None
    cpr_bytes_in_sweep = 0
    frame_size_history: Deque[int] = deque(maxlen=_FRAME_HISTORY_LEN)
    if cpr_bandwidth_bps > 0 and cpr_rtt_floor > 0:
        sweep_budget = cpr_bandwidth_bps * cpr_rtt_floor * _SWEEP_WINDOW / 8
        frames_per_sweep = max(1, int(sweep_budget / _CONSERVATIVE_FRAME_BYTES))

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

        # Read keys
        for event in app_session.input.read_keys():
            if event.key == "c-c":
                if cycle_color_on_ctrl_c:
                    color_mode = _next_color_mode(color_mode)
                    last_frame.fill(0xFFFFFFFF)
                else:
                    raise KeyboardInterrupt
            if event.key == "c-d":
                raise OSError
            if event.key == "<cursor-position-response>":
                screen_ready = True
                if cpr_sent_at is not None:
                    elapsed = time.perf_counter() - cpr_sent_at
                    if live_stats is not None:
                        live_stats.rtt_ms = elapsed * 1000
                    if (cpr_bytes_in_sweep > 0
                            and cpr_rtt_floor > 0 and cpr_bandwidth_bps > 0):
                        # Subtract RTT floor to isolate transmission time
                        overhead = elapsed - cpr_rtt_floor
                        if live_stats is not None:
                            live_stats.sweep_tx_ms = max(0.0, overhead * 1000)
                        if overhead > _MIN_OVERHEAD_S:
                            meas = cpr_bytes_in_sweep * 8 / overhead
                            # Dual-EWMA: fast reacts to drops, slow tracks sustained BW
                            fast_bw_ema = (1 - _FAST_EMA_ALPHA) * fast_bw_ema + _FAST_EMA_ALPHA * meas
                            slow_bw_ema = (1 - _SLOW_EMA_ALPHA) * slow_bw_ema + _SLOW_EMA_ALPHA * meas
                            # Use the lower of the two EMAs for a conservative estimate
                            safe_bw = min(fast_bw_ema, slow_bw_ema) * _SAFETY_FACTOR
                            sweep_budget = safe_bw * cpr_rtt_floor * _SWEEP_WINDOW / 8
                            if frame_size_history:
                                mean_dl = statistics.mean(frame_size_history)
                                stdev_dl = (statistics.stdev(frame_size_history)
                                            if len(frame_size_history) > 1 else mean_dl)
                                # p95 approximation: absorbs VBR burst frames
                                p95 = mean_dl + _FRAME_SIZE_P95_SIGMAS * stdev_dl
                                frames_per_sweep = max(1, int(sweep_budget / p95))
                            # Opportunistic ramp: in BW-limited regime (overhead >> RTT floor),
                            # each extra frame amortises one CPR wait.  Grow ramp_frames by 1
                            # when the last sweep was clean; shrink when tx >> RTT floor * bloat.
                            if overhead > _RAMP_BLOAT_FACTOR * cpr_rtt_floor:
                                ramp_frames = max(0, ramp_frames - 1)
                            elif frames_per_sweep + ramp_frames < _RAMP_MAX_FRAMES:
                                ramp_frames += 1
                            frames_per_sweep = min(
                                frames_per_sweep + ramp_frames, _RAMP_MAX_FRAMES
                            )
                frames_since_cpr = 0
                cpr_bytes_in_sweep = 0
                cpr_sent_at = None

        # Check terminal size (debounced)
        current_size = app_session.output.get_size()
        if current_size != seen_size:
            seen_size = current_size
            last_resize_time = time.time()

        # Render video
        with timing(video_deltas):
            # Send the frame
            shift = shifting and shifting[-1] > 1 / fps
            if i % frame_advance == 0 and new_frame and screen_ready and not shift:
                new_frame = False
                # Apply resize once terminal size has been stable for 100ms
                if last_resize_time is not None:
                    if time.time() - last_resize_time >= _RESIZE_DEBOUNCE:
                        height, width = seen_size
                        if sextant is None:
                            use_sextant = width < console.WIDTH + 6
                            blit_fn = blit_sextant if use_sextant else blit
                        refx, refy = get_ref(width, height, console, use_sextant)
                        last_resize_time = None
                        video_data = blit_fn(video, None, refx, refy, width - 1, height, color_mode)
                        video_data = b"\033[H\033[2J" + video_data
                        last_frame = video.copy()
                        data_length.append(len(video_data))
                        shown_frames.append(True)
                    else:
                        # Still debouncing - hold off rendering
                        video_data = None
                        data_length.append(0)
                        shown_frames.append(False)
                else:
                    # Normal render
                    video_data = blit_fn(video, last_frame, refx, refy, width - 1, height, color_mode)
                    last_frame = video.copy()
                    data_length.append(len(video_data))
                    shown_frames.append(True)
            # Ignore this video frame
            else:
                video_data = None
                data_length.append(0)
                shown_frames.append(False)

        if video_data:
            frame_size_history.append(len(video_data))

        with timing(sync_deltas):
            # Video sync
            if video_data:
                # Write video frame, might block
                write_bytes(app_session, b"\033[?2026h" + video_data + b"\033[?2026l")
                # NUL inter-frame pacing: pad frame to fill one bandwidth-slot so consecutive
                # frames arrive at the terminal spaced by 1/target_fps seconds.
                if use_cpr_sync and fast_bw_ema > 0 and frame_size_history:
                    bw_bytes_per_sec = fast_bw_ema / 8
                    mean_bytes = statistics.mean(frame_size_history)
                    bw_fps_cap = bw_bytes_per_sec / mean_bytes
                    target_fps_for_pacing = min(fps / frame_advance, bw_fps_cap)
                    slot_bytes = int(bw_bytes_per_sec / target_fps_for_pacing)
                    nul_count = max(0, slot_bytes - len(video_data))
                    if nul_count > 0:
                        write_bytes(app_session, bytes(nul_count))
                        cpr_bytes_in_sweep += nul_count
                    if live_stats is not None:
                        live_stats.nul_pad_last = nul_count
                # Send CPR request
                if use_cpr_sync:
                    frames_since_cpr += 1
                    cpr_bytes_in_sweep += len(video_data)
                    if frames_since_cpr >= frames_per_sweep:
                        app_session.output.ask_for_cpr()
                        screen_ready = False
                        cpr_sent_at = time.perf_counter()
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
            if live_stats is not None:
                live_stats.fps = video_fps
                live_stats.frames_per_sweep = frames_per_sweep
                live_stats.fast_bw_mbps = fast_bw_ema / 1_000_000
                live_stats.slow_bw_mbps = slow_bw_ema / 1_000_000
                if frame_size_history:
                    mean_dl = statistics.mean(frame_size_history)
                    stdev_dl = (statistics.stdev(frame_size_history)
                                if len(frame_size_history) > 1 else mean_dl)
                    live_stats.frame_mean_bytes = int(mean_dl)
                    live_stats.frame_p95_bytes = int(
                        mean_dl + _FRAME_SIZE_P95_SIGMAS * stdev_dl
                    )
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
            app_session.output.set_title(title)
            app_session.output.flush()
