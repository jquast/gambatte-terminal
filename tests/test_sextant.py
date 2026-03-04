from __future__ import annotations

import sys
from pathlib import Path
from subprocess import run as subprocess_run

import numpy as np
import pytest

from gambaterm.sextant import (
    SEXTANT,
    SEXTANT_BYTES,
    _display_cache,
    _last_params,
    _hsv_distance,
    _rgb_to_hsv,
    _select_bitonal_pair,
    _visual_pixel_diff,
    blit_sextant,
)

TEST_ROM = Path(__file__).parent / "test_rom.gb"


class TestSextantTable:
    def test_length(self) -> None:
        assert len(SEXTANT) == 64

    def test_all_single_characters(self) -> None:
        for ch in SEXTANT:
            assert len(ch) == 1

    def test_unique_characters(self) -> None:
        assert len(set(SEXTANT)) == 64

    def test_empty_pattern(self) -> None:
        assert SEXTANT[0] == ' '

    def test_full_block(self) -> None:
        assert SEXTANT[63] == '\u2588'

    def test_left_half(self) -> None:
        assert SEXTANT[21] == '\u258c'

    def test_right_half(self) -> None:
        assert SEXTANT[42] == '\u2590'

    def test_sextant_range(self) -> None:
        for i in range(64):
            if i in (0, 21, 42, 63):
                continue
            code = ord(SEXTANT[i])
            assert 0x1FB00 <= code <= 0x1FB3B

    def test_bytes_match_chars(self) -> None:
        for ch, bts in zip(SEXTANT, SEXTANT_BYTES):
            assert bts == ch.encode('utf-8')


class TestRgbToHsv:
    def test_red(self) -> None:
        h, s, v = _rgb_to_hsv(0xFF0000)
        assert h == 0.0
        assert s == 1.0
        assert v == 1.0

    def test_black(self) -> None:
        h, s, v = _rgb_to_hsv(0x000000)
        assert s == 0.0
        assert v == 0.0


class TestHsvDistance:
    def test_same_color_is_zero(self) -> None:
        assert _hsv_distance(0xFF0000, 0xFF0000) == 0.0

    def test_symmetry(self) -> None:
        d1 = _hsv_distance(0xFF0000, 0x00FF00)
        d2 = _hsv_distance(0x00FF00, 0xFF0000)
        assert d1 == d2

    def test_similar_closer_than_different(self) -> None:
        d_similar = _hsv_distance(0xFF0000, 0xFF3300)
        d_different = _hsv_distance(0xFF0000, 0x00FF00)
        assert d_similar < d_different

    def test_black_white_nonzero(self) -> None:
        assert _hsv_distance(0x000000, 0xFFFFFF) > 0


class TestSelectBitonalPair:
    def test_solid_color(self) -> None:
        pixels = [0xFF0000] * 6
        bg, fg, idx = _select_bitonal_pair(pixels)
        assert bg == 0xFF0000
        assert idx == 0

    def test_two_colors(self) -> None:
        red, blue = 0xFF0000, 0x0000FF
        pixels = [red, blue, red, blue, red, blue]
        bg, fg, idx = _select_bitonal_pair(pixels)
        assert {bg, fg} == {red, blue}
        assert 0 < idx < 63

    def test_two_colors_correct_pattern(self) -> None:
        red, blue = 0xFF0000, 0x0000FF
        pixels = [red, red, red, red, red, blue]
        bg, fg, idx = _select_bitonal_pair(pixels)
        assert bg == red
        assert fg == blue
        assert idx == 0b100000

    def test_three_colors(self) -> None:
        pixels = [0xFF0000, 0x00FF00, 0x0000FF, 0xFF0000, 0x00FF00, 0x0000FF]
        bg, fg, idx = _select_bitonal_pair(pixels)
        assert 0 < idx < 63

    def test_all_same_returns_zero_index(self) -> None:
        for color in [0x000000, 0xFFFFFF, 0x123456]:
            _, _, idx = _select_bitonal_pair([color] * 6)
            assert idx == 0

    def test_bg_is_more_frequent(self) -> None:
        red, blue = 0xFF0000, 0x0000FF
        pixels = [red, red, red, red, red, blue]
        bg, fg, idx = _select_bitonal_pair(pixels)
        assert bg == red


class TestVisualPixelDiff:
    def test_identical_outputs(self) -> None:
        assert _visual_pixel_diff(0xAA, 0xBB, 0b101010, 0xAA, 0xBB, 0b101010) == 0

    def test_bg_fg_swap_with_complement(self) -> None:
        assert _visual_pixel_diff(0xAA, 0xBB, 0b101010, 0xBB, 0xAA, 0b010101) == 0

    def test_one_pixel_different(self) -> None:
        assert _visual_pixel_diff(0xAA, 0xBB, 0b000000, 0xAA, 0xBB, 0b000001) == 1

    def test_all_pixels_different(self) -> None:
        assert _visual_pixel_diff(0xAA, 0xBB, 0b000000, 0xBB, 0xAA, 0b000000) == 6

    def test_pattern_one_bit_same_colors(self) -> None:
        assert _visual_pixel_diff(0xAA, 0xBB, 0b111111, 0xAA, 0xBB, 0b111110) == 1

    def test_all_bg_to_all_fg(self) -> None:
        assert _visual_pixel_diff(0xAA, 0xBB, 0b000000, 0xAA, 0xBB, 0b111111) == 6

    def test_same_color_bg_fg(self) -> None:
        assert _visual_pixel_diff(0xAA, 0xAA, 0b000000, 0xAA, 0xAA, 0b111111) == 0


class TestHysteresis:
    def setup_method(self) -> None:
        _display_cache.clear()
        _last_params.clear()

    def test_same_frame_suppressed(self) -> None:
        image = np.full((6, 4), 0x00FF0000, np.uint32)
        image[0, 0] = 0x000000FF
        blit_sextant(image, None, 1, 1, 2, 2, 4)
        result = blit_sextant(image, image, 1, 1, 2, 2, 4)
        assert result == b'\033[1;1H\033[0m'

    def test_one_pixel_change_renders_immediately(self) -> None:
        image1 = np.full((6, 4), 0x00FF0000, np.uint32)
        image1[0, 0] = 0x000000FF
        blit_sextant(image1, None, 1, 1, 2, 2, 4)

        image2 = image1.copy()
        image2[1, 0] = 0x000000FF
        result = blit_sextant(image2, image1, 1, 1, 2, 2, 4)
        assert len(result) > len(b'\033[1;1H\033[0m')

    def test_two_pixel_change_renders(self) -> None:
        image1 = np.full((6, 4), 0x00FF0000, np.uint32)
        blit_sextant(image1, None, 1, 1, 2, 2, 4)

        image2 = image1.copy()
        image2[0, 0] = 0x000000FF
        image2[0, 1] = 0x000000FF
        result = blit_sextant(image2, image1, 1, 1, 2, 2, 4)
        assert len(result) > len(b'\033[1;1H\033[0m')

    def test_resize_clears_cache(self) -> None:
        image = np.full((6, 4), 0x00FF0000, np.uint32)
        blit_sextant(image, None, 1, 1, 3, 3, 4)
        result = blit_sextant(image, image, 1, 2, 3, 3, 4)
        assert len(result) > len(b'\033[1;2H\033[0m')

    def test_last_none_clears_cache(self) -> None:
        image = np.full((6, 4), 0x00FF0000, np.uint32)
        blit_sextant(image, None, 1, 1, 2, 2, 4)
        assert len(_display_cache) > 0
        blit_sextant(image, None, 1, 1, 2, 2, 4)
        first_render = blit_sextant(image, None, 1, 1, 2, 2, 4)
        assert len(first_render) > len(b'\033[1;1H\033[0m')


class TestBlitSextant:
    def setup_method(self) -> None:
        _display_cache.clear()
        _last_params.clear()

    def test_solid_frame_returns_bytes(self) -> None:
        image = np.full((144, 160), 0x00FF0000, np.uint32)
        result = blit_sextant(image, None, 1, 1, 80, 48, 4)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_delta_unchanged_is_empty(self) -> None:
        image = np.full((144, 160), 0x00FF0000, np.uint32)
        blit_sextant(image, None, 1, 1, 80, 48, 4)
        result = blit_sextant(image, image, 1, 1, 80, 48, 4)
        assert result == b'\033[1;1H\033[0m'

    def test_delta_changed_cell(self) -> None:
        image1 = np.full((144, 160), 0x00FF0000, np.uint32)
        blit_sextant(image1, None, 1, 1, 80, 48, 4)
        image2 = image1.copy()
        image2[0, 0] = 0x0000FF00
        image2[0, 1] = 0x0000FF00
        result = blit_sextant(image2, image1, 1, 1, 80, 48, 4)
        assert len(result) > len(b'\033[1;1H\033[0m')

    def test_contains_reset(self) -> None:
        image = np.full((144, 160), 0x00FF0000, np.uint32)
        result = blit_sextant(image, None, 1, 1, 80, 48, 4)
        assert result.endswith(b'\033[0m')

    def test_color_mode_3(self) -> None:
        image = np.full((144, 160), 0x00FF0000, np.uint32)
        result = blit_sextant(image, None, 1, 1, 80, 48, 3)
        assert b'\033[48;5;' in result

    def test_color_mode_2(self) -> None:
        image = np.full((144, 160), 0x00FF0000, np.uint32)
        result = blit_sextant(image, None, 1, 1, 80, 48, 2)
        assert b'\033[' in result

    def test_color_mode_1(self) -> None:
        image = np.full((144, 160), 0x00FF0000, np.uint32)
        result = blit_sextant(image, None, 1, 1, 80, 48, 1)
        assert b'\033[' in result

    def test_clipping(self) -> None:
        image = np.full((144, 160), 0x00FF0000, np.uint32)
        result = blit_sextant(image, None, 1, 1, 10, 10, 4)
        assert isinstance(result, bytes)

    def test_zero_area(self) -> None:
        image = np.full((144, 160), 0x00FF0000, np.uint32)
        result = blit_sextant(image, None, 50, 1, 10, 10, 4)
        assert result == b''

    def test_two_color_frame(self) -> None:
        image = np.full((144, 160), 0x00FF0000, np.uint32)
        image[::2, :] = 0x000000FF
        result = blit_sextant(image, None, 1, 1, 80, 48, 4)
        assert isinstance(result, bytes)
        assert len(result) > 0


@pytest.mark.parametrize(
    "interactive", (False, True), ids=("non-interactive", "interactive")
)
def test_gambaterm_sextant(interactive: bool) -> None:
    assert TEST_ROM.exists()
    command = (
        f"gambaterm {TEST_ROM} --break-after 10 --input-file /dev/null"
        " --disable-audio --color-mode 4 --sextant"
    )
    result = subprocess_run(
        f"script -e -q -c '{command}' /dev/null" if interactive else command,
        shell=True,
        check=True,
        text=True,
        capture_output=True,
    )
    if interactive:
        assert result.stderr == ""
        assert "| test_rom.gb |" in result.stdout
    else:
        assert result.stderr == "Warning: Input is not a terminal (fd=0).\n"
    if sys.platform == "linux":
        decoded = result.stdout
        has_sextant = any(0x1FB00 <= ord(c) <= 0x1FB3B for c in decoded)
        has_block = '\u2588' in decoded or '\u258c' in decoded or '\u2590' in decoded
        assert has_sextant or has_block
