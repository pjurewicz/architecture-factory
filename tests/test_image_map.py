"""image_map: scan pixels <-> drawing points through view_screenshot's overlay_align."""

from __future__ import annotations

import math

import pytest

from engineering.arch.image_map import image_map

# A scan at 0.1 px per mm, turned 30 degrees, its pixel (0, 0) on WCS (1000, 2000).
SCALE, TURN, ORIGIN = 10.0, math.radians(30.0), (1000.0, 2000.0)


def truth(u: float, v: float) -> list[float]:
    """Pixel -> WCS written out independently: v points down, then rotate and shift."""
    x, y = u * SCALE, -v * SCALE
    c, s = math.cos(TURN), math.sin(TURN)
    return [ORIGIN[0] + x * c - y * s, ORIGIN[1] + x * s + y * c]


def pair(u, v, dx=0.0, dy=0.0):
    x, y = truth(u, v)
    return [u, v, x + dx, y + dy]


def test_two_pairs_map_there_and_back_but_say_they_are_unchecked():
    align = [pair(0, 0), pair(400, 300)]
    pixels = [[123.0, 45.0], [700.0, 20.0], [0.0, 999.0]]
    result = image_map(align, pixels=pixels)
    for got, (u, v) in zip(result["to_drawing"], pixels, strict=True):
        assert got == pytest.approx(truth(u, v), abs=1e-3)
    back = image_map(align, points=result["to_drawing"])
    for got, want in zip(back["to_image"], pixels, strict=True):
        assert got == pytest.approx(want, abs=0.01)
    fit = result["fit"]
    assert fit["controlled"] is False and "add a third pair" in fit["note"]
    assert "rms" not in fit and "drawing_segments" not in result
    assert fit["units_per_px"] == pytest.approx(SCALE)
    assert fit["rotation_deg"] == pytest.approx(30.0)


def test_more_pairs_are_fitted_and_report_the_pair_that_is_off():
    # one of four pairs was read 50 mm off along x: the fit spreads it, rms shows it
    align = [pair(0, 0), pair(800, 0), pair(800, 600, dx=50.0), pair(0, 600)]
    fit = image_map(align, pixels=[[0, 0]])["fit"]
    assert fit["controlled"] is True and fit["pairs"] == 4
    assert fit["max"] == max(fit["residuals"])
    assert fit["residuals"].index(fit["max"]) == 2
    rms = math.sqrt(sum(e * e for e in fit["residuals"]) / 4)
    assert fit["rms"] == pytest.approx(rms, abs=2e-3)
    assert 5.0 < fit["rms"] < 50.0
    exact = image_map([pair(0, 0), pair(400, 300), pair(800, 0)], pixels=[[10, 10]])
    assert exact["fit"]["rms"] == pytest.approx(0.0, abs=1e-3)
    assert exact["to_drawing"][0] == pytest.approx(truth(10, 10), abs=1e-3)


def test_measure_gives_lengths_and_angles_of_the_mapped_polyline():
    align = [pair(0, 0), pair(400, 300), pair(800, 0)]
    # a horizontal run of 100 px in the image is 1000 mm at 30 degrees (v is down)
    out = image_map(align, pixels=[[0, 0], [100, 0]], measure=True)
    (seg,) = out["drawing_segments"]
    assert seg["length"] == pytest.approx(1000.0, abs=1e-3)
    assert seg["angle"] == pytest.approx(30.0, abs=1e-3)


def test_refusals_name_their_argument():
    good = [pair(0, 0), pair(400, 300)]
    with pytest.raises(ValueError, match="at least two calibration pairs"):
        image_map([pair(0, 0)], pixels=[[0, 0]])
    with pytest.raises(ValueError, match="must differ"):
        image_map([pair(0, 0), pair(0, 0)], pixels=[[0, 0]])
    with pytest.raises(ValueError, match=r"pixels\[1\]: expected \[a, b\]"):
        image_map(good, pixels=[[0, 0], [1]])
    with pytest.raises(ValueError, match="give pixels"):
        image_map(good)
