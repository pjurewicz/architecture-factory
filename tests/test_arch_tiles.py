"""The laying engine of arch_floor_zone: the cut pattern and its takeoff, hand-derived."""

from __future__ import annotations

import math

import pytest

from engineering.arch.tiles import layout

ROOM = [[0, 0], [4000, 0], [4000, 3000], [0, 3000]]


def test_grid_without_joint_counts_full_and_cut_tiles():
    # 600 x 300 from (0, 0): 6 full columns + a 400 mm strip, 10 rows exactly.
    t = layout(ROOM, length=600, width=300, joint=0)["takeoff"]
    # The 200 mm offcut of a 400 mm strip piece fits no other piece: no reuse.
    assert (t["full"], t["cut"], t["tiles"], t["tiles_no_reuse"]) == (60, 10, 70, 70)
    assert t["floor_area_m2"] == 12.0 and t["tiled_area_m2"] == 12.0
    assert t["order_tiles"] == 77  # ceil(70 x 1.10)


def test_running_bond_reuses_row_end_offcuts():
    # Rows 0, 2, ..: 6 full + one 400 cut; rows 1, 3, ..: a 300 cut, 6 full, a 100 cut.
    # Five 400 cuts leave 197 mm and five 300 cuts 297 mm (3 mm kerf): the five
    # 100 mm pieces come out of those offcuts, so the cuts take 10 tiles, not 15.
    t = layout(ROOM, length=600, width=300, joint=0, pattern="running_bond")["takeoff"]
    assert (t["full"], t["cut"], t["cut_tiles"], t["tiles"]) == (60, 15, 10, 70)
    assert t["tiles_no_reuse"] == 75 and t["order_tiles"] == 77
    off = layout(ROOM, length=600, width=300, joint=0, pattern="running_bond", offcut_reuse=False)
    assert (off["takeoff"]["tiles"], off["takeoff"]["order_tiles"]) == (75, 83)


def test_kerf_can_make_an_offcut_too_short():
    # 1000 x 600 room, 600 x 300 tiles, half bond -> row 0: 600 full + a 400 cut,
    # row 1: a 300 cut + 600 full + a 100 cut. The 100 piece fits the 600 - 400 -
    # kerf offcut at kerf 0 (2 tiles for the cuts); at kerf 250 the offcuts are
    # -50 and 50 mm and it needs its own tile (3).
    room = [[0, 0], [1000, 0], [1000, 600], [0, 600]]
    args = {"length": 600, "width": 300, "joint": 0, "pattern": "running_bond"}
    assert layout(room, kerf=0, **args)["takeoff"]["cut_tiles"] == 2
    assert layout(room, kerf=250, **args)["takeoff"]["cut_tiles"] == 3


def test_a_cutout_is_left_untiled_and_its_area_taken_off():
    bath = {"points": [[0, 0], [900, 0], [900, 1700], [0, 1700]], "label": "bath"}
    r = layout(ROOM, length=600, width=300, joint=3, cutouts=[bath])
    assert r["takeoff"]["floor_area_m2"] == pytest.approx(12.0 - 0.9 * 1.7)
    assert r["cutout_labels"] == ["bath"]
    for piece in r["pieces"]:  # no piece reaches into the bath
        xs = [p[0] for loop in piece.loops for p in loop]
        ys = [p[1] for loop in piece.loops for p in loop]
        assert not (max(xs) <= 900 + 1e-6 and max(ys) <= 1700 + 1e-6)


def test_herringbone_at_45_degrees_on_an_l_room_tiles_the_exact_floor_without_joints():
    # Joint 0: the pieces must sum to the region exactly (no overlap, no gap).
    floor = [[0, 0], [5000, 0], [5000, 2000], [2000, 2000], [2000, 4000], [0, 4000]]
    notch = {"points": [[3000, 0], [4000, 0], [4000, 1500], [3500, 1500], [3500, 500], [3000, 500]]}
    r = layout(
        floor,
        length=600,
        width=100,
        joint=0,
        pattern="herringbone",
        angle_deg=45,
        start=(37, 11),
        cutouts=[notch],
    )
    expected = (5000 * 2000 + 2000 * 2000 - (1000 * 500 + 500 * 1000)) / 1e6
    assert r["takeoff"]["floor_area_m2"] == pytest.approx(expected)
    assert sum(p.area for p in r["pieces"]) / 1e6 == pytest.approx(expected, rel=1e-9)
    t = r["takeoff"]
    # reuse never orders less material than the pieces cover, nor more than no reuse
    assert t["tiles"] * 600 * 100 / 1e6 >= expected
    assert t["full"] < t["tiles"] <= t["tiles_no_reuse"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"pattern": "basket"}, "pattern"),
        ({"pattern": "herringbone", "width": 600}, "longer than it is wide"),
        ({"joint": -1}, "joint"),
        ({"kerf": -1}, "kerf"),
        ({"bond": 1.0}, "bond"),
        ({"outline": [[0, 0], [30, 10], [30, 0], [0, 10], [-5, 5]]}, "cross"),
        ({"length": 1, "width": 1}, "tiles would cover"),
        ({"pattern": "chevron", "cut_deg": 90}, "cut_deg"),
    ],
)
def test_refusals_name_their_cause(kwargs, match):
    args = {"outline": ROOM, "length": 600, "width": 300} | kwargs
    with pytest.raises(ValueError, match=match):
        layout(**args)


def _key(points):
    return tuple(sorted((round(x, 3), round(y, 3)) for x, y in points))  # 1 µm


def _repeats(pieces, vector) -> int:
    """How many full pieces, moved by ``vector``, land exactly on another full piece."""
    full = [p.corners for p in pieces if p.full]
    keys = {_key(c) for c in full}
    dx, dy = vector
    return sum(_key((x + dx, y + dy) for x, y in c) in keys for c in full)


def _direction(a, b) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0


BIG = [[0, 0], [6000, 0], [6000, 5000], [0, 5000]]


def test_chevron_bands_period_and_cut_angle_from_the_plank_size():
    # The source's French herringbone: joints every 253.7 mm along the axis, a 45°
    # zigzag. A 358.79 mm long edge x sin 45° = a 253.70 mm band; 90 mm planks
    # repeat every 90 / sin 45° = 127.28 mm along the axis.
    r = layout(BIG, length=358.79, width=90, joint=0, pattern="chevron", start=(0, 0))
    full = [p for p in r["pieces"] if p.full]
    band = 358.79 * math.sin(math.radians(45))
    assert band == pytest.approx(253.7, abs=0.01)
    for p in full:  # ends on the band edges, long sides at +45 (A) / -45 (B)
        ys = sorted({round(y, 6) for _x, y in p.corners})
        assert len(ys) == 2 and ys[1] - ys[0] == pytest.approx(band)
        assert ys[0] / band == pytest.approx(round(ys[0] / band))
        assert _direction(p.corners[1], p.corners[2]) == pytest.approx(
            45.0 if p.hand == "A" else 135.0
        )
        assert p.area == pytest.approx(358.79 * 90)
    step = 90 / math.sin(math.radians(45))
    assert _repeats(full, (step, 0)) > 0.8 * len(full)
    assert _repeats(full, (step * 0.5, 0)) == 0
    # two bands (one A, one B) repeat across the axis, one band does not
    assert _repeats(full, (0, 2 * band)) > 0.7 * len(full)
    assert _repeats(full, (0, band)) == 0
    t = r["takeoff"]
    assert t["by_hand"]["A"] + t["by_hand"]["B"] == t["tiles"] == t["tiles_no_reuse"]
    assert t["offcut_reuse"] is False and t["tiled_area_m2"] == pytest.approx(30.0)


def test_chevron_turned_and_started_cuts_the_notched_room_exactly():
    floor = [[0, 0], [5000, 0], [5000, 2000], [2000, 2000], [2000, 4000], [0, 4000]]
    notch = {"points": [[3000, 0], [4000, 0], [4000, 1500], [3500, 1500], [3500, 500], [3000, 500]]}
    start = (1234.5, 678.9)
    r = layout(
        floor, length=600, width=100, joint=0, pattern="chevron", angle_deg=30,
        cut_deg=60, start=start, cutouts=[notch],
    )  # fmt: skip
    expected = (5000 * 2000 + 2000 * 2000 - (1000 * 500 + 500 * 1000)) / 1e6
    assert sum(p.area for p in r["pieces"]) / 1e6 == pytest.approx(expected, rel=1e-9)
    assert any(math.dist(c, start) < 1e-6 for p in r["pieces"] if p.full for c in p.corners)
    for p in (p for p in r["pieces"] if p.full):  # ends along the 30° axis, sides at 30 ± 60
        assert _direction(p.corners[0], p.corners[1]) == pytest.approx(30.0)
        assert _direction(p.corners[1], p.corners[2]) == pytest.approx(
            90.0 if p.hand == "A" else 150.0
        )
    for p in r["pieces"]:  # nothing in the notch
        assert all(
            not (3000 + 1e-6 < x < 4000 - 1e-6 and y < 500 - 1e-6)
            for loop in p.loops
            for x, y in loop
        )


def test_herringbone_axis_follows_angle_and_repeats_every_width_root_two():
    r = layout(BIG, length=600, width=100, joint=2, pattern="herringbone", angle_deg=0)
    full = [p for p in r["pieces"] if p.full]
    # angle 0: the zigzag runs along x, its planks along +45 and -45 degrees
    long_edges = {
        round(_direction(*max(((c[0], c[1]), (c[1], c[2])), key=lambda e: math.dist(*e))))
        for c in (p.corners for p in full)
    }
    assert long_edges == {45, 135}
    period = 102 * math.sqrt(2)
    assert _repeats(full, (period, 0)) > 0.8 * len(full)
    assert _repeats(full, (0, period)) == 0
    assert _repeats(full, (0, 602 * math.sqrt(2))) > 0.6 * len(full)


def test_grid_and_offset_tiles_start_at_the_start_point():
    r = layout(BIG, length=600, width=300, joint=3, start=(100, 50), angle_deg=0)
    full = [p for p in r["pieces"] if p.full]
    assert any(
        _key(p.corners) == _key([(100, 50), (700, 50), (700, 350), (100, 350)]) for p in full
    )
    assert _repeats(full, (603, 0)) > 0.8 * len(full) and _repeats(full, (0, 303)) > 0.8 * len(full)
    third = layout(BIG, length=600, width=300, joint=0, pattern="running_bond", bond=1 / 3)
    full = [p for p in third["pieces"] if p.full]
    assert _repeats(full, (200, 300)) > 0.8 * len(full)  # each row a third further
    assert _repeats(full, (300, 300)) == 0
