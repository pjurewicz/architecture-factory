"""Door and window symbols: hinge, leaf, swing arc, door frame and glazing.

The wall runs along +X from the origin, 200 mm thick and centred, so its left
face is y = +100 and its right face y = -100. The door is 900 wide at offset
1000: its jambs are x = 1000 and x = 1900. For each swing/hand pair the hinge,
the open leaf and the arc angles are written out below by hand. The hinge
tables are the unframed door (``frame=False``, hinged on the jamb itself);
the frame section at the end adds the frame to the same walls.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from engineering.arch.model import Opening, Wall
from engineering.arch.openings import opening_prims
from engineering.mech.primitives import Arc, Line, Poly

WALL = Wall(id="a", axis=((0.0, 0.0), (5000.0, 0.0)), thickness=200.0)


def door(swing: str, hand: str) -> Opening:
    return Opening(
        id="d1",
        wall="a",
        kind="door",
        offset=1000.0,
        width=900.0,
        swing=swing,
        hand=hand,
        frame=False,
    )


def pt(p) -> tuple[float, float]:
    return (round(p[0], 9) + 0.0, round(p[1], 9) + 0.0)


@pytest.mark.parametrize(
    ("swing", "hand", "hinge", "leaf_end", "start_deg", "end_deg"),
    [
        # in = the axis's left (+Y). Facing the wall from +Y, the viewer's left is +X.
        ("in", "left", (1900.0, 100.0), (1900.0, 1000.0), 90.0, 180.0),
        ("in", "right", (1000.0, 100.0), (1000.0, 1000.0), 0.0, 90.0),
        # out = the axis's right (-Y). Facing the wall from -Y, the viewer's left is -X.
        ("out", "left", (1000.0, -100.0), (1000.0, -1000.0), 270.0, 0.0),
        ("out", "right", (1900.0, -100.0), (1900.0, -1000.0), 180.0, 270.0),
    ],
)
def test_a_door_hinges_on_its_hand_and_swings_a_quarter_arc_to_its_side(
    swing, hand, hinge, leaf_end, start_deg, end_deg
):
    leaf, arc = opening_prims(WALL, door(swing, hand))
    assert isinstance(leaf, Line) and leaf.role == "door"
    assert (pt(leaf.p1), pt(leaf.p2)) == (hinge, leaf_end)
    assert isinstance(arc, Arc) and arc.role == "door"
    assert pt(arc.center) == hinge
    assert arc.radius == 900.0
    assert (round(arc.start_deg, 9), round(arc.end_deg, 9)) == (start_deg, end_deg)


def test_the_leaf_is_as_long_as_the_opening_is_wide_and_square_to_the_wall():
    leaf, _arc = opening_prims(WALL, door("in", "left"))
    dx = leaf.p2[0] - leaf.p1[0]
    dy = leaf.p2[1] - leaf.p1[1]
    assert (dx, dy) == (0.0, 900.0)


def test_a_door_on_a_second_segment_is_placed_along_the_whole_axis():
    bent = Wall(id="b", axis=((0.0, 0.0), (4000.0, 0.0), (4000.0, 3000.0)), thickness=200.0)
    op = Opening(
        id="d2",
        wall="b",
        kind="door",
        offset=4500.0,
        width=800.0,
        swing="in",
        hand="right",
        frame=False,
    )
    leaf, arc = opening_prims(bent, op)
    # segment 1 runs +Y from (4000, 0); its left normal is -X, so "in" is x < 4000
    # and its left face is x = 3900. Facing +X from the left side, the viewer's
    # left is +Y, so a right hand hinges at the near jamb, y = 500.
    assert pt(leaf.p1) == (3900.0, 500.0)
    assert pt(leaf.p2) == (3100.0, 500.0)
    assert (round(arc.start_deg, 9), round(arc.end_deg, 9)) == (90.0, 180.0)


def test_a_window_is_two_frame_lines_and_two_glazing_lines():
    window = Opening(id="w1", wall="a", kind="window", offset=2500.0, width=1200.0)
    prims = opening_prims(WALL, window)
    assert len(prims) == 4
    assert all(isinstance(p, Line) and p.role == "window" for p in prims)
    assert [(pt(p.p1), pt(p.p2)) for p in prims] == [
        ((2500.0, 100.0), (3700.0, 100.0)),  # frame, left face
        ((2500.0, -100.0), (3700.0, -100.0)),  # frame, right face
        ((2500.0, round(-100.0 + 200.0 / 3.0, 9)), (3700.0, round(-100.0 + 200.0 / 3.0, 9))),
        ((2500.0, round(-100.0 + 400.0 / 3.0, 9)), (3700.0, round(-100.0 + 400.0 / 3.0, 9))),
    ]


def test_an_opening_of_another_wall_is_refused():
    with pytest.raises(ValueError, match="hosted in wall 'z'"):
        opening_prims(WALL, Opening(id="d1", wall="z", kind="door", offset=0.0, width=900.0))


def test_an_unknown_swing_is_refused_with_the_list():
    with pytest.raises(ValueError, match="swings are in, out"):
        opening_prims(WALL, door("sideways", "left"))


@pytest.mark.parametrize("kind", ["door", "double_door"])
@pytest.mark.parametrize("swing", ["in", "out"])
@pytest.mark.parametrize("hand", ["left", "right"])
def test_a_door_on_a_t_stem_typed_in_whole_mm_hinges_exactly_on_its_jamb(kind, swing, hand):
    # A host at -30 degrees and a stem square to it, both typed in whole mm: the
    # stem's start lies 0.2 mm off the host axis and the engine snaps it onto
    # it, turning the stem a little. The hinge must be the jamb's own vertex,
    # not a point recomputed from the typed axis.
    from engineering.arch.walls import wall_geometry

    host = Wall(id="h", axis=((0.0, 0.0), (6062.0, -3500.0)), thickness=250.0)
    stem = Wall(id="s", axis=((953.0, -550.0), (2453.0, 2048.0)), thickness=100.0)
    op = Opening(id="d", wall="s", kind=kind, offset=300.0, width=800.0, swing=swing, hand=hand)
    geometry = wall_geometry([host, stem], [op])
    (cut,) = geometry.cuts
    jamb_ends = {p for line in geometry.jambs for p in (line.p1, line.p2)}
    prims = opening_prims(stem, op, cut)
    frames = [p for p in prims if isinstance(p, Poly)]
    vertices = {v for f in frames for v in f.points}
    leaves = [p for p in prims if not isinstance(p, Poly)]
    assert len(frames) == 2
    for frame in frames:  # each frame stands on its jamb's own two vertices
        assert frame.points[0] in jamb_ends and frame.points[-1] in jamb_ends
    for leaf, arc in zip(leaves[::2], leaves[1::2], strict=True):
        assert leaf.p1 in vertices  # the hinge is the frame's own vertex: 0.0 mm
        assert arc.center == leaf.p1


# -- T-08: hinge start | end and swing_towards [x, y] ---------------------------
#
# One physical door per case: 900 wide, its jambs J1 and J2 at 1000 and 1900 mm
# from the wall's first typed point P0 along the wall, 200 mm thick, centred.
# The axis is typed both ways round; reversed, the offset is re-measured from
# the other end (5000 - 1900 = 3100), so J2 becomes the start jamb. The side
# is named by a WCS point 1000 mm off the door's middle. Every coordinate below
# is worked out by hand: hinge = jamb + 100 * n on the opening side, leaf end =
# jamb + 1000 * n, n the forward axis's left normal.

S3 = 3.0**0.5

# wall: (forward axis, toward-left point, toward-right point,
#        {(jamb, side): (hinge, leaf end)})
T08_WALLS = {
    "horizontal": (
        ((0.0, 0.0), (5000.0, 0.0)),
        (1450.0, 2000.0),
        (1450.0, -2000.0),
        {
            ("J1", "L"): ((1000.0, 100.0), (1000.0, 1000.0)),
            ("J2", "L"): ((1900.0, 100.0), (1900.0, 1000.0)),
            ("J1", "R"): ((1000.0, -100.0), (1000.0, -1000.0)),
            ("J2", "R"): ((1900.0, -100.0), (1900.0, -1000.0)),
        },
    ),
    "vertical": (
        ((0.0, 0.0), (0.0, 5000.0)),
        (-2000.0, 1450.0),
        (2000.0, 1450.0),
        {
            ("J1", "L"): ((-100.0, 1000.0), (-1000.0, 1000.0)),
            ("J2", "L"): ((-100.0, 1900.0), (-1000.0, 1900.0)),
            ("J1", "R"): ((100.0, 1000.0), (1000.0, 1000.0)),
            ("J2", "R"): ((100.0, 1900.0), (1000.0, 1900.0)),
        },
    ),
    # u = (sqrt3/2, -1/2), n = (1/2, sqrt3/2); J1 = (500 sqrt3, -500), J2 = (950 sqrt3, -950)
    "minus_30": (
        ((0.0, 0.0), (2500.0 * S3, -2500.0)),
        (725.0 * S3 + 500.0, -725.0 + 500.0 * S3),
        (725.0 * S3 - 500.0, -725.0 - 500.0 * S3),
        {
            ("J1", "L"): ((500 * S3 + 50, -500 + 50 * S3), (500 * S3 + 500, -500 + 500 * S3)),
            ("J2", "L"): ((950 * S3 + 50, -950 + 50 * S3), (950 * S3 + 500, -950 + 500 * S3)),
            ("J1", "R"): ((500 * S3 - 50, -500 - 50 * S3), (500 * S3 - 500, -500 - 500 * S3)),
            ("J2", "R"): ((950 * S3 - 50, -950 - 50 * S3), (950 * S3 - 500, -950 - 500 * S3)),
        },
    ),
}
#: Forward, J1 is at offset 1000 (start); reversed, J2 is at offset 3100 (start).
T08_HINGE = {
    (False, "J1"): "start",
    (False, "J2"): "end",
    (True, "J1"): "end",
    (True, "J2"): "start",
}


def t08_door(wall_name: str, reverse: bool, jamb: str, side: str, frame: bool = False):
    import math

    from engineering.arch.model import opening_from_dict, wall_from_dict

    axis, left_pt, right_pt, _table = T08_WALLS[wall_name]
    axis = axis[::-1] if reverse else axis
    wall = wall_from_dict({"id": "w", "axis": [list(p) for p in axis], "thickness": 200})
    offset = 5000.0 - 1900.0 if reverse else 1000.0
    assert math.isclose(math.dist(*axis), 5000.0)
    op = opening_from_dict(
        {
            "id": "d",
            "wall": "w",
            "kind": "door",
            "offset": offset,
            "width": 900,
            "hinge": T08_HINGE[(reverse, jamb)],
            "swing_towards": list(left_pt if side == "L" else right_pt),
            "frame": frame,
        }
    )
    return wall, op


def close(p, q) -> bool:
    return abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9


def arc_point(arc, deg):
    import math

    a = math.radians(deg)
    return (arc.center[0] + arc.radius * math.cos(a), arc.center[1] + arc.radius * math.sin(a))


@pytest.mark.parametrize("wall_name", list(T08_WALLS))
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
@pytest.mark.parametrize("jamb", ["J1", "J2"])
@pytest.mark.parametrize("side", ["L", "R"])
def test_hinge_and_swing_towards_place_one_physical_door_whatever_the_axis_order(
    wall_name, reverse, jamb, side
):
    wall, op = t08_door(wall_name, reverse, jamb, side)
    table = T08_WALLS[wall_name][3]
    hinge, leaf_end = table[(jamb, side)]
    closed_end = table[("J2" if jamb == "J1" else "J1", side)][0]  # the other jamb, same face
    leaf, arc = opening_prims(wall, op)
    assert close(leaf.p1, hinge) and close(leaf.p2, leaf_end)
    assert close(arc.center, hinge) and arc.radius == 900.0
    # a counter-clockwise quarter from one leaf position to the other
    assert abs(((arc.end_deg - arc.start_deg) % 360.0) - 90.0) < 1e-9
    ends = (arc_point(arc, arc.start_deg), arc_point(arc, arc.end_deg))
    assert (close(ends[0], closed_end) and close(ends[1], leaf_end)) or (
        close(ends[0], leaf_end) and close(ends[1], closed_end)
    )


@pytest.mark.parametrize("wall_name", list(T08_WALLS))
def test_the_same_hinge_on_a_reversed_axis_is_the_other_jamb_but_the_same_side(wall_name):
    # hinge names a jamb by the axis; swing_towards names a side in WCS.
    table = T08_WALLS[wall_name][3]
    for side in ("L", "R"):
        _wall, op = t08_door(wall_name, False, "J1", side)  # forward: hinge "start"
        wall_r, op_r = t08_door(wall_name, True, "J2", side)  # reversed: also "start"
        assert op.hinge == op_r.hinge == "start"
        leaf, _arc = opening_prims(wall_r, op_r)
        assert close(leaf.p1, table[("J2", side)][0])


def test_a_resolved_door_carries_swing_hand_and_hinge_that_agree():
    from engineering.arch.openings import resolve_door

    # Forward horizontal, hinge on J2 (end), opening to +Y (the axis's left): the
    # old pair is swing "in", hand "left" - the first row of the table at the top.
    wall, op = t08_door("horizontal", False, "J2", "L")
    resolved = resolve_door(wall, op)
    assert (resolved.swing, resolved.hand, resolved.hinge, resolved.swing_towards) == (
        "in",
        "left",
        "end",
        None,
    )
    # Reversed, the same door is "out" (now the axis's right), hinge "start"; the
    # person facing it from +Y still has the hinge on their left.
    wall, op = t08_door("horizontal", True, "J2", "L")
    resolved = resolve_door(wall, op)
    assert (resolved.swing, resolved.hand, resolved.hinge) == ("out", "left", "start")
    # hand alone still derives the hinge, and the record's pair is stable
    old = resolve_door(WALL, door("out", "right"))
    assert old.hinge == "end" and resolve_door(WALL, old) == old


@pytest.mark.parametrize("point", [(1450.0, 0.0), (1450.0, 60.0), (1450.0, -100.0)])
def test_a_swing_towards_point_on_or_inside_the_wall_is_refused_by_name(point):
    from engineering.arch.model import OpeningArgsError
    from engineering.errors import classify

    op = Opening(id="d1", wall="a", kind="door", offset=1000.0, width=900.0, swing_towards=point)
    with pytest.raises(OpeningArgsError, match=r"opening d1\.swing_towards: .*not clear") as info:
        opening_prims(WALL, op)
    assert classify(info.value)["kind"] == "invalid_args"


def test_swing_towards_on_a_left_justified_wall_is_judged_beyond_its_faces():
    from engineering.arch.model import OpeningArgsError
    from engineering.arch.openings import resolve_door

    # faces at y = 0 (the axis) and y = +200
    wall = Wall(id="a", axis=((0.0, 0.0), (5000.0, 0.0)), thickness=200.0, justification="left")
    at = {"id": "d1", "wall": "a", "kind": "door", "offset": 1000.0, "width": 900.0}
    assert resolve_door(wall, Opening(**at, swing_towards=(1450.0, -5.0))).swing == "out"
    assert resolve_door(wall, Opening(**at, swing_towards=(1450.0, 205.0))).swing == "in"
    for y in (0.0, 100.0, 200.0):
        with pytest.raises(OpeningArgsError, match="not clear of its faces"):
            resolve_door(wall, Opening(**at, swing_towards=(1450.0, y)))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"swing": "in", "swing_towards": [0, 500]}, "swing and swing_towards"),
        ({"hand": "left", "hinge": "start"}, "hinge and hand"),
        ({"kind": "double_door", "hinge": "start"}, r"double_door has no single hinge"),
        ({"kind": "garage_door", "hinge": "end"}, r"garage_door has no single hinge"),
        ({"kind": "sliding_door", "hinge": "end"}, r"sliding_door has no single hinge"),
        ({"kind": "sliding_door", "swing_towards": [0, 500]}, "sliding_door opens to no side"),
        ({"kind": "cased_opening", "swing_towards": [0, 500]}, "cased_opening opens to no"),
        ({"kind": "window", "swing_towards": [0, 500]}, "window opens to no side"),
        ({"hinge": "middle"}, r"hinge: 'middle' is not one of start, end"),
    ],
)
def test_hinge_and_swing_towards_are_refused_where_they_mean_nothing(fields, message):
    from engineering.arch.model import OpeningArgsError, opening_from_dict

    raw = {"id": "d", "wall": "a", "kind": "door", "offset": 1000, "width": 900, **fields}
    with pytest.raises(ValueError, match=message) as info:
        opening_from_dict(raw, "openings[0]")
    if "is not one of" not in message:
        assert isinstance(info.value, OpeningArgsError)


def test_a_double_and_a_garage_door_open_to_the_swing_towards_side():
    # -Y side: the right face, y = -100
    towards = (1450.0, -3000.0)
    double = Opening(
        id="d",
        wall="a",
        kind="double_door",
        offset=1000.0,
        width=900.0,
        swing_towards=towards,
        frame=False,
    )
    leaf_a, _, leaf_b, _ = opening_prims(WALL, double)
    assert (pt(leaf_a.p1), pt(leaf_a.p2)) == ((1000.0, -100.0), (1000.0, -550.0))
    assert (pt(leaf_b.p1), pt(leaf_b.p2)) == ((1900.0, -100.0), (1900.0, -550.0))
    garage = Opening(
        id="g",
        wall="a",
        kind="garage_door",
        offset=1000.0,
        width=900.0,
        height=2000.0,
        swing_towards=towards,
    )
    panel, side_a, back, _side_b = opening_prims(WALL, garage)
    assert (pt(panel.p1), pt(panel.p2)) == ((1000.0, -100.0), (1900.0, -100.0))
    assert (pt(back.p1), pt(back.p2)) == ((1000.0, -2100.0), (1900.0, -2100.0))


@pytest.mark.parametrize("hinge", ["start", "end"])
@pytest.mark.parametrize("towards", [(437.0, 556.0), (2169.0, -444.0)], ids=["left", "right"])
def test_a_hinge_by_name_on_a_t_stem_lies_exactly_on_its_jamb(hinge, towards):
    # The T-01b stem (axis direction ~ (0.5, 0.866)); the points lie 1000 mm
    # either side of the door's middle (1303, 56).
    from engineering.arch.walls import wall_geometry

    host = Wall(id="h", axis=((0.0, 0.0), (6062.0, -3500.0)), thickness=250.0)
    stem = Wall(id="s", axis=((953.0, -550.0), (2453.0, 2048.0)), thickness=100.0)
    op = Opening(
        id="d",
        wall="s",
        kind="door",
        offset=300.0,
        width=800.0,
        hinge=hinge,
        swing_towards=towards,
        frame=False,
    )
    geometry = wall_geometry([host, stem], [op])
    (cut,) = geometry.cuts
    jamb_ends = {p for line in geometry.jambs for p in (line.p1, line.p2)}
    leaf, arc = opening_prims(stem, op, cut)
    assert leaf.p1 in jamb_ends and arc.center == leaf.p1
    # start is the jamb at offset 300 along the stem, end the one at 1100
    along = ((leaf.p1[0] - 953.0) * 1500.0 + (leaf.p1[1] + 550.0) * 2598.0) / 3000.0
    assert abs(along - (300.0 if hinge == "start" else 1100.0)) < 1.0


# -- door frames (ościeżnice) -------------------------------------------------------
#
# The same three walls and the same 900 mm opening, now framed (the default):
# a 50 mm frame in each jamb across the full 200 mm, its 15 x 40 mm rebate on
# the swing-side face. The hinge is the rebate's vertex on that face, 35 mm in
# from the jamb; the leaf is 900 - 2 x 35 = 830 long and the clear passage
# between the stops 900 - 2 x 50 = 800. In the wall's own frame - a along the
# axis from the jamb into the gap, o off the axis on the left normal - the
# J1 frame on the left face is (0, 100) (35, 100) (35, 60) (50, 60) (50, -100)
# (0, -100), and every other jamb and side mirrors it.

#: wall -> (J1, u, n): the jamb at offset 1000, the axis direction, its left normal
FRAME_WALLS = {
    "horizontal": ((1000.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    "vertical": ((0.0, 1000.0), (0.0, 1.0), (-1.0, 0.0)),
    "minus_30": ((500.0 * S3, -500.0), (S3 / 2, -0.5), (0.5, S3 / 2)),
}


def local(wall_name: str, a: float, o: float) -> tuple[float, float]:
    """J1 + a u + o n, by hand."""
    (jx, jy), (ux, uy), (nx, ny) = FRAME_WALLS[wall_name]
    return (jx + a * ux + o * nx, jy + a * uy + o * ny)


def test_the_horizontal_frame_vertices_written_out():
    wall, op = t08_door("horizontal", False, "J2", "L", frame=True)
    leaf, arc, near, far = opening_prims(wall, op)
    assert op.frame is True and isinstance(near, Poly) and near.closed and near.role == "door"
    assert [pt(p) for p in near.points] == [
        (1000.0, 100.0),
        (1035.0, 100.0),
        (1035.0, 60.0),
        (1050.0, 60.0),
        (1050.0, -100.0),
        (1000.0, -100.0),
    ]
    assert [pt(p) for p in far.points] == [
        (1900.0, 100.0),
        (1865.0, 100.0),
        (1865.0, 60.0),
        (1850.0, 60.0),
        (1850.0, -100.0),
        (1900.0, -100.0),
    ]
    assert (pt(leaf.p1), pt(leaf.p2)) == ((1865.0, 100.0), (1865.0, 930.0))
    assert pt(arc.center) == (1865.0, 100.0) and arc.radius == 830.0


@pytest.mark.parametrize("wall_name", list(FRAME_WALLS))
@pytest.mark.parametrize("jamb", ["J1", "J2"])
@pytest.mark.parametrize("side", ["L", "R"])
def test_a_framed_door_hangs_on_its_frames_rebate_vertex(wall_name, jamb, side):
    wall, op = t08_door(wall_name, False, jamb, side, frame=True)
    leaf, arc, near, far = opening_prims(wall, op)
    s = 1.0 if side == "L" else -1.0
    # the frame on the hinge jamb, by hand: a runs into the gap from that jamb
    a0, sign = (0.0, 1.0) if jamb == "J1" else (900.0, -1.0)
    hand_frame = [
        local(wall_name, a0 + sign * a, s * o)
        for a, o in ((0, 100), (35, 100), (35, 60), (50, 60), (50, -100), (0, -100))
    ]
    frame = near if jamb == "J1" else far
    assert all(close(p, q) for p, q in zip(frame.points, hand_frame, strict=True))
    hinge = local(wall_name, a0 + sign * 35.0, s * 100.0)
    assert close(leaf.p1, hinge)
    assert close(leaf.p2, local(wall_name, a0 + sign * 35.0, s * 930.0))
    # 0.0 mm: the hinge is the frame polyline's own vertex, bit for bit
    assert leaf.p1 in frame.points and arc.center == leaf.p1
    assert arc.radius == pytest.approx(830.0)
    # the arc runs from the other frame's rebate vertex (closed) to the open leaf
    other = far if jamb == "J1" else near
    ends = (arc_point(arc, arc.start_deg), arc_point(arc, arc.end_deg))
    assert (close(ends[0], other.points[1]) and close(ends[1], leaf.p2)) or (
        close(ends[0], leaf.p2) and close(ends[1], other.points[1])
    )


def test_frame_false_hangs_the_leaf_on_the_jamb_with_no_frame():
    leaf, arc = opening_prims(*t08_door("horizontal", False, "J2", "L", frame=False))
    assert (pt(leaf.p1), pt(leaf.p2)) == ((1900.0, 100.0), (1900.0, 1000.0))
    assert arc.radius == 900.0


def test_a_framed_double_door_splits_the_leaf_between_the_rebates():
    op = Opening(id="d", wall="a", kind="double_door", offset=1000.0, width=1600.0)
    leaf_a, _arc_a, leaf_b, arc_b, near, far = opening_prims(WALL, op)
    assert (pt(leaf_a.p1), pt(leaf_a.p2)) == ((1035.0, 100.0), (1035.0, 865.0))
    assert (pt(leaf_b.p1), pt(leaf_b.p2)) == ((2565.0, 100.0), (2565.0, 865.0))
    assert leaf_a.p1 in near.points and leaf_b.p1 in far.points and arc_b.radius == 765.0


def test_a_sliding_door_frame_is_plain_and_its_panels_run_between_the_frames():
    op = Opening(id="d", wall="a", kind="sliding_door", offset=1000.0, width=1200.0)
    a, b, near, far = opening_prims(WALL, op)
    assert [pt(p) for p in near.points] == [
        (1000.0, 100.0),
        (1050.0, 100.0),
        (1050.0, -100.0),
        (1000.0, -100.0),
    ]
    assert (pt(a.p1), pt(b.p2)) == ((1050.0, 30.0), (2150.0, -30.0))
    assert pt(far.points[1]) == (2150.0, 100.0)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"kind": "window", "frame": True}, "window has no frame"),
        ({"kind": "garage_door", "frame": False}, "garage_door has no frame"),
        ({"frame": "yes"}, "frame: expected true or false"),
    ],
)
def test_frame_is_refused_where_it_means_nothing(fields, message):
    from engineering.arch.model import opening_from_dict

    raw = {"id": "d", "wall": "a", "kind": "door", "offset": 1000, "width": 900, **fields}
    with pytest.raises(ValueError, match=message):
        opening_from_dict(raw, "openings[0]")


def test_an_opening_too_narrow_for_its_frame_is_refused_by_name():
    op = Opening(id="d9", wall="a", kind="door", offset=1000.0, width=100.0)
    with pytest.raises(ValueError, match="opening d9: width 100 mm leaves no passage"):
        opening_prims(WALL, op)
    assert len(opening_prims(WALL, replace(op, frame=False))) == 2
