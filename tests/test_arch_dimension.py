"""Exterior dimension chains: three rows per side, from the wall engine's own faces.

The plan: a closed 8200 x 6200 axis rectangle of 200 mm walls (outer faces
-100..8300 x -100..6300), a 100 mm partition on x = 4100 teed into the bottom
and top walls (faces x = 4050 / 4150), a 1200 window at 1500 and a 1000 door
at 5500 in the bottom wall. Every point below is hand-computed from those
numbers; at 1:50 the rows sit 500, 900 and 1300 mm outside the building.
"""

from __future__ import annotations

import copy
import math

import pytest

from engineering.arch.dimension import (
    FIRST_OFFSET_PAPER_MM,
    ROWS,
    SIDES,
    STEP_PAPER_MM,
    TEXT_WIDTH_FACTOR,
    chain_text_positions,
    exterior_chains,
)
from engineering.arch.model import Opening, Wall
from engineering.mech.primitives import DimIntent

SHELL = Wall(
    id="shell",
    axis=((0.0, 0.0), (8200.0, 0.0), (8200.0, 6200.0), (0.0, 6200.0)),
    thickness=200.0,
    closed=True,
)
PARTITION = Wall(id="p", axis=((4100.0, 0.0), (4100.0, 6200.0)), thickness=100.0)
WINDOW = Opening(id="w1", wall="shell", kind="window", offset=1500.0, width=1200.0)
DOOR = Opening(id="d1", wall="shell", kind="door", offset=5500.0, width=1000.0)


def row(intents, feature) -> list[tuple]:
    out = []
    for intent in intents:
        if intent.feature == feature:
            out.append(
                (
                    tuple(round(v, 9) + 0.0 for v in intent.p1),
                    tuple(round(v, 9) + 0.0 for v in intent.p2),
                    tuple(round(v, 9) + 0.0 for v in intent.text_at),
                )
            )
    return out


def test_the_rows_and_the_offsets_are_declared():
    assert SIDES == ("bottom", "top", "left", "right")
    assert ROWS == ("openings", "walls", "overall")
    assert (FIRST_OFFSET_PAPER_MM, STEP_PAPER_MM) == (10.0, 8.0)


def test_the_bottom_side_carries_openings_walls_and_overall():
    intents = exterior_chains([SHELL, PARTITION], [WINDOW, DOOR], sides=("bottom",), scale=50)
    assert all(isinstance(i, DimIntent) and i.kind == "linear" for i in intents)
    assert row(intents, "bottom:openings") == [
        ((-100.0, -100.0), (1500.0, -100.0), (700.0, -600.0)),
        ((1500.0, -100.0), (2700.0, -100.0), (2100.0, -600.0)),
        ((2700.0, -100.0), (5500.0, -100.0), (4100.0, -600.0)),
        ((5500.0, -100.0), (6500.0, -100.0), (6000.0, -600.0)),
        ((6500.0, -100.0), (8300.0, -100.0), (7400.0, -600.0)),
    ]
    assert row(intents, "bottom:walls") == [
        ((-100.0, -100.0), (100.0, -100.0), (0.0, -1000.0)),
        ((100.0, -100.0), (4050.0, -100.0), (2075.0, -1000.0)),
        ((4050.0, -100.0), (4150.0, -100.0), (4100.0, -1000.0)),
        ((4150.0, -100.0), (8100.0, -100.0), (6125.0, -1000.0)),
        ((8100.0, -100.0), (8300.0, -100.0), (8200.0, -1000.0)),
    ]
    assert row(intents, "bottom:overall") == [
        ((-100.0, -100.0), (8300.0, -100.0), (4100.0, -1400.0)),
    ]
    assert len(intents) == 11


def test_a_row_that_repeats_a_farther_one_is_not_emitted():
    intents = exterior_chains([SHELL, PARTITION], [WINDOW, DOOR], sides=("left",), scale=50)
    # no opening in the left wall: that row would only restate the overall one
    assert row(intents, "left:openings") == []
    assert row(intents, "left:walls") == [
        ((-100.0, -100.0), (-100.0, 100.0), (-1000.0, 0.0)),
        ((-100.0, 100.0), (-100.0, 6100.0), (-1000.0, 3100.0)),
        ((-100.0, 6100.0), (-100.0, 6300.0), (-1000.0, 6200.0)),
    ]
    assert row(intents, "left:overall") == [((-100.0, -100.0), (-100.0, 6300.0), (-1400.0, 3100.0))]


def test_explicit_offsets_override_the_plot_scale_defaults():
    intents = exterior_chains(
        [SHELL], [], sides=("top",), first_offset=300.0, step=250.0, scale=100
    )
    # the top: the extremes only, so the walls row is the corner walls' faces
    assert row(intents, "top:overall") == [((-100.0, 6300.0), (8300.0, 6300.0), (4100.0, 7100.0))]
    assert {i.text_at[1] for i in intents if i.feature == "top:walls"} == {6850.0}


def test_bad_sides_and_scales_are_refused_by_name():
    with pytest.raises(ValueError, match=r"sides\[0\]: 'north'"):
        exterior_chains([SHELL], [], sides=("north",))
    with pytest.raises(ValueError, match="named twice"):
        exterior_chains([SHELL], [], sides=("left", "left"))
    with pytest.raises(ValueError, match="scale"):
        exterior_chains([SHELL], [], scale=0)
    with pytest.raises(ValueError, match="sides"):
        exterior_chains([SHELL], [], sides="bottom")


#: A stepped bottom facade: a garage, a vestibule pushed 1200 forward between
#: two 120 partitions, a wing set back 800 - every step projects onto one row.
STEPPED = Wall(
    id="ext",
    axis=(
        (0.0, 0.0),
        (6000.0, 0.0),
        (6000.0, 800.0),
        (6400.0, 800.0),
        (6400.0, -1200.0),
        (8800.0, -1200.0),
        (8800.0, 800.0),
        (9040.0, 800.0),
        (9040.0, 2400.0),
        (17000.0, 2400.0),
        (17000.0, 12000.0),
        (0.0, 12000.0),
    ),
    thickness=240.0,
    closed=True,
)
STEPPED_PARTITIONS = [
    Wall(id="p1", axis=((6000.0, 800.0), (6000.0, 12000.0)), thickness=120.0),
    Wall(id="p2", axis=((9040.0, 2400.0), (9040.0, 12000.0)), thickness=120.0),
]


def boxes(intents, texts, h):
    out = []
    for intent, (x, y) in zip(intents, texts, strict=True):
        label = f"{abs(intent.p2[0] - intent.p1[0]):.0f}"
        half = len(label) * TEXT_WIDTH_FACTOR * h / 2.0
        out.append((intent, (x - half, x + half, y - h / 2.0, y + h / 2.0)))
    return out


def test_stepped_facade_texts_never_overlap_and_fitting_ones_stay_centred():
    intents = exterior_chains([STEPPED, *STEPPED_PARTITIONS], [], sides=("bottom",), scale=50)
    h, gap = 125.0, 31.25  # ISO-25 at 1:50: DIMTXT 2.5, DIMGAP 0.625
    texts = chain_text_positions(intents, text_height=h, gap=gap, tad=1)
    shorts = [i for i in intents if abs(i.p2[0] - i.p1[0]) <= 240.0]
    assert len(shorts) >= 6  # the 60 / 120 / 160 / 240 steps that collided
    placed = boxes(intents, texts, h)
    for n, (a, ra) in enumerate(placed):
        for b, rb in placed[n + 1 :]:
            apart_x = ra[1] + h <= rb[0] + 1e-6 or rb[1] + h <= ra[0] + 1e-6
            apart_y = ra[3] <= rb[2] + 1e-6 or rb[3] <= ra[2] + 1e-6
            assert apart_x or apart_y, (a, b)
        # no text sits on another row's dimension line
        for line in {i.text_at[1] for i in intents} - {a.text_at[1]}:
            assert not ra[2] - gap < line < ra[3] + gap, a
    for intent, (x, y) in zip(intents, texts, strict=True):
        if abs(intent.p2[0] - intent.p1[0]) >= 1000.0:
            assert (x, y) == ((intent.p1[0] + intent.p2[0]) / 2.0, intent.text_at[1] + gap + h / 2)


#: A bottom facade kinked up by 500 over 4000 at x = 8000, 250 mm thick. The
#: junction engine mitres the kink: the outer faces meet at
#: (8000 + 125 * tan(a / 2), -125) and the inner ones at (8000 - 125 * tan(a / 2),
#: 125), a = atan(500 / 4000) - 15.6 mm apart along the facade, so the walls row
#: keeps the one on the outer face.
KINKED = Wall(
    id="k",
    axis=((0.0, 0.0), (8000.0, 0.0), (12000.0, 500.0), (12000.0, 6000.0), (0.0, 6000.0)),
    thickness=250.0,
    closed=True,
)
KINK_FACE = round(8000.0 + 125.0 * math.tan(math.atan2(500.0, 4000.0) / 2.0), 6)


def test_an_oblique_walls_chain_point_is_the_mitre_its_faces_are_drawn_to():
    intents = exterior_chains([KINKED], [], sides=("bottom",))
    spans = [(round(i.p1[0], 6), round(i.p2[0], 6)) for i in intents if i.feature == "bottom:walls"]
    assert spans == [(-125.0, 125.0), (125.0, KINK_FACE), (KINK_FACE, 12125.0)]
    kink = next(i.p2 for i in intents if round(i.p2[0], 6) == KINK_FACE)
    assert kink[1] == pytest.approx(-125.0, abs=1e-9)


def test_a_real_30_mm_gap_beside_an_oblique_wall_stays_dimensioned():
    # two partitions 130 apart on the axes: faces 4050 and 4080, 30 mm of real wall face
    partitions = [
        Wall(id="a", axis=((4000.0, 0.0), (4000.0, 6000.0)), thickness=100.0),
        Wall(id="b", axis=((4130.0, 0.0), (4130.0, 6000.0)), thickness=100.0),
    ]
    intents = exterior_chains([KINKED, *partitions], [], sides=("bottom",))
    spans = [(round(i.p1[0], 6), round(i.p2[0], 6)) for i in intents if i.feature == "bottom:walls"]
    assert (4050.0, 4080.0) in spans
    assert spans[-2:] == [(4180.0, KINK_FACE), (KINK_FACE, 12125.0)]


def _kinked_spec(degrees: float) -> dict:
    """A 250 ring whose bottom facade turns by ``degrees`` at x = 8000, a 100
    partition teed into it, a 60-degree partition teed into the top wall."""
    t = math.radians(degrees)
    x, y = 8000.0 + 4000.0 * math.cos(t), 4000.0 * math.sin(t)
    return {
        "walls": [
            {
                "id": "ext",
                "axis": [[0, 0], [8000, 0], [x, y], [x, 6000], [0, 6000]],
                "thickness": 250,
                "closed": True,
            },
            {"id": "p", "axis": [[4000, 0], [4000, 6000]], "thickness": 100},
            {
                "id": "q",
                "axis": [[1500, 6000], [3000, 6000 - 1500 * math.sqrt(3)]],
                "thickness": 100,
            },
        ],
        "openings": [{"id": "w1", "wall": "ext", "kind": "window", "offset": 5000, "width": 1500}],
        "chains": {"sides": list(SIDES)},
    }


#: The whole top facade oblique: no top wall runs parallel to the top side.
OBLIQUE_TOP = {
    "walls": [
        {
            "id": "ext",
            "axis": [[0, 0], [8000, 0], [12000, 500], [12000, 6000], [2500, 7000], [0, 4500]],
            "thickness": 250,
            "closed": True,
        },
        {"id": "p", "axis": [[5000, 0], [5000, 6400]], "thickness": 100},
    ],
    "openings": [{"id": "w1", "wall": "ext", "kind": "window", "offset": 2000, "width": 1500}],
    "chains": {"sides": list(SIDES)},
}


@pytest.mark.parametrize(
    "spec",
    [_kinked_spec(7.0), _kinked_spec(30.0), _kinked_spec(45.0), _kinked_spec(-30.0), OBLIQUE_TOP],
    ids=["7", "30", "45", "-30", "oblique_top"],
)
@pytest.mark.asyncio
async def test_every_chain_point_beside_an_oblique_wall_is_drawn_and_binds(backend, spec):
    from engineering.arch.spec import draw_plan_from_spec, validate_spec
    from engineering.arch.walls import wall_geometry

    result = await draw_plan_from_spec(backend, copy.deepcopy(spec))
    issues = await backend.drawing_critique(focus=["dimension_truth"])
    unbound = [i for i in issues if i.detail["code"] == "dim_unbound"]
    assert result["chains"]["count"] > 0 and unbound == []
    plan = validate_spec(copy.deepcopy(spec))
    faces = wall_geometry(plan["walls"], plan["openings"], poche=False).faces
    vertices = [p for line in faces for p in (line.p1, line.p2)]
    intents = exterior_chains(plan["walls"], plan["openings"], sides=SIDES)
    for intent in intents:
        ai = 0 if intent.feature.split(":")[0] in ("bottom", "top") else 1
        for point in (intent.p1, intent.p2):
            # a drawn corner, or (a wall teed square into the facade) that
            # corner's foot on the facade's outer face
            assert any(
                abs(point[ai] - v[ai]) <= 1e-6
                and (math.dist(point, v) <= 1e-6 or abs(point[1 - ai] - v[1 - ai]) > 1.0)
                for v in vertices
            ), (intent.feature, point)


def test_a_facade_oblique_all_along_does_not_repeat_the_opposite_one():
    plan_walls = [
        Wall(
            id=w["id"],
            axis=tuple(map(tuple, w["axis"])),
            thickness=w["thickness"],
            closed=w.get("closed", False),
        )
        for w in OBLIQUE_TOP["walls"]
    ]
    intents = exterior_chains(plan_walls, [], sides=("bottom", "top"))
    top = [i for i in intents if i.feature.startswith("top:")]
    # nothing parallel to the top side faces it: only the overall extremes,
    # each on the drawn corner nearest the top, not on the bottom wall
    assert [i.feature for i in top] == ["top:overall"]
    # the 45-degree face on x = -125; the top face on x = 12125: 125 off the axis
    # from (12000, 6000) along the normal (1000, 9500) / |(9500, 1000)|
    assert top[0].p1 == pytest.approx((-125.0, 4500.0 + 125.0 * math.sqrt(2.0) - 125.0))
    rise = (125.0 * math.hypot(9500.0, 1000.0) - 125.0 * 1000.0) / 9500.0
    assert top[0].p2 == pytest.approx((12125.0, 6000.0 + rise))


#: A user's report: walls at -30 degrees, 200 thick, teed into a 240 west wall
#: whose outer face is x = -120. Their faces meet its inner face x = 120 at
#: y0 - 120 * tan(30) -/+ 100 / cos(30): the T's span is 230.94 along the
#: facade, not the 2 * 100 * cos(30) = 173.21 of the faces projected from the
#: axis end.
COS30, SIN30 = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
TEE_SPEC = {
    "walls": [
        {
            "id": "ext",
            "axis": [[0, 0], [10000, 0], [10000, 7000], [0, 7000]],
            "thickness": 240,
            "closed": True,
        },
        *(
            {"id": f"t{y0}", "axis": [[0, y0], [3000 * COS30, y0 - 3000 * SIN30]], "thickness": 200}
            for y0 in (2500, 5000)
        ),
    ],
    "chains": {"sides": ["left"]},
}


@pytest.mark.asyncio
async def test_a_wall_teed_at_30_degrees_is_measured_at_its_t_corners(backend):
    from engineering.arch.spec import draw_plan_from_spec, validate_spec

    plan = validate_spec(copy.deepcopy(TEE_SPEC))
    intents = [
        i for i in exterior_chains(plan["walls"], [], sides=("left",)) if i.feature == "left:walls"
    ]
    half, drop = 100.0 / COS30, 120.0 * SIN30 / COS30
    corners = [(120.0, y0 - drop + k * half) for y0 in (2500.0, 5000.0) for k in (-1.0, 1.0)]
    points = [c for i in intents for c in i.p1] + list(intents[-1].p2)
    expected = [(-120.0, -120.0), (-120.0, 120.0), *corners, (-120.0, 6880.0), (-120.0, 7120.0)]
    assert points == pytest.approx([c for p in expected for c in p])
    assert [abs(i.p2[1] - i.p1[1]) for i in intents][2:5:2] == pytest.approx([2 * half] * 2)
    await draw_plan_from_spec(backend, copy.deepcopy(TEE_SPEC))
    issues = await backend.drawing_critique(focus=["dimension_truth"])
    assert [i for i in issues if i.detail["code"] == "dim_unbound"] == []
