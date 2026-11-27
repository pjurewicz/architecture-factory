"""The flexible arch layer: more opening kinds, corner openings, raw wall polygons
mixed with the engine's walls, centred and fitted room labels.

The L below is two centred 200 mm walls meeting at (6000, 0): S runs along +X,
E up +Y. Their faces mitre at the inner corner (5900, 100) and the outer one
(6100, -100); every expected value is written out from those numbers.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastmcp import Client

import server
from engineering.arch.model import Opening, Wall
from engineering.arch.openings import opening_prims
from engineering.arch.walls import wall_geometry
from engineering.mech.primitives import Arc, Line

S = Wall(id="S", axis=((0.0, 0.0), (6000.0, 0.0)), thickness=200.0)
E = Wall(id="E", axis=((6000.0, 0.0), (6000.0, 5000.0)), thickness=200.0)


def pt(p):
    return (round(p[0], 6) + 0.0, round(p[1], 6) + 0.0)


def seg(line):
    return tuple(sorted((pt(line.p1), pt(line.p2))))


# -- corner openings -------------------------------------------------------------


def test_two_corner_windows_meet_on_the_mitre_with_no_jamb_between_them():
    ws = Opening(id="W1", wall="S", kind="window", offset=4800.0, width=1200.0)
    we = Opening(id="W2", wall="E", kind="window", offset=0.0, width=1500.0)
    geo = wall_geometry([S, E], [ws, we], poche=False)
    assert geo.omitted == ()
    cuts = {cut["element"]: cut for cut in geo.cuts}
    # S: left face y=+100 ends on the inner mitre, right face y=-100 on the outer one
    assert [pt(p) for p in cuts["W1"]["left"]] == [(4800.0, 100.0), (5900.0, 100.0)]
    assert [pt(p) for p in cuts["W1"]["right"]] == [(4800.0, -100.0), (6100.0, -100.0)]
    assert cuts["W1"]["corner"] == ["end"] and cuts["W2"]["corner"] == ["start"]
    jambs = {seg(j) for j in geo.jambs}
    # the far jambs stand square to each axis; nothing closes the corner itself
    assert ((4800.0, -100.0), (4800.0, 100.0)) in jambs
    assert ((5900.0, 1500.0), (6100.0, 1500.0)) in jambs
    assert ((5900.0, 100.0), (6100.0, -100.0)) not in jambs
    # the glazing runs into the mitre, the way its neighbour's does
    glazing = opening_prims(S, ws, cuts["W1"])
    assert pt(glazing[0].p2) == (5900.0, 100.0) and pt(glazing[1].p2) == (6100.0, -100.0)


def test_one_corner_window_leaves_the_other_wall_closed_on_the_mitre():
    ws = Opening(id="W1", wall="S", kind="window", offset=4800.0, width=1200.0)
    geo = wall_geometry([S, E], [ws], poche=False)
    assert ((5900.0, 100.0), (6100.0, -100.0)) in {seg(j) for j in geo.jambs}


def test_a_door_may_not_run_into_a_corner_and_no_opening_into_a_tee():
    door = Opening(id="D1", wall="S", kind="door", offset=5100.0, width=900.0)
    (entry,) = wall_geometry([S, E], [door]).omitted
    assert (
        entry["element"] == "D1" and "only window, curtain_wall, cased_opening" in entry["reason"]
    )
    stem = Wall(id="T", axis=((3000.0, 0.0), (3000.0, 3000.0)), thickness=100.0)
    window = Opening(id="W1", wall="T", kind="window", offset=0.0, width=1000.0)
    (entry,) = wall_geometry([S, E, stem], [window]).omitted
    assert "not into a T or an X" in entry["reason"]


# -- the new kinds ---------------------------------------------------------------

WALL = Wall(id="a", axis=((0.0, 0.0), (5000.0, 0.0)), thickness=200.0)


def test_a_double_door_is_two_half_leaves_hinged_on_both_jambs():
    prims = opening_prims(
        WALL, Opening(id="d", wall="a", kind="double_door", offset=1000.0, width=1600.0)
    )
    leaves = [p for p in prims if isinstance(p, Line)]
    arcs = [p for p in prims if isinstance(p, Arc)]
    assert {(pt(leaf.p1), pt(leaf.p2)) for leaf in leaves} == {
        ((1000.0, 100.0), (1000.0, 900.0)),
        ((2600.0, 100.0), (2600.0, 900.0)),
    }
    assert [a.radius for a in arcs] == [800.0, 800.0]


def test_a_sliding_door_is_two_overlapping_panels_inside_the_wall():
    a, b = opening_prims(
        WALL, Opening(id="d", wall="a", kind="sliding_door", offset=1000.0, width=1200.0)
    )
    # 65 % and 35 % of the thickness from the right face (-100): y = +30 and -30
    assert (pt(a.p1), pt(a.p2)) == ((1000.0, 30.0), (1650.0, 30.0))
    assert (pt(b.p1), pt(b.p2)) == ((1550.0, -30.0), (2200.0, -30.0))


def test_a_curtain_wall_puts_a_mullion_at_least_every_1500():
    prims = opening_prims(
        WALL, Opening(id="w", wall="a", kind="curtain_wall", offset=500.0, width=3200.0)
    )
    mullions = sorted(pt(p.p1)[0] for p in prims if pt(p.p1)[0] == pt(p.p2)[0])
    # the ends are the wall's jambs, not mullions (they would be duplicates)
    assert mullions == pytest.approx([500.0 + 3200 / 3, 500.0 + 6400 / 3])


# -- the tools ---------------------------------------------------------------------


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setenv("AUTOCAD_MCP_BACKEND", "ezdxf")
    async with Client(server.mcp) as connected:
        status = (await connected.call_tool("system_status", {})).structured_content or {}
        assert status.get("backend") == "ezdxf"
        await connected.call_tool("drawing_new", {})
        yield connected


async def call(client, name, args):
    result = await client.call_tool(name, args, raise_on_error=False)
    return {"error": result.content[0].text} if result.is_error else result.structured_content


ROOM_BOX = [
    {
        "id": "S",
        "axis": [[0, 0], [6000, 0], [6000, 4000], [0, 4000]],
        "thickness": 200,
        "closed": True,
    }
]


@pytest.mark.parametrize(
    "openings",
    [
        [{"offset": 500, "width": 3200}],  # mid-wall: both ends on the wall's jambs
        [{"offset": 5000, "width": 1000}],  # one corner, the other wall solid there
        [{"offset": 5000, "width": 1000}, {"offset": 6000, "width": 1000}],  # glazed corner
    ],
)
@pytest.mark.asyncio
async def test_a_curtain_wall_draws_no_line_over_the_walls_own(client, openings):
    await call(client, "arch_wall", {"walls": ROOM_BOX})
    for opening in openings:
        placed = await call(
            client, "arch_opening", {"wall": "S", "kind": "curtain_wall", **opening}
        )
        assert "error" not in placed and not placed.get("omitted")
    critique = await call(client, "drawing_critique", {"focus": ["duplicate_entities"]})
    assert critique["result"] == []


@pytest.mark.asyncio
async def test_raw_polygons_mix_with_rooms_chains_and_schedules(client):
    await call(client, "arch_wall", {"walls": ROOM_BOX})
    bay = [[2000, -100], [2000, -900], [4000, -900], [4000, -100]]
    added = await call(client, "arch_wall", {"polygons": [{"id": "bay", "points": bay}]})
    assert added["added"] == ["bay"]
    # a hand-drawn column on another A-WALL layer, no record at all
    await call(client, "layer_create", {"name": "A-WALL-COL"})
    await call(
        client,
        "entity_create_polyline",
        {
            "points": [[2900, 1900], [3300, 1900], [3300, 2300], [2900, 2300]],
            "closed": True,
            "layer": "A-WALL-COL",
        },
    )
    await call(
        client, "arch_opening", {"wall": "S", "kind": "sliding_door", "offset": 500, "width": 1200}
    )
    room = await call(client, "arch_room", {"name": "HALL", "x": 1000, "y": 3000})
    # 5800 x 3800 net of the 400 x 400 column
    assert room["room"]["area"] == pytest.approx(5800 * 3800 - 400 * 400)
    refused = await call(client, "arch_room", {"name": "BAY", "x": 3000, "y": -500})
    assert "lies inside a wall polygon" in refused["error"]
    detected = await call(client, "arch_rooms_detect", {})
    assert [round(r["area"]) for r in detected["rooms"]] == [5800 * 3800 - 400 * 400]
    chains = await call(client, "arch_dimension_chains", {"sides": ["bottom"]})
    walls_row = [(d["from"], d["to"]) for d in chains["dimensions"] if d["row"] == "bottom:walls"]
    assert ([2000.0, -100.0], [4000.0, -100.0]) in [tuple(map(list, r)) for r in walls_row]
    overall = [d for d in chains["dimensions"] if d["row"] == "bottom:overall"]
    assert [d["value"] for d in overall] == [6200.0]
    doors = await call(client, "arch_schedule", {"kind": "doors", "x": 8000, "y": 0})
    assert [(r["tag"], r["type"], r["swing"], r["hand"]) for r in doors["rows"]] == [
        ("D1", "SLIDING DOOR", "", "")
    ]
    critique = await call(
        client, "drawing_critique", {"focus": ["arch_room_unlabelled", "dimension_truth"]}
    )
    assert critique["result"] == []


@pytest.mark.asyncio
async def test_a_centred_label_is_centred_and_fit_shrinks_it_into_a_narrow_room(client):
    await call(
        client,
        "arch_wall",
        {"walls": [*ROOM_BOX, {"id": "P", "axis": [[1500, 0], [1500, 4000]], "thickness": 100}]},
    )
    wide = await call(
        client, "arch_room", {"name": "LIVING", "x": 3800, "y": 2000, "align": "center"}
    )
    # name 175 high, area 125 high 200 lower: the stack 2187.5 .. 1812.5 is centred on 2000
    rows = [
        (await call(client, "entity_get", {"handle": h}))["properties"] for h in wide["handles"]
    ]
    assert [r["insertion"][1] for r in rows] == pytest.approx([2012.5, 1812.5])
    for row in rows:
        box = row["bounding_box"]
        assert (box["min"][0] + box["max"][0]) / 2.0 == pytest.approx(3800.0)
    narrow = await call(
        client,
        "arch_room",
        {"name": "STORAGE CLOSET", "x": 725, "y": 2000, "align": "center", "fit": True},
    )
    # 1250 mm clear width (1125 with the margin): 14 characters at 0.9 x 175 need
    # 2205, so the name wraps; "STORAGE" needs 1102.5 and keeps its 3.5 mm
    assert narrow["fit"]["name_lines"] == ["STORAGE", "CLOSET"]
    assert narrow["fit"]["name_height_mm"] == pytest.approx(3.5)
    assert narrow["fit"]["fits"] is True
    # measured, not estimated: every line's box lies inside the room's face
    assert wide["fits"] is True and narrow["fits"] is True
    assert "warning" not in wide and "warning" not in narrow
    heights = [
        (await call(client, "entity_get", {"handle": h}))["properties"]["height"]
        for h in narrow["handles"]
    ]
    assert heights == pytest.approx([175.0, 175.0, 125.0])
    small = await call(client, "arch_room", {"name": "X", "x": 725, "y": 3500, "fit": True})
    assert "already labelled" in small["error"]
