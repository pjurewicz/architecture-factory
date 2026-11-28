"""arch_floor_zone: a room minus what is fixed on it, hatched, its area measured.

The room is the inside of a 250 mm ring on the axis 0-5000 x 0-4000: faces at
125 / 4875 and 125 / 3875, 4750 x 3750 = 17.8125 m². A kitchen run 2400 x 600
in its corner (fitted) takes 1.44 m² out, a double bed (movable) nothing.
"""

from __future__ import annotations

import pytest

from engineering.annotation_record import read_annotation
from engineering.arch.catalogue import FITTED, ITEMS, insert_items
from engineering.arch.floor import floor_zone
from engineering.arch.legend import draw_legend
from engineering.arch.spec import draw_plan_from_spec
from engineering.arch.tiles import signed_area

SPEC = {
    "sheet": {"intent": "flat", "sheet_size": "A3", "scale": 50},
    "walls": [
        {
            "id": "EXT",
            "axis": [[0, 0], [5000, 0], [5000, 4000], [0, 4000]],
            "thickness": 250,
            "closed": True,
            "material": "brick",
        }
    ],
}


async def _room(backend):
    await draw_plan_from_spec(backend, SPEC)
    await insert_items(
        backend,
        [
            {"name": "kitchen_counter", "x": 125, "y": 125},
            {"name": "double_bed", "x": 2000, "y": 1500},
        ],
    )


def test_fitted_names_catalogue_items():
    assert FITTED <= {item.name for item in ITEMS}


@pytest.mark.asyncio
async def test_a_zone_stops_at_fitted_items_and_runs_under_furniture(backend):
    await _room(backend)
    zone = await floor_zone(backend, room=[2500, 2000], label="ogrzewanie podłogowe", name="UFH")
    assert zone["outline_area_m2"] == pytest.approx(17.8125)
    assert zone["area_m2"] == pytest.approx(17.8125 - 1.44)
    assert [e.get("item") for e in zone["excluded"]] == ["kitchen_counter"]
    hatch = backend._doc.entitydb[zone["handles"]["pattern"]]
    assert hatch.dxftype() == "HATCH" and hatch.dxf.layer == "A-FLOR-E-N"
    record = read_annotation(hatch)
    assert record["legend"] == "ogrzewanie podłogowe" and record["set"] == "UFH"
    # the bed too, by name; a rerun with the same name replaces the zone
    again = await floor_zone(
        backend, room=[2500, 2000], minus=["fitted", "double_bed"], joints=[1200, 200], name="UFH"
    )
    assert again["replaced"] == 1 and not backend._doc.entitydb[zone["handles"]["pattern"]].is_alive
    assert again["area_m2"] == pytest.approx(17.8125 - 1.44 - 1.6 * 2.0, abs=0.2)
    assert again["warnings"][0]["code"] == "no_label"


@pytest.mark.asyncio
async def test_an_outline_and_handles_and_the_refusals(backend):
    info = await backend.entity_create_polyline(
        [[100, 100], [300, 100], [300, 300], [100, 300]], closed=True
    )
    zone = await floor_zone(
        backend,
        outline=[[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
        minus_handles=[info.handle],
        dry_run=True,
    )
    assert zone["area_m2"] == pytest.approx(1.0 - 0.04) and "handles" not in zone
    with pytest.raises(ValueError, match=r"minus\[0\]: 'fited'.*nearest: fitted"):
        await floor_zone(backend, outline=[[0, 0], [1, 0], [1, 1]], minus=["fited"])
    with pytest.raises(ValueError, match="give exactly one"):
        await floor_zone(backend)
    with pytest.raises(ValueError, match="no room .R9."):
        await floor_zone(backend, room="R9")


ROOM = [[0, 0], [3300, 0], [3300, 2400], [0, 2400]]


@pytest.mark.asyncio
async def test_ar_patterns_are_building_mm_and_a_coarse_fill_warns(backend):
    # AR-HBONE's bricks are 101.6 x 304.8 mm in acadiso.pat: at 1:55 the old
    # rule hatched at 55, rows 7.9 m apart - one stroke in a 3.3 x 2.4 m room
    zone = await floor_zone(backend, outline=ROOM, pattern="AR-HBONE", scale=55, label="winyl")
    assert zone["pattern"] == {
        "name": "AR-HBONE", "hatch_scale": 1.0, "period_mm": 406.4, "repeats": 5.9
    }  # fmt: skip
    assert "warnings" not in zone
    assert backend._doc.entitydb[zone["handles"]["pattern"]].dxf.pattern_scale == 1.0
    # a sheet pattern still follows the plot scale: ANSI37 spacing 3.175 x 55
    sheet = await floor_zone(backend, outline=ROOM, pattern="ANSI37", scale=55, dry_run=True)
    assert sheet["pattern"]["hatch_scale"] == 55 and sheet["pattern"]["period_mm"] == 174.6
    # the bathroom's eight dots: AR-SAND blown up 55 times, 6.8 m against 1.8 m
    bath = [[0, 0], [2200, 0], [2200, 1800], [0, 1800]]
    sand = await floor_zone(
        backend, outline=bath, pattern="AR-SAND", pattern_scale=55, label="x", dry_run=True
    )
    assert sand["pattern"]["period_mm"] == pytest.approx(123.063 * 55, abs=0.1)
    assert [w["code"] for w in sand["warnings"]] == ["fill_period_exceeds_zone"]
    few = await floor_zone(
        backend, outline=bath, pattern="AR-SAND", pattern_scale=6, label="x", dry_run=True
    )
    assert few["pattern"]["repeats"] == 2.4
    assert [w["code"] for w in few["warnings"]] == ["fill_few_repeats"]
    joints = await floor_zone(backend, outline=bath, joints=[1200, 600], label="x", dry_run=True)
    assert "pattern" not in joints and joints["joints"]["repeats"] == 1.5
    assert joints["warnings"][0]["code"] == "fill_few_repeats"


@pytest.mark.asyncio
async def test_pattern_joints_and_edge_are_one_zone(backend):
    zone = await floor_zone(
        backend,
        outline=ROOM,
        minus_handles=[],
        pattern="AR-SAND",
        joints=[600, 300],
        edge=True,
        label="płytka 60x30",
        name="BATH",
    )
    db = backend._doc.entitydb
    pattern, joints = db[zone["handles"]["pattern"]], db[zone["handles"]["joints"]]
    assert pattern.dxf.pattern_name == "AR-SAND" and joints.dxf.pattern_name == "_USER"
    (edge,) = (db[h] for h in zone["handles"]["edge"])
    assert edge.dxftype() == "LWPOLYLINE" and edge.closed
    assert sorted(edge.get_points("xy")) == sorted(map(tuple, ROOM))
    parts = [pattern, joints, edge]
    assert {e.dxf.layer for e in parts} == {"A-FLOR-E-N"}
    assert all(read_annotation(e)["set"] == "BATH" for e in parts)
    assert all(read_annotation(e)["legend"] == "płytka 60x30" for e in parts)
    # the zone sits at the back, pattern under its joints under its outline
    order = [e.dxf.handle for e in backend._doc.modelspace().entities_in_redraw_order()]
    assert order[:3] == [e.dxf.handle for e in parts]
    again = await floor_zone(backend, outline=ROOM, pattern="AR-SAND", name="BATH", label="x")
    assert again["replaced"] == 3 and set(again["handles"]) == {"pattern"}


# -- skirting (round 8): along the faces, stopped at doors and fitted items, measured

DOOR_ROOM = {
    **SPEC,
    "walls": [*SPEC["walls"], {"id": "P1", "axis": [[3000, 0], [3000, 4000]], "thickness": 100}],
    "openings": [
        {"id": "D1", "wall": "EXT", "kind": "door", "offset": 1000, "width": 900},
        {"id": "W1", "wall": "EXT", "kind": "window", "offset": 12500, "width": 1200,
         "sill": 900, "height": 1200},
    ],
}  # fmt: skip
# the L: axis 6 x 3 m with a 3 x 2 m arm, 200 mm centred walls; its inner face
# has five convex corners (-200 each) and one re-entrant (+200): 22.0 - 0.8 m
L_ROOM = {
    "sheet": SPEC["sheet"],
    "walls": [
        {
            "id": "L",
            "axis": [[0, 0], [6000, 0], [6000, 3000], [3000, 3000], [3000, 5000], [0, 5000]],
            "thickness": 200,
            "closed": True,
        }
    ],
}


@pytest.mark.asyncio
async def test_skirting_runs_on_the_face_and_stops_at_the_door(backend):
    await draw_plan_from_spec(backend, DOOR_ROOM)
    zone = await floor_zone(
        backend, room=[1500, 2000], pattern="none", skirting={"P1": "flush"}, name="SK"
    )
    sk = zone["skirting"]
    # room 125..2950 x 125..3875: the partition's face flush (3.75 m), the rest
    # applied, the 0.9 m door out; the window (sill 900) does not stop it
    assert sk["flush_m"] == pytest.approx(3.75) and sk["stops_at"] == ["D1"]
    assert sk["applied_m"] == pytest.approx(2 * 2.825 + 3.75 - 0.9)
    assert set(zone["handles"]) == {"skirting"}  # pattern none: no hatch
    db = backend._doc.entitydb
    lines = [db[h] for h in zone["handles"]["skirting"]]
    assert {e.dxf.layer for e in lines} == {"A-FLOR-SKRT-E-N", "A-FLOR-SKRF-E-N"}
    flush = next(e for e in lines if e.dxf.layer == "A-FLOR-SKRF-E-N")
    assert flush.dxf.ltscale == 0.25 and flush.dxf.linetype == "BYLAYER"
    # 0.5 paper mm inside the face at 1:50: x = 2950 - 25
    assert {round(x) for x, _ in flush.get_points("xy")} == {2925}
    door_gap = sorted(
        x for e in lines if e.dxf.layer == "A-FLOR-SKRT-E-N" for x, y in e.get_points("xy")
        if abs(y - 150) < 1
    )  # fmt: skip
    assert door_gap[:2] == [150, 1000] and 1900 in door_gap  # the door 1000..1900 left open
    assert read_annotation(flush)["legend"] == "skirting_flush"
    again = await floor_zone(backend, room=[1500, 2000], pattern="none", skirting="none",
                             edge=True, name="SK")  # fmt: skip
    assert again["replaced"] == len(lines) and "skirting" not in again["handles"]
    assert not any(db[h].is_alive for h in zone["handles"]["skirting"])


@pytest.mark.asyncio
async def test_skirting_on_an_l_room_stops_behind_fitted_items(backend):
    await draw_plan_from_spec(backend, L_ROOM)
    bare = await floor_zone(backend, room=[1000, 1000], skirting="applied", dry_run=True)
    (piece,) = bare["skirting"]["pieces"]
    assert piece["closed"] and bare["skirting"]["applied_m"] == pytest.approx(21.2)
    assert "skirting_openings_unknown" not in str(bare.get("warnings"))
    await insert_items(backend, [{"name": "kitchen_counter", "x": 100, "y": 100}])
    fitted = await floor_zone(backend, room=[1000, 1000], skirting="applied", dry_run=True)
    # the 2400 x 600 run stands on 2.4 m of one face and 0.6 m of the other
    assert fitted["skirting"]["applied_m"] == pytest.approx(21.2 - 3.0)
    assert [p["closed"] for p in fitted["skirting"]["pieces"]] == [False]


@pytest.mark.asyncio
async def test_skirting_legend_rows_and_refusals(backend):
    await draw_plan_from_spec(backend, DOOR_ROOM)
    await floor_zone(backend, room=[1500, 2000], pattern="none", skirting={"P1": "flush"})
    result = await draw_legend(backend, at=(9000, 4000), lang="pl", only=["skirting_*"])
    assert [(r["label"], r["count"]) for r in result["rows"]] == [
        ("cokół szczelinowy (podcięcie)", 1),
        ("listwa przypodłogowa nakładana", 2),  # the flush wall splits it
    ]
    with pytest.raises(ValueError, match=r"skirting.X9: no wall 'X9'.*\['EXT', 'P1'\]"):
        await floor_zone(backend, room=[1500, 2000], skirting={"X9": "flush"})
    with pytest.raises(ValueError, match="skirting.default: 'glued'"):
        await floor_zone(backend, room=[1500, 2000], skirting="glued")
    with pytest.raises(ValueError, match="draws nothing"):
        await floor_zone(backend, room=[1500, 2000], pattern="none")
    typed = await floor_zone(backend, outline=ROOM, skirting="flush", dry_run=True)
    assert typed["skirting"]["flush_m"] == pytest.approx(11.4)
    assert "skirting_openings_unknown" in [w["code"] for w in typed["warnings"]]


@pytest.mark.asyncio
async def test_a_laid_floor_is_cut_to_the_zone_minus_the_fitted_run(backend):
    await _room(backend)
    args = {"room": [2500, 2000], "lay": "chevron", "plank": (358.79, 90), "gap": 0}
    dry = await floor_zone(
        backend, **args, label="dąb jodełka francuska", name="PARKIET", dry_run=True
    )
    assert "handles" not in dry and dry["lay"]["takeoff"]["by_hand"]["A"] > 0
    zone = await floor_zone(backend, **args, label="dąb jodełka francuska", name="PARKIET")
    db = backend._doc.entitydb
    pieces = [db[h] for h in zone["handles"]["pieces"]]
    assert set(zone["handles"]) == {"pieces"} and zone["lay"]["cut_angle"] == 45.0
    # the planks cover exactly the measured zone: the room less the kitchen run
    covered = sum(abs(signed_area(e.get_points("xy"))) for e in pieces) / 1e6
    assert covered == pytest.approx(zone["area_m2"], rel=1e-6)
    assert zone["area_m2"] == pytest.approx(17.8125 - 1.44)
    assert zone["lay"]["takeoff"]["floor_area_m2"] == zone["area_m2"]
    for e in pieces:  # nothing under the 2400 x 600 counter in the corner
        assert all(not (x < 2525 - 1e-6 and y < 725 - 1e-6) for x, y in e.get_points("xy"))
    assert {e.dxf.layer for e in pieces} == {"A-FLOR-E-N"}
    records = [read_annotation(e) for e in pieces]
    assert {(r["tool"], r["set"], r["legend"]) for r in records} == {
        ("arch_floor_zone", "PARKIET", "dąb jodełka francuska")
    }
    # a rerun replaces the whole set; a wood hatch under a grid of boards is one zone
    again = await floor_zone(
        backend, room=[2500, 2000], lay="running_bond", plank=(1200, 200), gap=0,
        pattern="AR-PARQ1", name="PARKIET",
    )  # fmt: skip
    assert again["replaced"] == len(pieces) and set(again["handles"]) == {"pattern", "pieces"}
    order = [e.dxf.handle for e in backend._doc.modelspace().entities_in_redraw_order()]
    assert order[0] == again["handles"]["pattern"]  # the hatch under its boards


@pytest.mark.asyncio
async def test_lay_refusals_name_this_tools_arguments(backend):
    room = {"outline": ROOM}
    with pytest.raises(ValueError, match="give one of lay / joints"):
        await floor_zone(backend, **room, lay="grid", plank=(600, 300), joints=[600])
    with pytest.raises(ValueError, match="plank_length / plank_width"):
        await floor_zone(backend, **room, lay="grid", plank=(600, None))
    with pytest.raises(ValueError, match="cut_angle: a chevron"):
        await floor_zone(backend, **room, lay="chevron", plank=(600, 100), cut_angle=5)
    with pytest.raises(ValueError, match="lay: 'basket'"):
        await floor_zone(backend, **room, lay="basket", plank=(600, 100))
    with pytest.raises(ValueError, match="plank_length: herringbone"):
        await floor_zone(backend, **room, lay="herringbone", plank=(100, 100))


@pytest.mark.asyncio
async def test_a_layout_that_fails_half_way_leaves_nothing_drawn(backend, monkeypatch):
    before = {e.handle for e in await backend.entity_list(limit=100000)}

    async def fail(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(backend, "draw_order", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        await floor_zone(backend, outline=ROOM, lay="grid", plank=(600, 300))
    assert {e.handle for e in await backend.entity_list(limit=100000)} == before
