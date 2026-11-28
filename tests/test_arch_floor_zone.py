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
from engineering.arch.spec import draw_plan_from_spec

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
