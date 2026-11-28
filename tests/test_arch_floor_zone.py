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
    hatch = backend._doc.entitydb[zone["handle"]]
    assert hatch.dxftype() == "HATCH" and hatch.dxf.layer == "A-FLOR-E-N"
    record = read_annotation(hatch)
    assert record["legend"] == "ogrzewanie podłogowe" and record["set"] == "UFH"
    # the bed too, by name; a rerun with the same name replaces the zone
    again = await floor_zone(
        backend, room=[2500, 2000], minus=["fitted", "double_bed"], joints=[1200, 200], name="UFH"
    )
    assert again["replaced"] == 1 and not backend._doc.entitydb[zone["handle"]].is_alive
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
    assert zone["area_m2"] == pytest.approx(1.0 - 0.04) and "handle" not in zone
    with pytest.raises(ValueError, match=r"minus\[0\]: 'fited'.*nearest: fitted"):
        await floor_zone(backend, outline=[[0, 0], [1, 0], [1, 1]], minus=["fited"])
    with pytest.raises(ValueError, match="give exactly one"):
        await floor_zone(backend)
    with pytest.raises(ValueError, match="no room .R9."):
        await floor_zone(backend, room="R9")
