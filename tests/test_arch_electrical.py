"""arch_electrical: a batch of outlets and devices, each wall device turned by its wall.

The plan is two rooms: a 250 mm brick shell 5000 x 4000 on its axis (inner faces
at 125 / 4875 and 125 / 3875) and a 100 mm partition on x = 3000 (faces 2950 and
3050). A wall device's origin must land on the face nearest its point and its
front (+Y of the block) must face the room - the side away from the wall body,
which for the partition differs between its two faces.
"""

from __future__ import annotations

import pytest

from engineering.arch.electrical import insert_devices
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
        },
        {"id": "P1", "axis": [[3000, 0], [3000, 4000]], "thickness": 100, "material": "aac"},
    ],
}


async def _plan(backend):
    await draw_plan_from_spec(backend, SPEC)


@pytest.mark.asyncio
async def test_each_wall_device_lands_on_its_face_facing_the_room(backend):
    await _plan(backend)
    result = await insert_devices(
        backend,
        [
            {"at": [1000, 140], "count": 2, "height": 300},  # south face, room above
            {"at": [140, 2000], "height": 300},  # west face, room to the right
            {"at": [1500, 3860]},  # north face, room below
            {"at": [2940, 2600], "kind": "switch", "height": 1100},  # partition, west room
            {"at": [3060, 1000], "kind": "tv"},  # partition, east room
            {"at": [1500, 2000], "kind": "light_outlet", "rotation": 30},
        ],
        kind="socket",
        scale=50,
    )
    rows = result["items"]
    assert [(r["at"], r.get("rotation")) for r in rows] == [
        ([1000.0, 125.0], 0.0),
        ([125.0, 2000.0], 270.0),
        ([1500.0, 3875.0], 180.0),
        ([2950.0, 2600.0], 90.0),
        ([3050.0, 1000.0], 270.0),
        ([1500.0, 2000.0], None),  # a free device stays where it is put
    ]
    assert result["counts"] == {"socket": 4, "switch": 1, "tv": 1, "light_outlet": 1}
    doc = backend._doc
    for row, turned in zip(rows, (0, 270, 180, 90, 270, 30), strict=True):
        ref = doc.entitydb[row["handle"]]
        assert ref.dxf.layer == "A-ELEC-E-N"
        assert ref.dxf.rotation == pytest.approx(turned)
        visible = [a for a in ref.attribs if not a.is_invisible and a.dxf.text]
        # the lettering reads horizontally whatever the wall
        assert all(a.dxf.rotation == 0.0 for a in visible)
    first = {a.dxf.tag: a.dxf.text for a in doc.entitydb[rows[0]["handle"]].attribs}
    assert first == {"ITEM": "socket", "N": "2x", "H": "H=30CM"}
    tv = {a.dxf.tag: a.dxf.text for a in doc.entitydb[rows[4]["handle"]].attribs}
    assert tv["CODE"] == "TV" and tv["N"] == "" and tv["H"] == ""
    # one block per kind and scale, reused by the second socket
    assert result["blocks_defined"] == [
        "ARCH_E_SOCKET_50",
        "ARCH_E_SWITCH_50",
        "ARCH_E_TV_50",
        "ARCH_E_LIGHT_OUTLET_50",
    ]


@pytest.mark.asyncio
async def test_a_bare_wall_line_takes_its_room_side_from_the_point(backend):
    await backend.layer_create("A-WALL-E-N")
    await backend.entity_create_line(0, 0, 4000, 0, layer="A-WALL-E-N")
    result = await insert_devices(backend, [{"at": [1000, -50]}, {"at": [2000, 80]}], kind="data")
    assert [(r["at"], r["rotation"]) for r in result["items"]] == [
        ([1000.0, 0.0], 180.0),
        ([2000.0, 0.0], 0.0),
    ]


@pytest.mark.asyncio
async def test_refusals_draw_nothing(backend):
    await _plan(backend)
    before = len(backend._doc.modelspace())
    with pytest.raises(ValueError, match=r"items\[1\]: no wall face within snap=300"):
        await insert_devices(backend, [{"at": [1000, 140]}, {"at": [1500, 2000]}], kind="socket")
    with pytest.raises(ValueError, match=r"items\[0\]\.kind: 'outlet' is not one of socket"):
        await insert_devices(backend, [{"kind": "outlet", "at": [0, 0]}])
    with pytest.raises(ValueError, match=r"items\[0\]\.count: a whole number 1-9"):
        await insert_devices(backend, [{"kind": "socket", "at": [1000, 140], "count": 0}])
    assert len(backend._doc.modelspace()) == before
