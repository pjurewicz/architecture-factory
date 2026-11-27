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
    assert first == {"ITEM": "socket", "L": "", "N": "2x", "H": "H=30CM"}
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


@pytest.mark.asyncio
async def test_a_label_never_hides_the_height_and_the_lettering_follows_text_height(backend):
    """Round 4 (A3a/b/c): label and height are two lines, text_height in paper mm."""
    await _plan(backend)
    result = await insert_devices(
        backend,
        [{"kind": "wall_light", "at": [1000, 140], "height": 1500, "label": "HALL 01"}],
        scale=55,
        text_height=1.0,
        layer="E-OSW",
        color=3,
    )
    ref = backend._doc.entitydb[result["items"][0]["handle"]]
    shown = {a.dxf.tag: (a.dxf.text, a.dxf.height) for a in ref.attribs if not a.is_invisible}
    assert shown["L"] == ("HALL 01", 55.0) and shown["H"] == ("H=150CM", 55.0)
    label, height = (next(a for a in ref.attribs if a.dxf.tag == t) for t in ("L", "H"))
    assert label.dxf.align_point.y > height.dxf.align_point.y  # the label above the height
    assert (ref.dxf.layer, ref.dxf.color, result["layer"]) == ("E-OSW", 3, "E-OSW")


@pytest.mark.asyncio
async def test_lighting_kinds_runs_frames_and_a_project_block_carry_their_circuit(backend):
    """Round 4 (A3d, B1): a project block anchored like ours with its record, a
    run along a path in its linetype, a switch frame listing its circuits."""
    from engineering.arch.electrical import APP_ID, read_records

    await _plan(backend)
    await backend.block_define(
        "K", [{"type": "circle", "cx": 0, "cy": 100, "radius": 100}], [], 0, 0
    )
    result = await insert_devices(
        backend,
        [
            {"kind": "wall_light", "at": [1500, 3860], "circuit": "SALON 01", "label": "K1"},
            {"kind": "pendant_light", "at": [1500, 2000], "circuit": "SALON 01"},
            {"kind": "led_strip", "path": [[200, 300], [2800, 300]], "circuit": "SALON 02"},
            {"kind": "track_pendant", "path": [[200, 1000], [2800, 1000]]},
            {"kind": "switch_frame", "at": [2940, 2600], "circuits": ["SALON 01", "SALON 02"]},
        ],
        symbols={"wall_light": "K"},
    )
    doc = backend._doc
    k1, pendant, strip, track, frame = (doc.entitydb[r["handle"]] for r in result["items"])
    assert (k1.dxf.name, k1.dxf.rotation) == ("K", pytest.approx(180.0))  # turned by its wall
    label = next(e for e in doc.modelspace() if e.dxftype() == "TEXT" and e.dxf.text == "K1")
    assert label.has_xdata(APP_ID)  # K has no L tag: lettered as TEXT beside it
    assert label.dxf.align_point.y < 3875 - 100  # below the symbol, in the room
    assert strip.dxftype() == "LWPOLYLINE" and strip.dxf.lineweight == 50
    assert track.dxf.linetype == "DASHED"
    assert result["items"][2]["length"] == 2600.0
    assert result["circuits"] == {"SALON 01": 2, "SALON 02": 1}
    frame_texts = sorted(
        e.dxf.text for e, p in read_records(doc) if p["kind"] == "frame" and e.dxftype() == "TEXT"
    )
    assert frame_texts == ["SALON 01", "SALON 02"]
    assert all(
        e.dxf.rotation == 0 for e, p in read_records(doc) if e.dxftype() == "TEXT"
    )  # horizontal on a vertical wall
    items = {p["item"] for _e, p in read_records(doc)}
    assert items == {"wall_light", "pendant_light", "led_strip", "track_pendant", "switch_frame"}
    with pytest.raises(ValueError, match=r"symbols.wall_light: no block 'NOPE'"):
        await insert_devices(
            backend, [{"kind": "wall_light", "at": [1500, 3860]}], symbols={"wall_light": "NOPE"}
        )
    with pytest.raises(ValueError, match=r"items\[0\]: a led_strip is a run along path"):
        await insert_devices(backend, [{"kind": "led_strip", "at": [0, 0]}])
    with pytest.raises(ValueError, match=r"items\[0\].circuits: a switch_frame lists"):
        await insert_devices(backend, [{"kind": "switch_frame", "at": [2940, 2600]}])
