"""arch_services: a batch of installation points, each wall point turned by its wall.

The plan is two rooms: a 250 mm brick shell 5000 x 4000 on its axis (inner faces
at 125 / 4875 and 125 / 3875) and a 100 mm partition on x = 3000 (faces 2950 and
3050). A wall device's origin must land on the face nearest its point and its
front (+Y of the block) must face the room - the side away from the wall body,
which for the partition differs between its two faces.
"""

from __future__ import annotations

import pytest

from engineering.arch.services import insert_points
from engineering.arch.spec import draw_plan_from_spec
from engineering.errors import InvalidArgsError

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
    result = await insert_points(
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
    result = await insert_points(backend, [{"at": [1000, -50]}, {"at": [2000, 80]}], kind="data")
    assert [(r["at"], r["rotation"]) for r in result["items"]] == [
        ([1000.0, 0.0], 180.0),
        ([2000.0, 0.0], 0.0),
    ]


@pytest.mark.asyncio
async def test_refusals_draw_nothing(backend):
    await _plan(backend)
    before = len(backend._doc.modelspace())
    with pytest.raises(InvalidArgsError, match=r"items\[1\]: no wall face within snap=300"):
        await insert_points(backend, [{"at": [1000, 140]}, {"at": [1500, 2000]}], kind="socket")
    with pytest.raises(
        InvalidArgsError,
        match=r"items\[0\]\.kind: 'outlet' is not a kind\. Nearest: light_outlet\. Kinds - electrical: socket",
    ):
        await insert_points(backend, [{"kind": "outlet", "at": [0, 0]}])
    with pytest.raises(InvalidArgsError, match=r"items\[0\]\.count: a whole number 1-9"):
        await insert_points(backend, [{"kind": "socket", "at": [1000, 140], "count": 0}])
    assert len(backend._doc.modelspace()) == before


@pytest.mark.asyncio
async def test_a_label_never_hides_the_height_and_the_lettering_follows_text_height(backend):
    """Round 4 (A3a/b/c): label and height are two lines, text_height in paper mm."""
    await _plan(backend)
    result = await insert_points(
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
    assert (ref.dxf.layer, ref.dxf.color, result["layers"]) == ("E-OSW", 3, ["E-OSW"])


@pytest.mark.asyncio
async def test_lighting_kinds_runs_frames_and_a_project_block_carry_their_circuit(backend):
    """Round 4 (A3d, B1): a project block anchored like ours with its record, a
    run along a path in its linetype, a switch frame listing its circuits."""
    from engineering.arch.services import APP_ID, read_records

    await _plan(backend)
    await backend.block_define(
        "K", [{"type": "circle", "cx": 0, "cy": 100, "radius": 100}], [], 0, 0
    )
    result = await insert_points(
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
    with pytest.raises(InvalidArgsError, match=r"symbols.wall_light: no block 'NOPE'"):
        await insert_points(
            backend, [{"kind": "wall_light", "at": [1500, 3860]}], symbols={"wall_light": "NOPE"}
        )
    with pytest.raises(InvalidArgsError, match=r"items\[0\]: a led_strip is a run along path"):
        await insert_points(backend, [{"kind": "led_strip", "at": [0, 0]}])
    with pytest.raises(InvalidArgsError, match=r"items\[0\].circuits: a switch_frame lists"):
        await insert_points(backend, [{"kind": "switch_frame", "at": [2940, 2600]}])


@pytest.mark.asyncio
async def test_plumbing_points_go_on_their_trade_layers_with_the_pair_at_150mm(backend):
    """B1: one engine for every trade - water on P-DOMW, drainage on P-SANR, the
    hot / cold pair's outlets 150 mm apart on the face, each record naming its trade."""
    from engineering.arch.services import read_records

    await _plan(backend)
    result = await insert_points(
        backend,
        [
            {"kind": "water_pair", "at": [1000, 140], "height": 600},
            {"kind": "drain", "at": [1500, 140], "height": 500},
            {"kind": "stack", "at": [2700, 300]},
            {"kind": "linear_drain", "path": [[200, 600], [1200, 600]]},
            {"kind": "socket", "at": [2000, 140]},
        ],
    )
    doc = backend._doc
    pair, drain, stack, gully, socket = (doc.entitydb[r["handle"]] for r in result["items"])
    assert [e.dxf.layer for e in (pair, drain, stack, gully, socket)] == [
        "P-DOMW-E-N",
        "P-SANR-E-N",
        "P-SANR-E-N",
        "P-SANR-E-N",
        "A-ELEC-E-N",
    ]
    assert result["trades"] == {"plumbing": 4, "electrical": 1}
    # the two stems start on the face 150 mm apart (real size at any scale)
    stems = [e for e in doc.blocks.get(pair.dxf.name) if e.dxftype() == "LINE"]
    assert sorted(round(e.dxf.start.x, 6) for e in stems) == [-75.0, 75.0]
    assert {e.dxf.start.y for e in stems} == {0.0}
    codes = {a.dxf.tag: a.dxf.text for a in pair.attribs}
    assert (codes["CODE"], codes["CODE2"], codes["H"]) == ("HW", "CW", "H=60CM")
    trades = {p["item"]: p["trade"] for _e, p in read_records(doc)}
    assert trades == {
        "water_pair": "plumbing",
        "drain": "plumbing",
        "stack": "plumbing",
        "linear_drain": "plumbing",
        "socket": "electrical",
    }
    assert gully.dxftype() == "LWPOLYLINE" and result["items"][3]["length"] == 1000.0


@pytest.mark.asyncio
async def test_dim_from_measures_the_point_square_to_the_named_wall(backend):
    """A9-style "in axis" dimension: from P1's face (2950) or axis (3000) to the
    point, whatever wall the point itself snapped to."""
    await _plan(backend)
    result = await insert_points(
        backend,
        [
            {"kind": "cold_water", "at": [2000, 140], "dim_from": "P1"},
            {
                "kind": "radiator",
                "at": [1500, 3860],
                "length": 900,
                "dim_from": {"wall": "P1", "ref": "axis"},
            },
        ],
        kind="socket",
    )
    first, second = result["dimensions"]
    assert first["value"] == pytest.approx(950.0)  # face x = 2950 to x = 2000
    assert second["value"] == pytest.approx(1500.0)  # axis x = 3000 to x = 1500
    dim = backend._doc.entitydb[first["handle"]]
    assert dim.dxftype() == "DIMENSION" and dim.dxf.layer == "P-DOMW-E-N"
    assert dim.get_measurement() == pytest.approx(950.0)
    assert dim.dxf.angle == pytest.approx(0.0)  # measured leftwards, its text still upright
    radiator = backend._doc.entitydb[result["items"][1]["handle"]]
    assert radiator.dxf.name == "ARCH_H_RADIATOR_50_L900"
    with pytest.raises(InvalidArgsError, match=r"items\[0\]\.dim_from\.wall: no arch_wall 'W9'"):
        await insert_points(backend, [{"kind": "drain", "at": [2000, 140], "dim_from": "W9"}])


@pytest.mark.asyncio
async def test_an_unknown_kind_is_refused_with_its_trades_list(backend):
    with pytest.raises(InvalidArgsError) as err:
        await insert_points(backend, [{"kind": "hot_watr", "at": [0, 0]}])
    message = str(err.value)
    assert "Nearest: hot_water" in message
    assert "plumbing: cold_water, hot_water, water_pair" in message
    assert "electrical:" not in message  # only the trade the near name belongs to


@pytest.mark.asyncio
async def test_the_underlay_greys_the_architecture_only_and_is_reversible(backend):
    """B2: every A-* architecture layer grey and thin, the poche light grey,
    the installation layers untouched; on=False puts back exactly what was there."""
    from engineering.arch.draw import UNDERLAY_STATE, arch_underlay

    await _plan(backend)
    await insert_points(
        backend,
        [{"kind": "drain", "at": [1500, 140]}, {"kind": "socket", "at": [2000, 140]}],
    )
    doc = backend._doc

    def props():
        return {
            n: (doc.layers.get(n).color, doc.layers.get(n).dxf.lineweight)
            for n in ("A-WALL-E-N", "A-WALL-H-N", "P-SANR-E-N", "A-ELEC-E-N")
        }

    before = props()
    result = await arch_underlay(backend)
    after = props()
    assert after["A-WALL-E-N"] == (8, 18)
    assert after["A-WALL-H-N"] == (253, 13)  # light grey, its own thinner weight kept
    assert after["P-SANR-E-N"] == before["P-SANR-E-N"]
    assert after["A-ELEC-E-N"] == before["A-ELEC-E-N"]  # A-* but an installation layer
    assert "A-ELEC-E-N" not in result["layers"]
    await arch_underlay(backend, color=9, lineweight=0.13)  # again: the originals survive
    await arch_underlay(backend, on=False)
    assert props() == before
    assert UNDERLAY_STATE not in {s["name"] for s in await backend.layer_state_list()}
    with pytest.raises(InvalidArgsError, match="the underlay is not on"):
        await arch_underlay(backend, on=False)
    with pytest.raises(InvalidArgsError, match="lineweight: an ISO 128 weight"):
        await arch_underlay(backend, lineweight=0.2)
