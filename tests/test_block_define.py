"""block_define: a definition with ATTDEFs from typed specs."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio

VALVE = [
    {"type": "polyline", "points": [[-4, -2], [-4, 2], [0, 0]], "closed": True},
    {"type": "polyline", "points": [[4, -2], [4, 2], [0, 0]], "closed": True},
    {"type": "circle", "cx": 0, "cy": 0, "radius": 1.5},
    {"type": "arc", "cx": 0, "cy": 2, "radius": 2, "start_angle": 0, "end_angle": 180},
    {"type": "line", "x1": 0, "y1": 0, "x2": 0, "y2": 4},
    {"type": "text", "text": "M", "x": 0, "y": 6, "height": 2, "align": "middle_center"},
    {"type": "solid", "points": [[0, 0], [-3, 1.2], [-3, -1.2]]},
]
ATTDEFS = [
    {"tag": "TAG", "x": 0, "y": 5.5, "height": 2.5, "align": "center"},
    {"tag": "LINK", "x": 0, "y": -5, "height": 2.0, "invisible": True, "default": "L-1"},
]


# ── headless engine ─────────────────────────────────────────────────────────


async def test_defines_geometry_and_attdefs(backend):
    result = await backend.block_define("PID_VALVE_TEST", VALVE, ATTDEFS)
    assert result == {
        "ok": True,
        "name": "PID_VALVE_TEST",
        "entity_count": 7,
        "attdef_count": 2,
        "replaced": False,
        "layers_created": [],
        "backend": "ezdxf",
    }
    blk = backend._doc.blocks.get("PID_VALVE_TEST")
    types_ = sorted(e.dxftype() for e in blk)
    assert types_ == [
        "ARC",
        "ATTDEF",
        "ATTDEF",
        "CIRCLE",
        "LINE",
        "LWPOLYLINE",
        "LWPOLYLINE",
        "SOLID",
        "TEXT",
    ]
    attdefs = {e.dxf.tag: e for e in blk if e.dxftype() == "ATTDEF"}
    assert attdefs["LINK"].is_invisible is True
    assert attdefs["LINK"].dxf.text == "L-1"
    assert attdefs["TAG"].is_invisible is False
    for e in blk:
        assert e.dxf.layer == "0"
        assert e.dxf.color == 0, "ByBlock colour so the INSERT's layer drives it"
        # W2-T3-06 — RED: the ATTDEFs kept BYLAYER, and no member had a ByBlock
        # lineweight; the live engine styles every member ByBlock (measured:
        # Linetype 'ByBlock', Lineweight -2 on AcDbLine and the ATTDEF).
        assert e.dxf.linetype.upper() == "BYBLOCK", e.dxftype()
        assert e.dxf.lineweight == -2, e.dxftype()


async def test_insert_after_define_carries_the_attributes(backend):
    await backend.block_define("PID_VALVE_TEST", VALVE, ATTDEFS)
    ref = await backend.block_insert("PID_VALVE_TEST", 10, 10, attributes={"TAG": "HV-101"})
    assert await backend.block_get_attributes(ref.handle) == {"TAG": "HV-101", "LINK": "L-1"}


async def test_name_clash_is_refused_unless_overwrite(backend):
    await backend.block_define("PID_X", VALVE)
    with pytest.raises(ValueError, match="overwrite"):
        await backend.block_define("PID_X", VALVE)
    result = await backend.block_define("PID_X", VALVE[:1], overwrite=True)
    assert result["replaced"] is True and result["entity_count"] == 1
    assert len(list(backend._doc.blocks.get("PID_X"))) == 1


async def test_overwrite_keeps_existing_inserts_pointing_at_the_name(backend):
    await backend.block_define("PID_X", VALVE)
    ref = await backend.block_insert("PID_X", 0, 0)
    await backend.block_define(
        "PID_X", [{"type": "circle", "cx": 0, "cy": 0, "radius": 9}], overwrite=True
    )
    info = await backend.entity_get(ref.handle)
    assert info.properties["block_name"] == "PID_X"
    virtual = list(backend._doc.entitydb[ref.handle].virtual_entities())
    assert [e.dxftype() for e in virtual] == ["CIRCLE"]


async def test_invalid_request_writes_nothing(backend):
    with pytest.raises(TypeError, match=r"entities\[1\]"):
        await backend.block_define("PID_BAD", [VALVE[0], {"type": "circle", "cx": 0, "cy": 0}])
    assert "PID_BAD" not in backend._doc.blocks
    with pytest.raises(ValueError, match="empty"):
        await backend.block_define("PID_EMPTY", [])
    assert "PID_EMPTY" not in backend._doc.blocks


async def test_anonymous_names_are_refused(backend):
    with pytest.raises(ValueError, match="anonymous"):
        await backend.block_define("*U9", VALVE)


async def test_solid_uses_bowtie_order(backend):
    await backend.block_define(
        "PID_SOLID", [{"type": "solid", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    )
    solid = next(e for e in backend._doc.blocks.get("PID_SOLID") if e.dxftype() == "SOLID")
    assert tuple(solid.dxf.vtx2)[:2] == (0.0, 1.0)
    assert tuple(solid.dxf.vtx3)[:2] == (1.0, 1.0)


async def test_a_wipeout_masks_its_outline_and_survives_a_reload(backend, tmp_path):
    """A non-rectangular outline, written before the lines it sits under."""
    outline = [[0, 0], [40, 0], [40, 10], [10, 30], [0, 30]]
    await backend.block_define(
        "MASKED",
        [
            {"type": "wipeout", "points": outline},
            {"type": "polyline", "points": outline, "closed": True},
        ],
    )
    path = tmp_path / "masked.dxf"
    await backend.drawing_save_as(str(path))
    import ezdxf

    blk = ezdxf.readfile(str(path)).blocks.get("MASKED")
    first = next(iter(blk))
    assert first.dxftype() == "WIPEOUT"
    assert [(round(v.x), round(v.y)) for v in first.boundary_path_wcs()][:5] == [
        tuple(p) for p in outline
    ]
    assert first.dxf.color == 0 and first.dxf.layer == "0"


async def test_block_text_takes_every_shared_alignment(backend):
    await backend.block_define(
        "LBL",
        [{"type": "text", "text": "M", "x": 3, "y": 4, "height": 1, "align": "top_right"}],
        [],
    )
    (text,) = backend._doc.blocks.get("LBL").query("TEXT")
    assert text.get_align_enum().name == "TOP_RIGHT" and tuple(text.dxf.align_point)[:2] == (3, 4)
