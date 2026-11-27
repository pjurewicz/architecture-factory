"""Tests for entity creation — all entity types via ezdxf backend."""

from __future__ import annotations

import pytest

from engineering.errors import InvalidArgsError

pytestmark = pytest.mark.asyncio


async def test_create_circle(backend):
    info = await backend.entity_create_circle(50, 50, 25)
    assert info.handle
    assert info.type == "CIRCLE"
    got = await backend.entity_get(info.handle)
    assert got.type == "CIRCLE"
    assert got.properties.get("radius") == pytest.approx(25, abs=0.01)


async def test_create_arc(backend):
    info = await backend.entity_create_arc(0, 0, 50, 0, 90)
    assert info.handle
    assert info.type == "ARC"


async def test_create_polyline(backend):
    pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    info = await backend.entity_create_polyline(pts, closed=True)
    assert info.handle
    assert info.type == "LWPOLYLINE"


async def test_create_polyline_open(backend):
    pts = [[0, 0], [50, 50], [100, 0]]
    info = await backend.entity_create_polyline(pts, closed=False)
    assert info.handle
    assert info.type == "LWPOLYLINE"


async def test_create_text(backend):
    info = await backend.entity_create_text("Hello", 10, 20, height=5.0)
    assert info.handle
    assert info.type == "TEXT"


async def test_create_mtext(backend):
    info = await backend.entity_create_mtext("Multi\\Pline Text", 0, 0, width=200)
    assert info.handle
    assert info.type == "MTEXT"


async def test_create_hatch(backend):
    boundary = [[0, 0], [100, 0], [100, 100], [0, 100]]
    info = await backend.entity_create_hatch("SOLID", boundary, scale=1.0, angle=0)
    assert info.handle
    assert info.type == "HATCH"


async def test_create_spline(backend):
    pts = [[0, 0], [25, 50], [50, 25], [75, 75], [100, 0]]
    info = await backend.entity_create_spline(pts)
    assert info.handle
    assert info.type == "SPLINE"


async def test_create_ellipse(backend):
    info = await backend.entity_create_ellipse(50, 50, 40, 0, ratio=0.5)
    assert info.handle
    assert info.type == "ELLIPSE"


async def test_create_point(backend):
    info = await backend.entity_create_point(10, 20)
    assert info.handle
    assert info.type == "POINT"


async def test_create_line_with_layer(backend):
    await backend.layer_create("TEST_LAYER", color=1)
    info = await backend.entity_create_line(0, 0, 100, 0, layer="TEST_LAYER")
    assert info.handle
    assert info.layer == "TEST_LAYER"


async def test_create_circle_with_color(backend):
    info = await backend.entity_create_circle(0, 0, 10, color=3)
    assert info.handle
    got = await backend.entity_get(info.handle)
    assert got.color == 3


async def test_create_entity_appears_in_list(backend):
    await backend.entity_create_line(0, 0, 10, 0)
    await backend.entity_create_circle(50, 50, 5)
    await backend.entity_create_point(0, 0)
    entities = await backend.entity_list()
    assert len(entities) == 3
    types = {e.type for e in entities}
    assert "LINE" in types
    assert "CIRCLE" in types
    assert "POINT" in types


async def test_batch_create_draws_every_basic_type_in_one_call(backend):
    await backend.block_define(
        "VALVE",
        [{"type": "circle", "cx": 0, "cy": 0, "radius": 2}],
        [{"tag": "TAG", "x": 0, "y": 3, "height": 1}],
    )
    specs = [
        {"type": "line", "x1": 0, "y1": 0, "x2": 100, "y2": 0},
        {"type": "polyline", "points": [[0, 0], [10, 0], [10, 10]], "bulges": [0.5, 0, 0]},
        {"type": "rectangle", "x1": 0, "y1": 0, "x2": 10, "y2": 5},
        {"type": "rectangle", "x": 20, "y": 1, "width": 10, "height": 5},
        {"type": "circle", "cx": 50, "cy": 50, "radius": 25},
        {"type": "arc", "cx": 0, "cy": 0, "radius": 3, "start_angle": 0, "end_angle": 90},
        {"type": "hatch", "pattern": "ANSI31", "boundary_points": [[0, 0], [5, 0], [5, 5]]},
        {"type": "text", "text": "A", "x": 5, "y": 6, "align": "Middle-Center"},
        {"type": "mtext", "text": "B", "x": 1, "y": 2, "align": "bottom_right"},
        {"type": "insert", "name": "VALVE", "x": 30, "y": 30, "attributes": {"TAG": "V-101"}},
        {
            "type": "dimension_linear",
            "x1": 0,
            "y1": 0,
            "x2": 20,
            "y2": 0,
            "dim_x": 10,
            "dim_y": -8,
            "fit": "H7",
        },
    ]
    result = await backend.entity_batch_create(specs)
    assert result["errors"] == []
    got = [e["type"] for e in result["entities"]]
    assert got == [
        "LINE",
        *["LWPOLYLINE"] * 3,
        "CIRCLE",
        "ARC",
        "HATCH",
        "TEXT",
        "MTEXT",
        "INSERT",
        "DIMENSION",
    ]
    ents = {e["type"]: backend._doc.entitydb[e["handle"]] for e in result["entities"]}
    rect = backend._doc.entitydb[result["entities"][3]["handle"]]
    assert [tuple(p) for p in rect.get_points("xy")][2] == (30, 6)
    text = ents["TEXT"]
    assert text.get_align_enum().name == "MIDDLE_CENTER"
    assert tuple(text.dxf.align_point)[:2] == (5, 6)  # aligned ON the point, not shifted
    assert ents["MTEXT"].dxf.attachment_point == 9
    assert [(a.dxf.tag, a.dxf.text) for a in ents["INSERT"].attribs] == [("TAG", "V-101")]
    dim = ents["DIMENSION"]
    assert dim.get_measurement() == pytest.approx(20) and dim.dxf.text == "<> H7"


async def test_batch_create_refuses_an_unknown_align_by_name(backend):
    result = await backend.entity_batch_create(
        [{"type": "text", "text": "A", "x": 0, "y": 0, "align": "diagonal"}]
    )
    assert result["created"] == 0 and "'diagonal'" in result["errors"][0]["error"]


async def test_batch_create_shared_linetype_style_and_line_spacing(backend):
    await backend.textstyle_create("ISO", "isocp.shx")
    result = await backend.entity_batch_create(
        [
            {
                "type": "arc",
                "cx": 0,
                "cy": 0,
                "radius": 5,
                "start_angle": 0,
                "end_angle": 90,
                "linetype": "DASHDOT",
            },
            {
                "type": "rectangle",
                "x1": 0,
                "y1": 0,
                "x2": 5,
                "y2": 5,
                "linetype": "HIDDEN",
                "color": 3,
            },
            {"type": "text", "text": "A", "x": 0, "y": 0, "style": "ISO"},
            {"type": "mtext", "text": "A\\PB", "x": 0, "y": 0, "style": "ISO", "line_spacing": 1.5},
        ]
    )
    assert result["errors"] == []
    arc, rect, text, mtext = (backend._doc.entitydb[e["handle"]] for e in result["entities"])
    assert (arc.dxf.linetype, rect.dxf.linetype) == ("DASHDOT", "HIDDEN")
    assert rect.dxf.color == 3
    assert "DASHDOT" in backend._doc.linetypes  # loaded, not just named
    assert text.dxf.style == mtext.dxf.style == "ISO"
    assert mtext.dxf.line_spacing_factor == pytest.approx(1.5)

    before = len(backend._doc.modelspace())
    refused = await backend.entity_batch_create(
        [{"type": "text", "text": "A", "x": 0, "y": 0, "style": "NOPE"}]
    )
    assert refused["created"] == 0 and "'NOPE'" in refused["errors"][0]["error"]
    assert len(backend._doc.modelspace()) == before  # the refused text was removed


async def test_a_text_given_a_style_takes_its_width_factor_unless_one_is_given(backend):
    await backend.textstyle_create("NARROW", "isocp.shx", width_factor=0.7)
    text = await backend.entity_create_text("A", 0, 0, height=2.5)
    mtext = await backend.entity_create_mtext("B", 0, 5, height=2.5)
    await backend.entity_edit_text(text.handle, style="NARROW")
    assert backend._doc.entitydb[text.handle].dxf.width == pytest.approx(0.7)
    await backend.entity_edit_text(text.handle, width_factor=1.2)
    assert backend._doc.entitydb[text.handle].dxf.width == pytest.approx(1.2)
    with pytest.raises(InvalidArgsError, match="width_factor is a TEXT's"):
        await backend.entity_edit_text(mtext.handle, width_factor=0.8)


async def test_edit_text_sets_height_and_style(backend):
    await backend.textstyle_create("ISO", "isocp.shx")
    info = await backend.entity_create_text("A", 0, 0, height=2.5)
    await backend.entity_edit_text(info.handle, height=2.0, style="ISO")
    ent = backend._doc.entitydb[info.handle]
    assert (ent.dxf.height, ent.dxf.style) == (2.0, "ISO")
    with pytest.raises(Exception, match="line_spacing"):
        await backend.entity_edit_text(info.handle, line_spacing=1.5)
