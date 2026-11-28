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


async def test_a_batch_answers_tags_and_dimension_texts_and_a_list_reads_them_back():
    # round 6: 64 dimensions came back as handles only, the agent counted indices
    # to find 29 of them, and entity_list refused properties.text / .measurement
    from fastmcp import Client

    import server

    async with Client(server.mcp) as client:
        await client.call_tool("drawing_new", {})
        await client.call_tool(
            "dimstyle_create", {"name": "CM", "overrides": {"DIMLFAC": 0.1, "DIMDEC": 0}}
        )
        dim = {"type": "dimension_linear", "x1": 0, "y1": 0, "x2": 2500, "y2": 0, "dim_x": 0}
        made = await client.call_tool(
            "entity_batch_create",
            {
                "entities": [
                    {"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 0},
                    dim | {"dim_y": 300, "tag": "wall-a", "style": "CM"},
                    dim | {"dim_y": 600},
                ]
            },
        )
        out = made.structured_content
        lines, cm, mm = out["handles"]
        assert out["tags"] == {"wall-a": cm}  # only the tagged item
        assert out["dim_texts"] == {cm: "250", mm: out["dim_texts"][mm]}
        assert out["dim_texts"][mm] != "250"  # the current style, not CM
        listed = await client.call_tool(
            "entity_list",
            {
                "type_filter": "DIMENSION",
                "fields": ["handle", "properties.text", "properties.measurement"],
            },
        )
        rows = {row["handle"]: row for row in listed.structured_content["result"]}
        assert rows[cm]["properties.text"] == "250"
        assert rows[cm]["properties.measurement"] == pytest.approx(2500)
        # a duplicate tag is refused before anything is drawn
        twice = await client.call_tool(
            "entity_batch_create",
            {"entities": [dim | {"dim_y": 900, "tag": "t"}, dim | {"dim_y": 990, "tag": "t"}]},
            raise_on_error=False,
        )
        assert twice.is_error and "tag" in twice.content[0].text
        aligned = await client.call_tool(
            "dimension_aligned",
            {"x1": 0, "y1": 0, "x2": 3000, "y2": 4000, "dim_x": -100, "dim_y": 100, "style": "CM"},
        )
        assert aligned.structured_content["properties"]["text"] == "500"
        nope = await client.call_tool(
            "dimension_linear",
            {"x1": 0, "y1": 0, "x2": 1, "y2": 0, "dim_x": 0, "dim_y": 1, "style": "NOPE"},
            raise_on_error=False,
        )
        assert nope.is_error and "invalid_args" in nope.content[0].text


class _Ctx:
    def __init__(self, backend):
        self.lifespan_context = {"backend": backend}

    async def info(self, message):
        pass

    async def report_progress(self, progress, total):
        pass


async def test_batch_defaults_fill_what_an_item_leaves_out_and_skip_types_that_cannot_take_them(
    backend,
):
    # round 8: an agent repeated "style": "ISO", "height": 60 on 70 items
    import server

    await backend.textstyle_create("ISO", "isocp.shx")
    out = await server.entity_batch_create(
        entities=[
            {"type": "text", "text": "A", "x": 0, "y": 0},
            {"type": "text", "text": "B", "x": 0, "y": 100, "height": 20, "color": 5},
            {"type": "mtext", "text": "C", "x": 0, "y": 200, "layer": "0"},
            # height / style mean something else here: not defaults for these types
            {"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 0},
            {"type": "rectangle", "x": 0, "y": 0, "width": 5, "height": 7},
            {
                "type": "dimension_linear",
                "x1": 0,
                "y1": 0,
                "x2": 50,
                "y2": 0,
                "dim_x": 0,
                "dim_y": 9,
            },
            {"type": "callout", "target": [0, 0], "landing": [20, 20], "text": "N"},
        ],
        defaults={"layer": "NOTES", "style": "ISO", "height": 60, "color": 3},
        ctx=_Ctx(backend),
    )
    assert out["ok"] and out["created"] == 7, out
    ents = [backend._doc.entitydb[h] for h in out["handles"]]
    text_a, text_b, mtext, line, rect, dim, _ = ents
    assert (text_a.dxf.layer, text_a.dxf.color) == ("NOTES", 3)
    assert (text_a.dxf.height, text_a.dxf.style) == (60, "ISO")
    assert (text_b.dxf.height, text_b.dxf.color, text_b.dxf.layer) == (20, 5, "NOTES")  # item wins
    assert (mtext.dxf.layer, mtext.dxf.char_height, mtext.dxf.style) == ("0", 60, "ISO")
    assert (line.dxf.layer, line.dxf.color) == ("NOTES", 3)
    # a rectangle's height is its extent, never the lettering height
    assert [tuple(p[:2]) for p in rect.get_points()][2] == (5, 7)
    assert dim.dxf.layer == "NOTES"
    assert dim.dxf.dimstyle != "ISO"  # a dimension's style is a dimension style


async def test_batch_defaults_refusals_name_the_key_and_draw_nothing(backend):
    import server

    line = {"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 0}
    before = len(backend._doc.modelspace())
    for defaults, needle in (
        ({"colour": 3}, "defaults.colour: unknown key; allowed: layer, color, style, height"),
        ({"height": 0}, "defaults.height"),
        ({"color": "red"}, "defaults.color"),
    ):
        with pytest.raises(InvalidArgsError, match=needle):
            await server.entity_batch_create(entities=[line], defaults=defaults, ctx=_Ctx(backend))
    # a default the text style does not exist for fails the item, all or nothing
    out = await server.entity_batch_create(
        entities=[line, {"type": "text", "text": "A", "x": 0, "y": 0}],
        defaults={"style": "NOPE"},
        ctx=_Ctx(backend),
    )
    assert out["ok"] is False and out["rolled_back"]
    assert len(backend._doc.modelspace()) == before


async def test_batch_defaults_reach_every_item_on_a_layout(backend):
    import server

    await backend.layout_create("S1")
    out = await server.entity_batch_create(
        entities=[
            {"type": "text", "text": "A", "x": 0, "y": 0},
            {"type": "point", "x": 1, "y": 1},
        ],
        defaults={"height": 9, "color": 2},
        layout="S1",
        ctx=_Ctx(backend),
    )
    assert out["created"] == 2
    text, point = (backend._doc.entitydb[h] for h in out["handles"])
    assert (text.dxf.height, text.dxf.color, point.dxf.color) == (9, 2, 2)
    assert text.dxf.owner == backend._doc.layouts.get("S1").layout_key
