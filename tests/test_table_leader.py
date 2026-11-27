from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_ezdxf_table_returns_composite_contract(backend):
    result = await backend.entity_create_table(
        10,
        100,
        rows=[["P-01", "4"], ["P-02", "2"]],
        headers=["Part", "Qty"],
        title="BOM",
        layer="TEXT",
    )

    assert result.type == "TABLE"
    assert result.layer == "TEXT"
    assert result.properties["representation"] == "composite"
    assert result.properties["logical_group_id"].startswith("table:")
    assert len(result.properties["child_handles"]) >= 10
    assert result.properties["bounds"] == {"min": [10.0, 72.0], "max": [30.0, 100.0]}


async def test_ezdxf_table_title_row_is_merged_across_the_table(backend):
    # AutoCAD's TABLE merges its title row. Drawn as one column's cell, a title
    # such as 'DOOR SCHEDULE' wrapped inside the first column and ran over the
    # header row below it. The title spans the table, and only the table's outer
    # edges cross its row.
    result = await backend.entity_create_table(
        0,
        100,
        rows=[["D1", "900"]],
        headers=["TAG", "WIDTH"],
        column_widths=[20, 30],
        row_height=10,
        text_height=2.5,
        title="DOOR SCHEDULE",
    )
    children = [backend._doc.entitydb[h] for h in result.properties["child_handles"]]
    (title,) = [e for e in children if e.dxftype() == "MTEXT" and e.text == "DOOR SCHEDULE"]
    assert title.dxf.width == pytest.approx(50.0 - 2 * 1.0)  # padding min(1.0, 10 x 0.15)
    verticals = [e for e in children if e.dxftype() == "LINE" and e.dxf.start.x == e.dxf.end.x]
    inner = [v for v in verticals if 0.0 < v.dxf.start.x < 50.0]
    assert inner and all(max(v.dxf.start.y, v.dxf.end.y) == pytest.approx(90.0) for v in inner)
    outer = [v for v in verticals if v.dxf.start.x in (0.0, 50.0)]
    assert len(outer) == 2 and all(max(v.dxf.start.y, v.dxf.end.y) == 100.0 for v in outer)


async def test_ezdxf_table_cell_margin_scales_with_the_lettering(backend):
    # A 1:50 schedule letters at 125 in rows of 350. A flat 1.0 margin is 0.02 mm
    # on that sheet: the text sat on the cell line, and 'LEFT' read as 'FT'.
    result = await backend.entity_create_table(
        0,
        0,
        rows=[["D1", "LEFT"]],
        column_widths=[750, 1000],
        row_height=350,
        text_height=125,
    )
    children = [backend._doc.entitydb[h] for h in result.properties["child_handles"]]
    (left,) = [e for e in children if e.dxftype() == "MTEXT" and e.text == "LEFT"]
    assert left.dxf.insert.x == pytest.approx(750 + 50.0)  # 0.4 x 125
    assert left.dxf.insert.y == pytest.approx(-50.0)
    assert left.dxf.width == pytest.approx(1000 - 2 * 50.0)


async def test_ezdxf_table_auto_width_fits_the_measured_text(backend):
    # Eight capitals W are 27.1 at 2.5 lettering; the old 0.7-per-character
    # estimate gave the column 14 and the text ran over the next cell.
    from ezdxf.fonts import fonts

    wide = fonts.make_font("DejaVuSans.ttf", 2.5).text_width("WWWWWWWW")
    result = await backend.entity_create_table(0, 0, rows=[["A1", "WWWWWWWW"]])
    assert result.properties["column_widths"][1] == pytest.approx(wide + 2 * 1.0)
    assert result.properties["text_fit"] == {"wrapped": 0, "overflowing": 0}


async def test_ezdxf_table_fixed_width_wraps_and_grows_only_that_row(backend):
    result = await backend.entity_create_table(
        0,
        0,
        rows=[["Pokrywa łożyska z uszczelnieniem", "x"], ["M8", "WWWWMMMM"]],
        column_widths=[30, 10],
    )
    # 28 wide at 2.5 lettering holds 11.2 text heights; "Pokrywa łożyska" is
    # 11.6, so three lines - 2 x 1.0 margin + 2.5 + 2 x 5/3 x 2.5 pitch + 0.3 x
    # 2.5 descent - and the second row keeps its 7.0; one word wider than its
    # cell cannot wrap and is reported, not hidden
    assert result.properties["bounds"]["min"][1] == pytest.approx(
        -(2 + 2.5 + 2 * 2.5 * 5 / 3 + 0.75) - 7.0
    )
    assert result.properties["text_fit"] == {"wrapped": 1, "overflowing": 1}


async def test_ezdxf_table_survives_save_and_reopen(backend, tmp_path):
    result = await backend.entity_create_table(0, 30, rows=[["A", "1"]], headers=["Name", "Qty"])
    path = tmp_path / "table.dxf"
    await backend.drawing_save_as(str(path), fmt="dxf")
    await backend.drawing_open(str(path))

    entities = await backend.entity_list(limit=100)
    handles = {entity.handle for entity in entities}
    texts = {
        entity.properties.get("text") for entity in entities if entity.type in {"TEXT", "MTEXT"}
    }
    assert set(result.properties["child_handles"]).issubset(handles)
    assert {"Name", "Qty", "A", "1"}.issubset(texts)


async def test_table_rejects_ragged_rows_and_size_limit(backend):
    with pytest.raises(RuntimeError, match="same number of columns"):
        await backend.entity_create_table(0, 0, rows=[["A", "B"], ["C"]])

    with pytest.raises(RuntimeError, match="200 rows"):
        await backend.entity_create_table(0, 0, rows=[["A"]] * 201)


async def test_ezdxf_mleader_returns_composite_contract(backend):
    result = await backend.leader_create_mleader(
        [[0, 0], [10, 5], [30, 5]],
        "SURFACE A",
        layer="DIM",
    )

    assert result.type == "MLEADER"
    assert result.properties["representation"] == "composite"
    assert result.properties["logical_group_id"].startswith("mleader:")
    assert result.properties["text"] == "SURFACE A"
    assert len(result.properties["child_handles"]) == 3


async def test_mleader_requires_two_points(backend):
    with pytest.raises(RuntimeError, match="at least two points"):
        await backend.leader_create_mleader([[0, 0]], "note")


async def test_server_registers_table_and_mleader_tools():
    import server

    names = {tool.name for tool in await server._registered_tools()}
    assert "entity_create_table" in names
    assert "leader_create_mleader" in names


async def test_table_and_mtext_draw_on_a_named_layout_without_switching(backend):
    await backend.layout_create("S")
    sheet = backend._doc.layouts.get("S")
    table = await backend.entity_create_table(0, 50, rows=[["A", "1"]], layout="s")
    note = await backend.entity_create_mtext("NOTE", 5, 5, layout="S")
    on_sheet = {e.dxf.handle for e in sheet}
    assert set(table.properties["child_handles"]) <= on_sheet
    assert note.handle in on_sheet
    assert backend._current_space == "Model", "the current layout is not switched"
    assert not list(backend._doc.modelspace())
    model_note = await backend.entity_create_mtext("M", 0, 0)
    assert model_note.handle in {e.dxf.handle for e in backend._doc.modelspace()}
    from engineering.errors import InvalidArgsError

    with pytest.raises(InvalidArgsError, match="layout 'NOPE' does not exist"):
        await backend.entity_create_table(0, 0, rows=[["x"]], layout="NOPE")
