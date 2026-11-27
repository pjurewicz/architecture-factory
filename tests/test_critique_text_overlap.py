"""The ``text_overlap`` focus: lettering printed over lettering or an arch symbol.

``dim_overlap`` compared dimensions with dimensions only: a room label
dropped on an elevation mark, a note over a chain's text or two labels on top
of each other critiqued clean. The finished sheet below - room labels,
elevation and section marks, a grid, dimension chains, a schedule TABLE,
furniture with labels inside, a numbered callout, a paper layout with a frame
and title block - must stay silent; each pair kind must be found on it.
"""

from __future__ import annotations

import pytest

from engineering.critique import run_critique
from engineering.qa_coverage import ledger_of

FOCUS = ["text_overlap"]
SCALE = 50

SPEC: dict = {
    "sheet": {"intent": "Flat, plan", "sheet_size": "A3", "scale": SCALE},
    "lang": "pl",
    "walls": [
        {
            "id": "EXT",
            "axis": [[0, 0], [7000, 0], [7000, 4500], [0, 4500]],
            "thickness": 250,
            "closed": True,
            "material": "brick",
        },
        {"id": "P1", "axis": [[4000, 0], [4000, 4500]], "thickness": 100, "material": "aac"},
    ],
    "openings": [
        {"id": "D1", "wall": "P1", "kind": "door", "offset": 1500, "width": 900, "tag": "D1"},
        {"id": "D2", "wall": "EXT", "kind": "door", "offset": 5000, "width": 1000, "tag": "D2"},
        {
            "id": "W1",
            "wall": "EXT",
            "kind": "window",
            "offset": 1200,
            "width": 1500,
            "sill": 900,
            "height": 1400,
            "tag": "W1",
        },
    ],
    "rooms": [
        {"id": "R1", "name": "POKÓJ DZIENNY", "number": "01", "at": [2000, 2600]},
        {"id": "R2", "name": "KUCHNIA", "number": "02", "at": [5500, 2600]},
    ],
    "schedules": [{"kind": "rooms", "at": [10500, 4500]}],
}

#: The elevation mark in the living room, well clear of its label.
ELEVATION_AT = (800.0, 3700.0)


async def _sheet(backend) -> dict:
    """A finished architectural sheet, drawn only with the server's own engines."""
    from engineering.arch.catalogue import insert_item
    from engineering.arch.draw import draw_chains, read_plan
    from engineering.arch.spec import draw_plan_from_spec
    from engineering.arch.symbols import draw_grid, draw_symbol
    from engineering.sheet.revision import add_revision
    from engineering.sheet.titleblock import TitleBlockMetadata, apply_titleblock

    plan = await draw_plan_from_spec(backend, SPEC)
    await draw_grid(backend, [0.0, 7000.0], [0.0, 4500.0], extension=1700.0, scale=SCALE)
    drawn = await read_plan(backend)
    chains = await draw_chains(
        backend, drawn["walls"], drawn["openings"], sides=("bottom", "left"), scale=SCALE
    )
    elevation = await draw_symbol(
        backend,
        "elevation_mark",
        ELEVATION_AT,
        scale=SCALE,
        label="1",
        directions={"up": "A", "right": "B"},
    )
    await draw_symbol(
        backend,
        "section_mark",
        (1300.0, -2600.0),
        scale=SCALE,
        p1=[0.0, 0.0],
        p2=[0.0, 9700.0],
        label="X",
        side="left",
    )
    await draw_symbol(backend, "north_arrow", (8600.0, 6000.0), scale=SCALE)
    # furniture: a catalogue sofa and wardrobe, a TEXT label inside the
    # wardrobe's outline, and a desk block whose ATTRIB label sits inside it
    await insert_item(backend, "sofa_3_seat", at=(600.0, 400.0))
    wardrobe = await insert_item(backend, "wardrobe", at=(4300.0, 3700.0))
    await backend.entity_create_text("SZAFA", 4500.0, 3950.0, 100.0, layer="TEXT")
    await backend.block_define(
        "DESK",
        [
            {
                "type": "polyline",
                "points": [[0, 0], [1400, 0], [1400, 700], [0, 700]],
                "closed": True,
            }
        ],
        attdefs=[{"tag": "LABEL", "default": "BIURKO", "x": 200, "y": 300, "height": 100}],
    )
    desk = await backend.block_insert("DESK", 5300.0, 400.0, attributes={"LABEL": "BIURKO"})
    callout = await backend.callout(
        [6800.0, 2000.0], [8000.0, 1200.0], number=3, text_height=2.5 * SCALE, layer="TEXT"
    )
    await backend.entity_create_mtext(
        "UWAGI: wymiary w mm", 10500.0, 6000.0, width=0, height=2.5 * SCALE, layer="TEXT"
    )
    await backend.layout_create("A3 RZUT")
    await backend.viewport_create(
        "A3 RZUT", 215.0, 150.0, 370.0, 270.0, 6000.0, 2000.0, scale=1.0 / SCALE
    )
    await apply_titleblock(
        backend,
        size="A3",
        metadata=TitleBlockMetadata(
            title="RZUT PARTERU",
            drawing_no="ST-01",
            scale=f"1:{SCALE}",
            drawn_by="autocad-mcp-pro",
            date="2026-11-27",
            revision="A",
        ),
        projection=None,
        frame=True,
        zones=True,
        layout="A3 RZUT",
    )
    await add_revision(
        backend,
        rev="A",
        description="Wydanie",
        date="2026-11-27",
        by="pj",
        size="A3",
        layout="A3 RZUT",
    )
    return {
        "plan": plan,
        "chains": chains,
        "elevation": elevation,
        "wardrobe": wardrobe,
        "desk": desk,
        "callout": callout,
    }


def _text_items(backend, kind):
    doc = backend._doc
    return [e for e in doc.modelspace() if e.dxftype() == kind]


def _room_label(backend, text):
    (label,) = [e for e in _text_items(backend, "TEXT") if e.dxf.text == text]
    return label


def _codes(issues):
    return sorted(i.detail["code"] for i in issues)


@pytest.mark.asyncio
async def test_a_finished_architectural_sheet_is_silent_and_every_space_is_covered(backend):
    await _sheet(backend)
    shared: dict = {}
    issues = await run_critique(backend, FOCUS, shared=shared)
    assert issues == []
    rows = {r.layout: r for r in ledger_of(shared).rows if r.focus == "text_overlap"}
    assert set(rows) == {"Model", "Layout1", "A3 RZUT"}
    assert all(r.status == "checked" for r in rows.values())
    # Not vacuous: room labels, grid labels, chain texts, symbol outlines in
    # model space; the title block's texts on the sheet.
    assert rows["Model"].n > 40 and rows["A3 RZUT"].n > 20


@pytest.mark.asyncio
async def test_each_pair_kind_is_found_on_the_finished_sheet(backend):
    await _sheet(backend)
    kitchen = _room_label(backend, "KUCHNIA")
    # 1. a TEXT over a room label, and a TEXT over the desk's ATTRIB
    over_label = await backend.entity_create_text(
        "KUCHNIA", kitchen.dxf.insert.x, kitchen.dxf.insert.y, 175.0, layer="TEXT"
    )
    over_attrib = await backend.entity_create_text("STOL", 5500.0, 700.0, 100.0, layer="TEXT")
    # 2. a TEXT over a chain dimension's text
    dim = next(e for e in backend._doc.modelspace() if e.dxftype() == "DIMENSION")
    dim_text = next(v for v in dim.virtual_entities() if v.dxftype() == "MTEXT")
    over_dim = await backend.entity_create_mtext(
        "NOTA",
        dim_text.dxf.insert.x,
        dim_text.dxf.insert.y,
        width=0,
        height=175.0,
        layer="TEXT",
        align="middle_center",
    )
    # 3. a room label dropped on the elevation mark's bubble
    over_symbol = await backend.entity_create_text(
        "ŁAZIENKA", ELEVATION_AT[0] - 150.0, ELEVATION_AT[1] - 50.0, 175.0, layer="A-AREA-IDEN"
    )
    issues = await run_critique(backend, FOCUS)
    found = {frozenset(i.handles): i for i in issues}
    label_pair = found[frozenset((over_label.handle, kitchen.dxf.handle))]
    assert label_pair.detail["code"] == "text_overlaps_text"
    (attrib_pair,) = [i for i in issues if over_attrib.handle in i.handles]
    assert attrib_pair.detail["code"] == "text_overlaps_text"
    assert attrib_pair.detail["items"][1]["type"] == "ATTRIB"
    assert found[frozenset((over_dim.handle, dim.dxf.handle))].detail["code"] == (
        "text_overlaps_dimension"
    )
    on_mark = [i for i in issues if over_symbol.handle in i.handles]
    (symbol_pair,) = [i for i in on_mark if i.detail["code"] == "text_overlaps_symbol"]
    # the mark's own lettering ("1", "A", "B") is lettering too
    assert sorted(i.detail["items"][1]["text"] for i in on_mark if i is not symbol_pair) == [
        "1",
        "A",
        "B",
    ]
    assert symbol_pair.handles[0] == over_symbol.handle
    bubble = backend._doc.entitydb[symbol_pair.handles[1]]
    assert bubble.dxftype() == "CIRCLE" and bubble.dxf.layer == "A-SYMB-T-N"
    assert len(issues) == 3 + len(on_mark)
    assert all(i.detail["repairable"] is False and i.severity == "warning" for i in issues)


@pytest.mark.parametrize(("shared_width", "reported"), [(0.12, False), (0.18, True)])
@pytest.mark.asyncio
async def test_lettering_that_only_touches_is_not_an_overlap(backend, shared_width, reported):
    # Two equal boxes side by side sharing 12 % / 18 % of their width share as
    # much of their area: under and over the 15 % threshold. The width is
    # measured, so the test holds whatever font the machine renders with.
    from engineering.viewport_clip import text_box

    a = await backend.entity_create_text("AAAA", 0, 0, 5)
    xs = [x for x, _ in text_box(backend._doc.entitydb[a.handle])]
    width = max(xs) - min(xs)
    b = await backend.entity_create_text("AAAA", width * (1 - shared_width), 0, 5)
    issues = await run_critique(backend, FOCUS)
    assert [sorted(i.handles) for i in issues] == (
        [sorted([a.handle, b.handle])] if reported else []
    )


@pytest.mark.asyncio
async def test_what_is_not_compared(backend):
    doc = backend._doc
    # a frozen layer and an off layer
    await backend.layer_create("FROZEN")
    await backend.layer_create("OFF")
    await backend.entity_create_text("SEEN", 0, 0, 5)
    await backend.entity_create_text("HIDDEN", 0, 0, 5, layer="FROZEN")
    await backend.entity_create_text("HIDDEN", 0, 0, 5, layer="OFF")
    doc.layers.get("FROZEN").freeze()
    doc.layers.get("OFF").off()
    # the same place in model space and on a paper layout
    doc.layouts.get("Layout1").add_text("PAPER", height=5)
    # two ATTRIBs of one INSERT on top of each other
    await backend.block_define(
        "TAGS",
        [{"type": "circle", "cx": 0, "cy": 0, "radius": 5}],
        attdefs=[
            {"tag": "A", "default": "ONE", "x": 100, "y": 0, "height": 5},
            {"tag": "B", "default": "TWO", "x": 100, "y": 0, "height": 5},
        ],
    )
    await backend.block_insert("TAGS", 0, 0)
    # a label inside a furniture block: an INSERT off the symbol layer
    await backend.block_define(
        "BED",
        [{"type": "polyline", "points": [[0, 0], [90, 0], [90, 200], [0, 200]], "closed": True}],
    )
    await backend.block_insert("BED", 200, 0)
    await backend.entity_create_text("LOZKO", 220, 100, 10)
    # cells of a table, and a note written into one of them
    table = await backend.entity_create_table(400, 0, [["10.50 m2", "PODLOGA"]], headers=["A", "B"])
    first_cell = backend._doc.entitydb[table.properties["child_handles"][-2]]
    await backend.entity_create_text(
        "X", first_cell.dxf.insert.x, first_cell.dxf.insert.y - 2.5, 2.5
    )
    assert await run_critique(backend, FOCUS) == []


@pytest.mark.asyncio
async def test_lettering_of_one_group_or_one_record_is_one_object(backend):
    record = {"ACADMCP_ARCH": ['{"kind": "room", "id": "R9"}']}
    a = await backend.entity_create_text("POKÓJ", 0, 0, 5, xdata=record)
    b = await backend.entity_create_text("POKÓJ", 2, 1, 5, xdata=record)
    c = await backend.entity_create_text("NOTE", 0, 100, 5)
    d = await backend.entity_create_text("NOTE", 2, 101, 5)
    group = backend._doc.groups.new()
    group.extend([backend._doc.entitydb[c.handle], backend._doc.entitydb[d.handle]])
    assert await run_critique(backend, FOCUS) == []
    other = await backend.entity_create_text("POKÓJ", 1, 0, 5)  # same place, no record
    issues = await run_critique(backend, FOCUS)
    assert sorted(sorted(i.handles) for i in issues) == sorted(
        [sorted([a.handle, other.handle]), sorted([b.handle, other.handle])]
    )


@pytest.mark.asyncio
async def test_isocp_lettering_is_measured_at_its_iso_3098_width(backend):
    # isocp.shx is not on this machine: ezdxf letters it with the fallback face,
    # 23 % wider than ISO 3098, and two ISOCP labels set side by side, as they
    # would stand in AutoCAD, were reported as an overlap.
    from ezdxf.fonts import fonts

    from backends.shx_substitute import SAMPLE, SHX_METRICS, mean_advance
    from engineering.viewport_clip import text_box

    manager = fonts.font_manager
    if manager.has_font("isocp.shx") or manager.has_font("isocp.ttf"):
        pytest.skip("this machine has the real isocp face")
    fallback = manager._font_cache.get(manager.fallback_font_name(), "").file_path
    await backend.textstyle_create("ISOCP", "isocp")

    def width(handle):
        xs = [x for x, _ in text_box(backend._doc.entitydb[handle])]
        return max(xs) - min(xs)

    plain = await backend.entity_create_text(SAMPLE, 0, -100, 5)
    iso = await backend.entity_create_text(SAMPLE, 0, 0, 5)
    backend._doc.entitydb[iso.handle].dxf.style = "ISOCP"
    expected = width(plain.handle) * SHX_METRICS["isocp.shx"] / mean_advance(fallback)
    assert width(iso.handle) == pytest.approx(expected, rel=0.01)
    # its neighbour starts where the ISO 3098 lettering ends: no overlap
    nxt = await backend.entity_create_text(SAMPLE, width(iso.handle), 0, 5)
    backend._doc.entitydb[nxt.handle].dxf.style = "ISOCP"
    issues = await run_critique(backend, FOCUS)
    assert [i for i in issues if iso.handle in i.handles] == []
