"""`callout`: a dot or arrow, a leader with a shelf, a note or a numbered circle,
over a WIPEOUT mask that works in model space and on a sheet."""

from __future__ import annotations

import io
import math

import pytest

from engineering.annotation import text_width
from engineering.errors import InvalidArgsError

pytestmark = pytest.mark.asyncio


def _entity(backend, handle):
    return backend._doc.entitydb[handle]


def _draw_rank(backend, space, handle):
    """Position of ``handle`` in the order the space is drawn."""
    order = [e.dxf.handle for e in space.entities_in_redraw_order()]
    return order.index(handle)


async def test_note_sits_past_the_shelf_over_a_mask_drawn_first(backend):
    h = 2.5
    result = await backend.callout([10, 10], [30, 25], text="Ø20 H7\nsecond line", style="arrow")
    msp = backend._doc.modelspace()

    # the shelf runs 2 h from the landing, away from the target
    assert result["side"] == "right"
    assert result["leader"] == [[10.0, 10.0], [30.0, 25.0], [35.0, 25.0]]
    note = _entity(backend, result["parts"]["text"])
    assert note.dxftype() == "MTEXT" and note.text == "Ø20 H7\\Psecond line"
    assert note.dxf.attachment_point == 4  # middle_left: the shelf hits the block's middle
    assert tuple(note.dxf.insert)[:2] == pytest.approx((35.0 + 0.5 * h, 25.0))

    # the mask covers the whole note, starts at the shelf end and lies under
    # the leader and the lettering
    wipeout = _entity(backend, result["parts"]["mask"])
    xs = [p.x for p in wipeout.boundary_path_wcs()]
    ys = [p.y for p in wipeout.boundary_path_wcs()]
    width = text_width("second line", h)
    assert min(xs) == pytest.approx(35.0)
    assert max(xs) >= 35.0 + 0.5 * h + width
    two_lines = h * (1 + 5 / 3)
    assert min(ys) < 25.0 - two_lines / 2 and max(ys) > 25.0 + two_lines / 2
    mask_rank = _draw_rank(backend, msp, wipeout.dxf.handle)
    assert all(
        mask_rank < _draw_rank(backend, msp, result["parts"][k])
        for k in ("leader", "terminator", "text")
    )

    # the arrow is a filled triangle whose tip is the target
    arrow = _entity(backend, result["parts"]["terminator"])
    assert arrow.dxftype() == "HATCH" and arrow.dxf.solid_fill == 1
    assert result["wipeout_frame"] == {"value": 0, "set": True}


async def test_landing_left_of_target_turns_the_callout_round(backend):
    result = await backend.callout([50, 0], [40, 10], text="LEFT", style="dot")
    note = _entity(backend, result["parts"]["text"])
    assert result["side"] == "left"
    assert result["leader"][-1] == [35.0, 10.0]
    assert note.dxf.attachment_point == 6  # middle_right
    xs = [p.x for p in _entity(backend, result["parts"]["mask"]).boundary_path_wcs()]
    assert max(xs) == pytest.approx(35.0) and min(xs) < 35.0 - text_width("LEFT", 2.5)
    # a dot: a filled circle on the target
    dot = _entity(backend, result["parts"]["terminator"])
    (edge,) = dot.paths[0].edges
    assert tuple(edge.center)[:2] == pytest.approx((50.0, 0.0))


async def test_number_circle_is_tangent_to_the_shelf_and_fully_masked(backend):
    result = await backend.callout([0, 0], [20, 10], number=128, text_height=3.5)
    circle = _entity(backend, result["parts"]["circle"])
    numeral = _entity(backend, result["parts"]["text"])
    r = circle.dxf.radius
    # three digits do not fit the minimum circle of one text height
    assert r > 3.5 and r >= text_width("128", 3.5) / 2
    shelf_end = result["leader"][-1]
    assert tuple(circle.dxf.center)[:2] == pytest.approx((shelf_end[0] + r, shelf_end[1]))
    assert numeral.dxf.text == "128" and numeral.dxf.height == 3.5
    assert tuple(numeral.dxf.align_point)[:2] == pytest.approx(tuple(circle.dxf.center)[:2])
    # every point of the circle lies inside the mask polygon: its closest edge
    # (the apothem) is not inside the circle
    poly = [p for p in _entity(backend, result["parts"]["mask"]).boundary_path_wcs()]
    cx, cy = circle.dxf.center.x, circle.dxf.center.y
    for a, b in zip(poly, poly[1:] + poly[:1], strict=True):
        mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
        assert math.hypot(mx - cx, my - cy) >= r - 1e-9


async def test_without_mask_no_wipeout_and_no_frame_setting(backend):
    result = await backend.callout([0, 0], [10, 10], text="PLAIN", mask=False)
    assert "mask" not in result["parts"] and result["mask"] is None
    assert result["wipeout_frame"] is None
    variables = backend._doc.rootdict.get("AcDbVariableDictionary")
    assert variables is None or variables.get("WIPEOUTFRAME") is None
    assert not list(backend._doc.modelspace().query("WIPEOUT"))


async def test_a_drawing_that_set_wipeoutframe_keeps_it(backend):
    backend._doc.set_wipeout_variables(frame=1)
    result = await backend.callout([0, 0], [10, 10], text="FRAMED")
    assert result["wipeout_frame"] == {"value": 1, "set": False}
    var = backend._doc.rootdict.get("AcDbVariableDictionary").get("WIPEOUTFRAME")
    assert str(var.value) == "1"


@pytest.mark.parametrize(
    ("kwargs", "words"),
    [
        ({"text": "A", "number": 1}, "exactly one"),
        ({}, "exactly one"),
        ({"text": "  \n "}, "empty"),
        ({"number": 0}, "positive"),
        ({"number": True}, "positive integer"),
        ({"text": "A", "style": "tick"}, "style"),
        ({"text": "A", "landing": [1.0, 1.0]}, "longer than its arrow"),
        ({"text": "A", "layout": "NoSuchSheet"}, "does not exist"),
    ],
)
async def test_refusals_draw_nothing(backend, kwargs, words):
    call = {"target": [0.0, 0.0], "landing": [20.0, 10.0], "style": "arrow", **kwargs}
    before = len(backend._doc.entitydb)
    with pytest.raises(InvalidArgsError, match=words):
        await backend.callout(**call)
    assert len(backend._doc.entitydb) == before


async def _sheet_over_red_model(backend):
    """A red SOLID model fill seen through a 1:1 viewport on sheet 'S'."""
    await backend.entity_create_hatch("SOLID", [[0, 0], [100, 0], [100, 60], [0, 60]], color=1)
    await backend.layout_create("S")
    await backend.viewport_create("S", 50, 30, 100, 60, 50, 30, scale=1.0)
    return backend._doc.layouts.get("S")


def _red_share(backend, layout, region):
    from PIL import Image

    from backends.region_render import render_layout

    png = render_layout(backend._doc, layout, region=region, max_px=400)["png"]
    image = Image.open(io.BytesIO(png)).convert("RGB")
    pixels = [image.getpixel((x, y)) for x in range(image.width) for y in range(image.height)]
    return sum(1 for r, g, b in pixels if r - max(g, b) > 80) / len(pixels)


async def test_sheet_mask_hides_the_model_seen_through_a_viewport(backend):
    # AutoCAD stacks viewports by "Plot paperspace last", not draw order; a
    # fresh layout is written with it clear, which put the model on top of
    # the note's mask. The callout sets it, and the sheet renders that way.
    sheet = await _sheet_over_red_model(backend)
    result = await backend.callout([20, 20], [40, 30], text="ON THE SHEET", layout="S")
    assert result["layout"] == "S"
    assert set(result["handles"]) <= {e.dxf.handle for e in sheet}
    assert result["sheet_order"] == {"paperspace_last": True, "set": True}
    assert sheet.dxf_layout.dxf.plot_layout_flags & 512

    xs = [p[0] for p in result["mask"]]
    ys = [p[1] for p in result["mask"]]
    # the mask's top margin, above the lettering: paper only
    top = (min(xs) + 1.0, max(ys) - 1.0, max(xs) - 1.0, max(ys) - 0.2)
    assert _red_share(backend, sheet, top) == 0.0
    # control: the same strip beside the mask shows the model
    assert _red_share(backend, sheet, (top[0], 50.0, top[2], 51.0)) > 0.5


async def test_model_callout_mask_works_through_a_viewport(backend):
    sheet = await _sheet_over_red_model(backend)
    result = await backend.callout([20, 20], [40, 30], text="IN THE MODEL")
    assert result["layout"] == "Model" and result["sheet_order"] is None
    xs = [p[0] for p in result["mask"]]
    ys = [p[1] for p in result["mask"]]
    top = (min(xs) + 1.0, max(ys) - 1.0, max(xs) - 1.0, max(ys) - 0.2)
    assert _red_share(backend, sheet, top) == 0.0
    assert _red_share(backend, backend._doc.modelspace(), top) == 0.0


async def test_side_overrides_the_direction_away_from_the_target(backend):
    # landing right of the target, shelf forced left; and the reverse
    left = await backend.callout([0, 0], [20, 10], text="L", side="left")
    assert left["side"] == "left" and left["leader"][-1] == [15.0, 10.0]
    assert _entity(backend, left["parts"]["text"]).dxf.attachment_point == 6
    right = await backend.callout([50, 0], [40, 10], number=3, side="right")
    assert right["side"] == "right" and right["leader"][-1] == [45.0, 10.0]
    assert _entity(backend, right["parts"]["circle"]).dxf.center.x > 45.0
    with pytest.raises(InvalidArgsError, match="side"):
        await backend.callout([0, 0], [20, 10], text="X", side="up")


async def test_deleting_any_part_erases_the_whole_callout(backend):
    keep = await backend.entity_create_line(0, 0, 5, 5)
    first = await backend.callout([0, 0], [20, 10], text="A")
    second = await backend.callout([0, 0], [20, -10], number=1)
    group = backend._doc.entitydb[first["group"]]
    assert group.dxftype() == "GROUP" and set(group.handles()) == set(first["handles"])

    result = await backend.entity_delete(first["parts"]["mask"])
    assert set(result["deleted_handles"]) == set(first["handles"])
    db = backend._doc.entitydb
    assert not any(db.get(h) is not None and db.get(h).is_alive for h in first["handles"])
    assert first["group"] not in [g.dxf.handle for _, g in backend._doc.groups]
    # the other callout and plain geometry are untouched
    assert all(db[h].is_alive for h in second["handles"])
    assert (await backend.entity_delete(keep.handle)) == {
        "ok": True,
        "deleted_handle": keep.handle,
    }

    # two parts of one callout in one call: one callout erased, no error
    many = await backend.entity_delete_many(
        [second["parts"]["text"], second["parts"]["circle"]], stop_on_error=True
    )
    assert many == {"deleted": 2, "errors": []}
    assert not list(backend._doc.modelspace())


async def test_max_width_wraps_the_note_and_the_mask_follows(backend):
    # 1:50: 2.5 mm lettering is 125 units, a 50 mm column 2500
    h, limit = 125.0, 2500.0
    text = "PŁYTA LAMIN. 18\u00a0mm DĄB BIELONY + ABS 2\u00a0mm\nKRAWĘDŹ"
    one_line = text_width(text.split("\n")[0], h)
    assert one_line > limit  # wrapping is actually needed
    result = await backend.callout([0, 0], [1000, 1000], text=text, text_height=h, max_width=limit)

    lines = result["text"].split("\n")
    assert result["lines"] == len(lines) >= 3 and lines[-1] == "KRAWĘDŹ"
    assert all(text_width(line, h) <= limit for line in lines)
    # no word lost or split, the no-break spaces kept "18 mm" and "2 mm" whole
    assert " ".join(lines[:-1]).split(" ") == text.split("\n")[0].split(" ")
    assert any("18\u00a0mm" in line for line in lines) and result["overflow"] == []
    # greedy: the next word would not have fitted on any line
    words = [line.split(" ") for line in lines[:-1]]
    for line, nxt in zip(lines[:-2], words[1:], strict=True):
        assert text_width(f"{line} {nxt[0]}", h) > limit
    note = _entity(backend, result["parts"]["text"])
    assert not note.dxf.get("width")  # explicit breaks only: AutoCAD never re-wraps
    xs = [p[0] for p in result["mask"]]
    widest = max(text_width(line, h) for line in lines)
    assert max(xs) - min(xs) == pytest.approx(widest + 2 * 0.5 * h)
    ys = [p[1] for p in result["mask"]]
    assert max(ys) - min(ys) == pytest.approx(h * (1 + (len(lines) - 1) * 5 / 3) + h)


async def test_max_width_word_wider_than_the_column_stands_alone(backend):
    result = await backend.callout(
        [0, 0], [20, 10], text="A SUPERCALIFRAGILISTIC B", max_width=10.0
    )
    assert result["text"].split("\n") == ["A", "SUPERCALIFRAGILISTIC", "B"]
    assert result["overflow"] == ["SUPERCALIFRAGILISTIC"]
    with pytest.raises(InvalidArgsError, match="max_width"):
        await backend.callout([0, 0], [20, 10], number=1, max_width=10.0)
    with pytest.raises(InvalidArgsError, match="max_width must be positive"):
        await backend.callout([0, 0], [20, 10], text="A", max_width=0)
