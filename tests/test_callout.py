"""`callout`: a dot or arrow, a leader with a shelf, a note or a numbered circle,
over a WIPEOUT mask that works in model space and on a sheet."""

from __future__ import annotations

import io
import math

import pytest
import pytest_asyncio

from engineering.annotation import style_text_width
from engineering.errors import InvalidArgsError

pytestmark = pytest.mark.asyncio


def _width(backend, text, h, style="Standard"):
    """A line's width in ``style``, as the callout measures it."""
    return style_text_width(backend._doc.styles.get(style))(text, h)


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
    width = _width(backend, "second line", h)
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
    assert max(xs) == pytest.approx(35.0) and min(xs) < 35.0 - _width(backend, "LEFT", 2.5)
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
    assert r > 3.5 and r >= _width(backend, "128", 3.5) / 2
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
        ({"text": "A", "style": "zigzag"}, "style"),
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
    one_line = _width(backend, text.split("\n")[0], h)
    assert one_line > limit  # wrapping is actually needed
    result = await backend.callout([0, 0], [1000, 1000], text=text, text_height=h, max_width=limit)

    lines = result["text"].split("\n")
    assert result["lines"] == len(lines) >= 3 and lines[-1] == "KRAWĘDŹ"
    assert all(_width(backend, line, h) <= limit for line in lines)
    # no word lost or split, the no-break spaces kept "18 mm" and "2 mm" whole
    assert " ".join(lines[:-1]).split(" ") == text.split("\n")[0].split(" ")
    assert any("18\u00a0mm" in line for line in lines) and result["overflow"] == []
    # greedy: the next word would not have fitted on any line
    words = [line.split(" ") for line in lines[:-1]]
    for line, nxt in zip(lines[:-2], words[1:], strict=True):
        assert _width(backend, f"{line} {nxt[0]}", h) > limit
    note = _entity(backend, result["parts"]["text"])
    assert not note.dxf.get("width")  # explicit breaks only: AutoCAD never re-wraps
    xs = [p[0] for p in result["mask"]]
    widest = max(_width(backend, line, h) for line in lines)
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


async def test_tick_none_under_shelf_right_aligned_with_style_and_bounds(backend):
    """Round 7: 25 of 27 notes were polyline + MTEXT pairs - no tick, no shelf
    under the note, no text style, no right alignment, bounds without the text."""
    backend._doc.styles.add("ARCH", font="arial.ttf")
    h = 2.5
    note = "STROP G-K\nDWIE LINIE"
    under = await backend.callout(
        [0, 0], [10, 10], text=note, style="tick", shelf="under", align="right",
        text_style="ARCH", mask=False, color=3,
    )  # fmt: skip
    width = max(_width(backend, line, h, "ARCH") for line in note.split("\n"))
    # the shelf runs under the whole note, from the landing
    assert under["leader"][1:] == [[10.0, 10.0], [pytest.approx(10.0 + width + h), 10.0]]
    mtext = _entity(backend, under["parts"]["text"])
    assert mtext.dxf.style == "ARCH" and mtext.dxf.color == 3
    assert mtext.dxf.attachment_point == 9  # bottom_right: above the shelf, right edge
    assert tuple(mtext.dxf.insert)[:2] == pytest.approx((10.0 + 0.5 * h + width, 10.0 + 0.5 * h))
    tick = _entity(backend, under["parts"]["terminator"])
    assert tick.dxftype() == "LINE"
    assert tick.dxf.start.distance(tick.dxf.end) == pytest.approx(h)
    # bounds reach the top of the note though there is no mask
    assert under["bounds"]["max"][1] > 10.0 + h * (1 + 5 / 3)
    bare = await backend.callout([0, 0], [1, 1], text="X", style="none")
    assert "terminator" not in bare["parts"]  # no terminator: a leader may be short
    with pytest.raises(InvalidArgsError, match="no text style 'NOPE'"):
        await backend.callout([0, 0], [10, 10], text="X", text_style="NOPE")
    with pytest.raises(InvalidArgsError, match="shelf 'under'"):
        await backend.callout([0, 0], [10, 10], number=1, shelf="under")


async def test_width_comes_from_the_text_style_width_factor_included(backend):
    """Round 9: after textstyle_modify(width_factor=0.72) the shelf and bounds
    stayed at width factor 1 (5294.14 both times), the shelf 40 % past the note."""
    from engineering.viewport_clip import mtext_box

    await backend.textstyle_create("OPIS", font="isocp.shx")
    h, note = 125.0, "SUFIT PODWIESZANY G-K 12,5\u00a0mm"
    call = {"text": note, "text_height": h, "text_style": "OPIS", "shelf": "under", "mask": False}
    ends = []
    for factor in (1.0, 0.72):
        await backend.textstyle_modify("OPIS", width_factor=factor)
        result = await backend.callout([0, 0], [4000, 1000], **call)
        # the box text_overlap reads (mtext_size), not the plan's own number
        right = max(x for x, _ in mtext_box(_entity(backend, result["parts"]["text"])))
        end = result["leader"][-1][0]
        assert end == pytest.approx(right + 0.5 * h, abs=1e-6)
        assert result["bounds"]["max"][0] == pytest.approx(end)
        ends.append(end - 4000)
    assert ends[1] / ends[0] == pytest.approx(0.72, abs=0.02)


async def test_long_shelf_over_the_note_to_its_end_from_three_targets(backend):
    """The source's layout: a long shelf from a knee by the targets, the note
    under it set against the shelf's end, three branches to one knee."""
    from engineering.viewport_clip import mtext_box

    h, note = 2.5, "TYNK GIPSOWY"
    targets = [[0, 0], [5, -8], [-6, -4]]
    result = await backend.callout(
        targets=targets, landing=[2, 10], text=note, style="dot", shelf="over", shelf_end=60
    )
    assert result["leader"] == [[0.0, 0.0], [2.0, 10.0], [60.0, 10.0]]
    assert result["branches"] == [[[5.0, -8.0], [2.0, 10.0]], [[-6.0, -4.0], [2.0, 10.0]]]
    parts = result["parts"]
    assert {"branch_2", "branch_3", "terminator", "terminator_2", "terminator_3"} <= set(parts)
    centres = [
        tuple(_entity(backend, parts[k]).paths[0].edges[0].center)[:2]
        for k in ("terminator", "terminator_2", "terminator_3")
    ]
    assert centres == [pytest.approx(tuple(t)) for t in targets]
    mtext = _entity(backend, parts["text"])
    assert mtext.dxf.attachment_point == 3  # top_right: under the shelf, at its end
    box = mtext_box(mtext)
    assert max(x for x, _ in box) == pytest.approx(60.0 - 0.5 * h, abs=1e-6)
    assert max(y for _, y in box) == pytest.approx(10.0 - 0.5 * h, abs=1e-6)
    # the mask is round the note, not the whole shelf
    assert min(x for x, _ in result["mask"]) == pytest.approx(min(x for x, _ in box) - 0.5 * h)
    assert result["bounds"]["min"] == [-6.0, -8.0]

    before = len(backend._doc.entitydb)
    with pytest.raises(InvalidArgsError, match="the note needs"):
        await backend.callout([0, 0], [2, 10], text=note, shelf="over", shelf_end=10)
    assert len(backend._doc.entitydb) == before


async def test_broken_leader_vertical_then_horizontal(backend):
    result = await backend.callout(
        leader=[[0, 0], [0, 50], [10, 50]], text="ŚCIANA", style="arrow", mask=False
    )
    assert result["side"] == "right"
    assert result["leader"] == [[0.0, 0.0], [0.0, 50.0], [10.0, 50.0], [15.0, 50.0]]
    tip, *base = _entity(backend, result["parts"]["terminator"]).paths[0].vertices
    assert tuple(tip)[:2] == pytest.approx((0.0, 0.0))
    assert all(v[1] == pytest.approx(2.5) for v in base)  # along the vertical leg


async def test_line_spacing_sets_the_mtext_and_the_mask_height(backend):
    from engineering.viewport_clip import mtext_box

    h = 2.5
    result = await backend.callout([0, 0], [20, 10], text="A\nB\nC", line_spacing=1.5)
    mtext = _entity(backend, result["parts"]["text"])
    assert mtext.dxf.line_spacing_factor == 1.5 and result["line_spacing"] == 1.5
    ys = [y for _, y in result["mask"]]
    box = [y for _, y in mtext_box(mtext)]
    assert max(ys) - min(ys) == pytest.approx(max(box) - min(box) + h, rel=1e-3)
    assert max(ys) - min(ys) == pytest.approx(h * (1 + 2 * 5 / 3 * 1.5) + h)


@pytest.mark.parametrize(
    ("kwargs", "words"),
    [
        ({"target": [0, 0], "targets": [[1, 1]], "landing": [20, 10]}, "exactly one of target"),
        ({"leader": [[0, 0], [20, 10]], "landing": [20, 10]}, "drop landing"),
        ({"targets": [[0, 0]]}, "landing"),
        ({"leader": [[0, 0]]}, "at least 2"),
        ({"target": [0, 0], "landing": [20, 10], "side": "left", "shelf_end": 40}, "not both"),
        ({"target": [0, 0], "landing": [20, 10], "shelf_end": 20}, "other than the knee"),
        ({"target": [0, 0], "landing": [20, 10], "line_spacing": 9}, "line_spacing"),
    ],
)
async def test_leader_refusals_draw_nothing(backend, kwargs, words):
    before = len(backend._doc.entitydb)
    with pytest.raises(InvalidArgsError, match=words):
        await backend.callout(text="A", **kwargs)
    with pytest.raises(InvalidArgsError, match="line_spacing"):
        await backend.callout([0, 0], [20, 10], number=1, line_spacing=1.5)
    assert len(backend._doc.entitydb) == before


@pytest_asyncio.fixture
async def client():
    from fastmcp import Client

    import server

    async with Client(server.mcp) as c:
        await c.call_tool("drawing_new", {})
        yield c


async def test_items_draw_many_callouts_and_a_bad_one_draws_none(client):
    async def count():
        rows = await client.call_tool("call_tool", {"name": "entity_list", "arguments": {}})
        data = rows.structured_content
        return len(data.get("result", data) if isinstance(data, dict) else data)

    before = await count()
    bad = await client.call_tool(
        "call_tool",
        {
            "name": "callout",
            "arguments": {
                "style": "tick",
                "items": [
                    {"target": [0, 0], "landing": [20, 10], "text": "A"},
                    {"target": [0, 0], "landing": [20, 10], "text": "B", "number": 2},
                ],
            },
        },
        raise_on_error=False,
    )
    assert bad.is_error and "items[1]" in bad.content[0].text
    assert await count() == before
    done = (
        await client.call_tool(
            "call_tool",
            {
                "name": "callout",
                "arguments": {
                    "style": "tick",
                    "text_height": 125,
                    "items": [
                        {"target": [0, 0], "landing": [2000, 1000], "text": "A"},
                        {"target": [0, 0], "landing": [-2000, 1000], "number": 3},
                    ],
                },
            },
        )
    ).structured_content
    assert done["count"] == 2 and all(row["handles"] for row in done["callouts"])
    batch = (
        await client.call_tool(
            "entity_batch_create",
            {
                "entities": [
                    {"type": "callout", "target": [0, 0], "landing": [30, 20], "text": "C",
                     "style": "arrow", "tag": "c"},
                    {"type": "line", "x1": 0, "y1": 0, "x2": 5, "y2": 0},
                    {"type": "callout", "targets": [[0, 0], [9, 0]], "landing": [4, 20],
                     "text": "D", "shelf": "over", "shelf_end": 40, "line_spacing": 1.2},
                ]
            },
        )
    ).structured_content  # fmt: skip
    assert batch["ok"] and batch["created"] == 3 and "c" in batch["tags"]
