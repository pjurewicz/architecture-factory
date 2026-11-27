"""The ``viewport_clip`` focus: model lettering a paper viewport's frame cuts.

A multi-view sheet whose VP2 cut the start of view A's title and whose VP3 let
in the tails of view A's callouts ("ABS 2 mm") critiqued clean: no focus
compared a viewport's window with the text behind it.
"""

from __future__ import annotations

import pytest

from engineering.critique import run_critique
from engineering.qa_coverage import ledger_of

FOCUS = ["viewport_clip"]


def _window(sheet, x0, y0, x1, y1, **kw):
    """A 1:1 viewport whose model window is exactly ``x0..x1`` x ``y0..y1``."""
    w, h = x1 - x0, y1 - y0
    return sheet.add_viewport(
        center=(w / 2 + 10, h / 2 + 10),
        size=(w, h),
        view_center_point=((x0 + x1) / 2, (y0 + y1) / 2),
        view_height=h,
        **kw,
    )


def _rows(shared, layout="Layout1"):
    return [r for r in ledger_of(shared).rows if r.focus == "viewport_clip" and r.layout == layout]


@pytest.mark.asyncio
async def test_a_frame_through_a_title_is_reported_and_whole_texts_are_not(backend):
    title = await backend.entity_create_text("WIDOK A - RZUT", 0, 0, height=5)
    inside = await backend.entity_create_text("WIDOK B", 100, 0, height=5)
    await backend.entity_create_text("FAR AWAY", 500, 0, height=5)
    vp = _window(backend._doc.layouts.get("Layout1"), 20, -50, 220, 50)
    shared: dict = {}
    (issue,) = await run_critique(backend, FOCUS, shared=shared)
    assert issue.handles == [title.handle]
    assert issue.detail["viewport"] == vp.dxf.handle and issue.detail["count"] == 1
    assert inside.handle not in issue.handles
    (row,) = _rows(shared)
    assert row.status == "checked" and row.n == 3


@pytest.mark.asyncio
async def test_an_mtext_tail_is_measured_on_one_line_not_wrapped_per_word(backend):
    # "ABS 2 mm" at 5 mm, no width: one line about 36 mm long (x 150..186).
    # ezdxf.bbox wraps it at every word (x 150..164), which would call it
    # outside a window starting at x 170.
    callout = await backend.entity_create_mtext("ABS 2 mm", 150, 50, width=0, height=5)
    _window(backend._doc.layouts.get("Layout1"), 170, 0, 370, 100)
    (issue,) = await run_critique(backend, FOCUS)
    assert issue.handles == [callout.handle]


@pytest.mark.parametrize(
    ("align", "window", "cut"),
    [
        # "ABS 2 mm" at 5 mm is about 36 x 5 mm; every insert is (100, 0).
        # Centred: x 81.9..118.1 - read as top-left (100..136) it would be inside.
        ("middle_center", (90, -50, 300, 50), True),
        # Bottom-right: x 63.8..100 - read as top-left it would be outside.
        ("bottom_right", (0, -50, 95, 50), True),
        # A 100 mm column, centred: the frame is x 50..150, the lettering
        # x 81.9..118.1 - a window from x 125 sees none of it.
        ("middle_center", (125, -50, 300, 50), False),
    ],
)
@pytest.mark.asyncio
async def test_an_mtext_box_follows_its_attachment_point_and_its_ink(backend, align, window, cut):
    label = await backend.entity_create_mtext("ABS 2 mm", 100, 0, width=100, height=5, align=align)
    _window(backend._doc.layouts.get("Layout1"), *window)
    issues = await run_critique(backend, FOCUS)
    assert [i.handles for i in issues] == ([[label.handle]] if cut else [])


@pytest.mark.asyncio
async def test_a_cut_dimension_is_reported_and_a_frozen_layer_is_not(backend):
    dim = await backend.dimension_linear(0, 0, 100, 0, 0, 10)
    hidden = await backend.entity_create_text("SKIP ME", 45, 30, height=5, layer="HIDDEN")
    sheet = backend._doc.layouts.get("Layout1")
    vp = _window(sheet, 50, -50, 250, 50)
    vp.frozen_layers = ["HIDDEN"]
    (issue,) = await run_critique(backend, FOCUS)
    assert issue.handles == [dim.handle]
    assert hidden.handle not in issue.handles


@pytest.mark.asyncio
async def test_a_twisted_window_cuts_by_its_own_frame(backend):
    # Above the untwisted window (y 45 > 40) the text would be outside; the
    # window twisted by 45 degrees reaches up to y 56.6, and its edge crosses
    # the text between x 0 and x ~28.
    text = await backend.entity_create_text("CORNER", 0, 45, height=5)
    vp = _window(backend._doc.layouts.get("Layout1"), -40, -40, 40, 40)
    vp.dxf.view_twist_angle = 45.0
    (issue,) = await run_critique(backend, FOCUS)
    assert issue.handles == [text.handle]


@pytest.mark.asyncio
async def test_windows_it_cannot_read_are_not_checked_never_clean(backend):
    await backend.entity_create_text("WIDOK A", 0, 0, height=5)
    sheet = backend._doc.layouts.get("Layout1")
    _window(sheet, 20, -50, 220, 50).dxf.view_height = 0.0
    shared: dict = {}
    assert await run_critique(backend, FOCUS, shared=shared) == []
    (row,) = _rows(shared)
    assert row.status == "not_checked" and row.reason == "viewport_scale_deferred"

    await backend.layout_create("A1")
    boundary = backend._doc.layouts.get("A1").add_lwpolyline([(0, 0), (50, 0), (0, 50)])
    vp = _window(backend._doc.layouts.get("A1"), 20, -50, 220, 50)
    vp.dxf.flags |= 0x10000
    vp.dxf.clipping_boundary_handle = boundary.dxf.handle
    shared = {}
    assert await run_critique(backend, FOCUS, shared=shared) == []
    (row,) = _rows(shared, "A1")
    assert row.status == "not_checked" and row.reason == "viewport_clip_boundary"
