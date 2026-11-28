"""Round 4 (A4): a catalogue block's WIPEOUT hid LED strips and symbols drawn
before it, and nothing said so."""

from __future__ import annotations

import pytest

from engineering.arch.catalogue import insert_item
from engineering.arch.services import insert_points
from engineering.critique import run_critique

pytestmark = pytest.mark.asyncio


async def _strip(backend):
    await backend.layer_create("A-WALL-E-N")
    await backend.entity_create_line(0, 0, 4000, 0, layer="A-WALL-E-N")
    result = await insert_points(
        backend, [{"kind": "led_strip", "path": [[1100, 300], [1700, 300]]}]
    )
    return result["items"][0]["handle"]


async def test_a_wardrobe_on_top_masks_the_strip_and_the_focus_names_it(backend):
    strip = await _strip(backend)
    on_top = await insert_item(
        backend, "built_in_wardrobe", at=(1000, 0), width=1000, send_to_back=False
    )
    issues = await run_critique(backend, focus=["wipeout_mask"])
    (issue,) = [i for i in issues if i.focus == "wipeout_mask"]
    assert issue.detail["code"] == "hidden_by_wipeout"
    assert issue.detail["insert"] == on_top["handle"]
    assert strip in issue.detail["hidden"]
    # the fix the message names: a filter, not 56 handles copied by hand
    moved = await backend.draw_order(None, where={"block": "ARCH_BUILT_IN_WARDROBE*"})
    assert moved["count"] == 1
    assert [i for i in await run_critique(backend, focus=["wipeout_mask"])] == []


async def test_by_default_the_wardrobe_goes_under_what_is_already_drawn(backend):
    await _strip(backend)
    placed = await insert_item(backend, "built_in_wardrobe", at=(1000, 0), width=1000)
    assert placed["draw_order"] == "back"
    order = [e.dxf.handle for e in backend._doc.modelspace().entities_in_redraw_order()]
    assert order[0] == placed["handle"]
    assert await run_critique(backend, focus=["wipeout_mask"]) == []


async def test_an_unknown_focus_answers_the_catalogue_a_tool_only_client_can_read(backend):
    """Round 4 (A9a): the focus list was only a resource, out of call_tool's reach."""
    (issue,) = await run_critique(backend, focus=["?"])
    assert issue.detail["code"] == "unknown_focus"
    assert "hidden_by_wipeout" in issue.detail["focuses"]["wipeout_mask"]
    assert "domain_absent" in issue.detail["not_checked_reasons"]
