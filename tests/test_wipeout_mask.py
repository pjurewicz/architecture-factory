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
    on_top = await insert_item(backend, "built_in_wardrobe", at=(1000, 0), width=1000)
    await backend.draw_order([on_top["handle"]], "bring_to_front")
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
    assert placed["tier"] == "furniture" and placed["warnings"] == []
    order = [e.dxf.handle for e in backend._doc.modelspace().entities_in_redraw_order()]
    assert order[0] == placed["handle"]
    assert await run_critique(backend, focus=["wipeout_mask"]) == []


async def test_an_unknown_focus_answers_the_catalogue_a_tool_only_client_can_read(backend):
    """Round 4 (A9a): the focus list was only a resource, out of call_tool's reach."""
    (issue,) = await run_critique(backend, focus=["?"])
    assert issue.detail["code"] == "unknown_focus"
    assert "hidden_by_wipeout" in issue.detail["focuses"]["wipeout_mask"]
    assert "domain_absent" in issue.detail["not_checked_reasons"]


# ── round 9: two send_to_back calls; one order in the model, sheet and PDF renders ──

SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]
MASK = [[30, 30], [70, 30], [70, 70], [30, 70]]


async def _hatch_and_mask(backend):
    hatch = await backend.entity_create_hatch("SOLID", SQUARE, color=1)
    mask = await backend.entity_create_wipeout(MASK)
    return hatch.handle, mask["handle"]


def _order(backend, layout=None):
    space = layout or backend._doc.modelspace()
    return [e.dxf.handle for e in space.entities_in_redraw_order()]


async def test_of_two_send_to_back_calls_the_later_set_goes_under(backend, tmp_path):
    """The round 9 report: WIPEOUT then HATCH sent back masks the hatch; the
    other way round the hatch covers the mask. AutoCAD's DRAWORDER Back moves
    its selection under everything, so the last call decides, not the pair."""
    hatch, mask = await _hatch_and_mask(backend)

    first = await backend.draw_order(None, where={"type": "WIPEOUT"})
    await backend.draw_order(None, where={"type": "HATCH"})
    assert _order(backend) == [hatch, mask]
    assert (first["handles"], first["count"]) == ([mask], 1), "handles, not entity reprs"

    await backend.draw_order(None, where={"type": "HATCH"})
    await backend.draw_order(None, where={"type": "WIPEOUT"})
    assert _order(backend) == [mask, hatch]

    import ezdxf

    path = tmp_path / "order.dxf"
    await backend.drawing_save_as(str(path))
    reread = ezdxf.readfile(str(path)).modelspace()
    assert _order(backend, reread) == [mask, hatch], "kept in the SORTENTSTABLE AutoCAD reads"


def _pdf_fills(path) -> list[str]:
    """The colour of every fill (``f``) in the PDF's content stream, in drawing
    order: ``rg`` / ``g`` set it, ``q`` / ``Q`` save and restore it."""
    import re
    import zlib

    fills = []
    for raw in re.findall(rb"stream\r?\n(.*?)endstream", path.read_bytes(), re.S):
        try:
            text = zlib.decompress(raw).decode("latin-1")
        except zlib.error:
            continue
        colour, saved = (0.0, 0.0, 0.0), []
        ops = r"([\d.]+) ([\d.]+) ([\d.]+) rg\b|([\d.]+) g\b|\b(q|Q|f)\b"
        for r, g, b, grey, op in re.findall(ops, text):
            if r:
                colour = (float(r), float(g), float(b))
            elif grey:
                colour = (float(grey),) * 3
            elif op == "q":
                saved.append(colour)
            elif op == "Q":
                colour = saved.pop()
            else:
                fills.append({(1.0, 0.0, 0.0): "red", (1.0, 1.0, 1.0): "white"}.get(colour))
    return fills


async def test_model_render_sheet_render_and_pdf_follow_the_same_order(backend, tmp_path):
    """Model screenshot, sheet screenshot and the sheet's PDF: one order, the
    SORTENTSTABLE's, before and after the mask moves under the hatch."""
    import io

    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    Image = pytest.importorskip("PIL.Image")

    hatch, mask = await _hatch_and_mask(backend)
    await backend.draw_order([hatch], "send_to_back")
    await backend.layout_create("S1")
    backend._doc.layouts.get("S1").page_setup(size=(420, 297), margins=(0, 0, 0, 0), units="mm")
    await backend.viewport_create("S1", 210, 148.5, 200, 150, 50, 50, scale=1.0)

    def centre(report, x, y):
        image = Image.open(io.BytesIO(report["png"])).convert("RGB")
        x0, _y0, _x1, y1 = report["region"]
        pixel = image.getpixel(
            (int((x - x0) * report["px_per_unit"]), int((y1 - y) * report["px_per_unit"]))
        )
        return "red" if pixel == (255, 0, 0) else "masked"

    async def seen(tag):
        model = await backend.view_render(region=[-10, -10, 110, 110], background="white")
        sheet = await backend.view_render(layout="S1", region=[150, 90, 270, 210])
        pdf = tmp_path / f"{tag}.pdf"
        await backend.drawing_export_pdf(str(pdf), layout="S1")
        return centre(model, 50, 50), centre(sheet, 210, 148.5), _pdf_fills(pdf)[-1]

    assert await seen("mask_on_top") == ("masked", "masked", "white"), "the mask is filled last"
    await backend.draw_order([hatch], "bring_to_front")
    assert await seen("hatch_on_top") == ("red", "red", "red")
