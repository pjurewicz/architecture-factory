"""1.8-07 - an honest region image (ergo-9, narrowed).

Before 1.8.0 the headless zoom-and-screenshot tool (since removed) passed its
window to a no-op ``view_zoom_window`` and then rendered the whole model space: the caller
asked for one corner of the sheet and got the sheet, with nothing saying so
(the C2 class of false success). ``view_screenshot`` had no region at all, so a
detail cost a full-page PNG.

The gate: of two far-apart circles, a window on one draws only that one - its
ink bounding box sits where ``px_per_unit`` says it should - the report says
``applied: true`` with ``{px, region, px_per_unit}``, and ``max_px`` is obeyed.
"""

from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
Image = pytest.importorskip("PIL.Image", reason="the ink check reads pixels with Pillow")

from backends.region_render import (  # noqa: E402
    DEFAULT_MAX_PX,
    png_size,
)

LEFT = (0.0, 0.0)
RIGHT = (1000.0, 0.0)
RADIUS = 5.0


def _ink_box(png: bytes) -> tuple[int, int, int, int] | None:
    """(left, top, right, bottom) of every pixel unlike the background corner."""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    background = image.getpixel((0, 0))
    width, height = image.size
    pixels = image.load()
    xs, ys = [], []
    for y in range(height):
        for x in range(width):
            if pixels[x, y] != background:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _expected_box(report: dict, cx: float, cy: float, r: float) -> tuple[float, ...]:
    """The circle's pixel box from the report's own region and scale."""
    x0, _y0, _x1, y1 = report["region"]
    ppu = report["px_per_unit"]
    return ((cx - r - x0) * ppu, (y1 - (cy + r)) * ppu, (cx + r - x0) * ppu, (y1 - (cy - r)) * ppu)


async def _two_circles(backend):
    await backend.entity_create_circle(*LEFT, RADIUS)
    await backend.entity_create_circle(*RIGHT, RADIUS)


# ── headless: the window is applied ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_window_on_one_of_two_far_circles_draws_only_that_one(backend):
    await _two_circles(backend)
    report = await backend.view_render(region=[-10, -10, 10, 10], max_px=256)

    assert report["applied"] is True
    assert report["source"] == "headless"
    assert report["framing"] == "region"
    assert report["px"] == [256, 256]
    assert report["region"] == pytest.approx([-10, -10, 10, 10])
    assert report["px_per_unit"] == pytest.approx(256 / 20)

    ink = _ink_box(report["png"])
    assert ink is not None, "the circle inside the window was not drawn"
    expected = _expected_box(report, *LEFT, RADIUS)
    for got, want in zip(ink, expected, strict=True):
        assert got == pytest.approx(want, abs=3), (ink, expected)


@pytest.mark.asyncio
async def test_a_white_background_draws_colours_as_on_paper(backend):
    """A thin green line on the dark model background read as grey, twice."""
    await backend.entity_create_circle(*LEFT, RADIUS, color=3)

    def pixels(report):
        image = Image.open(io.BytesIO(report["png"])).convert("RGB")
        return image.getpixel((0, 0)), set(image.getdata())

    paper, white = pixels(
        await backend.view_render(region=[-10, -10, 10, 10], max_px=128, background="white")
    )
    screen, _ = pixels(await backend.view_render(region=[-10, -10, 10, 10], max_px=128))
    assert paper == (255, 255, 255) and sum(screen) < 200
    assert any(g == 255 and r < 40 and b < 40 for r, g, b in white), "ACI 3 is green on paper"


@pytest.mark.asyncio
async def test_a_window_between_the_circles_draws_nothing(backend):
    await _two_circles(backend)
    report = await backend.view_render(region=[400, -10, 600, 10], max_px=200)
    assert report["applied"] is True
    assert _ink_box(report["png"]) is None, "geometry outside the window leaked into the image"


@pytest.mark.asyncio
async def test_the_window_on_the_far_circle_puts_it_where_the_scale_says(backend):
    await _two_circles(backend)
    report = await backend.view_render(region=[990, -20, 1020, 10], max_px=300)
    ink = _ink_box(report["png"])
    expected = _expected_box(report, *RIGHT, RADIUS)
    for got, want in zip(ink, expected, strict=True):
        assert got == pytest.approx(want, abs=3), (ink, expected)


@pytest.mark.asyncio
async def test_max_px_is_obeyed_and_the_scale_is_one_number(backend):
    await _two_circles(backend)
    report = await backend.view_render(region=[0, 0, 100, 37], max_px=400)
    width, height = report["px"]
    assert max(width, height) == 400
    assert png_size(report["png"]) == (width, height), "the reported px is the file's own size"
    x0, y0, x1, y1 = report["region"]
    ppu = report["px_per_unit"]
    assert width / (x1 - x0) == pytest.approx(ppu, rel=1e-9)
    assert height / (y1 - y0) == pytest.approx(ppu, rel=1e-9)
    # the drawn region holds the requested one, widened by under a pixel
    assert x0 <= 0 and y0 <= 0 and x1 >= 100 and y1 >= 37
    assert (x1 - x0) - 100 < 1 / ppu + 1e-9 and (y1 - y0) - 37 < 1 / ppu + 1e-9
    assert report["requested_region"] == [0, 0, 100, 37]


@pytest.mark.asyncio
async def test_the_default_size_is_the_model_image_limit(backend):
    await _two_circles(backend)
    report = await backend.view_render(region=[-10, -10, 10, 10])
    assert DEFAULT_MAX_PX == 1568
    assert report["px"] == [1568, 1568]


@pytest.mark.asyncio
async def test_corners_in_any_order_are_one_region(backend):
    await _two_circles(backend)
    report = await backend.view_render(region=[10, 10, -10, -10], max_px=128)
    assert report["requested_region"] == [-10, -10, 10, 10]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "region",
    [[0, 0, 0, 10], [0, 5, 10, 5], [0, 0, 10], [0, 0, float("nan"), 1], "abcd"],
)
async def test_a_degenerate_region_is_refused_by_name(backend, region):
    with pytest.raises(ValueError, match="region"):
        await backend.view_render(region=region)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_px", [0, 63, 4097])
async def test_max_px_out_of_range_is_refused_by_name(backend, max_px):
    with pytest.raises(ValueError, match="max_px"):
        await backend.view_render(region=[0, 0, 1, 1], max_px=max_px)


@pytest.mark.asyncio
async def test_no_region_frames_the_extents_and_says_so(backend):
    await _two_circles(backend)
    report = await backend.view_render(max_px=800)
    assert report["applied"] is True
    assert report["framing"] == "extents"
    assert report["requested_region"] is None
    x0, y0, x1, y1 = report["region"]
    assert x0 <= -RADIUS and x1 >= 1000 + RADIUS and y0 <= -RADIUS and y1 >= RADIUS
    assert max(report["px"]) == 800


@pytest.mark.asyncio
@pytest.mark.parametrize("sign", ["\u00d8", "\u2300", "\u2205"])  # Ø ⌀ ∅
async def test_a_diameter_sign_is_a_ring_not_a_half_filled_blob(backend, sign):
    # The slash pokes out of the ring; re-nested by bounding box, one half of
    # the counter was filled and the sign drew far heavier than the digits.
    await backend.entity_create_text(sign, 0, 0, height=10)
    png = (await backend.view_render(region=[-2, -4, 14, 14], max_px=256))["png"]
    image = Image.open(io.BytesIO(png)).convert("RGB")
    background = image.getpixel((0, 0))
    left, top, right, bottom = _ink_box(png)
    y = (top + bottom) // 2
    for f in (0.3, 0.7):  # inside the ring, either side of the slash
        assert image.getpixel((round(left + f * (right - left)), y)) == background, (sign, f)


@pytest.mark.asyncio
@pytest.mark.parametrize(("psltscale", "ltscale"), [(1, 1.0), (0, 20.0)])
async def test_a_hidden_line_seen_through_a_viewport_plots_dashed(backend, psltscale, ltscale):
    # ezdxf scaled a viewport's pattern by 1 / scale on paper coordinates: at
    # 1:20 a HIDDEN dash (5 + 2.5 gap) became 100 mm or more of paper and the
    # 75 mm line plotted solid. Both cases put the period at 7.5 mm of paper.
    backend._doc.header["$PSLTSCALE"] = psltscale
    backend._doc.header["$LTSCALE"] = ltscale
    await backend.entity_create_line(0, 0, 1500, 0, linetype="HIDDEN")
    await backend.layout_create("A3")
    backend._doc.layouts.get("A3").page_setup(size=(420, 297), margins=(0, 0, 0, 0), units="mm")
    await backend.viewport_create("A3", 210, 148.5, 400, 280, 750, 0, scale=1 / 20)
    await backend.layout_set_current("A3")
    png = (await backend.view_render(region=[160, 138.5, 260, 158.5], max_px=1000))["png"]
    image = Image.open(io.BytesIO(png)).convert("RGB")
    background = image.getpixel((0, 0))
    width, height = image.size
    rows = [[image.getpixel((x, y)) != background for x in range(width)] for y in range(height)]
    row = max(rows, key=sum)  # the line's own row
    dashes = sum(1 for x in range(width) if row[x] and (x == 0 or not row[x - 1]))
    assert 9 <= dashes <= 11, dashes  # 75 mm of line / 7.5 mm period


async def _a2_sheet_at_1_50(backend, view_x, view_y):
    await backend.layout_create("A2")
    backend._doc.layouts.get("A2").page_setup(size=(594, 420), margins=(0, 0, 0, 0), units="mm")
    await backend.viewport_create("A2", 297, 210, 500, 340, view_x, view_y, scale=1 / 50)
    await backend.layout_set_current("A2")


def _gaps_mm(report, *, vertical=False) -> list[float]:
    """Breaks (paper mm) along the inkiest pixel row - or column - of a render."""
    image = Image.open(io.BytesIO(report["png"])).convert("RGB")
    if vertical:
        image = image.transpose(Image.Transpose.ROTATE_90)
    background = image.getpixel((0, 0))
    width, height = image.size
    rows = [[image.getpixel((x, y)) != background for x in range(width)] for y in range(height)]
    row = max(rows, key=sum)  # the line's own row
    ink = [x for x in range(width) if row[x]]
    pairs = zip(ink, ink[1:], strict=False)
    return [(b - a - 1) / report["px_per_unit"] for a, b in pairs if b - a > 1]


@pytest.mark.asyncio
async def test_a_center_line_on_an_a2_sheet_at_1_50_plots_with_visible_gaps(backend):
    # ezdxf's standard CENTER is acad.lin's, in inches: loaded into a metric
    # drawing it repeated every 2 mm with 0.25 mm gaps and plotted solid.
    # acadiso.lin's CENTER: 31.75 dash, 6.35 gap, 6.35 dot, 6.35 gap (mm).
    await backend.entity_create_line(0, 0, 7500, 0, linetype="CENTER")
    await _a2_sheet_at_1_50(backend, 3750, 0)
    gaps = _gaps_mm(await backend.view_render(region=[222, 200, 372, 220], max_px=1500))
    assert 5 <= len(gaps) <= 7, gaps  # 150 mm of line / 50.8 mm period, two gaps each
    assert min(gaps) > 5, gaps  # 6.35 mm of paper, less the pen's round caps


@pytest.mark.asyncio
async def test_a_drawing_saved_with_inch_center_reopens_with_visible_gaps(backend, tmp_path):
    # Every build before 8632da0 stored acad.lin's inch CENTER (period 2.0) in
    # metric drawings; reopened, it still plotted solid through the viewport.
    from ezdxf.tools import standards

    doc = backend._doc
    _, description, inch = next(lt for lt in standards.linetypes() if lt[0] == "CENTER")
    doc.linetypes.add("CENTER", pattern=inch, description=description)
    await backend.entity_create_line(0, 0, 7500, 0, linetype="CENTER")
    await _a2_sheet_at_1_50(backend, 3750, 0)
    path = str(tmp_path / "legacy.dxf")
    await backend.drawing_save(path)
    opened = await backend.drawing_open(path)
    assert opened["linetypes_rescaled"] == ["CENTER"]
    await backend.layout_set_current("A2")
    gaps = _gaps_mm(await backend.view_render(region=[222, 200, 372, 220], max_px=1500))
    assert 5 <= len(gaps) <= 7 and min(gaps) > 5, gaps


@pytest.mark.asyncio
async def test_a_sheet_plot_warns_when_ltscale_makes_its_viewport_patterns_plot_solid(
    backend, tmp_path
):
    # LTSCALE 50 under PSLTSCALE 1 draws HIDDEN's dash 250 mm on paper: the
    # line plots solid, as in AutoCAD, and the plot said nothing.
    from engineering.standards.plot import batch_plot

    await backend.entity_create_line(0, 0, 7500, 0, linetype="HIDDEN")
    await _a2_sheet_at_1_50(backend, 3750, 0)
    quiet = await backend.drawing_export_pdf(str(tmp_path / "quiet.pdf"), layout="A2")
    assert "warnings" not in quiet
    backend._doc.header["$LTSCALE"] = 50.0
    loud = await backend.drawing_export_pdf(str(tmp_path / "loud.pdf"), layout="A2")
    assert len(loud["warnings"]) == 1 and "50x their size" in loud["warnings"][0]
    backend._doc.header["$PSLTSCALE"] = 0  # model units: LTSCALE 50 is right at 1:50
    assert "warnings" not in await backend.drawing_export_pdf(str(tmp_path / "ok.pdf"), layout="A2")
    backend._doc.header["$LTSCALE"] = 1.0
    plotted = await batch_plot(backend, ["A2"], str(tmp_path / "set"))
    assert "0.02x their size" in plotted["sheets"][0]["warnings"][0]


@pytest.mark.asyncio
async def test_an_axis_extension_line_plots_dash_dot_at_1_50_and_after_a_rerender(backend):
    # dimension_chain(axis=) gives the station's extension line CENTER, but a
    # 20 mm line holds no 50.8 mm sequence: AutoCAD draws it solid, and so did
    # the plot. The dimension's linetype scale now fits two sequences.
    from engineering.dimension_chain import draw_chain

    out = await draw_chain(
        backend, [[0, 0], [3000, 0], [7000, 0]], 1000, axis=[1], overall=False, scale=50
    )
    await _a2_sheet_at_1_50(backend, 3000, 500)
    window = [292, 200, 302, 230]  # paper, around the axis station's extension line
    for _ in range(2):
        gaps = _gaps_mm(await backend.view_render(region=window, max_px=1500), vertical=True)
        assert len([g for g in gaps if g > 0.8]) >= 2, gaps  # 1.29 mm, less the round caps
        await backend.dimension_rerender([d["handle"] for d in out["dimensions"]], True)


@pytest.mark.asyncio
async def test_handle_labels_only_for_entities_in_the_region(backend):
    left = await backend.entity_create_circle(*LEFT, RADIUS)
    await backend.entity_create_circle(*RIGHT, RADIUS)
    report = await backend.view_render(region=[-10, -10, 10, 10], max_px=256, overlay_handles=True)
    assert set(report["labels"]) == {left.handle}
    assert report["total"] == 1 and report["truncated"] is False


# ── the tools ────────────────────────────────────────────────────────────────


def _tool_report(result) -> tuple[bytes, dict]:
    import base64

    images = [c for c in result.content if getattr(c, "type", None) == "image"]
    assert len(images) == 1, result.content
    report = result.structured_content
    texts = [c for c in result.content if getattr(c, "type", None) == "text"]
    assert texts and json.loads(texts[0].text) == report, "the text block carries the report"
    return base64.b64decode(images[0].data), report


@pytest.mark.asyncio
async def test_view_screenshot_tool_reports_the_region():
    from fastmcp import Client

    import server

    async with Client(server.mcp) as client:
        await client.call_tool("drawing_new", {})
        await client.call_tool("entity_create_circle", {"cx": 0, "cy": 0, "radius": RADIUS})
        await client.call_tool("entity_create_circle", {"cx": 1000, "cy": 0, "radius": RADIUS})
        result = await client.call_tool(
            "view_screenshot", {"region": [-10, -10, 10, 10], "max_px": 256}
        )
    png, report = _tool_report(result)
    assert report["ok"] is True and report["applied"] is True
    assert report["source"] == "headless"
    assert report["px"] == [256, 256] and png_size(png) == (256, 256)
    assert report["px_per_unit"] == pytest.approx(12.8)
    assert "png" not in report
