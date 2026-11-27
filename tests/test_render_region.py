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
