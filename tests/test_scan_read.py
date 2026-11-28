"""scan_read: map mode (pixels <-> WCS through overlay_align), grid crop, colour features, compare."""

from __future__ import annotations

import io
import math

import pytest
import pytest_asyncio

from engineering.arch.image_map import image_map

# A scan at 0.1 px per mm, turned 30 degrees, its pixel (0, 0) on WCS (1000, 2000).
SCALE, TURN, ORIGIN = 10.0, math.radians(30.0), (1000.0, 2000.0)


def truth(u: float, v: float) -> list[float]:
    """Pixel -> WCS written out independently: v points down, then rotate and shift."""
    x, y = u * SCALE, -v * SCALE
    c, s = math.cos(TURN), math.sin(TURN)
    return [ORIGIN[0] + x * c - y * s, ORIGIN[1] + x * s + y * c]


def pair(u, v, dx=0.0, dy=0.0):
    x, y = truth(u, v)
    return [u, v, x + dx, y + dy]


def test_two_pairs_map_there_and_back_but_say_they_are_unchecked():
    align = [pair(0, 0), pair(400, 300)]
    pixels = [[123.0, 45.0], [700.0, 20.0], [0.0, 999.0]]
    result = image_map(align, pixels=pixels)
    for got, (u, v) in zip(result["to_drawing"], pixels, strict=True):
        assert got == pytest.approx(truth(u, v), abs=1e-3)
    back = image_map(align, points=result["to_drawing"])
    for got, want in zip(back["to_image"], pixels, strict=True):
        assert got == pytest.approx(want, abs=0.01)
    fit = result["fit"]
    assert fit["controlled"] is False and "add a third pair" in fit["note"]
    assert "rms" not in fit and "drawing_segments" not in result
    assert fit["units_per_px"] == pytest.approx(SCALE)
    assert fit["rotation_deg"] == pytest.approx(30.0)


def test_more_pairs_are_fitted_and_report_the_pair_that_is_off():
    # one of four pairs was read 50 mm off along x: the fit spreads it, rms shows it
    align = [pair(0, 0), pair(800, 0), pair(800, 600, dx=50.0), pair(0, 600)]
    fit = image_map(align, pixels=[[0, 0]])["fit"]
    assert fit["controlled"] is True and fit["pairs"] == 4
    assert fit["max"] == max(fit["residuals"])
    assert fit["residuals"].index(fit["max"]) == 2
    rms = math.sqrt(sum(e * e for e in fit["residuals"]) / 4)
    assert fit["rms"] == pytest.approx(rms, abs=2e-3)
    assert 5.0 < fit["rms"] < 50.0
    exact = image_map([pair(0, 0), pair(400, 300), pair(800, 0)], pixels=[[10, 10]])
    assert exact["fit"]["rms"] == pytest.approx(0.0, abs=1e-3)
    assert exact["to_drawing"][0] == pytest.approx(truth(10, 10), abs=1e-3)


def test_measure_gives_lengths_and_angles_of_the_mapped_polyline():
    align = [pair(0, 0), pair(400, 300), pair(800, 0)]
    # a horizontal run of 100 px in the image is 1000 mm at 30 degrees (v is down)
    out = image_map(align, pixels=[[0, 0], [100, 0]], measure=True)
    (seg,) = out["drawing_segments"]
    assert seg["length"] == pytest.approx(1000.0, abs=1e-3)
    assert seg["angle"] == pytest.approx(30.0, abs=1e-3)


def test_refusals_name_their_argument():
    good = [pair(0, 0), pair(400, 300)]
    with pytest.raises(ValueError, match="at least two calibration pairs"):
        image_map([pair(0, 0)], pixels=[[0, 0]])
    with pytest.raises(ValueError, match="must differ"):
        image_map([pair(0, 0), pair(0, 0)], pixels=[[0, 0]])
    with pytest.raises(ValueError, match=r"pixels\[1\]: expected \[a, b\]"):
        image_map(good, pixels=[[0, 0], [1]])
    with pytest.raises(ValueError, match="give pixels"):
        image_map(good)


# ── round 5: the scan is read (B4 grid crop and colour features, B5 compare) ──

Image = pytest.importorskip("PIL.Image", reason="scan_read opens the scan with Pillow")
ImageDraw = pytest.importorskip("PIL.ImageDraw")

from engineering.arch import scan_read as sr  # noqa: E402

#: 10 mm per px, pixel (0, 0) on WCS (1000, 5000), v down: (u, v) -> (1000 + 10u, 5000 - 10v).
#: Three pairs on purpose: a wrong sign of v or a wrong scale moves every answer by metres.
ALIGN = [[0, 0, 1000, 5000], [400, 0, 5000, 5000], [0, 300, 1000, 2000]]
GREY, RED = (128, 128, 128), (255, 0, 0)


def wcs(u: float, v: float) -> list[float]:
    return [1000 + 10 * u, 5000 - 10 * v]


def _scan(tmp_path, name="scan.png") -> str:
    image = Image.new("RGB", (400, 300), "white")
    pen = ImageDraw.Draw(image)
    pen.rectangle([50, 40, 149, 59], fill=GREY)  # px u 50-150, v 40-60
    pen.rectangle([200, 100, 219, 199], fill=GREY)  # px u 200-220, v 100-200
    pen.rectangle([300, 200, 309, 209], fill=RED)  # dot, centre (305, 205)
    pen.ellipse([340, 30, 369, 59], outline=RED, width=3)  # ring, centre (355, 45)
    pen.line([(0, 250), (399, 250)], fill="black", width=3)
    path = tmp_path / name
    image.save(path)
    return str(path)


def test_rects_are_found_in_drawing_units(tmp_path):
    out = sr.find_rects(_scan(tmp_path), ALIGN, color="#808080")
    assert out["count"] == 2
    wide, tall = sorted(out["rects"], key=lambda r: r["size"][1])
    assert wide["center"] == pytest.approx(wcs(100, 50), abs=1)
    assert wide["size"] == pytest.approx([1000, 200], abs=1)
    assert wide["box"] == pytest.approx([1500, 4400, 2500, 4600], abs=1)
    assert tall["center"] == pytest.approx(wcs(210, 150), abs=1)
    assert tall["size"] == pytest.approx([200, 1000], abs=1)
    assert wide["pixels"] == 2000 and wide["fill"] == 1.0
    # size filter, limit and region all bound the answer
    assert sr.find_rects(_scan(tmp_path), ALIGN, color=GREY, size=[0, 500])["count"] == 0
    capped = sr.find_rects(_scan(tmp_path), ALIGN, color=GREY, limit=1)
    assert capped["count"] == 2 and len(capped["rects"]) == 1 and capped["truncated"]
    left = sr.find_rects(_scan(tmp_path), ALIGN, color=GREY, region=[1000, 4000, 2900, 5000])
    assert [r["center"] for r in left["rects"]] == [pytest.approx(wcs(100, 50), abs=1)]


def test_an_l_shaped_poche_is_its_two_arms(tmp_path):
    image = Image.new("RGB", (200, 200), "white")
    pen = ImageDraw.Draw(image)
    pen.rectangle([20, 20, 119, 39], fill=GREY)  # arm along u
    pen.rectangle([20, 40, 39, 139], fill=GREY)  # arm along v, touching it
    path = tmp_path / "l.png"
    image.save(path)
    rects = sr.find_rects(str(path), ALIGN, color=GREY)["rects"]
    assert sorted(r["px"] for r in rects) == [[20, 20, 120, 40], [20, 40, 40, 140]]


def test_blobs_give_centroids_of_dots_and_rings(tmp_path):
    out = sr.find_blobs(_scan(tmp_path), ALIGN, color=[255, 0, 0], tolerance=60)
    assert out["count"] == 2
    centres = sorted(b["center"] for b in out["blobs"])
    assert centres[0] == pytest.approx(wcs(305, 205), abs=2)  # the dot
    assert centres[1] == pytest.approx(wcs(355, 45), abs=2)  # the ring, hollow
    ring = next(b for b in out["blobs"] if b["center"][0] > 4000)
    assert ring["size"] == pytest.approx([300, 300], abs=11) and ring["fill"] < 0.6


def test_grid_crop_puts_the_scan_where_the_calibration_says(tmp_path):
    report = sr.grid_crop(_scan(tmp_path), ALIGN, region=[1400, 4300, 2600, 4700], grid=100)
    picture = Image.open(io.BytesIO(report["png"])).convert("RGB")
    assert list(picture.size) == report["px"]
    x0, _y0, _x1, y1 = report["region"]
    ppu = report["px_per_unit"]

    def at(x, y):
        return picture.getpixel((int((x - x0) * ppu), int((y1 - y) * ppu)))

    assert at(2030, 4430) == pytest.approx(GREY, abs=12)  # inside the wide rect, off the grid
    assert min(at(2030, 4630)) > 200  # above it: paper (a v-up error would put grey here)
    assert report["grid"]["step"] == 100 and report["scan"]["units_per_px"] == 10


def test_grid_crop_auto_zoom_and_px_window(tmp_path):
    report = sr.grid_crop(_scan(tmp_path), ALIGN, region_px=[0, 0, 100, 50], max_px=300)
    assert max(report["px"]) <= 300
    assert report["region"][0] == pytest.approx(1000) and report["region"][3] == pytest.approx(5000)
    with pytest.raises(ValueError, match="region_px: the window lies outside"):
        sr.grid_crop(_scan(tmp_path), ALIGN, region_px=[500, 500, 600, 600])


def test_detection_refusals_name_their_argument(tmp_path):
    scan = _scan(tmp_path)
    with pytest.raises(ValueError, match="color:"):
        sr.find_rects(scan, ALIGN, color="grey")
    with pytest.raises(ValueError, match="tolerance:"):
        sr.find_blobs(scan, ALIGN, color=RED, tolerance=300)
    with pytest.raises(ValueError, match="limit:"):
        sr.find_blobs(scan, ALIGN, color=RED, limit=0)
    with pytest.raises(ValueError, match="image: .* does not exist"):
        sr.find_rects(str(tmp_path / "none.png"), ALIGN, color=RED)


def test_without_pillow_a_file_mode_is_a_capability_refusal(monkeypatch, tmp_path):
    from backends.base import UnsupportedCapabilityError

    monkeypatch.setattr(sr, "pillow_available", lambda: False)
    with pytest.raises(UnsupportedCapabilityError) as refused:
        sr.find_rects(_scan(tmp_path), ALIGN, color=RED)
    assert refused.value.capability == "scan_image"


# ── through the server: modes, the image result, compare against the drawing ──


@pytest_asyncio.fixture
async def client():
    from fastmcp import Client

    import server

    async with Client(server.mcp) as connected:
        await connected.call_tool("drawing_new", {})
        yield connected


def _lines_scan(tmp_path) -> str:
    """Four black horizontal lines 3 px thick at v = 50, 100, 150, 200, u 20-380."""
    image = Image.new("RGB", (400, 300), "white")
    pen = ImageDraw.Draw(image)
    for v in (50, 100, 150, 200):
        pen.line([(20, v), (380, v)], fill="black", width=3)
    path = tmp_path / "lines.png"
    image.save(path)
    return str(path)


@pytest.mark.asyncio
async def test_compare_reports_half_the_source_drawn(client, tmp_path):
    pytest.importorskip("matplotlib")
    scan = _lines_scan(tmp_path)
    for v in (50, 150):  # two of the four source lines
        (x1, y1), (x2, y2) = wcs(20, v), wcs(380, v)
        await client.call_tool(
            "entity_create_line", {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "layer": "0"}
        )
    result = await client.call_tool("scan_read", {"align": ALIGN, "image": scan, "mode": "compare"})
    out = result.structured_content
    assert result.content[0].type == "image"
    assert out["source_covered_pct"] == pytest.approx(50, abs=5)
    assert out["drawing_on_source_pct"] > 90
    assert out["method"]["tolerance_px"] == sr.COMPARE_TOLERANCE_PX
    # the worst region holds an undrawn line: v = 100 or 200 -> y = 4000 or 3000
    y0, y1 = out["regions"][0]["box"][1], out["regions"][0]["box"][3]
    assert y0 <= 4000 <= y1 or y0 <= 3000 <= y1
    assert out["regions"][0]["missing_px"] > 0


@pytest.mark.asyncio
async def test_compare_with_a_shifted_drawing_covers_nothing(client, tmp_path):
    """A drawing one line spacing (500 mm) off - a scale or flip error - must not pass."""
    pytest.importorskip("matplotlib")
    scan = _lines_scan(tmp_path)
    for v in (75, 125):  # between the source lines, 25 px from each
        (x1, y1), (x2, y2) = wcs(20, v), wcs(380, v)
        await client.call_tool("entity_create_line", {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    out = (
        await client.call_tool("scan_read", {"align": ALIGN, "image": scan, "mode": "compare"})
    ).structured_content
    assert out["source_covered_pct"] < 5 and out["drawing_on_source_pct"] < 5


@pytest.mark.asyncio
async def test_tool_modes_and_refusals(client, tmp_path):
    scan = _scan(tmp_path)
    grid = await client.call_tool("scan_read", {"align": ALIGN, "image": scan, "mode": "grid"})
    assert grid.content[0].type == "image" and grid.structured_content["grid"]["step"] > 0
    rects = await client.call_tool(
        "scan_read", {"align": ALIGN, "image": scan, "mode": "rects", "color": "#808080"}
    )
    assert rects.structured_content["count"] == 2
    mapped = await client.call_tool("scan_read", {"align": ALIGN, "pixels": [[100, 50]]})
    assert mapped.structured_content["to_drawing"] == [wcs(100, 50)]
    for args, word in (
        ({"mode": "rects"}, "image:"),
        ({"mode": "blobs", "image": scan}, "color:"),
    ):
        bad = await client.call_tool("scan_read", {"align": ALIGN, **args}, raise_on_error=False)
        assert bad.is_error and bad.structured_content["kind"] == "invalid_args"
        assert word in bad.structured_content["message"]


@pytest.mark.asyncio
async def test_a_page_at_one_to_n_or_two_points_and_a_length_calibrate_in_one_call(
    client, tmp_path
):
    # round 6: the agent derived A3 at 300 dpi and checked two dimensions itself
    page = tmp_path / "a3.png"
    Image.new("RGB", (4961, 3508), "white").save(page)  # A3 landscape at 300 dpi
    out = (
        await client.call_tool(
            "scan_read", {"align": {"sheet": "A3", "scale": 20}, "image": str(page)}
        )
    ).structured_content["calibration"]
    assert out["align"] == [[0, 3508, 0, 0], [4961, 0, 8400, 5940]]
    assert out["dpi"] == pytest.approx(300, abs=0.1) and out["aspect_delta_pct"] < 0.1
    # the align it answered is the one view_screenshot / scan_read take: pixel -> WCS
    mapped = await client.call_tool(
        "scan_read", {"align": out["align"], "pixels": [[4961 / 2, 3508 / 2]]}
    )
    assert mapped.structured_content["to_drawing"][0] == pytest.approx([4200, 2970], abs=0.01)
    # two pixels 250 px apart along u are 1000 mm: 4 mm a pixel, the first on (500, 0)
    two = (
        await client.call_tool(
            "scan_read",
            {
                "align": {"points": [[100, 40], [350, 40]], "length": 1000, "at": [500, 0]},
                "pixels": [[350, 140]],
            },
        )
    ).structured_content
    assert two["calibration"]["units_per_px"] == 4
    assert two["to_drawing"] == [[1500, -400]]  # v down: 100 px below is 400 mm lower
    bad = await client.call_tool(
        "scan_read",
        {"align": {"sheet": "A3", "scale": 0}, "image": str(page)},
        raise_on_error=False,
    )
    assert bad.is_error and "calibrate.scale" in bad.structured_content["message"]


@pytest.mark.asyncio
async def test_compare_counts_coloured_lines_and_grey_fills_of_the_source(client, tmp_path):
    """Round 6: a luminance threshold (< 128) lost orange lines and a grey poche,
    so a correctly drawn one counted as extra. Synthetic source, known truth."""
    pytest.importorskip("matplotlib")
    image = Image.new("RGB", (400, 300), "white")
    pen = ImageDraw.Draw(image)
    for v in (50, 150):
        pen.line([(20, v), (380, v)], fill=(255, 140, 0), width=3)  # orange, luminance 158
    pen.rectangle([220, 180, 319, 259], fill=(200, 200, 200))  # grey poche, luminance 200
    scan = tmp_path / "colour.png"
    image.save(scan)
    ents = [
        {"type": "line", "x1": wcs(20, v)[0], "y1": wcs(20, v)[1], "x2": wcs(380, v)[0],
         "y2": wcs(380, v)[1]}
        for v in (50, 150)
    ]  # fmt: skip
    box = [wcs(220, 260), wcs(320, 260), wcs(320, 180), wcs(220, 180)]
    ents.append({"type": "hatch", "boundary_points": box, "pattern": "SOLID"})
    await client.call_tool("entity_batch_create", {"entities": ents})
    out = (
        await client.call_tool("scan_read", {"align": ALIGN, "image": str(scan), "mode": "compare"})
    ).structured_content
    assert out["source_covered_pct"] > 95 and out["drawing_on_source_pct"] > 95
    assert out["source"]["fills"]["px"] > 5000 and out["source"]["fills"]["covered_pct"] > 95
    assert out["source"]["strokes"]["px"] > 1500  # the two lines, not counted as fill
    assert out["drawing"]["fills"]["on_source_pct"] > 95
    assert out["drawing"]["lines"]["on_source_pct"] > 95
    assert out["drawing"]["text"] == {"drawn_px": 0, "on_source_pct": None}


@pytest.mark.asyncio
async def test_compare_on_a_layout_fits_the_page_and_sees_paper_space(client, tmp_path):
    """Round 6: model-only compare called the frame and title block always missing."""
    pytest.importorskip("matplotlib")
    await client.call_tool("layout_create", {"name": "A3"})
    await client.call_tool("page_setup_apply", {"layout": "A3", "paper": "A3"})
    frame = {"type": "rectangle", "x1": 20, "y1": 10, "x2": 410, "y2": 287}
    await client.call_tool("entity_batch_create", {"entities": [frame], "layout": "A3"})
    image = Image.new("RGB", (1680, 1188), "white")  # the page at 4 px a mm
    ImageDraw.Draw(image).rectangle([80, 40, 1640, 1148], outline="black", width=3)
    scan = tmp_path / "page.png"
    image.save(scan)
    out = (
        await client.call_tool("scan_read", {"image": str(scan), "mode": "compare", "layout": "A3"})
    ).structured_content
    assert out["source_covered_pct"] > 90 and out["drawing_on_source_pct"] > 90
    bad = await client.call_tool(
        "scan_read", {"image": str(scan), "mode": "grid"}, raise_on_error=False
    )
    assert bad.is_error and "align:" in bad.structured_content["message"]
