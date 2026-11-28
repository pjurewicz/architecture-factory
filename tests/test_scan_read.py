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
@pytest.mark.parametrize("mode", ["map", "grid", "rects", "blobs", "snap", "lines", "compare"])
async def test_the_sheet_shortcut_calibrates_every_mode(client, tmp_path, mode):
    # round 7: rects took the scan's pixel size for its `size` argument
    if mode == "compare":
        pytest.importorskip("matplotlib")
    page = tmp_path / "a4.png"
    image = Image.new("RGB", (1169, 827), "white")  # A4 landscape at 100 dpi
    ImageDraw.Draw(image).rectangle([200, 200, 399, 239], fill=(153, 153, 153))
    image.save(page)
    args = {"align": {"sheet": "A4", "scale": 50}, "image": str(page), "mode": mode}
    if mode in ("rects", "blobs"):
        args["color"] = "#999999"
    if mode in ("map", "snap"):
        args["pixels"] = [[200, 200]]
    out = await client.call_tool("scan_read", args, raise_on_error=False)
    assert not out.is_error, out.content
    data = out.structured_content
    assert data["calibration"]["align"][0][:2] == [0, 827]
    if mode in ("rects", "blobs"):
        (row,) = data[mode]
        # 200 px of a 1169 px page across 297 mm at 1:50
        assert row["size"][0] == pytest.approx(200 * 297 / 1169 * 50, rel=1e-3)


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


def test_the_grid_avoids_the_scan_colour_and_letters_y_on_its_line(tmp_path):
    # round 6: a red grid over red dimension lines; a y label above its line
    image = Image.new("RGB", (400, 300), "white")
    pen = ImageDraw.Draw(image)
    for v in range(20, 300, 20):
        pen.line([(0, v), (399, v)], fill=(220, 0, 0), width=2)  # red dimension lines
    red = tmp_path / "red.png"
    image.save(red)
    report = sr.grid_crop(str(red), ALIGN, grid=500, zoom=2)
    assert "e60000" not in report["read"]  # not the red the scan uses
    assert sr.grid_crop(str(red), ALIGN, grid=500, zoom=2, color="#00a000")["read"].startswith(
        "lines in #00a000"
    )
    blank = tmp_path / "blank.png"
    Image.new("RGB", (400, 300), "white").save(blank)
    report = sr.grid_crop(str(blank), ALIGN, grid=500, zoom=2)
    pic = Image.open(io.BytesIO(report["png"])).convert("RGB")
    ppu, top = report["px_per_unit"], report["region"][3]
    for y in (4000, 3000):
        py = round((top - y) * ppu)
        inked = [
            j for j in range(max(0, py - 20), min(pic.size[1], py + 20))
            if any(min(pic.getpixel((i, j))) < 150 for i in range(3, 12)) and abs(j - py) > 1
        ]  # fmt: skip
        assert min(inked) < py < max(inked), y  # the label straddles its line


def test_the_x_labels_are_centred_on_their_line(tmp_path):
    blank = tmp_path / "blank.png"
    Image.new("RGB", (400, 300), "white").save(blank)
    report = sr.grid_crop(str(blank), ALIGN, grid=500, zoom=2)
    pic = Image.open(io.BytesIO(report["png"])).convert("RGB")
    x0, ppu = report["region"][0], report["px_per_unit"]
    for x in (2500, 3000, 3500):
        px = round((x - x0) * ppu)
        cols = [
            i for i in range(px - 40, px + 40)
            if any(min(pic.getpixel((i, j))) < 150 for j in range(2, 11))
        ]  # fmt: skip
        assert (min(cols) + max(cols)) / 2 == pytest.approx(px, abs=2), x  # was 10-20 mm right


@pytest.mark.parametrize("size, width", [((400, 300), 1), ((4000, 3000), 1), ((4000, 3000), 2)])
def test_the_grid_colour_is_read_from_the_full_resolution_ink(tmp_path, size, width):
    # a thumbnail washes thin lines on a big scan out to paper: red on red again
    image = Image.new("RGB", size, "white")
    pen = ImageDraw.Draw(image)
    for v in range(20, size[1], 40):
        pen.line([(0, v), (size[0] - 1, v)], fill=(220, 0, 0), width=width)
    for u in range(20, size[0], 40):
        pen.line([(u, 0), (u, size[1] - 1)], fill=(0, 90, 250), width=width)
    path = tmp_path / "ink.png"
    image.save(path)
    align = [[0, 0, 0, size[1] * 10], [size[0], size[1], size[0] * 10, 0]]
    read = sr.grid_crop(str(path), align, grid=500, zoom=1, max_px=1200)["read"]
    assert "e60000" not in read and "005aff" not in read  # neither colour the scan uses


def test_a_small_window_is_enlarged_to_read(tmp_path):
    scan = _scan(tmp_path)
    big = sr.grid_crop(scan, ALIGN, region_px=[0, 0, 100, 50])  # 1000 mm: was 4x, 400 px
    assert big["zoom"] == pytest.approx(15.68) and max(big["px"]) == 1568
    tiny = sr.grid_crop(scan, ALIGN, region_px=[0, 0, 20, 20])  # the default stops at 16x
    assert tiny["zoom"] == 16 and tiny["px"] == [320, 320]


def test_a_downsampled_strip_warns_and_says_how_to_split(tmp_path):
    strip = tmp_path / "strip.png"
    Image.new("RGB", (5600, 900), "white").save(strip)
    align = [[0, 0, 0, 900], [5600, 900, 5600, 0]]  # 1 mm a px
    report = sr.grid_crop(str(strip), align, max_px=2400)
    assert report["px"] == [2400, 386]
    assert "split region into 3 windows" in report["warning"]
    assert "warning" not in sr.grid_crop(
        str(strip), align, region_px=[0, 0, 1800, 900], max_px=2400
    )


def test_snap_finds_the_corner_and_the_end_near_a_pixel(tmp_path):
    image = Image.new("RGB", (400, 300), "white")
    pen = ImageDraw.Draw(image)
    pen.line([(100, 100), (300, 100)], fill="black", width=3)  # rows 99-101
    pen.line([(100, 100), (100, 250)], fill="black", width=3)  # columns 99-101
    scan = tmp_path / "l.png"
    image.save(scan)
    out = sr.snap_points(str(scan), ALIGN, pixels=[[108, 93], [296, 108], [20, 20]])
    corner, end, none = out["snapped"]
    assert corner["kind"] == "corner" and corner["px"] == pytest.approx([100.5, 100.5], abs=1)
    assert corner["point"] == pytest.approx(wcs(100.5, 100.5), abs=10)
    assert end["kind"] == "end" and end["px"] == pytest.approx([302, 100.5], abs=2)
    assert none["found"] is False


def _plan_scan(tmp_path) -> str:
    """A grey-poche plan at 10 mm/px: a 200 mm shell, two 100 mm walls in a T,
    an 800 mm gap in the bottom wall, a red dimension line and a red band
    across walls, blue hatching over one wall and thin grey lettering."""
    image = Image.new("RGB", (400, 300), "white")
    pen = ImageDraw.Draw(image)
    for box in ([40, 40, 359, 59], [40, 240, 99, 259], [180, 240, 359, 259],
                [40, 40, 59, 259], [340, 40, 359, 259],  # shell
                [195, 60, 204, 239], [60, 145, 194, 154]):  # fmt: skip
        pen.rectangle(box, fill=GREY)
    pen.line([(250, 20), (250, 80)], fill=RED, width=2)  # a dimension line across the top
    pen.rectangle([330, 100, 370, 105], fill=RED)  # a red band across the right wall
    for d in range(0, 220, 8):
        pen.line([(40, 40 + d), (59, 59 + d)], fill=(0, 0, 255), width=1)  # hatching
    pen.line([(80, 100), (150, 100)], fill=GREY, width=2)  # lettering, not a wall
    path = tmp_path / "plan.png"
    image.save(path)
    return str(path)


def test_walls_come_back_as_spec_axes_joined_at_their_junctions(tmp_path):
    out = sr.find_walls(
        _plan_scan(tmp_path), ALIGN, color=GREY, skip=["#ff0000"], gap=1000, size=[60, 400]
    )
    got = sorted((tuple(map(tuple, w["axis"])), w["thickness"]) for w in out["walls"])
    assert got == sorted(
        [
            (((1500, 4500), (4500, 4500)), 200),  # top, through the dimension line
            (((1500, 2500), (4500, 2500)), 200),  # bottom, across the 800 mm gap
            (((1500, 2500), (1500, 4500)), 200),  # left, under its hatching
            (((4500, 2500), (4500, 4500)), 200),  # right, through the red band
            (((3000, 2500), (3000, 4500)), 100),  # the T at both ends
            (((1500, 3500), (3000, 3500)), 100),  # between the shell and the T wall
        ]
    )
    bottom = next(w["id"] for w in out["walls"] if w["axis"] == [[1500, 2500], [4500, 2500]])
    assert out["openings"] == [{"wall": bottom, "offset": 500, "width": 800}]
    # without skip the red band cuts the right wall; without gap the bottom is two walls
    plain = sr.find_walls(_plan_scan(tmp_path), ALIGN, color=GREY, size=[60, 400])
    assert len(plain["walls"]) == 8 and plain["openings"] == []
    local = sr.find_walls(
        _plan_scan(tmp_path), ALIGN, color=GREY, skip="#ff0000", gap=1000,
        size=[60, 400], origin=[1500, 2500], step=50,
    )  # fmt: skip
    assert [[0, 0], [3000, 0]] in [w["axis"] for w in local["walls"]]


@pytest.mark.asyncio
async def test_scan_walls_paste_into_arch_plan_from_spec(client, tmp_path):
    scan = _plan_scan(tmp_path)
    args = {"align": ALIGN, "image": scan, "mode": "walls", "color": "#808080",
            "skip": ["#ff0000"], "gap": 1000, "size": [60, 400]}  # fmt: skip
    read = (await client.call_tool("scan_read", args)).structured_content
    await client.call_tool("drawing_new", {})
    walls = [{**w, "material": "brick"} for w in read["walls"]]
    openings = [{**o, "id": f"D{i}", "kind": "door"} for i, o in enumerate(read["openings"])]
    plan = await client.call_tool(
        "arch_plan_from_spec", {"spec": {"walls": walls, "openings": openings}}
    )
    data = plan.structured_content
    assert not plan.is_error and data["omitted"] == []
    bad = await client.call_tool(
        "scan_read", {**args, "mode": "blobs", "skip": "#ff0000"}, raise_on_error=False
    )
    assert bad.is_error and "rects, walls and poche only" in bad.structured_content["message"]


@pytest.mark.asyncio
async def test_compare_does_not_let_a_dense_hatch_cover_the_lines_under_it(client, tmp_path):
    """Round 7: within the 2 px tolerance a dense drawing hatch reached every
    furniture line under it, so an undrawn sofa scored as drawn."""
    pytest.importorskip("matplotlib")
    image = Image.new("RGB", (400, 300), "white")
    pen = ImageDraw.Draw(image)
    for v in (120, 160, 200):  # three furniture lines, never drawn
        pen.line([(120, v), (280, v)], fill="black", width=1)
    scan = tmp_path / "under.png"
    image.save(scan)
    box = [wcs(100, 220), wcs(300, 220), wcs(300, 100), wcs(100, 100)]
    await client.call_tool(
        "entity_create_hatch",
        {"boundary_points": box, "pattern": "ANSI31", "scale": 0.5, "layer": "0"},
    )
    out = (
        await client.call_tool("scan_read", {"align": ALIGN, "image": str(scan), "mode": "compare"})
    ).structured_content
    assert out["source_covered_pct"] < 10
    assert out["source_covered_by_fills_only_pct"] > 50
    assert "font" in out["method"]["text"]


def _sheet_scan(tmp_path) -> tuple[str, list]:
    """Round 8: a sheet at 5 mm/px turned 0.2 degrees, everything one grey: a
    250 mm shell (0..6000 x 0..4000 on the axes) with a 900 mm door in the top
    wall, 120 mm walls at x 3500 and y 2000, hatching and furniture outlines
    (30 mm) in the rooms, a title block's rows and a solid logo 2 m below the
    plan and a sheet frame - only the walls may come back."""
    turn = math.radians(0.2)

    def px(x, y):  # WCS -> scan px; scan (0, 0) on WCS (-1000, 5000)
        dx, dy = x + 1000, y - 5000
        c, s = math.cos(-turn), math.sin(-turn)
        return (dx * c - dy * s) / 5.0, -(dx * s + dy * c) / 5.0

    image = Image.new("RGB", (1700, 1700), "white")
    pen = ImageDraw.Draw(image)
    fill = (185, 185, 185)
    for x0, y0, x1, y1 in (
        (-125, -125, 6125, 125), (-125, 3875, 2000, 4125), (2900, 3875, 6125, 4125),
        (-125, -125, 125, 4125), (5875, -125, 6125, 4125),
        (3440, 125, 3560, 3875), (125, 1940, 3440, 2060), (6600, -3200, 6900, -2900),
    ):  # fmt: skip
        pen.polygon([px(x0, y0), px(x1, y0), px(x1, y1), px(x0, y1)], fill=fill)
    for d in range(0, 3000, 100):
        pen.line([px(300 + d, 300), px(1800 + d, 1800)], fill=fill, width=1)
    for x0, y0, x1, y1 in ((4000, 500, 5500, 1500), (600, 2500, 2400, 3400)):
        pen.rectangle([*px(x0, y1), *px(x1, y0)], outline=fill, width=6)
    for i in range(6):
        pen.line([px(6500, -2200 - i * 200), px(7000, -2200 - i * 200)], fill=fill, width=5)
    pen.rectangle([2, 2, 1697, 1697], outline=fill, width=3)
    path = tmp_path / "sheet.png"
    image.save(path)
    align = []
    for x, y in ((-1000, 5000), (7000, 5000), (-1000, -1000)):
        align.append([*px(x, y), x, y])
    return str(path), align


def test_walls_on_a_sheet_are_the_plan_only_square_on_the_grid(tmp_path):
    """Round 8: without size the thinnest wall is 80 mm, without a region only
    the biggest cluster of fill is read, every axis is square on the grid and
    a carried end lies on the crossing wall's axis - pasted as they come."""
    scan, align = _sheet_scan(tmp_path)
    out = sr.find_walls(scan, align, color="#b9b9b9", tolerance=25, gap=3000, step=5)
    got = sorted((tuple(map(tuple, w["axis"])), w["thickness"]) for w in out["walls"])
    want = [
        (((0, 0), (6000, 0)), 250), (((0, 4000), (6000, 4000)), 250),
        (((0, 0), (0, 4000)), 250), (((6000, 0), (6000, 4000)), 250),
        (((3500, 0), (3500, 4000)), 120), (((0, 2000), (3500, 2000)), 120),
    ]  # fmt: skip
    assert [a for a, _ in got] == sorted(a for a, _ in want)
    for (_, t), (_, w) in zip(got, sorted(want), strict=True):
        assert abs(t - w) <= 5
    top = next(w["id"] for w in out["walls"] if w["axis"] == [[0, 4000], [6000, 4000]])
    (door,) = out["openings"]
    assert (
        door["wall"] == top and abs(door["offset"] - 2000) <= 5 and abs(door["width"] - 900) <= 10
    )
    assert out["cluster"][1] > -500 and "warning" not in out  # the title block left out
    thin = sr.find_walls(
        scan, align, color="#b9b9b9", tolerance=25, size=[5], region=[-1000, -4000, 7000, 5000]
    )
    assert "median thickness" in thin["warning"]


def test_overlapping_and_collinear_pieces_become_one_wall():
    segs = [
        {"o": "x", "c": 0, "a": [0, 3000], "t": 200, "gaps": []},
        {"o": "x", "c": 0, "a": [0, 3000], "t": 190, "gaps": []},  # a duplicate
        {"o": "x", "c": 5, "a": [3800, 6000], "t": 200, "gaps": []},  # across a 800 gap
        {"o": "x", "c": 0, "a": [9500, 9800], "t": 200, "gaps": []},  # beyond gap
    ]
    joined = sr._join_square(segs, 5.0, 1000.0)
    assert [(s["a"], s["gaps"]) for s in joined] == [
        ([0, 6000], [(3000, 3800)]),
        ([9500, 9800], []),
    ]


@pytest.mark.asyncio
async def test_poche_and_walls_from_a_sheet_paste_into_arch_plan_from_spec(client, tmp_path):
    scan, align = _sheet_scan(tmp_path)
    args = {
        "align": align,
        "image": scan,
        "color": "#b9b9b9",
        "tolerance": 25,
        "gap": 3000,
        "grid": 5,
    }
    poche = (await client.call_tool("scan_read", {**args, "mode": "poche"})).structured_content
    polygons = poche["wall_polygons"]
    for polygon in polygons:  # square on the grid: every edge horizontal or vertical
        pts = polygon["points"]
        for a, b in zip(pts, pts[1:] + pts[:1], strict=True):
            assert a[0] == b[0] or a[1] == b[1]
            assert a[0] % 5 == 0 and a[1] % 5 == 0

    def area_of(pts):
        ring = zip(pts, pts[1:] + pts[:1], strict=True)
        return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in ring)) / 2

    area = sum(area_of(p["points"]) for p in polygons)
    shell = 6250 * 4250 - 5750 * 3750 - 900 * 250 + 120 * 3750 + 120 * 3315
    assert area == pytest.approx(shell, rel=0.02)
    walls = (await client.call_tool("scan_read", {**args, "mode": "walls"})).structured_content
    # polygons host no opening: the room behind the door gap reads open
    for spec, closed in (({"wall_polygons": polygons}, 2), ({"walls": walls["walls"]}, 3)):
        await client.call_tool("drawing_new", {})
        plan = await client.call_tool("arch_plan_from_spec", {"spec": spec})
        assert not plan.is_error and plan.structured_content["omitted"] == []
        rooms = (await client.call_tool("arch_rooms_detect", {})).structured_content
        assert len(rooms["rooms"]) == closed, spec.keys()


# ── round 9: lines along the WCS axes ──


def _elevation_scan(tmp_path) -> str:
    """A wall elevation: flutes every 20 px (200 mm) at u 60-180, a zone station
    5 px wide at u 200, tile joints every 30 px at v 60-180 right of it, a fill."""
    image = Image.new("RGB", (400, 300), "white")
    pen = ImageDraw.Draw(image)
    for u in range(60, 181, 20):
        pen.line([(u, 40), (u, 240)], fill="black", width=1)
    pen.rectangle([198, 20, 202, 260], fill="black")  # u 198-203: centre 200.5
    for v in range(60, 181, 30):
        pen.line([(210, v), (380, v)], fill=(90, 90, 90), width=1)
    pen.rectangle([300, 200, 360, 280], fill=GREY)  # a fill, not a line
    path = tmp_path / "elevation.png"
    image.save(path)
    return str(path)


def test_lines_give_flute_pitch_zone_station_and_tile_joints(tmp_path):
    out = sr.find_lines(_elevation_scan(tmp_path), ALIGN)
    vert, horiz = out["vertical"], out["horizontal"]
    # a 1 px line at u covers [u, u + 1): centre u + 0.5 -> x = 1000 + 10 (u + 0.5)
    flutes = [1000 + 10 * (u + 0.5) for u in range(60, 181, 20)]
    assert [r["at"] for r in vert["lines"]] == pytest.approx([*flutes, 3005], abs=0.01)
    assert vert["spacing"] == pytest.approx([200] * 7, abs=0.01)
    assert vert["pitch"]["median"] == pytest.approx(200, abs=0.01)
    assert vert["lines"][-1]["width"] == pytest.approx(50, abs=0.01)
    assert vert["lines"][0]["span"] == pytest.approx([2590, 4600], abs=0.01)
    assert vert["fills_dropped"] == 1  # the grey box, 61 px wide
    # joints: y = 5000 - 10 (v + 0.5), ascending
    joints = sorted(5000 - 10 * (v + 0.5) for v in range(60, 181, 30))
    assert [r["at"] for r in horiz["lines"]] == pytest.approx(joints, abs=0.01)
    assert horiz["pitch"] == {"median": 300, "min": 300, "max": 300}
    # one colour, a region and the size bounds narrow the answer
    grey = sr.find_lines(_elevation_scan(tmp_path), ALIGN, color=(90, 90, 90), tolerance=10)
    assert grey["vertical"]["count"] == 0 and grey["horizontal"]["count"] == 5
    left = sr.find_lines(_elevation_scan(tmp_path), ALIGN, region=[1500, 2000, 2900, 5000])
    assert left["vertical"]["count"] == 7 and left["horizontal"]["count"] == 0
    thin = sr.find_lines(_elevation_scan(tmp_path), ALIGN, size=[None, 20])
    assert thin["vertical"]["count"] == 7  # the 50 mm station is a fill now
    capped = sr.find_lines(_elevation_scan(tmp_path), ALIGN, limit=2)
    assert capped["vertical"]["count"] == 8 and len(capped["vertical"]["lines"]) == 2
    assert capped["vertical"]["truncated"]
    with pytest.raises(ValueError, match="size:"):
        sr.find_lines(_elevation_scan(tmp_path), ALIGN, size="big")


def test_lines_on_a_turned_scan_are_read_along_wcs(tmp_path):
    """The scan of test 1's calibration (30 degrees): lines drawn along WCS x."""
    image = Image.new("RGB", (500, 500), "white")
    pen = ImageDraw.Draw(image)
    c, s = math.cos(TURN), math.sin(TURN)

    def px(x, y):  # the inverse of truth(): WCS -> scan pixel
        dx, dy = x - ORIGIN[0], y - ORIGIN[1]
        return ((dx * c + dy * s) / SCALE, -(-dx * s + dy * c) / SCALE)

    for x in (2400, 3000, 3600):  # vertical in WCS, 600 mm apart
        pen.line([px(x, 300), px(x, 1500)], fill="black", width=2)
    path = tmp_path / "turned.png"
    image.save(path)
    align = [pair(0, 0), pair(400, 0), pair(0, 300)]
    out = sr.find_lines(str(path), align, region=[2000, 200, 4000, 1600], size=[800])
    assert out["vertical"]["spacing"] == pytest.approx([600, 600], abs=SCALE)
    assert [r["at"] for r in out["vertical"]["lines"]] == pytest.approx(
        [2400, 3000, 3600], abs=SCALE
    )
    assert out["horizontal"]["count"] == 0


@pytest.mark.asyncio
async def test_lines_through_the_tool(client, tmp_path):
    scan = _elevation_scan(tmp_path)
    lines = await client.call_tool(
        "scan_read",
        {"align": {"points": [[0, 0], [400, 0]], "length": 4000}, "image": scan, "mode": "lines"},
    )
    out = lines.structured_content
    assert out["calibration"]["units_per_px"] == 10
    assert out["vertical"]["pitch"]["median"] == pytest.approx(200, abs=0.01)
    bad = await client.call_tool(
        "scan_read",
        {"align": ALIGN, "image": scan, "mode": "lines", "gap": 10},
        raise_on_error=False,
    )
    assert bad.is_error and "gap" in bad.structured_content["message"]
