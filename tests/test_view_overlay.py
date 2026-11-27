"""T-10 - a scan under the CAD geometry in ``view_screenshot`` (overlay_image).

A redraw of a scan is checked by laying the CAD over the scan: the gate puts a
red marker at an asymmetric pixel of a synthetic scan, calibrates it with a
rotation, a scale and the image's v-down axis, and reads the marker back where
the two calibration pairs say it lands - and the CAD stays drawn on top.
"""

from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
Image = pytest.importorskip("PIL.Image", reason="the overlay is read with Pillow")

from backends.region_render import overlay_transform  # noqa: E402

#: Scan 200 x 100 px; the marker block covers pixels u 150-169, v 10-29.
SCAN = (200, 100)
MARK_UV = (160.0, 20.0)
#: Pixel (0, 0) -> (100, 0); pixel (0, 100) -> (50, 0): v grows towards -x,
#: so u grows towards -y (a -90 degree turn at 0.5 units per pixel; a
#: similarity never mirrors, and image v points down).
ALIGN = [[0, 0, 100, 0], [0, 100, 50, 0]]
MARK_XY = (100 - 0.5 * MARK_UV[1], -0.5 * MARK_UV[0])  # (90, -80)
REGION = [0, -110, 120, 10]


def _scan(path) -> str:
    image = Image.new("RGB", SCAN, (255, 255, 255))
    for u in range(150, 170):
        for v in range(10, 30):
            image.putpixel((u, v), (255, 0, 0))
    image.save(path)
    return str(path)


def _pixel(report: dict, x: float, y: float) -> tuple[int, int, int]:
    x0, _y0, _x1, y1 = report["region"]
    ppu = report["px_per_unit"]
    image = Image.open(io.BytesIO(report["png"])).convert("RGB")
    return image.getpixel((int((x - x0) * ppu), int((y1 - y) * ppu)))


def _reddish(rgb) -> bool:
    r, g, b = rgb
    return r > g + 60 and r > b + 60


def test_transform_maps_both_pairs_and_keeps_v_down():
    a, b = overlay_transform(ALIGN)
    for u, v, x, y in ALIGN:
        z = a * complex(u, -v) + b
        assert (z.real, z.imag) == pytest.approx((x, y))
    z = a * complex(MARK_UV[0], -MARK_UV[1]) + b
    assert (z.real, z.imag) == pytest.approx(MARK_XY)
    assert abs(a) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_scan_lands_where_the_pairs_say_under_the_cad(backend, tmp_path):
    await backend.entity_create_line(0, -40, 200, -40, color=3)  # green, crosses the scan
    report = await backend.view_render(
        region=REGION,
        max_px=480,
        overlay_image=_scan(tmp_path / "scan.png"),
        overlay_align=ALIGN,
        overlay_opacity=0.8,
    )
    assert _reddish(_pixel(report, *MARK_XY))
    # The marker's mirror (v not flipped) and its unrotated spot stay unmarked.
    assert not _reddish(_pixel(report, 100 - 0.5 * (100 - MARK_UV[1]), MARK_XY[1]))
    assert not _reddish(_pixel(report, 0.5 * MARK_UV[0], -0.5 * MARK_UV[1]))
    # Scan area is paper-light, outside it is the dark background.
    assert sum(_pixel(report, 60, -10)) > sum(_pixel(report, 110, 5)) + 200
    # The CAD line is drawn over the scan, not hidden under it.
    plain = await backend.view_render(region=REGION, max_px=480)
    line = _pixel(plain, 70, -40)
    assert line[1] > line[0] + 100  # green on the plain render
    assert _pixel(report, 70, -40) == line
    overlay = report["overlay"]
    assert overlay["pixels"] == list(SCAN)
    assert overlay["units_per_px"] == pytest.approx(0.5)
    assert overlay["rotation_deg"] == pytest.approx(-90.0)
    assert overlay["bounds"] == pytest.approx([50, -100, 100, 0])


@pytest.mark.asyncio
async def test_without_region_the_framing_covers_the_scan(backend, tmp_path):
    report = await backend.view_render(
        max_px=300, overlay_image=_scan(tmp_path / "scan.png"), overlay_align=ALIGN
    )
    x0, y0, x1, y1 = report["region"]
    assert x0 <= 50 and y0 <= -100 and x1 >= 100 and y1 >= 0
    assert x1 - x0 < 60 and y1 - y0 < 110  # framed on the scan, not on (0, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"overlay_align": ALIGN}, "go together"),
        ({"image": True, "overlay_align": [[0, 0, 1, 1], [0, 0, 2, 2]]}, "must differ"),
        ({"image": True, "overlay_align": [[0, 0, 1, 1]]}, "two calibration pairs"),
        ({"image": True, "overlay_align": ALIGN, "overlay_opacity": 1.5}, "overlay_opacity"),
    ],
)
async def test_bad_overlay_is_refused_by_name(backend, tmp_path, kwargs, message):
    if kwargs.pop("image", False):
        kwargs["overlay_image"] = _scan(tmp_path / "scan.png")
    with pytest.raises(ValueError, match=message):
        await backend.view_render(region=[0, 0, 10, 10], **kwargs)


@pytest.mark.asyncio
async def test_unreadable_scan_is_refused(backend, tmp_path):
    bogus = tmp_path / "scan.png"
    bogus.write_text("not an image")
    with pytest.raises(ValueError, match="not an image"):
        await backend.view_render(overlay_image=str(bogus), overlay_align=ALIGN)


@pytest.mark.asyncio
async def test_tool_validates_the_path_and_reports_the_overlay(tmp_path):
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    import server

    scan = _scan(tmp_path / "scan.png")
    async with Client(server.mcp) as client:
        await client.call_tool("drawing_new", {})
        result = await client.call_tool(
            "view_screenshot",
            {
                "region": REGION,
                "max_px": 256,
                "overlay_image": scan,
                "overlay_align": ALIGN,
            },
        )
        report = json.loads(result.content[1].text)
        assert report["overlay"]["path"] == scan and report["overlay"]["opacity"] == 0.5
        with pytest.raises(ToolError):
            await client.call_tool(
                "view_screenshot",
                {"overlay_image": str(tmp_path / ".." / "x.png"), "overlay_align": ALIGN},
            )
