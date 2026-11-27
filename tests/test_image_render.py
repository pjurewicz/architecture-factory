"""IMAGE entities (``image_attach``) drawn in ``view_screenshot`` and the plot.

A screenshot shows every image on a visible layer, under the vectors and
faded; a plot (PDF, or the same path written as PNG) only an image whose layer
plots, as AutoCAD plots. The scan is asymmetric - a red block in its top-left
corner on blue - so a mirrored or wrongly rotated transform puts the red
somewhere else.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
Image = pytest.importorskip("PIL.Image", reason="images are read with Pillow")

RED, BLUE = (255, 0, 0), (0, 0, 255)
#: 40 x 20 px; red covers u 0-9, v 0-9 (top-left), blue everything else.
SCAN = (40, 20)


def _scan(path) -> str:
    image = Image.new("RGB", SCAN, BLUE)
    image.paste(RED, (0, 0, 10, 10))
    image.save(path)
    return str(path)


def _pixel(report: dict, x: float, y: float) -> tuple[int, int, int]:
    x0, _y0, _x1, y1 = report["region"]
    ppu = report["px_per_unit"]
    image = Image.open(io.BytesIO(report["png"])).convert("RGB")
    return image.getpixel((int((x - x0) * ppu), int((y1 - y) * ppu)))


def _red(rgb) -> bool:
    r, g, b = rgb[:3]
    return r > g + 60 and r > b + 60


def _blue(rgb) -> bool:
    r, g, b = rgb[:3]
    return b > r + 60 and b > g + 60


async def _plot(backend, tmp_path, name: str, **kwargs):
    """The plot written as PDF and, through the same renderer, as PNG."""
    pdf = await backend.drawing_export_pdf(str(tmp_path / f"{name}.pdf"), **kwargs)
    png = await backend.drawing_export_pdf(str(tmp_path / f"{name}.png"), **kwargs)
    assert pdf["ok"] and png["ok"], (pdf, png)
    data = (tmp_path / f"{name}.pdf").read_bytes()
    return pdf, b"/Subtype /Image" in data, Image.open(tmp_path / f"{name}.png").convert("RGB")


def _pixels(image):
    """(x, y, rgb) of every pixel."""
    data, width = image.tobytes(), image.width
    for i in range(image.width * image.height):
        yield i % width, i // width, tuple(data[3 * i : 3 * i + 3])


def _any(image, predicate) -> bool:
    return any(predicate(rgb) for _x, _y, rgb in _pixels(image))


@pytest.mark.asyncio
async def test_rotated_image_lies_under_the_line_unmirrored(backend, tmp_path):
    # The line comes first, so the image is later in draw order - and still under it.
    await backend.entity_create_line(21, -2, 21, 22, color=3)
    # At (30, 0), 0.5 mm/px, turned 90 degrees: the image covers x 20-30, y 0-20,
    # and its top-left red block x 20-25, y 0-5.
    await backend.image_attach(_scan(tmp_path / "scan.png"), (30.0, 0.0), 0.5, 90.0, fade=0)
    report = await backend.view_render(region=[10, -5, 40, 25], max_px=600)
    assert _red(_pixel(report, 23.0, 2.5))
    assert _blue(_pixel(report, 23.0, 17.5))  # mirrored in u
    assert _blue(_pixel(report, 27.5, 2.5))  # mirrored in v
    line = [_pixel(report, x, 2.5) for x in (20.95, 21.0, 21.05)]
    assert any(g > 150 and r < 100 and b < 100 for r, g, b in line), line
    assert "warnings" not in report


@pytest.mark.asyncio
async def test_fade_moves_the_image_toward_the_background(backend, tmp_path):
    scan = _scan(tmp_path / "scan.png")
    region = [-5, -5, 45, 25]
    plain = await backend.image_attach(scan, (0.0, 0.0), 1.0, 0.0, fade=0, plot=True)
    shot_plain = await backend.view_render(region=region, max_px=500)
    _, _, plot_plain = await _plot(backend, tmp_path, "plain", background="white")
    backend._doc.entitydb[plain["handle"]].dxf.fade = 80
    shot_faded = await backend.view_render(region=region, max_px=500)
    _, _, plot_faded = await _plot(backend, tmp_path, "faded", background="white")

    background = _pixel(shot_plain, 44, 24)
    red, faded = _pixel(shot_plain, 5, 15), _pixel(shot_faded, 5, 15)
    assert _red(red)
    gap = [abs(a - b) for a, b in zip(red, background, strict=True)]
    gap_faded = [abs(a - b) for a, b in zip(faded, background, strict=True)]
    assert sum(gap_faded) < sum(gap) / 2, (red, faded, background)
    # On white paper fading lightens: the darkest pixel of the plot gets lighter.
    darkest = [min(sum(rgb) for _x, _y, rgb in _pixels(p)) for p in (plot_plain, plot_faded)]
    assert darkest[1] > darkest[0] + 150, darkest


@pytest.mark.asyncio
async def test_underlay_is_on_screen_and_not_plotted_a_logo_in_both(backend, tmp_path):
    underlay = await backend.image_attach(_scan(tmp_path / "scan.png"), (0.0, 0.0), 1.0, 0.0)
    assert (underlay["layer"], underlay["plot"], underlay["fade"]) == ("UNDERLAY", False, 50)
    assert backend._doc.layers.get("UNDERLAY").dxf.plot == 0
    shot = await backend.view_render(region=[-5, -5, 45, 25], max_px=500)
    assert _red(_pixel(shot, 5, 15))
    _, embedded, plot = await _plot(backend, tmp_path, "underlay")
    assert not embedded and not _any(plot, _red) and not _any(plot, _blue)

    logo = await backend.image_attach(str(tmp_path / "scan.png"), (100.0, 0.0), 1.0, 0.0, plot=True)
    assert (logo["layer"], logo["plot"], logo["fade"]) == ("IMAGE", True, 0)
    assert backend._doc.layers.get("IMAGE").dxf.plot == 1
    shot = await backend.view_render(region=[-5, -5, 145, 25], max_px=600)
    assert _red(_pixel(shot, 5, 15)) and _red(_pixel(shot, 105, 15))
    _, embedded, plot = await _plot(backend, tmp_path, "logo", background="white")
    assert embedded and _any(plot, _red) and _any(plot, _blue)
    # Only the logo plotted: its red block is left of its blue, nothing further left.
    reds = [x for x, _y, rgb in _pixels(plot) if _red(rgb)]
    blues = [x for x, _y, rgb in _pixels(plot) if _blue(rgb)]
    assert max(reds) < max(blues) and min(reds) <= min(blues)


@pytest.mark.asyncio
@pytest.mark.parametrize("hide", ["off", "freeze"])
async def test_an_off_or_frozen_layer_is_in_neither(backend, tmp_path, hide):
    await backend.image_attach(_scan(tmp_path / "scan.png"), (0.0, 0.0), 1.0, 0.0, plot=True)
    getattr(backend._doc.layers.get("IMAGE"), hide)()
    shot = await backend.view_render(region=[-5, -5, 45, 25], max_px=500)
    assert not _red(_pixel(shot, 5, 15)) and not _blue(_pixel(shot, 30, 5))
    _, embedded, plot = await _plot(backend, tmp_path, hide, background="white")
    assert not embedded and not _any(plot, _red)


@pytest.mark.asyncio
async def test_a_missing_or_refused_file_is_a_frame_and_a_warning(backend, tmp_path, monkeypatch):
    scan = tmp_path / "gone.png"
    await backend.image_attach(_scan(scan), (0.0, 0.0), 1.0, 0.0, plot=True)
    scan.unlink()
    region = [-5, -5, 45, 25]
    report = await backend.view_render(region=region, max_px=500)
    assert "gone.png was not found" in report["warnings"][0]
    background = _pixel(report, 44, 24)
    assert _pixel(report, 20, 0) != background  # the frame's bottom edge
    assert not _red(_pixel(report, 5, 15))
    pdf, embedded, _ = await _plot(backend, tmp_path, "missing")
    assert "gone.png was not found" in pdf["warnings"][0] and not embedded

    # A path the policy refuses is never opened: drawn as missing, said so.
    _scan(scan)
    import security
    from engineering.errors import PathRefusedError

    def refuse(path, **_kwargs):
        raise PathRefusedError(f"refused: {path}")

    monkeypatch.setattr(security, "validate_path", refuse)
    report = await backend.view_render(region=region, max_px=500)
    assert "refused by the path policy" in report["warnings"][0]
    assert not _red(_pixel(report, 5, 15))


@pytest.mark.asyncio
async def test_a_plotted_image_shows_through_a_paper_viewport(backend, tmp_path):
    await backend.image_attach(_scan(tmp_path / "scan.png"), (0.0, 0.0), 1.0, 0.0, plot=True)
    await backend.layout_create("A3")
    backend._doc.layouts.get("A3").page_setup(size=(420, 297), margins=(0, 0, 0, 0), units="mm")
    await backend.viewport_create("A3", 210, 148.5, 100, 60, 25, 10, 1.0)
    _, embedded, sheet = await _plot(backend, tmp_path, "sheet", layout="A3")
    assert embedded

    def at(x, y):  # model point -> sheet pixel, through the viewport (1:1)
        px, py = 210 + x - 25, 148.5 + y - 10
        return sheet.getpixel((int(px / 420 * sheet.width), int((1 - py / 297) * sheet.height)))

    assert _red(at(5, 15)) and _blue(at(35, 5)) and _blue(at(5, 5))


@pytest.mark.asyncio
async def test_bad_fade_or_layer_is_refused_before_writing(backend, tmp_path):
    scan = _scan(tmp_path / "scan.png")
    before = len(await backend.entity_list())
    for kwargs in ({"fade": 95}, {"fade": -1}, {"layer": "a<b"}):
        with pytest.raises(ValueError):
            await backend.image_attach(scan, (0.0, 0.0), 1.0, 0.0, **kwargs)
    assert len(await backend.entity_list()) == before
    assert "UNDERLAY" not in backend._doc.layers
