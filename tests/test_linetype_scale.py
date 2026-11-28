"""Round 7: a dashed run plots 3 mm dashes on paper at any sheet scale.

ACAD_ISO02W100 (acadiso.lin, 12 / 3 for a 1.0 mm pen) at the 0.25 pen is a
3 mm dash with a 0.75 mm gap. The sheet scale sets CANNOSCALE 1:N with
MSLTSCALE 1, so model space scales the pattern by N and a PSLTSCALE 1
viewport sizes it in paper units: model preview, layout preview and PDF agree,
and LTSCALE stays 1.
"""

from __future__ import annotations

import io
import re
import zlib

import pytest

from engineering.arch.services import insert_points

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("matplotlib")

pytestmark = pytest.mark.asyncio

PX_PER_MM = 20  # 100 mm of paper rendered 2000 px wide


def _dashes_mm(png: bytes) -> list[float]:
    """Ink runs (paper mm) along the most inked row, the cut ends dropped."""
    grey = np.asarray(Image.open(io.BytesIO(png)).convert("L")).astype(int)
    ink = np.abs(grey - int(np.median(grey))) > 64
    row = ink[int(ink.sum(axis=1).argmax())]
    runs, n = [], 0
    for v in row:
        if v:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    return [r / PX_PER_MM for r in runs[1:-1]]


def _pdf_dashes_mm(path: str) -> list[float]:
    """Horizontal segments the PDF strokes, in mm (72 pt per inch)."""
    data = open(path, "rb").read()
    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            text = zlib.decompress(m.group(1)).decode("latin1")
        except zlib.error:
            continue
        for x1, y1, x2, y2 in re.findall(
            r"([\d.\-]+) ([\d.\-]+) m\s+([\d.\-]+) ([\d.\-]+) l", text
        ):
            if abs(float(y1) - float(y2)) < 1e-3:
                out.append(abs(float(x2) - float(x1)) * 25.4 / 72)
    return out


@pytest.mark.parametrize("n", [20, 50, 100])
async def test_a_dashed_run_is_3_mm_on_paper_in_model_layout_and_pdf(backend, tmp_path, n):
    from backends.region_render import render_layout

    await backend.drawing_new()
    set_ = await backend.sheet_linetype_scale(n)
    assert set_ == {"annotation_scale": f"1:{n}", "set": True}
    length = 100.0 * n  # 100 mm of paper
    await insert_points(
        backend, [{"kind": "track_pendant", "path": [[0, 0], [length, 0]]}], scale=n
    )
    doc = backend._doc
    assert doc.header["$LTSCALE"] == 1.0
    model = render_layout(
        doc, doc.modelspace(), region=(0, -2 * n, length, 2 * n), max_px=2000, background="white"
    )
    await backend.layout_create("S")
    await backend.viewport_create("S", 150, 100, 120, 20, length / 2, 0, scale=1 / n)
    sheet = render_layout(doc, doc.layouts.get("S"), region=(100, 98, 200, 102), max_px=2000)
    pdf = tmp_path / "s.pdf"
    await backend.drawing_export_pdf(str(pdf), layout="S")
    for where, dashes in (
        ("model", _dashes_mm(model["png"])),
        ("layout", _dashes_mm(sheet["png"])),
        ("pdf", [d for d in _pdf_dashes_mm(str(pdf)) if d < 10][1:-1]),
    ):
        assert dashes, where
        assert all(d == pytest.approx(3.0, abs=0.15) for d in dashes), (where, dashes)


async def test_a_named_linetype_answers_its_paper_dash_and_iso_equivalent(backend):
    await backend.drawing_new()
    named = await backend.linetype_load("DASHED")
    assert named["paper_dash_mm"] == 12.7 and named["iso"] == "ACAD_ISO02W100"
    iso = await backend.linetype_load("ACAD_ISO04W100")
    assert "iso" not in iso and iso["source"] == "autocad_fallback"
    assert list(backend._doc.linetypes.get("ACAD_ISO04W100").simplified_line_pattern()) == [
        24.0, 3.0, 0.0, 3.0
    ]  # fmt: skip
