"""pdf_read (B6): a PDF page's vectors in page mm (origin top-left, v down) or WCS."""

from __future__ import annotations

import math

import pytest

from engineering import pdfread
from engineering.pdfread import read_pdf

pytest.importorskip("pypdf", reason="pdf_read needs the [pdfread] extra")


def _pdf(content: bytes, pages: int = 1, rotate: int = 0) -> bytes:
    """A minimal hand-written PDF: A4 pages (595 x 842 pt), Helvetica, one stream."""
    font, stream = 3 + pages, 4 + pages
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [%s] /Count %d >>"
        % (b" ".join(b"%d 0 R" % (3 + i) for i in range(pages)), pages),
    ]
    for _ in range(pages):
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Rotate %d "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (rotate, font, stream)
        )
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objs.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content))
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (n, obj)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


MM = 72.0 / 25.4
# A 50 x 20 mm rectangle whose lower-left corner is 30 mm from the left and
# 40 mm from the bottom, written as plotters do (m l l l h), and a text 100 mm
# from the left whose baseline is 200 mm above the bottom.
PAGE = b"0.5 w %.4f %.4f m %.4f %.4f l %.4f %.4f l %.4f %.4f l h S " % (
    30 * MM,
    40 * MM,
    80 * MM,
    40 * MM,
    80 * MM,
    60 * MM,
    30 * MM,
    60 * MM,
) + b"BT /F1 10 Tf %.4f %.4f Td (KITCHEN) Tj ET" % (100 * MM, 200 * MM)
H = 842 / MM  # page height in mm


def test_text_and_rect_land_on_wcs_through_the_calibration(tmp_path):
    path = tmp_path / "plan.pdf"
    path.write_bytes(_pdf(PAGE))
    # 1:50 (1 page mm = 50 WCS mm), page top-left on WCS (1000, 9000), three pairs
    align = [[u, v, 1000 + 50 * u, 9000 - 50 * v] for u, v in ((0, 0), (200, 0), (0, 280))]
    result = read_pdf(path, align=align)
    assert result["units"] == "wcs" and result["fit"]["controlled"] is True
    assert result["fit"]["rms"] < 1e-6 and math.isclose(result["fit"]["mm_to_units"], 50)
    (text,) = [o for o in result["objects"] if o["kind"] == "text"]
    assert text["text"] == "KITCHEN"
    assert math.isclose(text["u"], 100, abs_tol=0.01)
    assert math.isclose(text["v"], H - 200, abs_tol=0.01)
    assert math.isclose(text["x"], 1000 + 50 * 100, abs_tol=0.5)
    assert math.isclose(text["y"], 9000 - 50 * (H - 200), abs_tol=0.5)
    assert math.isclose(text["height"], 50 * 10 / MM, abs_tol=0.5)  # 10 pt at 1:50
    (rect,) = [o for o in result["objects"] if o["kind"] == "rect"]
    xs = sorted({round(p[0]) for p in rect["corners"]})
    ys = sorted({round(p[1]) for p in rect["corners"]})
    assert xs == [round(1000 + 50 * 30), round(1000 + 50 * 80)]
    assert ys == [round(9000 - 50 * (H - 40)), round(9000 - 50 * (H - 60))]
    assert not [o for o in result["objects"] if o["kind"] == "line"]  # a rect, not a line too


def test_region_kinds_limit_and_rotation(tmp_path):
    path = tmp_path / "plan.pdf"
    path.write_bytes(_pdf(PAGE))
    only_text = read_pdf(path, kinds=["text"])
    assert only_text["counts"] == {"text": 1}
    # a page-mm window around the rectangle only
    window = read_pdf(path, region=[20, H - 70, 90, H - 30])
    assert [o["kind"] for o in window["objects"]] == ["rect"]
    cut = read_pdf(path, limit=1)
    assert len(cut["objects"]) == 1 and cut["truncated"]["reason"] == "limit"
    rotated = tmp_path / "rotated.pdf"
    rotated.write_bytes(_pdf(PAGE, rotate=90))
    turned = read_pdf(rotated, kinds=["text"])
    assert turned["page_mm"][0] > turned["page_mm"][1]  # landscape as displayed
    (text,) = turned["objects"]
    assert math.isclose(text["u"], 200, abs_tol=0.01) and math.isclose(text["v"], 100, abs_tol=0.01)


def test_limits_are_refused_by_name(tmp_path, monkeypatch):
    path = tmp_path / "three.pdf"
    path.write_bytes(_pdf(PAGE, pages=3))
    with pytest.raises(ValueError, match="page: the PDF has 3"):
        read_pdf(path, page=4)
    monkeypatch.setattr(pdfread, "MAX_PAGES", 2)
    with pytest.raises(ValueError, match="MAX_PAGES=2"):
        read_pdf(path)
    monkeypatch.setattr(pdfread, "MAX_BYTES", 100)
    with pytest.raises(ValueError, match="MAX_BYTES=100"):
        read_pdf(path)
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"<html>not a pdf</html>")
    monkeypatch.setattr(pdfread, "MAX_BYTES", 1 << 20)
    with pytest.raises(ValueError, match="no %PDF- header"):
        read_pdf(fake)
    with pytest.raises(ValueError, match="limit: 1-"):
        read_pdf(path, limit=pdfread.MAX_OBJECTS + 1)
    monkeypatch.setattr(pdfread, "MAX_SECONDS", -1.0)
    many = tmp_path / "many.pdf"
    many.write_bytes(_pdf(b" ".join([b"0 0 m 10 10 l S"] * 600)))
    assert read_pdf(many)["truncated"]["reason"] == "max_seconds"


def test_without_pypdf_the_capability_refuses(tmp_path, monkeypatch):
    from backends.capability import UnsupportedCapabilityError

    path = tmp_path / "plan.pdf"
    path.write_bytes(_pdf(PAGE))
    monkeypatch.setattr(pdfread, "available", lambda: False)
    with pytest.raises(UnsupportedCapabilityError) as refused:
        read_pdf(path)
    assert refused.value.capability == "pdf_read" and "pdfread" in str(refused.value)
