"""Page setup; the PDF's own /MediaBox is the evidence.

ezdxf: apply a paper, export the layout, parse the PDF back and compare to the
paper size.
"""

from __future__ import annotations

import re
import zlib

import pytest

from engineering.standards.papers import (
    CTB_CATALOG,
    DXF_STANDARD_SCALE_TYPE,
    DXF_STD_SCALES,
    PAPER_SIZES,
    SCALES,
    canonical_media_name,
    dxf_scale_label,
    paper_from_size,
    paper_size_mm,
    paper_units_name,
    parse_canonical_media,
    resolve_page_setup,
    scale_label,
    scale_ratio,
    turned_sheet,
)
from engineering.standards.pdfinfo import read_mediabox

SHEET = "Sheet"


# ── pure data ───────────────────────────────────────────────────────────────


def test_paper_sizes_are_iso_216_and_ansi_y14_1():
    assert PAPER_SIZES == {
        "ISO_A0": (841, 1189),
        "ISO_A1": (594, 841),
        "ISO_A2": (420, 594),
        "ISO_A3": (297, 420),
        "ISO_A4": (210, 297),
        "ANSI_A": (216, 279),
        "ANSI_B": (279, 432),
        "ANSI_C": (432, 559),
        "ANSI_D": (559, 864),
        "ANSI_E": (864, 1118),
    }
    assert CTB_CATALOG[0] == "monochrome.ctb" and "acad.ctb" in CTB_CATALOG
    assert SCALES["fit"] is None and SCALES["1:50"] == 0.02 and SCALES["10:1"] == 10.0


def test_canonical_media_name_uses_autocad_spelling():
    assert canonical_media_name("ISO_A3", "landscape") == "ISO_A3_(420.00_x_297.00_MM)"
    assert canonical_media_name("ISO_A3", "portrait") == "ISO_A3_(297.00_x_420.00_MM)"
    assert canonical_media_name("ANSI_B", "landscape") == "ANSI_B_(17.00_x_11.00_Inches)"


FULL_BLEED_A3 = "ISO_full_bleed_A3_(420.00_x_297.00_MM)"


def test_a_canonical_media_name_is_accepted_and_parsed_issue_18():
    """Issue #18: the device's full-bleed media could not be named at all
    (``paper: unknown paper``). AutoCAD's own spelling is now taken as is."""
    assert parse_canonical_media(FULL_BLEED_A3) == (420.0, 297.0)
    assert parse_canonical_media("ISO_expand_A4_(210.00_x_297.00_MM)") == (210.0, 297.0)
    assert parse_canonical_media("ANSI_B_(17.00_x_11.00_Inches)") == (431.8, 279.4)
    assert parse_canonical_media("ISO_A3") is None
    assert parse_canonical_media("ISO_A3_(420.00 x 297.00 MM)") is None, "not AutoCAD's spelling"
    assert canonical_media_name(FULL_BLEED_A3) == FULL_BLEED_A3
    assert canonical_media_name(FULL_BLEED_A3, None) == FULL_BLEED_A3
    assert paper_size_mm(FULL_BLEED_A3, None) == (420.0, 297.0)

    setup = resolve_page_setup(FULL_BLEED_A3)
    assert setup["canonical_media_name"] == FULL_BLEED_A3
    assert setup["size_mm"] == [420.0, 297.0]
    assert setup["orientation"] == "landscape", "derived from the numbers"
    assert setup["paper"] == "ISO_A3" and setup["bleed"] == "none"

    portrait = resolve_page_setup("ISO_expand_A4_(210.00_x_297.00_MM)")
    assert portrait["orientation"] == "portrait" and portrait["size_mm"] == [210.0, 297.0]
    assert resolve_page_setup(FULL_BLEED_A3, "landscape")["orientation"] == "landscape"


def test_a_contradicting_orientation_is_refused_not_flipped():
    with pytest.raises(ValueError, match=r"orientation: 'portrait' contradicts paper"):
        resolve_page_setup(FULL_BLEED_A3, "portrait")
    with pytest.raises(ValueError, match="orientation"):
        paper_size_mm(FULL_BLEED_A3, "portrait")
    with pytest.raises(ValueError, match="orientation"):
        canonical_media_name(FULL_BLEED_A3, "portrait")


def test_bleed_selects_the_full_bleed_and_expand_media():
    assert canonical_media_name("ISO_A3", "landscape", "full_bleed") == FULL_BLEED_A3
    assert canonical_media_name("A3", "portrait", "full_bleed") == (
        "ISO_full_bleed_A3_(297.00_x_420.00_MM)"
    )
    assert canonical_media_name("ISO_A4", "portrait", "expand") == (
        "ISO_expand_A4_(210.00_x_297.00_MM)"
    )
    assert canonical_media_name("ANSI_B", "landscape", "full_bleed") == (
        "ANSI_full_bleed_B_(17.00_x_11.00_Inches)"
    )
    assert canonical_media_name("ISO_A3", "landscape", "none") == "ISO_A3_(420.00_x_297.00_MM)"
    setup = resolve_page_setup("ISO_A3", bleed="full_bleed")
    assert setup["canonical_media_name"] == FULL_BLEED_A3 and setup["bleed"] == "full_bleed"
    assert setup["size_mm"] == [420.0, 297.0] and setup["paper"] == "ISO_A3"
    with pytest.raises(ValueError, match="bleed: must be one of"):
        resolve_page_setup("ISO_A3", bleed="borderless")


def test_bleed_beside_a_canonical_media_name_is_refused():
    with pytest.raises(ValueError, match="already names the media"):
        resolve_page_setup(FULL_BLEED_A3, bleed="full_bleed")
    with pytest.raises(ValueError, match="already names the media"):
        resolve_page_setup("ISO_A3_(420.00_x_297.00_MM)", bleed="expand")


def test_paper_from_size_matches_either_orientation_within_half_a_millimetre():
    assert paper_from_size(420, 297) == "ISO_A3"
    assert paper_from_size(297, 420) == "ISO_A3"
    assert paper_from_size(431.8, 279.4) == "ANSI_B"
    assert paper_from_size(400, 300) is None


def test_scale_ratio_and_label_round_trip():
    assert scale_ratio("1:50") == (1.0, 50.0)
    assert scale_ratio("2:1") == (2.0, 1.0)
    assert scale_label(1.0, 50.0) == "1:50"
    assert scale_label(1.0, 3.0) == "1:3"
    with pytest.raises(ValueError, match="scale"):
        scale_ratio("fit")


def test_resolve_page_setup_defaults():
    setup = resolve_page_setup("ISO_A3")
    assert setup["paper"] == "ISO_A3"
    assert setup["orientation"] == "landscape"
    assert setup["size_mm"] == [420.0, 297.0]
    assert setup["canonical_media_name"] == "ISO_A3_(420.00_x_297.00_MM)"
    assert setup["plot_style"] == "monochrome.ctb" and setup["plot_style_known"] is True
    assert setup["scale"] == "fit"
    assert setup["dxf_standard_scale_type"] == 0
    assert setup["scale_ratio"] == [1.0, 1.0]
    assert setup["plot_area"] == "layout" and setup["plot_type"] == 5
    assert setup["device"] == "DWG To PDF.pc3"
    assert setup["margins_mm"] is None
    assert setup["center"] is None, "a layout plot has no centring (AutoCAD greys it out)"


def test_resolve_page_setup_center_applies_to_an_extents_plot_only():
    """ActiveX Reference, CenterPlot: 'This property cannot be set to True on a
    layout object whose PlotType property is set to acLayout.' The resolver
    carries the rule so both engines write the same thing: a bool for an
    extents plot, ``None`` (not applicable, never written) for a layout plot."""
    assert resolve_page_setup("ISO_A3", plot_area="extents")["center"] is True
    assert resolve_page_setup("ISO_A3", plot_area="extents", center=False)["center"] is False
    assert resolve_page_setup("ISO_A3", plot_area="layout", center=True)["center"] is None
    assert resolve_page_setup("ISO_A3", plot_area="layout", center=False)["center"] is None
    with pytest.raises(ValueError, match="center"):
        resolve_page_setup("ISO_A3", plot_area="layout", center="yes")


def test_resolve_page_setup_accepts_lowercase_and_short_paper_names():
    assert resolve_page_setup("iso_a4", "portrait")["size_mm"] == [210.0, 297.0]
    assert resolve_page_setup("A3")["paper"] == "ISO_A3"
    assert resolve_page_setup("ansi_b")["canonical_media_name"] == "ANSI_B_(17.00_x_11.00_Inches)"


def test_resolve_page_setup_scale_codes():
    one_to_five = resolve_page_setup("ISO_A3", scale="1:5")
    # Code 19 is 1:5 in the file enum, but the typelib's ac1_5 = 19 is refused
    # by AutoCAD 2026 ("Invalid input", measured live), so both engines store
    # 1:5 the same way: the custom route, 142/143 = 1/5.
    assert one_to_five["dxf_standard_scale_type"] is None
    assert one_to_five["scale_ratio"] == [1.0, 5.0]
    five_to_one = resolve_page_setup("ISO_A3", scale="5:1")
    assert five_to_one["dxf_standard_scale_type"] is None
    assert five_to_one["scale_ratio"] == [5.0, 1.0]
    # One numbering: AutoCAD 2026 saves StandardScale = 26 (ac1_50) as
    # group 75 = 26 (measured, COM SaveAs DXF) — not ezdxf's 25.
    assert resolve_page_setup("ISO_A3", scale="1:50")["dxf_standard_scale_type"] == 26


def test_dxf_standard_scale_codes_are_the_enum_autocad_writes():
    """Group 75 is the ObjectARX ``StdScaleType`` enum, the same numbering as
    ActiveX ``AcPlotScale`` — measured on AutoCAD 2026 (``StandardScale = n``
    then ``SaveAs(..., 61)``): 1:2→17, 1:10→21, 1:50→26, 2:1→28, 10:1→31.
    ezdxf's ``STD_SCALES`` has no 19 = 1:5 and is one lower above 18; read
    through it, an AutoCAD 1:50 sheet is "1:100" and a 1:10 sheet plots 1:16.
    """
    measured = {"fit": 0, "1:1": 16, "1:2": 17, "1:10": 21, "1:50": 26, "2:1": 28, "10:1": 31}
    for label, code in measured.items():
        assert DXF_STANDARD_SCALE_TYPE[label] == code, label
    # Every code's authored ratio is the catalogue factor.
    for label, code in DXF_STANDARD_SCALE_TYPE.items():
        if code:
            numerator, denominator = DXF_STD_SCALES[code]
            assert numerator / denominator == pytest.approx(SCALES[label]), label
    # The full file enum, including what the writer never emits.
    assert DXF_STD_SCALES[19] == (1.0, 5.0)
    assert DXF_STD_SCALES[20] == (1.0, 8.0)
    assert DXF_STD_SCALES[22] == (1.0, 16.0)
    assert DXF_STD_SCALES[25] == (1.0, 40.0)
    assert DXF_STD_SCALES[33] == (1000.0, 1.0)
    assert set(DXF_STD_SCALES) == set(range(1, 34))
    # Read-back labels come from the code, never from a stale 142/143.
    assert dxf_scale_label(0, 3.0, 7.0) == "fit"
    assert dxf_scale_label(26, 1.0, 100.0) == "1:50"
    assert dxf_scale_label(21, 1.0, 1.0) == "1:10"
    assert dxf_scale_label(19, 1.0, 1.0) == "1:5"
    assert dxf_scale_label(22, 1.0, 1.0) == "1:16"
    assert dxf_scale_label(28, 1.0, 0.5) == "2:1"
    assert dxf_scale_label(8, 1.0, 1.0) == "1/4\"=1'"
    assert dxf_scale_label(99, 1.0, 3.0) == "1:3"  # unknown code: the stored ratio


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"paper": "ISO_A9"}, "paper"),
        ({"paper": "ISO_A3", "orientation": "sideways"}, "orientation"),
        ({"paper": "ISO_A3", "scale": "1:3"}, "scale"),
        ({"paper": "ISO_A3", "plot_area": "window"}, "plot_area"),
        ({"paper": "ISO_A3", "device": ""}, "device"),
        ({"paper": "ISO_A3", "plot_style": ""}, "plot_style"),
        ({"paper": "ISO_A3", "margins_mm": [1, 2, 3]}, "margins_mm"),
        ({"paper": "ISO_A3", "margins_mm": [-1, 0, 0, 0]}, "margins_mm"),
        ({"paper": "ISO_A3", "margins_mm": [150, 150, 0, 0]}, "margins_mm"),
        ({"paper": "ISO_A3", "center": "yes"}, "center"),
    ],
)
def test_resolve_page_setup_refuses_naming_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        resolve_page_setup(**kwargs)


def test_unknown_ctb_is_allowed_but_flagged():
    setup = resolve_page_setup("ISO_A3", plot_style="company.ctb")
    assert setup["plot_style"] == "company.ctb" and setup["plot_style_known"] is False


def test_resolve_page_setup_paper_units_default_mm_and_keep():
    assert resolve_page_setup("ISO_A3")["paper_units"] == "mm"
    assert resolve_page_setup("ANSI_B", paper_units="inches")["paper_units"] == "inches"
    assert resolve_page_setup("ANSI_B", paper_units="inch")["paper_units"] == "inches"
    assert resolve_page_setup("ANSI_B", paper_units=None)["paper_units"] is None, "keep"
    with pytest.raises(ValueError, match="paper_units"):
        resolve_page_setup("ISO_A3", paper_units="cm")


def test_paper_units_name_covers_the_three_dxf_and_activex_codes():
    assert paper_units_name(0) == "inches" and paper_units_name(1) == "mm"
    assert paper_units_name(2) == "pixels"
    assert paper_units_name(None) is None and paper_units_name(7) == "code:7"


@pytest.mark.parametrize(
    "rotation, size, margins",
    [
        # stored media 210 x 297 with (top, bottom, left, right) = (1, 3, 4, 2),
        # i.e. ezdxf's clockwise (top, right, bottom, left) = (1, 2, 3, 4).
        (0, [210.0, 297.0], [1.0, 3.0, 4.0, 2.0]),
        (1, [297.0, 210.0], [2.0, 4.0, 1.0, 3.0]),
        (2, [210.0, 297.0], [3.0, 1.0, 2.0, 4.0]),
        (3, [297.0, 210.0], [4.0, 2.0, 3.0, 1.0]),
    ],
)
def test_turned_sheet_maps_size_and_margins_like_ezdxf_page_from_dxf_layout(
    rotation, size, margins
):
    """The one mapping ``page_setup_list``, ``_paper_frame`` and the COM row share.

    Cross-checked against ezdxf's ``Page.from_dxf_layout`` (the renderer's own
    reading of PLOTSETTINGS 40-43 under ``plot_rotation``), so the row a
    client reads is the sheet the same engine plots.
    """
    assert turned_sheet(210, 297, [1, 3, 4, 2], rotation) == (size, margins)


def test_turned_sheet_agrees_with_ezdxf_page_from_dxf_layout():
    ezdxf = pytest.importorskip("ezdxf")
    from ezdxf.addons.drawing.layout import Page

    doc = ezdxf.new()
    lay = doc.layouts.new("Turn")
    for rotation in (0, 1, 2, 3):
        lay.page_setup(size=(210, 297), margins=(1, 2, 3, 4), units="mm", rotation=rotation)
        page = Page.from_dxf_layout(lay.dxf_layout)
        size, (top, bottom, left, right) = turned_sheet(210, 297, [1, 3, 4, 2], rotation)
        assert size == [page.width, page.height], rotation
        assert (page.margins.top, page.margins.bottom) == (top, bottom), rotation
        assert (page.margins.left, page.margins.right) == (left, right), rotation


# ── the PDF reader ──────────────────────────────────────────────────────────


def test_read_mediabox_refuses_a_non_pdf(tmp_path):
    bogus = tmp_path / "x.pdf"
    bogus.write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="PDF"):
        read_mediabox(str(bogus))


def test_read_mediabox_finds_a_box_inside_a_flate_stream(tmp_path):
    body = zlib.compress(b"<< /Type /Page /MediaBox [0 0 595.276 841.89] >>")
    pdf = tmp_path / "c.pdf"
    pdf.write_bytes(
        b"%PDF-1.5\n1 0 obj\n<< /Type /ObjStm /Filter /FlateDecode >>\nstream\n"
        + body
        + b"\nendstream\nendobj\n%%EOF"
    )
    width, height = read_mediabox(str(pdf))
    assert abs(width - 210.0) < 0.05 and abs(height - 297.0) < 0.05


def test_read_mediabox_reports_a_missing_box(tmp_path):
    pdf = tmp_path / "n.pdf"
    pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF")
    with pytest.raises(ValueError, match="MediaBox"):
        read_mediabox(str(pdf))


# ── headless engine ─────────────────────────────────────────────────────────


async def _sheet(backend, name: str = SHEET) -> str:
    result = await backend.layout_create(name)
    assert result["ok"], result
    return name


async def _mediabox_after_export(backend, tmp_path, layout: str):
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    pdf = tmp_path / f"{layout}.pdf"
    result = await backend.drawing_export_pdf(str(pdf), layout=layout)
    assert result["ok"] is True, result
    return read_mediabox(str(pdf))


@pytest.mark.asyncio
async def test_apply_iso_a3_landscape_and_the_pdf_mediabox_agrees(backend, tmp_path):
    await _sheet(backend)
    result = await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3", "landscape"))
    assert result["ok"] is True and result["layout"] == SHEET
    assert result["applied"]["canonical_media_name"] == "ISO_A3_(420.00_x_297.00_MM)"
    assert result["plot_style_known"] is True
    assert result["viewports_kept"] is True

    rows = await backend.page_setup_list(SHEET)
    assert len(rows) == 1
    row = rows[0]
    assert row["layout"] == SHEET
    assert row["paper"] == "ISO_A3"
    assert row["canonical_media_name"] == "ISO_A3_(420.00_x_297.00_MM)"
    assert row["size_mm"] == [420.0, 297.0]
    assert row["orientation"] == "landscape"
    assert row["plot_style"] == "monochrome.ctb"
    assert row["scale"] == "fit"
    assert row["plot_area"] == "layout"
    assert row["device"] == "DWG To PDF.pc3"
    assert row["center"] is None, "layout plot: centring is not applicable"
    assert len(row["margins_mm"]) == 4

    width, height = await _mediabox_after_export(backend, tmp_path, SHEET)
    assert abs(width - 420.0) <= 0.5 and abs(height - 297.0) <= 0.5


def _pdf_strokes(path) -> tuple[list[float], set[str]]:
    """Stroke widths (pt) and stroke colours (``G`` / ``RG`` operands) of a PDF."""
    widths, colours = [], set()
    for stream in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", path.read_bytes(), re.S):
        try:
            text = zlib.decompress(stream.group(1)).decode("latin-1")
        except zlib.error:
            continue
        widths += [float(w) for w in re.findall(r"([\d.]+)\s+w\b", text)]
        colours |= set(re.findall(r"((?:[\d.]+\s+){3}RG|[\d.]+\s+G)\b", text))
    return widths, {" ".join(c.split()) for c in colours}


@pytest.mark.asyncio
@pytest.mark.parametrize(("ctb", "expect_colour"), [("monochrome.ctb", False), ("acad.ctb", True)])
async def test_a_sheet_plots_its_ctb_and_its_lineweights(backend, tmp_path, ctb, expect_colour):
    """The plot style and the ISO 128 hierarchy reach the PDF.

    Before: ezdxf found no ``monochrome.ctb`` file and fell back to object
    colours, and a lineweight in mm was stroked as points under a one-pixel
    floor, so 0.25, 0.35 and 0.50 mm were all 0.72 pt.
    """
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    await _sheet(backend)
    await backend.page_setup_apply(
        SHEET, resolve_page_setup("ISO_A3", "landscape", plot_style=ctb, scale="1:1")
    )
    lay = backend._doc.layouts.get(SHEET)
    for y, (lw, aci) in enumerate([(25, 1), (35, 3), (50, 5)]):
        attribs = {"lineweight": lw, "color": aci}
        lay.add_line((50, 50 + 20 * y), (300, 50 + 20 * y), dxfattribs=attribs)
    lay.add_line((50, 150), (300, 150), dxfattribs={"lineweight": 70}).rgb = (200, 30, 30)

    pdf = tmp_path / "plot.pdf"
    result = await backend.drawing_export_pdf(str(pdf), layout=SHEET)
    ink = "object" if expect_colour else "monochrome"
    assert result["plot_style"] == {"name": ctb, "ink": ink, "exact": True}
    widths, colours = _pdf_strokes(pdf)
    for mm in (0.25, 0.35, 0.50, 0.70):
        assert any(abs(w - mm * 72 / 25.4) < 1e-3 for w in widths), (mm, sorted(set(widths)))
    inked = colours - {"1 G"}  # white: the paper, not ink
    assert (inked != {"0 G"}) is expect_colour, colours


def _grey(value: int) -> str:
    return f"{value / 255:.10f}".rstrip("0") + " G"


@pytest.mark.asyncio
async def test_a_sheet_plots_screening_50_on_bylayer_and_direct_colours(backend, tmp_path):
    """Screening is ink intensity: channel c plots as 255 - (255 - c) * 0.5.

    Before: Screening 50%.ctb plotted full-strength object colours.
    """
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    await _sheet(backend)
    await backend.page_setup_apply(
        SHEET, resolve_page_setup("ISO_A3", "landscape", plot_style="Screening 50%.ctb")
    )
    backend._doc.layers.add("RED", color=1)
    lay = backend._doc.layouts.get(SHEET)
    lay.add_line((50, 50), (300, 50), dxfattribs={"layer": "RED"})  # ByLayer red
    lay.add_line((50, 70), (300, 70), dxfattribs={"color": 8})  # (128, 128, 128)

    pdf = tmp_path / "screen.pdf"
    result = await backend.drawing_export_pdf(str(pdf), layout=SHEET)
    assert result["plot_style"]["exact"] is True
    assert result["plot_style"]["pens"] == [{"aci": "all", "screen": 50, "lineweight_mm": None}]
    _, colours = _pdf_strokes(pdf)
    assert {"1 0.5019607843 0.5019607843 RG", _grey(192)} <= colours, colours
    assert "1 0 0 RG" not in colours and _grey(128) not in colours, colours


@pytest.mark.asyncio
async def test_model_plots_arch_monochrome_grey_pens_thin_by_their_resolved_colour(
    backend, tmp_path
):
    """arch_monochrome.ctb: ACI 8/252/253 at 35 % grey and 0.13 mm, ACI 9/254 at
    35 % and 0.09 mm (below the 0.13 mm vector floor), every other pen black at
    its own weight. The grey pens are layer colours: ezdxf's own ctb lookup reads
    only an entity's direct colour and would miss them."""
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    doc = backend._doc
    doc.layers.add("HATCH-GREY", color=8, lineweight=50)
    doc.layers.add("FURN", color=254, lineweight=35)
    doc.layers.add("WALL", color=1, lineweight=50)
    msp = doc.modelspace()
    for y, layer in enumerate(("HATCH-GREY", "FURN", "WALL")):
        msp.add_line((0, 10 * y), (100, 10 * y), dxfattribs={"layer": layer})

    pdf = tmp_path / "model.pdf"
    result = await backend.drawing_export_pdf(str(pdf), plot_style="arch_monochrome.ctb")
    assert result["plot_style"]["name"] == "arch_monochrome.ctb"
    assert result["plot_style"]["ink"] == "monochrome" and result["background"] == "auto"
    widths, colours = _pdf_strokes(pdf)
    rounded = {round(w * 25.4 / 72, 3) for w in widths}
    assert {0.09, 0.13, 0.5} <= rounded, sorted(rounded)
    assert 0.35 not in rounded, "FURN's 0.35 mm is replaced by its pen's 0.09 mm"
    assert colours - {"1 G"} == {"0 G", _grey(166)}, colours  # 35 % of black, and black

    # the drawing is not touched: the next Model export plots its own (no) table
    plain = await backend.drawing_export_pdf(str(tmp_path / "plain.pdf"))
    assert plain["plot_style"] == {"name": None, "ink": "object", "exact": True}


@pytest.mark.asyncio
async def test_model_background_white_turns_aci7_black(backend, tmp_path):
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    backend._doc.modelspace().add_line((0, 0), (100, 0))  # ACI 7 via layer 0
    dark, white = tmp_path / "dark.pdf", tmp_path / "white.pdf"
    await backend.drawing_export_pdf(str(dark))
    result = await backend.drawing_export_pdf(str(white), background="white")
    assert result["background"] == "white"

    def line_colours(path) -> set[str]:  # the G set beside a nonzero width: a stroke
        text = "".join(
            zlib.decompress(m.group(1)).decode("latin-1")
            for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", path.read_bytes(), re.S)
        )
        return {g for w, g in re.findall(r"([\d.]+) w ([\d.]+ G)", text) if float(w) > 0}

    assert line_colours(dark) == {"1 G"}, "Model's own dark background: ACI 7 is white"
    assert line_colours(white) == {"0 G"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"layout": SHEET, "plot_style": "monochrome.ctb"}, "page_setup_apply"),
        ({"layout": SHEET, "background": "white"}, "page_setup_apply"),
        ({"background": "grey"}, "background"),
        ({"plot_style": "company.stb"}, ".ctb"),
    ],
)
@pytest.mark.asyncio
async def test_export_pdf_refuses_model_options_on_a_sheet_and_bad_values(
    backend, tmp_path, kwargs, match
):
    await _sheet(backend)
    pdf = tmp_path / "x.pdf"
    with pytest.raises(ValueError, match=re.escape(match)):
        await backend.drawing_export_pdf(str(pdf), **kwargs)
    assert not pdf.exists()


@pytest.mark.asyncio
async def test_apply_full_bleed_a3_round_trips_and_the_pdf_mediabox_agrees(backend, tmp_path):
    """Issue #18 on the headless engine: the full-bleed name is written into
    the LAYOUT's plot settings, read back by page_setup_list, carried by the
    DXF on disk, and the exported PDF is 420 x 297 mm."""
    import ezdxf

    await _sheet(backend)
    result = await backend.page_setup_apply(SHEET, resolve_page_setup(FULL_BLEED_A3))
    assert result["ok"] is True
    assert result["changed"]["canonical_media_name"][1] == FULL_BLEED_A3
    row = (await backend.page_setup_list(SHEET))[0]
    assert row["canonical_media_name"] == FULL_BLEED_A3
    assert row["size_mm"] == [420.0, 297.0] and row["orientation"] == "landscape"
    assert row["paper"] == "ISO_A3", "the catalogue size the media measures"

    saved = tmp_path / "bleed.dxf"
    await backend.drawing_save(str(saved))
    plot = ezdxf.readfile(str(saved)).layouts.get(SHEET).dxf_layout.dxf
    assert plot.paper_size == FULL_BLEED_A3
    assert (plot.paper_width, plot.paper_height) == (420.0, 297.0)

    width, height = await _mediabox_after_export(backend, tmp_path, SHEET)
    assert abs(width - 420.0) <= 0.5 and abs(height - 297.0) <= 0.5


@pytest.mark.asyncio
async def test_a_sheet_frame_from_the_origin_fits_a_new_layout_and_a_full_bleed_one(backend):
    """A new layout kept ezdxf's [20, 20, 7.5, 7.5] margins, so a 1:1 layout
    plot started at (-7.5, -20) and a sheet_frame drawn over 0..420 x 0..297
    lost its right 7.5 mm and top 20 mm off the PDF page; full-bleed media
    kept those margins too."""
    await backend.drawing_new()
    default = (await backend.page_setup_list())[0]
    assert default["margins_mm"] == [0.0, 0.0, 0.0, 0.0], default["layout"]

    await _sheet(backend)
    lay = backend._doc.layouts.get(SHEET)
    await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3", scale="1:1"))
    frame = backend._paper_frame(lay)
    assert frame["xlim"] == pytest.approx((0.0, 420.0))
    assert frame["ylim"] == pytest.approx((0.0, 297.0))

    dxf = lay.dxf_layout.dxf
    dxf.top_margin = dxf.bottom_margin = 20.0
    dxf.left_margin = dxf.right_margin = 7.5
    await backend.page_setup_apply(
        SHEET, resolve_page_setup("ISO_A3", scale="1:1", bleed="full_bleed")
    )
    assert (await backend.page_setup_list(SHEET))[0]["margins_mm"] == [0.0, 0.0, 0.0, 0.0]
    assert backend._paper_frame(lay)["xlim"] == pytest.approx((0.0, 420.0))


@pytest.mark.asyncio
async def test_apply_bleed_expand_portrait_a4(backend, tmp_path):
    await _sheet(backend)
    await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A4", "portrait", bleed="expand"))
    row = (await backend.page_setup_list(SHEET))[0]
    assert row["canonical_media_name"] == "ISO_expand_A4_(210.00_x_297.00_MM)"
    assert row["size_mm"] == [210.0, 297.0] and row["orientation"] == "portrait"


@pytest.mark.asyncio
async def test_apply_ansi_b(backend, tmp_path):
    await _sheet(backend)
    await backend.page_setup_apply(SHEET, resolve_page_setup("ANSI_B", "landscape", scale="1:1"))
    row = (await backend.page_setup_list(SHEET))[0]
    assert row["paper"] == "ANSI_B" and row["size_mm"] == [432.0, 279.0]
    assert row["scale"] == "1:1"
    width, height = await _mediabox_after_export(backend, tmp_path, SHEET)
    assert abs(width - 432.0) <= 0.5 and abs(height - 279.0) <= 0.5


@pytest.mark.asyncio
async def test_apply_portrait_a4(backend, tmp_path):
    await _sheet(backend)
    await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A4", "portrait"))
    row = (await backend.page_setup_list(SHEET))[0]
    assert row["orientation"] == "portrait" and row["size_mm"] == [210.0, 297.0]
    width, height = await _mediabox_after_export(backend, tmp_path, SHEET)
    assert abs(width - 210.0) <= 0.5 and abs(height - 297.0) <= 0.5


def _ink_columns(png_path: str) -> tuple[int, int, int]:
    """(first inked column, last inked column, image width) of a rendered PNG."""
    from PIL import Image

    image = Image.open(png_path).convert("L")
    width, height = image.size
    pixels = image.load()
    inked = [x for x in range(width) if any(pixels[x, y] < 128 for y in range(height))]
    assert inked, "the render is blank"
    return inked[0], inked[-1], width


@pytest.mark.asyncio
async def test_rotated_layout_renders_the_whole_sheet(backend, tmp_path):
    """plot_rotation 1 is AutoCAD's landscape-on-portrait-media convention.

    A drawing authored in AutoCAD stores ISO A4 landscape as paper 210 x 297
    with ``plot_rotation`` 1; paper-space X then runs along the 297 mm side.
    The MediaBox alone cannot see a frame that has been clipped inside a
    correctly sized page, so this test looks at the ink: the frame's right
    edge (x = 277, the printable width) must reach the right of the sheet.
    """
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    pytest.importorskip("PIL", reason="pixel check needs Pillow")
    await _sheet(backend, "Rot")
    lay = backend._doc.layouts.get("Rot")
    lay.page_setup(
        size=(210, 297),
        margins=(20, 7.5, 20, 7.5),
        units="mm",
        rotation=1,
        scale=16,
        name="ISO_A4",
        device="DWG To PDF.pc3",
    )
    lay.add_lwpolyline([(0, 0), (277, 0), (277, 190), (0, 190)], close=True)

    row = (await backend.page_setup_list("Rot"))[0]
    assert row["size_mm"] == [297.0, 210.0] and row["orientation"] == "landscape"

    png = tmp_path / "rot.png"
    result = await backend.drawing_export_pdf(str(png), layout="Rot")
    assert result["ok"] is True, result
    assert result["paper_mm"] == [297.0, 210.0]
    assert result["rotation_applied"] is False

    first, last, width = _ink_columns(str(png))
    # x = 277 sits at 277 / 297 of the sheet from the left margin, i.e. well
    # past the 0.707-scaled box the unrotated window produced (col 1314/1753).
    assert last >= int(width * 0.9), (first, last, width)

    pdf = tmp_path / "rot.pdf"
    await backend.drawing_export_pdf(str(pdf), layout="Rot")
    mm_w, mm_h = read_mediabox(str(pdf))
    assert abs(mm_w - 297.0) <= 0.5 and abs(mm_h - 210.0) <= 0.5


async def _export_png(backend, tmp_path, layout: str, stem: str):
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    pytest.importorskip("PIL", reason="pixel check needs Pillow")
    png = tmp_path / f"{stem}.png"
    result = await backend.drawing_export_pdf(str(png), layout=layout)
    assert result["ok"] is True, result
    return result, _ink_columns(str(png))


@pytest.mark.asyncio
async def test_layout_plot_honours_the_plot_scale(backend, tmp_path):
    """A 1:2 sheet holds twice the paper-space units a 1:1 sheet holds.

    The MediaBox is the same 420 x 297 either way, so it cannot see this;
    the ink can. An 800-unit-wide frame at 1:2 spans 400 mm and ends inside
    the sheet; at 1:1 it runs off the right edge. Before the fix the two
    exports were pixel-identical (both clipped at the edge) with ``ok: True``.
    """
    await _sheet(backend)
    # The frame starts at (0, 0): with zero margins its left edge would lie on
    # the page edge, so the sheet keeps a printable border.
    margins = [20.0, 20.0, 7.5, 7.5]
    await backend.page_setup_apply(
        SHEET, resolve_page_setup("ISO_A3", scale="1:2", margins_mm=margins)
    )
    lay = backend._doc.layouts.get(SHEET)
    lay.add_lwpolyline([(0, 0), (800, 0), (800, 500), (0, 500)], close=True)
    _, _, left_mm, _ = (await backend.page_setup_list(SHEET))[0]["margins_mm"]

    frame = backend._paper_frame(lay)
    assert frame["scale"] == "1:2" and frame["scale_applied"] is True
    assert frame["xlim"][1] - frame["xlim"][0] == pytest.approx(840.0)
    assert frame["ylim"][1] - frame["ylim"][0] == pytest.approx(594.0)

    result, (first, last, width) = await _export_png(backend, tmp_path, SHEET, "half")
    assert result["scale"] == "1:2" and result["effective_scale"] == "1:2"
    assert result["scale_applied"] is True
    assert result["plot_area"] == "layout" and result["plot_area_applied"] is True
    expected_last = int(width * (left_mm + 400.0) / 420.0)
    assert abs(last - expected_last) <= 3, (first, last, width, expected_last)

    await backend.page_setup_apply(
        SHEET, resolve_page_setup("ISO_A3", scale="1:1", margins_mm=margins)
    )
    result, (_, last_full, _) = await _export_png(backend, tmp_path, SHEET, "full")
    assert result["scale"] == "1:1"
    assert last_full >= width - 2, "at 1:1 the 800-unit frame runs off the 420 mm sheet"
    assert last < last_full - 20, "1:2 and 1:1 must not render the same"


@pytest.mark.asyncio
async def test_custom_scale_sizes_the_window_too(backend):
    """1:5 takes the custom route (as on the live engine), so it rides on the
    numerator / denominator."""
    await _sheet(backend)
    await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3", scale="1:5"))
    frame = backend._paper_frame(backend._doc.layouts.get(SHEET))
    assert frame["scale"] == "1:5" and frame["scale_applied"] is True
    assert frame["xlim"][1] - frame["xlim"][0] == pytest.approx(2100.0)


@pytest.mark.asyncio
async def test_fit_on_a_layout_plot_is_one_to_one(backend):
    """AutoCAD greys "Fit to paper" out under the layout plot area."""
    await _sheet(backend)
    await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3", scale="fit"))
    frame = backend._paper_frame(backend._doc.layouts.get(SHEET))
    assert frame["scale"] == "fit" and frame["effective_scale"] == "1:1"
    assert frame["scale_applied"] is True
    assert frame["xlim"][1] - frame["xlim"][0] == pytest.approx(420.0)


@pytest.mark.parametrize(
    ("paper", "paper_units", "sheet_mm", "frame_units"),
    [
        pytest.param("ISO_A3", "mm", (420.0, 297.0), (800.0, 500.0), id="mm"),
        pytest.param("ANSI_B", "inches", (432.0, 279.0), (16.0, 10.0), id="inches"),
    ],
)
@pytest.mark.asyncio
async def test_extents_plot_fits_the_printable_area_and_centres(
    backend, tmp_path, paper, paper_units, sheet_mm, frame_units
):
    """extents + fit: the paper-space extents (not the main viewport, which
    ezdxf's bbox would count) scale to the printable area and sit centred.

    The inch sheet is the label's regression: ``effective_scale`` is paper
    units per drawing unit, and 25.4 mm per paper unit *divides* the mm ratio.
    Measured before the fix: the ANSI B fit reported ``607.06:1`` for a fit
    that is ``1:1.06276`` (off by 25.4 squared) while the ink landed right,
    and the mm sheet could not see it because there the unit is 1.
    """
    await _sheet(backend)
    await backend.page_setup_apply(
        SHEET,
        resolve_page_setup(
            paper,
            "landscape",
            scale="fit",
            plot_area="extents",
            center=True,
            paper_units=paper_units,
            # A printable area smaller than the sheet, so the fit is below 1:1.
            margins_mm=[20.0, 20.0, 7.5, 7.5],
        ),
    )
    lay = backend._doc.layouts.get(SHEET)
    sheet_w, sheet_h = sheet_mm
    frame_w, frame_h = frame_units
    lay.add_lwpolyline([(0, 0), (frame_w, 0), (frame_w, frame_h), (0, frame_h)], close=True)
    row = (await backend.page_setup_list(SHEET))[0]
    assert row["paper_units"] == paper_units and row["size_mm"] == list(sheet_mm)
    top, bottom, left_mm, right_mm = row["margins_mm"]
    printable_w, printable_h = sheet_w - left_mm - right_mm, sheet_h - top - bottom
    mm_per_unit = min(printable_w / frame_w, printable_h / frame_h)
    mm_per_paper_unit = 25.4 if paper_units == "inches" else 1.0

    frame = backend._paper_frame(lay)
    assert frame["plot_area"] == "extents" and frame["plot_area_applied"] is True
    assert frame["scale"] == "fit"
    # The axes window is unit-free: the sheet in drawing units.
    assert frame["xlim"][1] - frame["xlim"][0] == pytest.approx(sheet_w / mm_per_unit)

    result, (first, last, width) = await _export_png(backend, tmp_path, SHEET, "extents")
    # Paper units per drawing unit, not millimetres per drawing unit.
    assert result["effective_scale"] == scale_label(1.0, mm_per_paper_unit / mm_per_unit)
    # The frame sits centred in the printable area; the mm case is width-bound
    # (it spans the printable width exactly), the inch case is height-bound.
    frame_mm = frame_w * mm_per_unit
    ink_left_mm = left_mm + (printable_w - frame_mm) / 2.0
    assert abs(first - int(width * ink_left_mm / sheet_w)) <= 3, (first, last, width)
    assert abs(last - int(width * (ink_left_mm + frame_mm) / sheet_w)) <= 3, (first, last, width)


@pytest.mark.asyncio
async def test_extents_plot_at_a_fixed_scale_starts_at_the_plot_origin(backend):
    await _sheet(backend)
    await backend.page_setup_apply(
        SHEET, resolve_page_setup("ISO_A3", scale="1:2", plot_area="extents", center=False)
    )
    lay = backend._doc.layouts.get(SHEET)
    lay.add_lwpolyline([(100, 50), (900, 50), (900, 550), (100, 550)], close=True)
    _, _, left_mm, _ = (await backend.page_setup_list(SHEET))[0]["margins_mm"]
    frame = backend._paper_frame(lay)
    assert frame["effective_scale"] == "1:2" and frame["plot_area_applied"] is True
    # The extents' lower-left corner lands on the printable corner: sheet x0
    # is that corner less the left margin, in paper-space units at 1:2.
    assert frame["xlim"][0] == pytest.approx(100.0 - left_mm * 2.0)
    assert frame["xlim"][1] - frame["xlim"][0] == pytest.approx(840.0)


@pytest.mark.asyncio
async def test_extents_plot_of_an_empty_layout_falls_back_to_the_sheet(backend):
    await _sheet(backend)
    await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3", plot_area="extents"))
    frame = backend._paper_frame(backend._doc.layouts.get(SHEET))
    assert frame["plot_area_applied"] is False
    assert "no entities" in frame["plot_area_note"]
    assert frame["xlim"][1] - frame["xlim"][0] == pytest.approx(420.0)


@pytest.mark.asyncio
async def test_unrenderable_plot_area_is_reported_not_hidden(backend, tmp_path):
    """display / view need a screen; the headless engine plots the sheet and says so."""
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    await _sheet(backend)
    await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3", scale="1:1"))
    lay = backend._doc.layouts.get(SHEET)
    lay.dxf_layout.dxf.plot_type = 0  # display
    lay.add_line((0, 0), (100, 100))
    result = await backend.drawing_export_pdf(str(tmp_path / "display.pdf"), layout=SHEET)
    assert result["ok"] is True
    assert result["plot_area"] == "display" and result["plot_area_applied"] is False
    assert "display" in result["plot_area_note"]
    assert result["scale_applied"] is True


@pytest.mark.asyncio
async def test_apply_materialises_the_default_flags_before_setting_centre(backend):
    """Measured: set_flag_state starts from 0 on a fresh layout, so a naive
    plot_centered(True) writes 4 and drops lineweights/plot-styles/viewports-first."""
    await _sheet(backend)
    extents = resolve_page_setup("ISO_A3", plot_area="extents", center=True)
    result = await backend.page_setup_apply(SHEET, extents)
    flags = int(backend._doc.layouts.get(SHEET).dxf_layout.dxf.plot_layout_flags)
    assert flags & 4, "centre-plot bit"
    assert flags & 32, "plot with plot styles"
    assert flags & 128, "plot entity lineweights"
    assert flags & 512, "draw viewports first"
    assert flags & 16, "use standard scale (fit)"
    assert result["applied"]["center"] is True
    assert (await backend.page_setup_list(SHEET))[0]["center"] is True
    await backend.page_setup_apply(
        SHEET, resolve_page_setup("ISO_A3", plot_area="extents", center=False)
    )
    flags = int(backend._doc.layouts.get(SHEET).dxf_layout.dxf.plot_layout_flags)
    assert not flags & 4 and flags & 32 and flags & 128 and flags & 512
    assert (await backend.page_setup_list(SHEET))[0]["center"] is False


@pytest.mark.asyncio
async def test_layout_plot_never_sets_the_centre_bit_and_reports_it_not_applicable(backend):
    """A layout plot is the whole sheet from its origin: AutoCAD disables
    'Center the plot' (ActiveX refuses CenterPlot=True under acLayout). The
    headless engine mirrors that: bit 4 is not written, and the read-back says
    None rather than a bool AutoCAD would ignore."""
    await _sheet(backend)
    result = await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3", center=True))
    flags = int(backend._doc.layouts.get(SHEET).dxf_layout.dxf.plot_layout_flags)
    assert not flags & 4, "centre bit must not be set for a layout plot"
    assert result["applied"]["center"] is None
    assert "center" not in result["changed"], "None before (fresh layout plot) and None after"
    assert (await backend.page_setup_list(SHEET))[0]["center"] is None
    # Switching an extents+centred sheet to a layout plot clears the stale bit
    # (what AutoCAD stores for a layout plot) and reports the move honestly.
    await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3", plot_area="extents"))
    assert (await backend.page_setup_list(SHEET))[0]["center"] is True
    result = await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3"))
    flags = int(backend._doc.layouts.get(SHEET).dxf_layout.dxf.plot_layout_flags)
    assert not flags & 4
    assert result["changed"]["center"] == [True, None]
    assert result["changed"]["plot_area"] == ["extents", "layout"]


@pytest.mark.asyncio
async def test_custom_scale_clears_the_standard_scale_bit_and_writes_the_ratio(backend):
    await _sheet(backend)
    await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3", scale="1:5"))
    dxf = backend._doc.layouts.get(SHEET).dxf_layout.dxf
    assert not int(dxf.plot_layout_flags) & 16
    assert (float(dxf.scale_numerator), float(dxf.scale_denominator)) == (1.0, 5.0)
    assert float(dxf.unit_factor) == pytest.approx(0.2)
    assert (await backend.page_setup_list(SHEET))[0]["scale"] == "1:5"


@pytest.mark.parametrize(
    "label, code, factor",
    [
        ("fit", 0, 1.0),
        ("1:1", 16, 1.0),
        ("1:2", 17, 0.5),
        ("1:10", 21, 0.1),
        ("1:20", 23, 0.05),
        ("1:50", 26, 0.02),
        ("1:100", 27, 0.01),
        ("2:1", 28, 2.0),
        ("10:1", 31, 10.0),
    ],
)
@pytest.mark.asyncio
async def test_standard_scale_writes_the_code_and_the_factor_autocad_plots_by(
    backend, label, code, factor
):
    """Under USE_STANDARD_SCALE AutoCAD's plot engine scales by group 147
    alone (measured, AutoCAD 2026: a headless 1:10 sheet with code 21 and
    147 = 1.0 plotted a 100-unit line at 99.99 mm — 1:1 — while the row,
    ``applied`` and 142/143 all said 1:10; patching 147 to 0.1 alone gave
    9.99 mm). AutoCAD writes 147 = numerator / denominator, 1.0 for fit; so
    does the headless writer, on every route, with the code AutoCAD uses.
    """
    await _sheet(backend)
    result = await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3", scale=label))
    assert result["ok"] is True and result["applied"]["scale"] == label
    dxf = backend._doc.layouts.get(SHEET).dxf_layout.dxf
    assert int(dxf.plot_layout_flags) & 16
    assert int(dxf.standard_scale_type) == code
    assert float(dxf.unit_factor) == pytest.approx(factor)
    ratio = [1.0, 1.0] if label == "fit" else list(scale_ratio(label))
    assert [float(dxf.scale_numerator), float(dxf.scale_denominator)] == ratio
    assert (await backend.page_setup_list(SHEET))[0]["scale"] == label
    frame = backend._paper_frame(backend._doc.layouts.get(SHEET))
    assert frame["scale"] == label and frame["scale_applied"] is True
    if label != "fit":
        assert frame["xlim"][1] - frame["xlim"][0] == pytest.approx(420.0 / factor)


def _autocad_authored_scale(layout, code, numerator, denominator, factor):
    """Store the plot scale exactly as AutoCAD 2026 saves it (measured via COM
    ``SaveAs(..., 61)``): group 75 from the ObjectARX enum, 142/143 as the
    ratio (AutoCAD normalises 2:1 to 1 / 0.5), 147 as the factor, bit 16 set.
    """
    dxf = layout.dxf_layout.dxf
    if not dxf.hasattr("plot_layout_flags"):
        dxf.plot_layout_flags = dxf.get_default("plot_layout_flags")
    dxf.paper_size = "ISO_A3_(420.00_x_297.00_MM)"
    dxf.paper_width, dxf.paper_height = 420.0, 297.0
    dxf.plot_paper_units = 1
    dxf.plot_rotation = 0
    dxf.standard_scale_type = code
    dxf.scale_numerator, dxf.scale_denominator = numerator, denominator
    dxf.unit_factor = factor
    layout.use_standard_scale(True)
    layout.set_plot_type(5)


@pytest.mark.parametrize(
    "code, numerator, denominator, factor, label, units_across",
    [
        (21, 1.0, 10.0, 0.1, "1:10", 4200.0),
        (23, 1.0, 20.0, 0.05, "1:20", 8400.0),
        (26, 1.0, 50.0, 0.02, "1:50", 21000.0),
        (27, 1.0, 100.0, 0.01, "1:100", 42000.0),
        (28, 1.0, 0.5, 2.0, "2:1", 210.0),
        (31, 1.0, 0.1, 10.0, "10:1", 42.0),
        (19, 1.0, 5.0, 0.2, "1:5", 2100.0),
        (22, 1.0, 16.0, 0.0625, "1:16", 6720.0),
    ],
)
@pytest.mark.asyncio
async def test_list_and_render_read_an_autocad_authored_sheet_at_its_own_scale(
    backend, code, numerator, denominator, factor, label, units_across
):
    """A sheet AutoCAD saved at 1:50 (group 75 = 26) reads back as 1:50 and
    renders 21000 paper-space units across the 420 mm — not "1:100" / 42000
    (what ezdxf's one-off ``STD_SCALES`` made of it: 1:10 rendered at 1:16,
    1:20 at 1:30, 1:50 at 1:100, 1:100 at 2:1, 2:1 at 4:1, 10:1 at 100:1,
    with the row and the plot disagreeing). Codes 19 and 22 are ones this
    server never writes, so this reads the file enum, not the writer's table.
    """
    await _sheet(backend)
    lay = backend._doc.layouts.get(SHEET)
    _autocad_authored_scale(lay, code, numerator, denominator, factor)
    row = (await backend.page_setup_list(SHEET))[0]
    assert row["scale"] == label
    frame = backend._paper_frame(lay)
    assert frame["scale"] == label and frame["effective_scale"] == label
    assert frame["scale_applied"] is True
    assert frame["xlim"][1] - frame["xlim"][0] == pytest.approx(units_across)


@pytest.mark.asyncio
async def test_apply_keeps_the_viewports(backend):
    await _sheet(backend)
    await backend.viewport_create(SHEET, 210, 148.5, 260, 180, 0, 0, scale=1.0)
    before = (await backend.viewport_list(SHEET))["count"]
    await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3"))
    assert (await backend.viewport_list(SHEET))["count"] == before


@pytest.mark.asyncio
async def test_changed_reports_only_what_moved(backend):
    await _sheet(backend)
    first = await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3"))
    assert first["changed"], "a fresh layout differs from an A3 setup somewhere"
    second = await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3"))
    assert second["changed"] == {}
    third = await backend.page_setup_apply(
        SHEET, resolve_page_setup("ISO_A3", plot_style="acad.ctb")
    )
    assert third["changed"] == {"plot_style": ["monochrome.ctb", "acad.ctb"]}


@pytest.mark.asyncio
async def test_unknown_ctb_is_written_and_flagged(backend):
    await _sheet(backend)
    result = await backend.page_setup_apply(
        SHEET, resolve_page_setup("ISO_A3", plot_style="company.ctb")
    )
    assert result["plot_style_known"] is False
    assert (await backend.page_setup_list(SHEET))[0]["plot_style"] == "company.ctb"


@pytest.mark.asyncio
async def test_refusals_happen_before_any_write(backend):
    await _sheet(backend)
    dxf = backend._doc.layouts.get(SHEET).dxf_layout.dxf
    before = dxf.get("paper_size")
    with pytest.raises(ValueError, match="Model"):
        await backend.page_setup_apply("Model", resolve_page_setup("ISO_A3"))
    with pytest.raises(ValueError, match="Nope"):
        await backend.page_setup_apply("Nope", resolve_page_setup("ISO_A3"))
    with pytest.raises(ValueError, match="canonical_media_name"):
        await backend.page_setup_apply(SHEET, {"paper": "ISO_A3"})
    assert dxf.get("paper_size") == before
    with pytest.raises(ValueError, match="Nope"):
        await backend.page_setup_list("Nope")


@pytest.mark.asyncio
async def test_list_covers_every_paper_layout_and_skips_model(backend):
    await _sheet(backend, "A")
    await _sheet(backend, "B")
    names = [row["layout"] for row in await backend.page_setup_list()]
    assert "Model" not in names and {"A", "B"} <= set(names)


def _rotated_a4(backend, name: str = "Rot", rotation: int = 1):
    """AutoCAD's landscape-on-portrait-media convention: media 210 x 297,
    ezdxf clockwise margins (top, right, bottom, left) = (20, 7.5, 20, 7.5),
    turned by ``plot_rotation``."""
    lay = backend._doc.layouts.get(name)
    lay.page_setup(
        size=(210, 297),
        margins=(20, 7.5, 20, 7.5),
        units="mm",
        rotation=rotation,
        scale=16,
        name="ISO_A4",
        device="DWG To PDF.pc3",
    )
    return lay


@pytest.mark.asyncio
async def test_list_turns_the_margins_with_the_sheet(backend):
    """The row used to swap ``size_mm`` for rotation 1/3 but report the
    margins un-turned: [20, 20, 7.5, 7.5] on a 297 x 210 sheet whose true
    margins (ezdxf's Page, and the window ``_paper_frame`` plots) are
    [7.5, 7.5, 20, 20] — a printable area of 282 x 170 from the row against
    257 x 195 on paper."""
    await _sheet(backend, "Rot")
    lay = _rotated_a4(backend)
    row = (await backend.page_setup_list("Rot"))[0]
    assert row["size_mm"] == [297.0, 210.0] and row["orientation"] == "landscape"
    assert row["margins_mm"] == [7.5, 7.5, 20.0, 20.0]
    top, bottom, left, right = row["margins_mm"]
    assert (297 - left - right, 210 - top - bottom) == (257.0, 195.0)

    # The same sheet the renderer plots: at 1:1 the axes window starts one
    # (turned) left margin before x = 0 and one (turned) bottom margin below y = 0.
    frame = backend._paper_frame(lay)
    assert frame["size_mm"] == row["size_mm"]
    assert frame["xlim"][0] == pytest.approx(-left) and frame["ylim"][0] == pytest.approx(-bottom)
    assert frame["xlim"][1] - frame["xlim"][0] == pytest.approx(297.0)
    assert frame["ylim"][1] - frame["ylim"][0] == pytest.approx(210.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("rotation", [1, 2, 3])
async def test_apply_on_a_turned_sheet_keeps_the_effective_margins(backend, rotation):
    """``page_setup_apply`` writes ``plot_rotation`` 0 (orientation rides on
    width/height). With no ``margins_mm`` asked for, the stored margins used
    to stay as they were while the turn that gave them their meaning went
    away, so the effective margins moved from [7.5, 7.5, 20, 20] to
    [20, 20, 7.5, 7.5] (printable 257 x 195 -> 282 x 170) and ``changed``
    said nothing. The stored margins are now re-mapped so the sheet keeps
    the margins it had, and the row before equals the row after."""
    await _sheet(backend, "Rot")
    lay = _rotated_a4(backend, rotation=rotation)
    before = (await backend.page_setup_list("Rot"))[0]
    result = await backend.page_setup_apply(
        "Rot", resolve_page_setup("ISO_A4", before["orientation"], scale="1:1")
    )
    after = (await backend.page_setup_list("Rot"))[0]
    assert int(lay.dxf_layout.dxf.plot_rotation) == 0
    assert after["margins_mm"] == before["margins_mm"]
    assert after["size_mm"] == before["size_mm"]
    assert "margins_mm" not in result["changed"] and "size_mm" not in result["changed"]
    dxf = lay.dxf_layout.dxf
    top, bottom, left, right = after["margins_mm"]
    assert (dxf.top_margin, dxf.bottom_margin, dxf.left_margin, dxf.right_margin) == (
        top,
        bottom,
        left,
        right,
    )


@pytest.mark.asyncio
async def test_apply_with_margins_on_a_turned_sheet_writes_them_as_asked(backend):
    await _sheet(backend, "Rot")
    _rotated_a4(backend)
    result = await backend.page_setup_apply(
        "Rot", resolve_page_setup("ISO_A4", "landscape", margins_mm=[5, 6, 7, 8])
    )
    row = (await backend.page_setup_list("Rot"))[0]
    assert row["margins_mm"] == [5.0, 6.0, 7.0, 8.0]
    assert result["changed"]["margins_mm"] == [[7.5, 7.5, 20.0, 20.0], [5.0, 6.0, 7.0, 8.0]]


def _inch_sheet(backend, name: str = "In"):
    """An inch-authored ANSI B sheet (what ANSI_A-E exist for): paper-space
    units are inches, so at 1:1 a 16-unit frame spans 16 inches of paper."""
    lay = backend._doc.layouts.get(name)
    lay.page_setup(
        size=(17, 11),
        margins=(0.5, 0.5, 0.5, 0.5),
        units="inch",
        scale=16,
        name="ANSI_B",
        device="DWG To PDF.pc3",
    )
    lay.add_lwpolyline([(0, 0), (16, 0), (16, 10), (0, 10)], close=True)
    return lay


@pytest.mark.asyncio
async def test_row_reports_the_paper_units(backend):
    await _sheet(backend, "In")
    lay = _inch_sheet(backend)
    row = (await backend.page_setup_list("In"))[0]
    assert row["paper_units"] == "inches"
    assert row["size_mm"] == [431.8, 279.4] and row["paper"] == "ANSI_B"
    assert row["margins_mm"] == [12.7, 12.7, 12.7, 12.7]
    lay.dxf_layout.dxf.plot_paper_units = 1
    assert (await backend.page_setup_list("In"))[0]["paper_units"] == "mm"


@pytest.mark.asyncio
async def test_apply_names_a_paper_units_flip_and_its_consequence(backend):
    """The default setup plots in millimetres. On an inch-authored sheet that
    is a 25.4x rescale of every paper-space unit: the 16-unit frame used to
    go from 406.4 mm to 16 mm of paper with ``changed`` naming only the
    media and the plot style — ``ok: True``, ``scale_applied: True``."""
    await _sheet(backend, "In")
    lay = _inch_sheet(backend)
    result = await backend.page_setup_apply(
        "In", resolve_page_setup("ANSI_B", "landscape", scale="1:1")
    )
    assert result["changed"]["paper_units"] == ["inches", "mm"]
    assert int(lay.dxf_layout.dxf.plot_paper_units) == 1
    assert result["warnings"], "a units flip is never silent"
    assert any("inches" in w and "mm" in w and "25.4" in w for w in result["warnings"])
    assert result["applied"]["paper_units"] == "mm"
    # Re-applying the same setup moves nothing and warns of nothing.
    again = await backend.page_setup_apply(
        "In", resolve_page_setup("ANSI_B", "landscape", scale="1:1")
    )
    assert again["changed"] == {} and again["warnings"] == []


@pytest.mark.asyncio
async def test_apply_can_keep_or_set_the_paper_units(backend):
    await _sheet(backend, "In")
    lay = _inch_sheet(backend)
    kept = await backend.page_setup_apply(
        "In", resolve_page_setup("ANSI_B", "landscape", scale="1:1", paper_units=None)
    )
    assert int(lay.dxf_layout.dxf.plot_paper_units) == 0
    assert "paper_units" not in kept["changed"] and kept["warnings"] == []
    assert kept["applied"]["paper_units"] is None
    # The catalogue's ANSI_B is 432 x 279 mm (ANSI Y14.1's metric value), so
    # at 1:1 under inches the sheet holds 432 / 25.4 units across.
    frame = backend._paper_frame(lay)
    assert frame["xlim"][1] - frame["xlim"][0] == pytest.approx(432 / 25.4), "inches at 1:1"

    to_mm = await backend.page_setup_apply(
        "In", resolve_page_setup("ANSI_B", "landscape", scale="1:1", paper_units="mm")
    )
    assert to_mm["changed"]["paper_units"] == ["inches", "mm"] and to_mm["warnings"]
    frame = backend._paper_frame(lay)
    assert frame["xlim"][1] - frame["xlim"][0] == pytest.approx(432.0), "432 mm at 1:1"

    back = await backend.page_setup_apply(
        "In", resolve_page_setup("ANSI_B", "landscape", scale="1:1", paper_units="inches")
    )
    assert back["changed"]["paper_units"] == ["mm", "inches"] and back["warnings"]
    assert int(lay.dxf_layout.dxf.plot_paper_units) == 0


@pytest.mark.asyncio
async def test_apply_on_a_mm_sheet_warns_of_nothing(backend):
    await _sheet(backend)
    result = await backend.page_setup_apply(SHEET, resolve_page_setup("ISO_A3"))
    assert result["warnings"] == [] and "paper_units" not in result["changed"]


@pytest.mark.asyncio
async def test_plot_style_list_headless_is_the_catalog_then_the_authored_tables(backend):
    rows = await backend.plot_style_list()
    assert [row["name"] for row in rows] == list(CTB_CATALOG) + ["arch_monochrome.ctb"]
    assert [row["source"] for row in rows] == ["catalog"] * len(CTB_CATALOG) + ["authored"]
    assert all(row["installed"] is None for row in rows)


@pytest.mark.asyncio
async def test_dwt_write_capability_is_declared_false_headlessly(backend):
    assert backend.capabilities().to_dict()["features"]["dwt_write"]["supported"] is False
