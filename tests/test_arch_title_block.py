"""arch_title_block: an architect's title block and numbered notes in a sheet corner."""

from __future__ import annotations

import pytest

from engineering.annotation_record import APP_ID
from engineering.arch.sheetblock import draw_title_block
from engineering.errors import InvalidArgsError
from engineering.standards.papers import resolve_page_setup
from engineering.viewport_clip import text_box

FIELDS = {
    "project": "Remont mieszkania przy ul. Długiej 12/4 w Gdańsku - aranżacja wnętrz",
    "investor": "Anna Kowalska",
    "stage": "projekt wykonawczy",
    "scale": "1:50",
    "title": "RZUT INSTALACJI ELEKTRYCZNEJ",
    "Branża": "wnętrza",  # not a known key: lettered as written
}
NOTES = [
    "Wysokości montażu gniazd podano od poziomu posadzki wykończonej.",
    "Wszystkie wymiary sprawdzić na budowie.",
]


async def _sheet(backend):
    await backend.layout_create("A3")
    await backend.page_setup_apply("A3", resolve_page_setup(paper="A3", orientation="landscape"))


def _texts(backend, handles):
    return [
        backend._doc.entitydb[h] for h in handles if backend._doc.entitydb[h].dxftype() == "TEXT"
    ]


@pytest.mark.asyncio
async def test_the_block_sits_in_the_frame_corner_with_the_notes_above(backend):
    await _sheet(backend)
    result = await draw_title_block(backend, layout="A3", fields=FIELDS, notes=NOTES, lang="pl")
    # A3 landscape's ISO 5457 frame runs 20..410 x 10..287: the block's corner is its corner
    assert result["at"] == [410.0, 10.0]
    assert result["bbox"]["max"][0] == pytest.approx(410.0)
    assert result["bbox"]["min"] == pytest.approx([230.0, 10.0])
    assert result["rows"] == ["TEMAT", "INWESTOR", "ETAP", "SKALA", "TYTUŁ RYSUNKU", "Branża"]
    texts = _texts(backend, result["handles"])
    assert all(t.has_xdata(APP_ID) for t in texts)
    words = {t.dxf.text: t for t in texts}
    assert words["UWAGI"].dxf.insert.y > words["TEMAT"].dxf.insert.y  # notes stacked above
    assert words["1."].dxf.insert.y > words["2."].dxf.insert.y
    assert words["RZUT INSTALACJI ELEKTRYCZNEJ"].dxf.height == 3.5
    assert words["Anna Kowalska"].dxf.height == 2.5
    # every letter stays inside the block
    lo, hi = result["bbox"]["min"], result["bbox"]["max"]
    for t in texts:
        for x, y in text_box(t):
            assert lo[0] - 1e-6 <= x <= hi[0] + 1e-6 and lo[1] - 1e-6 <= y <= hi[1] + 1e-6, (
                t.dxf.text
            )
    assert backend._doc.layouts.get("A3").name == "A3"
    assert all(
        backend._doc.entitydb[h].dxf.owner == backend._doc.layouts.get("A3").layout_key
        for h in result["handles"]
    )


@pytest.mark.asyncio
async def test_a_long_value_wraps_and_its_row_grows(backend):
    narrow = await draw_title_block(
        backend, at=(0, 0), fields=FIELDS, width=100, scale=1, name="narrow"
    )
    wide = await draw_title_block(backend, at=(0, 500), fields=FIELDS, width=300, scale=1)

    def height(r):
        return r["bbox"]["max"][1] - r["bbox"]["min"][1]

    assert height(narrow) > height(wide)
    lines = [t.dxf.text for t in _texts(backend, narrow["handles"])]
    assert FIELDS["project"] not in lines  # broken at its spaces
    assert " ".join(lines[1:5]).startswith("Remont mieszkania")


@pytest.mark.asyncio
async def test_model_space_scales_and_a_top_corner_hangs_the_notes_below(backend):
    result = await draw_title_block(
        backend,
        at=(10000, 8000),
        corner="top_left",
        fields={"date": "2026-11-27"},
        notes=["x"],
        scale=50,
    )
    assert result["bbox"]["min"][0] == 10000 and result["bbox"]["max"][1] == 8000
    assert result["bbox"]["max"][0] == pytest.approx(10000 + 180 * 50)
    words = {t.dxf.text: t for t in _texts(backend, result["handles"])}
    assert words["NOTES"].dxf.insert.y < words["DATE"].dxf.insert.y
    assert words["2026-11-27"].dxf.height == 2.5 * 50


@pytest.mark.asyncio
async def test_refusals_draw_nothing(backend):
    before = len(backend._doc.modelspace())
    with pytest.raises(ValueError, match="give a layout"):
        await draw_title_block(backend, fields=FIELDS)
    with pytest.raises(ValueError, match="corner must be one of"):
        await draw_title_block(backend, at=(0, 0), fields=FIELDS, corner="middle")
    with pytest.raises(ValueError, match=r"fields\.investor: must not be empty"):
        await draw_title_block(backend, at=(0, 0), fields={"investor": "  "})
    with pytest.raises(ValueError, match="give fields, notes or both"):
        await draw_title_block(backend, at=(0, 0))
    assert len(backend._doc.modelspace()) == before


@pytest.mark.asyncio
async def test_cells_side_by_side_fit_a_short_block_and_the_notes_go_apart(backend):
    """Round 4 (A2): 8 values in 85 mm - as stacked rows 56 mm high, as cells
    side by side with smaller lettering within 40; the notes in their own box;
    a block over a viewport says so."""
    await _sheet(backend)
    await backend.viewport_create(
        layout="A3",
        center_x=250,
        center_y=130,
        width=300,
        height=220,
        view_center_x=0,
        view_center_y=0,
    )
    pairs = [
        ["Inwestor", "Anna Kowalska"],
        ["Adres", "ul. Długa 12/4, Gdańsk"],
        ["Tytuł", "RZUT OŚWIETLENIA"],
        ["Autor", "arch. Jan Nowak"],
        ["Data", "11.2026"],
        ["Etap", "PW"],
        ["Skala", "1:50"],
        ["Nr", "E-02"],
    ]
    with pytest.raises(InvalidArgsError, match=r"height: the block needs 56\.0 mm, over 40"):
        await draw_title_block(
            backend,
            layout="A3",
            corner="bottom_left",
            at=(4.2, 5.7),
            width=85.3,
            fields=pairs,
            height=40,
        )
    rows = [pairs[0], pairs[1], [pairs[3], pairs[2]], [pairs[4], pairs[5], pairs[6], pairs[7]]]
    result = await draw_title_block(
        backend,
        layout="A3",
        corner="bottom_left",
        at=(4.2, 5.7),
        width=85.3,
        fields=rows,
        text_height=2.0,
        label_position="above",
        lang="pl",
        notes={"items": NOTES, "x": 100, "y": 5.7, "width": 120},
    )
    height = result["bbox"]["max"][1] - result["bbox"]["min"][1]
    assert height <= 40 and result["bbox"]["min"] == pytest.approx([4.2, 5.7])
    assert result["rows"] == ["Inwestor", "Adres", "Autor | Tytuł", "Data | Etap | Skala | Nr"]
    texts = {t.dxf.text: t for t in _texts(backend, result["handles"])}
    date, stage = texts["11.2026"], texts["PW"]
    assert date.dxf.insert.y == pytest.approx(stage.dxf.insert.y)  # one row, side by side
    assert date.dxf.insert.x < stage.dxf.insert.x
    assert texts["Data"].dxf.insert.y > date.dxf.insert.y  # the label above its value
    assert texts["1."].dxf.insert.x > 100  # the notes in their own box
    assert result["notes_bbox"]["min"] == [100.0, 5.7]
    assert any(w.startswith("notes covers_viewport") for w in result["warnings"])


@pytest.mark.asyncio
async def test_explicit_lines_cell_styles_and_a_fixed_height(backend):
    """Round 5 (A3): "\\n" is a line break and a short line never wraps; a
    cell sets its own alignment, height and width; the block stretches to
    `height` exactly."""
    await _sheet(backend)
    rows = [
        {"value": "PROJEKT BUDOWLANY\nDOM JEDNORODZINNY", "align": "center", "text_height": 5},
        [
            {"label": "Opracował", "value": "mgr inż. arch. Łucja Żółć\nupr. 12/34\nWarszawa"},
            {"label": "Źródło", "value": "pomiar 2026", "align": "right", "width": 60},
        ],
    ]
    result = await draw_title_block(
        backend, layout="A3", fields=rows, width=180, height=69.6, lang="pl"
    )
    assert result["height"] == pytest.approx(69.6)
    assert result["bbox"]["max"][1] - result["bbox"]["min"][1] == pytest.approx(69.6)
    texts = {t.dxf.text: t for t in _texts(backend, result["handles"])}
    author = ["mgr inż. arch. Łucja Żółć", "upr. 12/34", "Warszawa"]
    assert all(line in texts for line in author)  # one TEXT per line given, none broken
    ys = [texts[line].dxf.insert.y for line in author]
    assert ys[0] > ys[1] > ys[2]
    title = texts["PROJEKT BUDOWLANY"]
    assert title.dxf.height == 5 and title.dxf.halign == 1  # centred on the block
    assert title.dxf.align_point.x == pytest.approx(230 + 90)
    source = texts["pomiar 2026"]
    assert source.dxf.halign == 2  # right-aligned at the 60 mm cell's right edge
    assert source.dxf.align_point.x == pytest.approx(410 - 1.5)
    assert texts["Źródło"].dxf.halign == 2
    with pytest.raises(InvalidArgsError, match=r"fields\[0\]\[0\]\.align: one of"):
        await draw_title_block(
            backend, at=(0, 0), fields=[[{"value": "x", "align": "middle"}, ["a", "b"]]]
        )


@pytest.mark.asyncio
async def test_a_rerun_replaces_its_set_and_another_name_coexists(backend):
    """Round 5 (A5): every entity carries the record with its set; a second
    call with the same name deletes the first block, another name keeps it."""
    await _sheet(backend)
    paper = backend._doc.layouts.get("A3")
    first = await draw_title_block(backend, layout="A3", fields=FIELDS, notes=NOTES)
    count = len(paper)
    assert first["replaced"] == 0
    assert all(backend._doc.entitydb[h].has_xdata(APP_ID) for h in first["handles"])
    again = await draw_title_block(backend, layout="A3", fields=FIELDS, notes=NOTES)
    assert again["replaced"] == len(first["handles"]) and len(paper) == count
    other = await draw_title_block(
        backend, layout="A3", corner="top_left", fields={"date": "1"}, name="stamp"
    )
    assert other["replaced"] == 0 and len(paper) == count + len(other["handles"])


@pytest.mark.asyncio
async def test_notes_without_frame_or_title_in_their_own_colour(backend):
    result = await draw_title_block(
        backend,
        at=(0, 0),
        scale=1,
        notes={
            "items": ["Pierwsza\ndruga linia"],
            "title": None,
            "frame": False,
            "color": 1,
            "text_height": 3,
        },
    )
    kinds = [backend._doc.entitydb[h].dxftype() for h in result["handles"]]
    assert kinds == ["TEXT", "TEXT", "TEXT"]  # "1.", two lines, no frame, no title
    texts = _texts(backend, result["handles"])
    assert {t.dxf.color for t in texts} == {1} and {t.dxf.height for t in texts} == {3}


@pytest.mark.asyncio
async def test_a_narrow_source_block_is_drawn_at_its_own_width(backend):
    # round 6: the source's title block is 48.75 mm wide; 60 mm was a caution
    result = await draw_title_block(backend, at=(0, 0), fields=FIELDS, width=48.75, scale=1)
    lo, hi = result["bbox"]["min"], result["bbox"]["max"]
    assert hi[0] - lo[0] == pytest.approx(48.75)
    # a word is never broken: the one wider than its column is named
    assert {w.split(":")[0] for w in result["warnings"]} == {"fields[4]"}  # the 3.5 mm title
    assert any("'ELEKTRYCZNEJ'" in w for w in result["warnings"])
    fits = await draw_title_block(
        backend, at=(0, -500), fields={"date": "2026-11"}, width=48.75, scale=1, name="c"
    )
    assert "warnings" not in fits
    with pytest.raises(ValueError, match="outside 30-400 mm"):
        await draw_title_block(backend, at=(0, 0), fields=FIELDS, width=29, scale=1, name="b")


@pytest.mark.asyncio
async def test_an_office_frame_with_a_band_anchors_the_block_and_notes_number_their_way(backend):
    """Round 7: sheet_frame was ISO 5457 only (drawn, then deleted), the block
    anchored to ISO's 20/10 frame, and notes could only be '1.'."""
    from engineering.sheet.frames import draw_sheet_frame

    await _sheet(backend)
    frame = await draw_sheet_frame(
        backend, layout="A3", margins=5, band=40, zones=False, marks=False
    )
    assert frame["metrics"]["frame"] == [5.0, 5.0, 415.0, 292.0]
    assert frame["metrics"]["band"] == [5.0, 5.0, 415.0, 45.0]
    assert frame["metrics"]["standard"] == "custom margins"
    band_line = backend._doc.entitydb[frame["handles"][2]]
    assert band_line.dxftype() == "LINE" and band_line.dxf.start.y == 45.0
    out = await draw_title_block(
        backend,
        fields={"title": "RZUT: WYTYCZNE C.O."},
        notes={"items": ["Pierwsza.", "Druga."], "numbering": "{n}/"},
        layout="A3",
        width=60,
    )
    assert out["at"] == [415.0, 5.0]  # the drawn frame's corner, not ISO's (410, 10)
    assert out["anchor"] == {"frame": frame["inner_border"]}
    texts = [t.dxf.text for t in _texts(backend, out["handles"])]
    assert "1/" in texts and "2/" in texts and "1." not in texts
    # 60 mm: the title wraps, and says how wide one line would be
    assert any("wrapped to" in w for w in out["wrapped"]), out
    with pytest.raises(InvalidArgsError, match="notes.numbering"):
        await draw_title_block(backend, notes={"items": ["x"], "numbering": "#"}, layout="A3")
    with pytest.raises(ValueError, match="margins: one number or"):
        await draw_sheet_frame(backend, layout="A3", margins=[5, 5])
