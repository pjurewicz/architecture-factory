"""arch_title_block: an architect's title block and numbered notes in a sheet corner."""

from __future__ import annotations

import pytest

from engineering.annotation_record import APP_ID
from engineering.arch.sheetblock import draw_title_block
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
    narrow = await draw_title_block(backend, at=(0, 0), fields=FIELDS, width=100, scale=1)
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
    with pytest.raises(ValueError, match=r"max_height: the block needs 56\.0 mm, over 40"):
        await draw_title_block(
            backend,
            layout="A3",
            corner="bottom_left",
            at=(4.2, 5.7),
            width=85.3,
            fields=pairs,
            max_height=40,
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
        max_height=40,
        lang="pl",
        notes=NOTES,
        notes_box={"x": 100, "y": 5.7, "width": 120},
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
    assert any(w.startswith("notes_box covers_viewport") for w in result["warnings"])
