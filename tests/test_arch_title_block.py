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
