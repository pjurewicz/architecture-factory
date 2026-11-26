"""``text_find_replace`` searches the whole drawing (spec section 5, Task 12).

The headless engine read the ATTRIBs of model-space references only, so an
attribute on a paper layout or on a reference nested in a block definition was
missed. It now searches ``TEXT_SEARCH_SCOPE``: model space, every paper layout,
every block definition that is not a layout block or an xref, and the ATTRIBs
of every reference found in any of them.
"""

from __future__ import annotations

import pytest

from backends.contracts.entity_modification import TEXT_SEARCH_SCOPE

pytestmark = pytest.mark.asyncio

#: The one logical drawing the engine is given, as (DXF type, value): model
#: space TEXT, the ATTRIB of a model-space reference, paper-layout MTEXT, the
#: ATTRIB of a paper-layout reference, the ATTDEF default and a TEXT inside
#: block B, the ATTRIB of a reference to B nested inside block C.
EXPECTED = sorted(
    [
        ("TEXT", "OLD m", "NEW m"),
        ("ATTRIB", "OLD a1", "NEW a1"),
        ("MTEXT", "OLD p", "NEW p"),
        ("ATTRIB", "OLD a2", "NEW a2"),
        ("ATTDEF", "OLD d", "NEW d"),
        ("TEXT", "OLD b", "NEW b"),
        ("ATTRIB", "OLD n", "NEW n"),
    ]
)


def _rows(result: dict) -> list[tuple[str, str, str]]:
    return sorted((row["type"], row["before"], row["after"]) for row in result["entities"])


# -- the headless engine --------------------------------------------------------


def _build_headless(backend):
    doc = backend._doc
    block_b = doc.blocks.new("B")
    block_b.add_attdef("TAG", (0, 0), text="OLD d")
    block_b.add_text("OLD b", dxfattribs={"insert": (0, 5)})
    model = doc.modelspace()
    model.add_text("OLD m", dxfattribs={"insert": (0, 0)})
    model.add_blockref("B", (10, 0)).add_auto_attribs({"TAG": "OLD a1"})
    paper = doc.layouts.get("Layout1")
    paper.add_mtext("OLD p", dxfattribs={"insert": (5, 5)})
    paper.add_blockref("B", (20, 20)).add_auto_attribs({"TAG": "OLD a2"})
    block_c = doc.blocks.new("C")
    block_c.add_blockref("B", (0, 0)).add_auto_attribs({"TAG": "OLD n"})


async def test_a_paper_layout_attribute_is_found_headlessly(backend):
    doc = backend._doc
    doc.blocks.new("P").add_attdef("TAG", (0, 0), text="")
    ref = doc.layouts.get("Layout1").add_blockref("P", (0, 0))
    ref.add_auto_attribs({"TAG": "OLD sheet"})

    result = await backend.text_find_replace("OLD", "NEW")

    assert result["replaced"] == 1
    assert ref.attribs[0].dxf.text == "NEW sheet"


async def test_a_nested_reference_attribute_is_found_headlessly(backend):
    doc = backend._doc
    doc.blocks.new("INNER").add_attdef("TAG", (0, 0), text="")
    nested = doc.blocks.new("OUTER").add_blockref("INNER", (0, 0))
    nested.add_auto_attribs({"TAG": "OLD nested"})

    result = await backend.text_find_replace("OLD", "NEW")

    assert result["replaced"] == 1
    assert nested.attribs[0].dxf.text == "NEW nested"


async def test_the_headless_engine_covers_the_whole_scope(backend):
    _build_headless(backend)

    result = await backend.text_find_replace("OLD", "NEW", dry_run=True)

    assert _rows(result) == EXPECTED
    assert result["searched_scope"] == list(TEXT_SEARCH_SCOPE)


async def test_an_xref_block_is_not_searched_headlessly(backend):
    doc = backend._doc
    doc.add_xref_def("other.dxf", "XR")
    xref = doc.blocks.get("XR")
    xref.add_text("OLD x", dxfattribs={"insert": (0, 0)})
    assert xref.block_record.is_xref

    result = await backend.text_find_replace("OLD", "NEW")

    assert result["replaced"] == 0


async def test_a_dimension_rendering_block_is_not_searched_headlessly(backend):
    # The review's repro: the measurement "10" lives as MTEXT in the
    # dimension's anonymous *D block; rewriting it showed 77 over a 10 mm line.
    await backend.entity_create_line(0, 0, 10, 0)
    await backend.dimension_linear(0, 0, 10, 0, 5, 5)
    dim = backend._doc.modelspace().query("DIMENSION").first
    rendered = dim.get_geometry_block()
    assert rendered is not None and rendered.name.startswith("*D")
    assert [e.text for e in rendered.query("MTEXT")] == ["10"]

    preview = await backend.text_find_replace("10", "77", dry_run=True)
    applied = await backend.text_find_replace("10", "77")

    assert preview["entities"] == [] and applied["replaced"] == 0
    assert [e.text for e in rendered.query("MTEXT")] == ["10"]
    assert dim.get_measurement() == pytest.approx(10.0)


async def test_an_anonymous_block_is_not_searched_but_a_named_one_is(backend):
    doc = backend._doc
    doc.blocks.new_anonymous_block("U").add_text("OLD u", dxfattribs={"insert": (0, 0)})
    doc.blocks.new("NAMED").add_text("OLD named", dxfattribs={"insert": (0, 0)})

    result = await backend.text_find_replace("OLD", "NEW", dry_run=True)

    assert [row["before"] for row in result["entities"]] == ["OLD named"]
