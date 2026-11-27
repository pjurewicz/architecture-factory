"""arch_legend: one row per block and hatch pattern model space uses, sample + description."""

from __future__ import annotations

import pytest

from engineering.annotation_record import APP_ID
from engineering.arch.catalogue import insert_item
from engineering.arch.electrical import insert_devices
from engineering.arch.legend import draw_legend
from engineering.arch.spec import draw_plan_from_spec

SPEC = {
    "sheet": {"intent": "flat", "sheet_size": "A3", "scale": 50},
    "walls": [
        {
            "id": "EXT",
            "axis": [[0, 0], [5000, 0], [5000, 4000], [0, 4000]],
            "thickness": 250,
            "closed": True,
            "material": "brick",
        },
        {"id": "P1", "axis": [[3000, 0], [3000, 4000]], "thickness": 100, "material": "aac"},
    ],
}


async def _drawing(backend):
    await draw_plan_from_spec(backend, SPEC)
    await insert_devices(
        backend, [{"kind": "socket", "at": [1000, 140]}, {"kind": "socket", "at": [2000, 140]}]
    )
    await insert_item(backend, "wc", at=(4000, 125))
    await backend.entity_create_hatch(
        "ANSI37", [[3200, 2500], [4700, 2500], [4700, 3700], [3200, 3700]], scale=50
    )


@pytest.mark.asyncio
async def test_a_paper_legend_explains_what_model_space_uses(backend):
    await _drawing(backend)
    await backend.layout_create("A3")
    result = await draw_legend(backend, at=(300, 280), layout="A3", lang="pl")
    assert [(r["key"], r["label"], r["count"]) for r in result["rows"]] == [
        ("ANSI31", "mur z cegły", 8),  # the brick shell's hatch at 1:50 density
        ("ANSI31", "ściana z betonu komórkowego", 1),  # aac: twice the spacing
        ("ANSI37", "ANSI37", 1),  # no table names it: its pattern
        ("ARCH_E_SOCKET_50", "gniazdo wtyczkowe", 2),
        ("ARCH_WC", "miska ustępowa ze spłuczką", 1),
    ]
    paper = backend._doc.layouts.get("A3")
    mine = [e for e in paper if e.has_xdata(APP_ID)]
    assert {e.dxf.handle for e in mine} == set(result["handles"])
    hatches = sorted(
        (e.dxf.pattern_name, round(e.dxf.pattern_scale, 6)) for e in mine if e.dxftype() == "HATCH"
    )
    # a sample prints at the plan's density: model scale / plot scale on paper
    assert hatches == [("ANSI31", 0.4), ("ANSI31", 0.8), ("ANSI37", 1.0)]
    socket = next(e for e in mine if e.dxftype() == "INSERT" and e.dxf.name == "ARCH_E_SOCKET_50")
    assert socket.dxf.xscale == pytest.approx(1 / 50)  # its plotted size, not enlarged
    wc = next(e for e in mine if e.dxftype() == "INSERT" and e.dxf.name == "ARCH_WC")
    assert wc.dxf.xscale < 1 / 50  # shrunk into its 15 x 6 mm cell
    texts = [e.dxf.text for e in mine if e.dxftype() == "TEXT"]
    assert texts[0] == "LEGENDA"
    assert len(paper.query("*")) == len(mine)  # nothing else was drawn on the sheet


@pytest.mark.asyncio
async def test_labels_override_or_drop_a_row_and_a_second_legend_skips_the_first(backend):
    await _drawing(backend)
    await draw_legend(backend, at=(6000, 4000))  # in model space this time
    again = await draw_legend(
        backend, at=(9000, 4000), labels={"ansi37": "floor tiles 300 x 300", "ARCH_WC": None}
    )
    assert [r["label"] for r in again["rows"]] == [
        "aerated concrete (AAC) wall",
        "brick wall",
        "floor tiles 300 x 300",
        "socket outlet",
    ]
    assert again["rows"][-1]["count"] == 2  # the first legend's sample is not counted


@pytest.mark.asyncio
async def test_refusals_draw_nothing(backend):
    with pytest.raises(ValueError, match="uses no block and no hatch pattern"):
        await draw_legend(backend, at=(0, 0))
    await _drawing(backend)
    before = len(backend._doc.modelspace())
    with pytest.raises(ValueError, match=r"labels: no block or hatch pattern \['SOFA'\]"):
        await draw_legend(backend, at=(0, 0), labels={"sofa": "x"})
    with pytest.raises(ValueError, match="lang must be one of"):
        await draw_legend(backend, at=(0, 0), lang="de")
    assert len(backend._doc.modelspace()) == before
