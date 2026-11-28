"""arch_legend: one row per item model space uses, sample + description."""

from __future__ import annotations

import pytest

from engineering.annotation_record import APP_ID
from engineering.arch.catalogue import insert_item
from engineering.arch.legend import draw_legend
from engineering.arch.services import insert_points
from engineering.arch.spec import draw_plan_from_spec
from engineering.errors import InvalidArgsError
from engineering.standards.papers import resolve_page_setup

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
    await insert_points(
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
    result = await draw_legend(
        backend, at=(300, 280), layout="A3", lang="pl", only=["ANSI*", "socket", "wc"]
    )
    assert [(r["key"], r["label"], r["count"]) for r in result["rows"]] == [
        ("ANSI31", "mur z cegły", 8),  # the brick shell's hatch at 1:50 density
        ("ANSI31", "ściana z betonu komórkowego", 1),  # aac: twice the spacing
        ("ANSI37", "ANSI37", 1),  # no table names it: its pattern
        ("socket", "gniazdo wtyczkowe", 2),
        ("wc", "miska ustępowa ze spłuczką", 1),  # furniture only when named
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
    assert texts[0] == "OZNACZENIA"
    assert len(paper.query("*")) == len(mine)  # nothing else was drawn on the sheet


@pytest.mark.asyncio
async def test_labels_override_or_drop_a_row_and_a_second_legend_skips_the_first(backend):
    await _drawing(backend)
    await draw_legend(backend, at=(6000, 4000))  # in model space this time
    again = await draw_legend(
        backend, at=(9000, 4000), labels={"ansi37": "floor tiles 300 x 300", "socket": None}
    )
    assert [r["label"] for r in again["rows"]] == [
        "aerated concrete (AAC) wall",
        "brick wall",
        "floor tiles 300 x 300",
    ]
    assert again["left_out"]["keys"] == ["wc"]  # furniture: out unless only names it
    first = await draw_legend(backend, at=(12000, 4000), only=["socket"])
    assert first["rows"][-1]["count"] == 2  # the earlier legends' samples are not counted


@pytest.mark.asyncio
async def test_refusals_draw_nothing(backend):
    with pytest.raises(ValueError, match="no row left to draw"):
        await draw_legend(backend, at=(0, 0))
    await _drawing(backend)
    before = len(backend._doc.modelspace())
    with pytest.raises(ValueError, match=r"labels: no item, block or hatch pattern \['SOFA'\]"):
        await draw_legend(backend, at=(0, 0), labels={"sofa": "x"})
    with pytest.raises(ValueError, match="lang must be one of"):
        await draw_legend(backend, at=(0, 0), lang="de")
    assert len(backend._doc.modelspace()) == before


@pytest.mark.asyncio
async def test_a_lighting_sheet_legend_merges_filters_folds_and_warns(backend):
    """Round 4 (A1, B2): every size of a catalogue item is one row; a project
    block is named by its record; LED strip and pendant track get a line
    sample; an office row with no occurrence is added; columns fold; a legend
    past the paper's edge says so."""
    await draw_plan_from_spec(backend, SPEC)
    for width in (600, 800, 1000):  # three sizes of one wardrobe
        await insert_item(backend, "wardrobe", at=(500 + width, 125), width=width)
    await backend.block_define(
        "K", [{"type": "circle", "cx": 0, "cy": 100, "radius": 100}], [], 0, 0
    )
    await insert_points(
        backend,
        [
            {"kind": "wall_light", "at": [1000, 3860], "circuit": "SALON 01"},
            {"kind": "wall_light", "at": [2000, 3860]},
            {"kind": "led_strip", "path": [[500, 2000], [2500, 2000]]},
            {"kind": "track_pendant", "path": [[500, 2500], [2500, 2500]]},
        ],
        symbols={"wall_light": "K"},
    )
    await backend.layout_create("A3")
    await backend.page_setup_apply("A3", resolve_page_setup("A3", "landscape"))
    result = await draw_legend(
        backend,
        at=(395, 60),
        layout="A3",
        lang="pl",
        only=["wardrobe", "wall_light", "led_strip", "track_pendant"],
        extra=[
            {"label": "gniazdo 230 V hermetyczne"},
            {"label": "czujka ruchu", "kind": "motion_sensor"},
        ],
        columns=2,
    )
    rows = [(r["key"], r["label"], r["count"]) for r in result["rows"]]
    assert rows[:2] == [("track_pendant", "szynoprzewód wiszący", 1), ("led_strip", "taśma LED", 1)]
    assert ("wall_light", "wypust oświetleniowy ścienny", 2) in rows  # block K, named by its record
    assert ("wardrobe", next(r[1] for r in rows if r[0] == "wardrobe"), 3) in rows  # one row
    assert rows[-2:] == [
        ("gniazdo 230 V hermetyczne", "gniazdo 230 V hermetyczne", 0),
        ("czujka ruchu", "czujka ruchu", 0),
    ]
    paper = backend._doc.layouts.get("A3")
    lines = [e for e in paper if e.dxftype() == "LINE" and e.dxf.linetype == "DASHED"]
    assert len(lines) == 1 and lines[0].dxf.lineweight == 70  # the pendant track's sample
    samples = {e.dxf.name for e in paper if e.dxftype() == "INSERT"}
    assert {"K", "ARCH_E_MOTION_SENSOR_50"} <= samples
    # 6 rows in 2 columns: 3 rows high
    assert result["bbox"]["min"][1] == pytest.approx(60 - (3.5 + 4 + 3 * 8 + 2))
    assert any(w.startswith("off_paper") for w in result["warnings"])


@pytest.mark.asyncio
async def test_title_frame_style_order_multiline_and_a_two_symbol_row(backend):
    """Round 5 (A4): no frame by default, a title of the caller's, text height,
    colour and layer; an explicit order; a 2-line label is two TEXTs and a
    taller row; a row with two samples side by side takes over their rows."""
    await _drawing(backend)
    await backend.block_define("ZW", [{"type": "circle", "cx": 0, "cy": 0, "radius": 50}], [], 0, 0)
    await backend.block_define("CW", [{"type": "circle", "cx": 0, "cy": 0, "radius": 50}], [], 0, 0)
    await backend.block_insert("CW", 1500, 2000)
    await backend.layout_create("A3")
    result = await draw_legend(
        backend,
        at=(300, 280),
        layout="A3",
        lang="pl",
        title="OZNACZENIA",
        text_height=2.0,
        color=7,
        layer="A-ANNO-T-N",
        only=["socket", "ANSI37", "CW"],
        order=["socket"],
        extra=[
            {"label": "woda ciepła / zimna\n(pion)", "blocks": ["ZW", "CW"]},
            {"label": "przybór", "keys": ["wc"]},
        ],
    )
    keys = [r["key"] for r in result["rows"]]
    assert keys[0] == "socket" and "wc" not in keys  # ordered; wc moved into the keys row
    paper = backend._doc.layouts.get("A3")
    mine = [e for e in paper if e.has_xdata(APP_ID)]
    assert not [
        e
        for e in mine
        if e.dxftype() == "LWPOLYLINE"
        and len(e) == 4
        and e.closed
        and abs(e.get_points()[0][0] - 300) < 1e-6
    ]  # no outer frame
    texts = [e for e in mine if e.dxftype() == "TEXT"]
    assert texts[0].dxf.text == "OZNACZENIA"
    assert {t.dxf.layer for t in texts} == {"A-ANNO-T-N"} and {t.dxf.color for t in texts} == {7}
    body = {t.dxf.text: t for t in texts[1:]}
    assert body["gniazdo wtyczkowe"].dxf.height == 2.0
    a, b = body["woda ciepła / zimna"], body["(pion)"]
    assert a.dxf.insert.x == b.dxf.insert.x and a.dxf.insert.y > b.dxf.insert.y
    pair = [e for e in mine if e.dxftype() == "INSERT" and e.dxf.name in ("ZW", "CW")]
    assert len(pair) == 3  # CW in its own row, ZW + CW side by side
    row = next(r for r in result["rows"] if r["key"] == "woda ciepła / zimna\n(pion)")
    assert row["samples"] == ["ZW", "CW"]
    with pytest.raises(InvalidArgsError, match=r"order\[0\]: 'sofa' names no row"):
        await draw_legend(backend, at=(0, 0), order=["sofa"])
    with pytest.raises(InvalidArgsError, match=r"extra\[0\]\.blocks\[1\]: no block 'Q'"):
        await draw_legend(backend, at=(0, 0), extra=[{"label": "q", "blocks": ["ZW", "Q"]}])


@pytest.mark.asyncio
async def test_a_rerun_replaces_its_legend_and_another_name_coexists(backend):
    await _drawing(backend)
    await backend.layout_create("A3")
    paper = backend._doc.layouts.get("A3")
    first = await draw_legend(backend, at=(300, 280), layout="A3", frame=True)
    count = len(paper)
    again = await draw_legend(backend, at=(300, 280), layout="A3", frame=True)
    assert first["replaced"] == 0 and again["replaced"] == count and len(paper) == count
    other = await draw_legend(backend, at=(20, 280), layout="A3", only=["wc"], name="furniture")
    assert other["replaced"] == 0 and len(paper) == count + len(other["handles"])
    model = await draw_legend(backend, at=(9000, 4000))  # another space: nothing replaced
    assert model["replaced"] == 0


@pytest.mark.asyncio
async def test_one_legend_names_the_points_of_every_trade(backend):
    """B1: the legend reads the ACADMCP_SERVICE records of plumbing and
    electrical alike - a project block, a run and our own symbols."""
    await draw_plan_from_spec(backend, SPEC)
    await backend.block_define(
        "WC_POINT", [{"type": "circle", "cx": 0, "cy": 100, "radius": 100}], [], 0, 0
    )
    await insert_points(
        backend,
        [
            {"kind": "water_pair", "at": [1000, 140]},
            {"kind": "drain", "at": [1500, 140]},
            {"kind": "linear_drain", "path": [[200, 600], [1200, 600]]},
            {"kind": "socket", "at": [2000, 140]},
        ],
        symbols={"drain": "WC_POINT"},
    )
    result = await draw_legend(backend, at=(6000, 4000), lang="pl")
    rows = {r["key"]: r["label"] for r in result["rows"] if r["kind"] != "hatch"}
    assert rows == {
        "linear_drain": "odwodnienie liniowe",
        "water_pair": "punkt wody ciepłej + zimnej (150 mm)",
        "drain": "podejście kanalizacyjne",  # the project block, named by its record
        "socket": "gniazdo wtyczkowe",
    }
