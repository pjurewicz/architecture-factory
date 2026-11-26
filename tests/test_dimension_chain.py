"""dimension_chain: one continuous chain through any points, tight texts staggered."""

from __future__ import annotations

import math

import pytest

from engineering.dimension_chain import draw_chain, plan_chain

# 1:20 on the ISO-25 style: text 2.5 x 20 = 50, gap 0.625 x 20 = 12.5, DIMEXE 1.25 x 20 = 25
H, G = 50.0, 12.5


def _dims(backend, handles):
    doc = backend._require_doc()
    return [doc.entitydb[h] for h in handles]


def _box(row):
    half = len(f"{row['value']:g}") * 0.75 * H / 2
    x, y = row["text_at"]
    return (x - half, x + half, y - H / 2, y + H / 2)


@pytest.mark.asyncio
async def test_tight_spans_are_staggered_beyond_the_line_and_the_overall_row_clears_them(backend):
    # "60" (75 wide) and "105" (112.5 wide) do not fit their spans; "150" does
    points = [[315, 0], [0, 0], [60, 0], [165, 0], [915, 0]]  # given out of order
    out = await draw_chain(backend, points, -200, scale=20)
    chain = [r for r in out["dimensions"] if r["row"] == "chain"]
    assert [(r["value"], r["placement"]) for r in chain] == [
        (60, "staggered"),
        (105, "staggered"),
        (150, "inline"),
        (600, "inline"),
    ]
    # inline: above the line (towards the part); tier 1 below it past DIMEXE + gap,
    # tier 2 one text height + gap farther; each stays centred on its own span
    assert [r["text_at"] for r in chain] == [
        [30, -262.5],
        [112.5, -325],
        [240, -162.5],
        [615, -162.5],
    ]

    dims = _dims(backend, out["handles"])
    assert sorted(round(d.get_measurement(), 6) for d in dims) == [60, 105, 150, 600, 915]
    # the overall line: tier 2's far edge (-350) + gap, + its own text below the
    # line (gap + height) -> 225 beyond the chain line, not the 160 default step
    assert {round(d.dxf.defpoint.y, 6) for d in dims} == {-200.0, -425.0}
    boxes = [_box(r) for r in out["dimensions"]]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert not (a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]), (a, b)


@pytest.mark.asyncio
async def test_a_text_with_no_free_tier_slides_out_with_a_leader(backend):
    out = await draw_chain(backend, [[i * 60, 0] for i in range(6)], -200, scale=20)
    placements = [r["placement"] for r in out["dimensions"] if r["row"] == "chain"]
    assert placements == ["staggered"] * 4 + ["moved"]
    for r, d in zip(out["dimensions"], _dims(backend, out["handles"]), strict=True):
        middle = ((r["from"][0] + r["to"][0]) / 2, d.dxf.defpoint.y)
        # the leader starts at the dimension line's middle and leaves the line
        leaders = [
            e
            for e in d.get_geometry_block()
            if e.dxftype() == "LINE"
            and math.dist((e.dxf.start.x, e.dxf.start.y), middle) < 1e-6
            and abs(e.dxf.end.y - e.dxf.start.y) > 1e-6
        ]
        assert bool(leaders) == (r["placement"] == "moved"), r
        assert r["leader"] == (r["placement"] == "moved")


@pytest.mark.asyncio
async def test_the_drawing_dimscale_is_put_back_and_dry_run_draws_nothing(backend):
    before = len(await backend.entity_list())
    plan = await draw_chain(backend, [[0, 0], [10, 0], [30, 0]], 10, scale=5, dry_run=True)
    assert plan["count"] == 3 and "handles" not in plan
    assert len(await backend.entity_list()) == before
    await draw_chain(backend, [[0, 0], [10, 0], [30, 0]], 10, scale=5)
    assert float(await backend.system_get_variable("DIMSCALE")) == 1.0


@pytest.mark.asyncio
async def test_vertical_and_aligned_chains_measure_along_their_direction(backend):
    # a section's steps: the stations are the y values, x only decides the outermost
    out = await draw_chain(backend, [[0, 0], [5, 20], [30, 60]], 10)
    assert out["orientation"] == "vertical"
    dims = _dims(backend, out["handles"])
    assert sorted(round(d.get_measurement(), 6) for d in dims) == [20, 40, 60]
    assert {round(d.dxf.defpoint.x, 6) for d in dims} == {40.0, 48.0}

    # aligned: a 45° run of three points 30 apart in x is 2 x 30√2 plus the overall
    out = await draw_chain(backend, [[0, 0], [30, 30], [60, 60]], 10, orientation="aligned")
    measured = sorted(d.get_measurement() for d in _dims(backend, out["handles"]))
    assert measured == pytest.approx([30 * math.sqrt(2)] * 2 + [60 * math.sqrt(2)])


def test_one_station_and_one_span_are_handled_by_name():
    plan = plan_chain([[0, 0], [0, 20], [5, 20]], 10, orientation="vertical")
    # two points on y = 20: the one nearer the dimension line keeps the station
    assert plan["merged"] == [{"point": [0.0, 20.0], "kept": [5.0, 20.0]}]
    # one span restates the overall: no overall row
    assert [r["row"] for r in plan["rows"]] == ["chain"]
    with pytest.raises(ValueError, match="one station"):
        plan_chain([[0, 0], [0, 50]], 10, orientation="horizontal")
    with pytest.raises(ValueError, match="offset"):
        plan_chain([[0, 0], [50, 0]], 0)
    with pytest.raises(ValueError, match="orientation"):
        plan_chain([[0, 0], [50, 0]], 5, orientation="diagonal")


@pytest.mark.asyncio
async def test_the_tool_refuses_by_name_and_draws_on_the_dim_layer():
    from fastmcp import Client

    import server

    async with Client(server.mcp) as client:
        await client.call_tool("drawing_new", {})
        made = await client.call_tool(
            "dimension_chain", {"points": [[0, 0], [60, 0], [165, 0], [315, 0]], "offset": -15}
        )
        assert made.structured_content["layer"] == "DIM"
        assert made.structured_content["count"] == 4
        bad = await client.call_tool(
            "dimension_chain",
            {"points": [[0, 0], [0, 5]], "offset": 5, "orientation": "horizontal"},
            raise_on_error=False,
        )
    assert bad.is_error
    assert bad.structured_content["kind"] == "invalid_args"
