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


@pytest.mark.asyncio
async def test_offset_zero_puts_the_line_on_the_points_texts_above_or_left(backend):
    # internal dimensions: the line through the points, the style's text side
    out = await draw_chain(backend, [[0, 0], [100, 0], [250, 0]], 0)
    chain = [r for r in out["dimensions"] if r["row"] == "chain"]
    dims = _dims(backend, out["handles"])
    assert [round(d.dxf.defpoint.y, 6) for d in dims[:2]] == [0.0, 0.0]
    assert all(r["placement"] == "inline" and r["text_at"][1] > 0 for r in chain)
    assert dims[2].dxf.defpoint.y > 0  # the overall row on the text's side too

    out = await draw_chain(backend, [[10, 0], [10, 40], [10, 100]], 0)
    assert out["orientation"] == "vertical"
    dims = _dims(backend, out["handles"])
    assert [round(d.dxf.defpoint.x, 6) for d in dims[:2]] == [10.0, 10.0]
    assert all(r["text_at"][0] < 10 for r in out["dimensions"])
    assert dims[2].dxf.defpoint.x < 10


@pytest.mark.asyncio
async def test_ext_length_puts_a_fixed_tick_through_each_point_at_offset_zero(backend):
    def ticks(handle):
        dim = backend._require_doc().entitydb[handle]
        lines = [e for e in dim.get_geometry_block() if e.dxftype() == "LINE"]
        return sorted(
            (round(e.dxf.start.x, 6), round(e.dxf.start.y, 6), round(e.dxf.end.y, 6))
            for e in lines
            if math.isclose(e.dxf.start.x, e.dxf.end.x, abs_tol=1e-9)
        )

    plain = await draw_chain(backend, [[0, 0], [100, 0]], 0, overall=False)
    assert ticks(plain["handles"][0]) == [], "a zero-length extension line is suppressed"
    # DIMSCALE 2: 6 drawing units below the line, DIMEXE 1.25 x 2 above
    out = await draw_chain(backend, [[0, 0], [100, 0]], 0, overall=False, scale=2, ext_length=6)
    assert ticks(out["handles"][0]) == [(0.0, -6.0, 2.5), (100.0, -6.0, 2.5)]
    with pytest.raises(ValueError, match="ext_length"):
        await draw_chain(backend, [[0, 0], [100, 0]], 0, ext_length=0)


# a joist run at 1:50: 300, then 4 x 205 whose texts (281 wide at h 125) do not fit, then 300
JOISTS = [[0, 0], [300, 0], [505, 0], [710, 0], [915, 0], [1120, 0], [1420, 0]]


@pytest.mark.asyncio
async def test_tight_repeats_are_warned_and_merged_into_one_honest_dimension(backend):
    from engineering.critique import run_critique

    warned = await draw_chain(backend, JOISTS, -500, scale=50, dry_run=True)
    assert [r["value"] for r in warned["dimensions"]] == [300, 205, 205, 205, 205, 300, 1420]
    assert [(w["code"], w["count"], w["value"]) for w in warned["warnings"]] == [
        ("tight_repeats", 4, 205)
    ]
    # a 300 next to the run is not part of it, though it is tight too
    assert warned["repeats"] == [{"count": 4, "value": 205, "from": [300, 0], "to": [1120, 0]}]

    msp = backend._require_doc().modelspace()
    for x, _ in JOISTS:
        msp.add_line((x, 0), (x, 100))
    out = await draw_chain(backend, JOISTS, -500, scale=50, merge_repeats=True)
    assert out["warnings"] == []
    chain = [r for r in out["dimensions"] if r["row"] == "chain"]
    assert [r["value"] for r in chain] == [300, 820, 300]
    assert chain[1]["repeat"] == {"count": 4, "value": 205}
    dims = _dims(backend, out["handles"])
    merged = dims[1]
    assert round(merged.get_measurement(), 6) == 820
    shown = [e.plain_text() for e in merged.get_geometry_block() if e.dxftype() == "MTEXT"]
    assert shown == ["4 × 205 = 820"]
    # the claimed size is the measurement (820), not the count's 205 or 4
    issues = await run_critique(backend, ["dimension_truth"])
    assert [i for i in issues if i.severity != "info"] == []


@pytest.mark.asyncio
async def test_axis_points_get_center_extension_lines_and_none_has_zero_length(backend):
    out = await draw_chain(backend, [[0, -50], [100, 0], [250, -50]], 0, axis=[0])
    dims = _dims(backend, out["handles"])

    def extensions(dim):
        lines = [e for e in dim.get_geometry_block() if e.dxftype() == "LINE"]
        return {
            round(e.dxf.start.x, 6): e.dxf.linetype
            for e in lines
            if abs(e.dxf.start.x - e.dxf.end.x) < 1e-9  # vertical: an extension line
        }

    # 100 lies on the line (offset 0): no extension line, no DIMEXE tick
    assert extensions(dims[0]) == {0.0: "CENTER"}
    assert extensions(dims[1]) == {250.0: "BYLAYER"}
    assert extensions(dims[2]) == {0.0: "CENTER", 250.0: "BYLAYER"}
    assert [r["extension"] for r in out["dimensions"]] == [
        ["axis", "suppressed"],
        ["suppressed", "line"],
        ["axis", "line"],
    ]

    on_line = plan_chain([[0, 0], [100, 0], [250, 0]], 0, axis=[1])
    assert [w["code"] for w in on_line["warnings"]] == ["axis_on_line"]
    with pytest.raises(ValueError, match=r"axis\[0\]"):
        plan_chain([[0, 0], [100, 0]], 5, axis=[2])


def test_one_station_and_one_span_are_handled_by_name():
    plan = plan_chain([[0, 0], [0, 20], [5, 20]], 10, orientation="vertical")
    # two points on y = 20: the one nearer the dimension line keeps the station
    assert plan["merged"] == [{"point": [0.0, 20.0], "kept": [5.0, 20.0]}]
    # one span restates the overall: no overall row
    assert [r["row"] for r in plan["rows"]] == ["chain"]
    with pytest.raises(ValueError, match="one station"):
        plan_chain([[0, 0], [0, 50]], 10, orientation="horizontal")
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
        on_points = await client.call_tool(
            "dimension_chain", {"points": [[0, 0], [60, 0]], "offset": 0}
        )
        assert on_points.structured_content["count"] == 1
        bad = await client.call_tool(
            "dimension_chain",
            {"points": [[0, 0], [0, 5]], "offset": 5, "orientation": "horizontal"},
            raise_on_error=False,
        )
    assert bad.is_error
    assert bad.structured_content["kind"] == "invalid_args"
