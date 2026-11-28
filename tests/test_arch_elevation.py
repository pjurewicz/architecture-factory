"""interior_wall_elevation: the pure model and the tool through a real MCP client."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.exceptions import ToolError

import server
from engineering.arch.elevation import elevation_dimensions, elevation_levels, parse_elevation

BATH = {
    "length": 3600,
    "height": 2700,
    "openings": [
        {"id": "D1", "kind": "door", "x": 200, "width": 800, "height": 2050, "hinge": "left"},
        {"id": "O1", "kind": "window", "x": 2600, "width": 700, "height": 900, "sill": 1500},
    ],
    "fittings": [
        {"kind": "skirting", "top": 80},
        {"id": "BLAT", "kind": "worktop", "x": 1300, "width": 1200, "top": 850},
        {"id": "G1", "kind": "socket", "x": 2400, "y": 1050, "gangs": 2},
    ],
    "services": [{"id": "ZW", "kind": "cold_water", "x": 1800, "y": 550}],
}


def test_levels_name_what_sets_them_and_merge_equal_heights():
    spec = dict(
        BATH, fittings=[*BATH["fittings"], {"id": "S", "kind": "switch", "x": 1150, "y": 1050}]
    )
    levels = elevation_levels(parse_elevation(spec))
    assert [row["level"] for row in levels] == [0, 80, 550, 850, 1050, 1500, 2050, 2400, 2700]
    assert levels[4]["from"] == ["G1.centre", "S.centre"]
    # a worktop's underside is not a level a drafter dimensions
    assert all("BLAT.bottom" not in row["from"] for row in levels)


def test_height_chain_adds_up_to_floor_to_ceiling():
    intents = elevation_dimensions(parse_elevation(BATH), scale=20)
    chain = [i for i in intents if i.feature == "right:levels"]
    assert sum(abs(i.p2[1] - i.p1[1]) for i in chain) == pytest.approx(2700)
    # the position chain: jambs, the socket centre (x 2400) and the service axis
    xs = sorted({i.p1[0] for i in intents if i.feature == "bottom:positions"})
    assert xs == [0, 200, 1000, 1800, 2400, 2600, 3300]


def test_a_bare_wall_draws_only_the_overall_rows():
    intents = elevation_dimensions(parse_elevation({"length": 3000, "height": 2500}), scale=25)
    assert [i.feature for i in intents] == ["right:overall", "bottom:overall"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"height": 0}, "spec.height: must be greater than 0"),
        ({"colour": "red"}, "unknown field"),
        (
            {"openings": [{"kind": "window", "x": 0, "width": 500, "height": 900}]},
            r"openings\[0\].sill: required",
        ),
        ({"openings": [{"kind": "door", "x": 3000, "width": 900, "height": 2000}]}, "past the end"),
        ({"services": [{"kind": "steam", "x": 1, "y": 1}]}, "is not one of"),
    ],
)
def test_refusals_name_their_path(change, message):
    with pytest.raises(ValueError, match=message):
        parse_elevation({"length": 3600, "height": 2700, **change})


def test_a_fitting_inside_an_opening_is_refused_by_both_ids():
    spec = dict(
        BATH,
        fittings=[
            {"id": "M", "kind": "mirror", "x": 300, "width": 400, "bottom": 1000, "top": 1600}
        ],
    )
    with pytest.raises(ValueError, match="M lies inside opening D1"):
        parse_elevation(spec)


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setenv("AUTOCAD_MCP_BACKEND", "ezdxf")
    async with Client(server.mcp) as connected:
        await connected.call_tool("drawing_new", {})
        yield connected


@pytest.mark.asyncio
async def test_typed_elevation_draws_real_dimensions_on_elev_layers(client):
    result = (
        await client.call_tool(
            "interior_wall_elevation", {"x": 0, "y": 0, "spec": BATH, "lang": "pl"}
        )
    ).structured_content
    assert result["source"] == "typed"
    assert result["counts"]["dimension"] == len(result["dimension_handles"]) > 2
    assert "A-WALL-E-N" not in result["layers"]
    entities = (
        await client.call_tool("entity_list", {"layer_filter": "A-DIMEN-T-N"})
    ).structured_content
    assert str(entities).count("DIMENSION") >= result["counts"]["dimension"]


@pytest.mark.asyncio
async def test_from_plan_reads_the_clear_face_and_mirrors_the_openings(client):
    await client.call_tool(
        "arch_wall",
        {
            "walls": [
                {
                    "id": "W1",
                    "axis": [[0, 0], [4000, 0], [4000, 3500], [0, 3500]],
                    "thickness": 200,
                    "closed": True,
                },
                {"id": "W2", "axis": [[2000, 0], [2000, 3500]], "thickness": 100},
            ]
        },
    )
    await client.call_tool(
        "arch_opening",
        {
            "wall": "W1",
            "offset": 2300,
            "width": 1200,
            "kind": "window",
            "sill": 900,
            "height": 1200,
        },
    )
    args = {
        "x": 6000,
        "y": 0,
        "spec": {"height": 2600},
        "from_plan": {"wall": "W1", "side": "left", "at": 3000},
    }
    result = (await client.call_tool("interior_wall_elevation", args)).structured_content
    # inner face: from W2's face (2050) to the corner's inner face (3900)
    assert result["length"] == pytest.approx(1850)
    assert result["plan"]["mirrored"] is True
    # seen from inside the axis runs right to left: the far jamb (3500) is 400 from the corner face
    assert result["openings"][0]["x"] == pytest.approx(400)
    assert result["openings"][0]["head"] == pytest.approx(2100)
    # an opening length typed beside from_plan is refused, nothing drawn
    with pytest.raises(ToolError, match="spec.length: read from the plan"):
        await client.call_tool(
            "interior_wall_elevation", {**args, "spec": {"height": 2600, "length": 1000}}
        )


@pytest.mark.asyncio
async def test_a_plan_door_without_a_height_is_refused_until_given(client):
    await client.call_tool(
        "arch_wall", {"walls": [{"id": "W", "axis": [[0, 0], [4000, 0]], "thickness": 200}]}
    )
    await client.call_tool(
        "arch_opening", {"wall": "W", "offset": 1000, "width": 900, "kind": "door", "tag": "D9"}
    )
    args = {
        "x": 0,
        "y": 5000,
        "spec": {"height": 2600},
        "from_plan": {"wall": "W", "side": "right"},
    }
    with pytest.raises(ToolError, match="opening D9 has no height"):
        await client.call_tool("interior_wall_elevation", args)
    args["from_plan"]["heights"] = {"D9": {"height": 2000}}
    result = (await client.call_tool("interior_wall_elevation", args)).structured_content
    assert result["openings"][0]["head"] == pytest.approx(2000)


TILED = dict(
    BATH,
    zones=[
        {"id": "T", "label": "tile 60x30", "lay": "grid", "plank_length": 600,
         "plank_width": 300, "gap": 0, "bottom": 80, "top": 2100},
    ],
)  # fmt: skip


def test_a_tiled_zone_is_its_band_minus_openings_and_fitted_units():
    from engineering.arch.elevation import zone_fill

    model = parse_elevation(TILED)
    # band 3600 x 2020, minus the door (800 x 1970), the window (700 x 600) and the
    # 40 mm worktop (1200 x 40); the socket sits on the tiles
    expected = 3600 * 2020 - 800 * 1970 - 700 * 600 - 1200 * 40
    fill = zone_fill(model, 0, scale=20)
    assert fill["area"] == pytest.approx(expected)
    assert fill["lay"]["takeoff"]["floor_area_m2"] == pytest.approx(expected / 1e6, abs=1e-4)
    assert fill["cutouts"] == ["D1", "O1", "BLAT"]
    levels = [row["level"] for row in elevation_levels(model)]
    assert 2100 in levels and 80 in levels


def test_a_later_zone_covers_an_earlier_one_and_its_edges_join_the_position_chain():
    from engineering.arch.elevation import zone_fill

    spec = dict(
        BATH,
        zones=[
            {"id": "PAINT", "label": "paint", "pattern": "SOLID"},
            {"id": "TILE", "label": "tile", "joints": [300, 600], "x": 1100, "width": 1500,
             "top": 1400},
        ],
    )  # fmt: skip
    model = parse_elevation(spec)
    paint, tile = zone_fill(model, 0, scale=20), zone_fill(model, 1, scale=20)
    # the tile band 1100..2600 x 0..1400 minus the 40 mm worktop (1300..2500 at 810..850)
    assert tile["area"] == pytest.approx(1500 * 1400 - 1200 * 40)
    assert tile["cutouts"] == ["BLAT"]
    # the paint is the wall minus both openings, the worktop outside the tiles and the tiles
    wall = 3600 * 2700 - 800 * 2050 - 700 * 900
    assert paint["area"] == pytest.approx(wall - 1500 * 1400)
    assert paint["cutouts"] == ["D1", "O1", "BLAT", "TILE"]
    xs = {i.p1[0] for i in elevation_dimensions(model, scale=20) if i.feature == "bottom:positions"}
    assert {1100, 2600} <= xs


def test_overlapping_fittings_follow_list_order_the_later_one_hides_the_earlier():
    from engineering.arch.elevation import elevation_prims
    from engineering.mech.primitives import Line, Poly

    spec = {
        "length": 2400,
        "height": 2850,
        "fittings": [
            {"id": "blat", "kind": "worktop", "x": 510, "width": 1680, "top": 750},
            {"id": "p1", "kind": "panel", "x": 510, "width": 150, "bottom": 0, "top": 2850},
        ],
    }
    prims = elevation_prims(parse_elevation(spec), scale=20)

    def inside(p):
        return 510 + 1e-6 < p[0] < 660 - 1e-6 and 1e-6 < p[1] < 2850 - 1e-6

    segments = []
    for prim in prims:
        if isinstance(prim, Line):
            segments.append((prim.p1, prim.p2))
        elif isinstance(prim, Poly):
            pts = prim.points
            segments += [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    mids = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in segments]
    assert not any(inside(m) for m in mids)  # nothing of the worktop shows through the panel
    # the worktop's top edge still runs from the panel to its right end
    assert any(a[1] == b[1] == 750 and {a[0], b[0]} == {660, 2190} for a, b in segments)
    # reversed, the worktop lies over the panel: the panel's edges break at it
    flipped = dict(spec, fittings=spec["fittings"][::-1])
    lines = [p for p in elevation_prims(parse_elevation(flipped), scale=20) if isinstance(p, Line)]
    assert any(p.p1[0] == p.p2[0] == 660 and max(p.p1[1], p.p2[1]) == 710 for p in lines)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"zones": [{"lay": "grid", "plank_length": 600, "plank_width": 300,
                     "cut_around": ["G9"]}]}, r"zones\[0\].cut_around\[0\]: 'G9' is not a fitting"),
        ({"zones": [{"lay": "grid", "plank_length": 600, "plank_width": 0}]},
         r"zones\[0\].plank_width: must be positive"),
        ({"zones": [{"label": "x"}]}, r"zones\[0\]: give pattern, joints, lay or edge"),
        ({"zones": [{"pattern": "ANSI31"}, {"pattern": "SOLID"}]},
         r"zones\[0\]: nothing of it is left once D1, O1, BLAT, Z2 cut it"),
        ({"zones": [{"pattern": "nope"}]}, r"zones\[0\].pattern: invalid_args"),
        ({"zones": [{"joints": [20, 0], "gap": 2}]}, r"zones\[0\].gap: only with lay"),
        ({"fittings": [{"kind": "shelf", "x": 0, "width": 600, "top": 1200}]},
         r"fittings\[0\].thickness: required for a shelf"),
    ],
)  # fmt: skip
def test_zone_and_shelf_refusals_name_their_path(change, message):
    with pytest.raises(ValueError, match=message):
        parse_elevation({**BATH, **change})


@pytest.mark.asyncio
async def test_dry_run_draws_nothing_and_the_tiles_land_on_their_layer(client):
    args = {"x": 0, "y": 0, "spec": TILED}
    dry = (
        await client.call_tool("interior_wall_elevation", {**args, "dry_run": True})
    ).structured_content
    assert dry["dry_run"] is True and dry["handles"] == [] and dry["zones"][0]["handles"] == {}
    assert (await client.call_tool("entity_list", {})).structured_content["result"] == []
    drawn = (await client.call_tool("interior_wall_elevation", args)).structured_content
    zone = drawn["zones"][0]
    assert zone["lay"]["takeoff"] == dry["zones"][0]["lay"]["takeoff"]
    assert zone["cutouts"] == ["D1", "O1", "BLAT"] and zone["layer"] == "A-ELEV-TILE-E-N"
    assert [r["value"] for r in drawn["dimensions"]] == pytest.approx(
        [r["value"] for r in dry["dimensions"]]
    )
    tiles = (
        await client.call_tool("entity_list", {"layer_filter": "A-ELEV-TILE-E-N"})
    ).structured_content
    assert {row["handle"] for row in tiles["result"]} == set(zone["handles"]["pieces"])
    assert zone["handles"]["pieces"]


FINISHES = dict(
    BATH,
    zones=[
        {"id": "PAINT", "label": "paint", "pattern": "SOLID", "color": 254},
        {"id": "FLUTE", "label": "fluted oak", "joints": [20, 0], "x": 1300, "width": 1200,
         "bottom": 850, "top": 2400},
    ],
)  # fmt: skip


@pytest.mark.asyncio
async def test_zones_are_legend_rows_and_flutes_are_one_vertical_family(backend):
    from engineering.annotation_record import read_annotation
    from engineering.arch.draw import draw_interior_elevation
    from engineering.arch.legend import draw_legend

    result = await draw_interior_elevation(backend, FINISHES, at=(0, 0))
    paint, flute = result["zones"]
    assert flute["area_m2"] == pytest.approx(1.2 * 1.55)
    hatch = backend._doc.entitydb[flute["handles"]["joints"]]
    # [20, 0]: the across family dropped, vertical lines 20 mm apart only
    lines = hatch.pattern.lines
    assert len(lines) == 1 and lines[0].angle == pytest.approx(90)
    assert abs(lines[0].offset[0]) == pytest.approx(20)
    assert read_annotation(hatch)["legend"] == "fluted oak"
    solid = backend._doc.entitydb[paint["handles"]["pattern"]]
    assert solid.dxf.solid_fill and solid.dxf.color == 254
    legend = await draw_legend(backend, at=(9000, 0), only=["paint", "fluted oak"])
    assert sorted((r["label"], r["count"]) for r in legend["rows"]) == [
        ("fluted oak", 1),
        ("paint", 1),
    ]


@pytest.mark.asyncio
async def test_dimensions_none_draws_none_and_cm_uses_the_arch_cm_style(client):
    spec = dict(BATH, dimensions="none")
    off = (
        await client.call_tool("interior_wall_elevation", {"x": 0, "y": 0, "spec": spec})
    ).structured_content
    assert off["dimension_handles"] == [] and off["dimensions"] == []
    with pytest.raises(ToolError, match="dimstyle_create"):
        await client.call_tool(
            "interior_wall_elevation", {"x": 0, "y": 5000, "spec": dict(BATH, dimensions="cm")}
        )
    await client.call_tool("dimstyle_create", {"name": "ARCH", "preset": "arch-cm"})
    before = (
        await client.call_tool("system_get_variable", {"name": "DIMSTYLE"})
    ).structured_content
    cm = (
        await client.call_tool(
            "interior_wall_elevation", {"x": 0, "y": 10000, "spec": dict(BATH, dimensions="cm")}
        )
    ).structured_content
    assert cm["dimstyle"] == "ARCH"
    dim = await client.call_tool("entity_get", {"handle": cm["dimension_handles"][-1]})
    assert "ARCH" in str(dim.structured_content)
    after = (await client.call_tool("system_get_variable", {"name": "DIMSTYLE"})).structured_content
    assert after == before  # the current style is put back
