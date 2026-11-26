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
        (
            {
                "fittings": [
                    {"kind": "socket", "x": 500, "y": 1000},
                    {"kind": "switch", "x": 520, "y": 1010},
                ]
            },
            "overlap",
        ),
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
    tiling={"tile_length": 600, "tile_width": 300, "joint": 0, "bottom": 80, "top": 2100},
)


def test_tiling_area_is_the_band_minus_openings_and_fitted_units():
    from engineering.arch.elevation import tile_pieces

    model = parse_elevation(TILED)
    # band 3600 x 2020, minus the door (800 x 1970), the window (700 x 600) and the
    # 40 mm worktop (1200 x 40); the socket sits on the tiles
    expected = 3600 * 2020 - 800 * 1970 - 700 * 600 - 1200 * 40
    takeoff = tile_pieces(model.tiling)["takeoff"]
    assert takeoff["floor_area_m2"] == pytest.approx(expected / 1e6, abs=1e-4)
    assert [c["label"] for c in model.tiling.layout["cutouts"]] == ["D1", "O1", "BLAT"]
    levels = [row["level"] for row in elevation_levels(model)]
    assert 2100 in levels and 80 in levels


def test_tiling_refusals_name_the_tiling_path():
    with pytest.raises(ValueError, match=r"spec.tiling.cut_around\[0\]: 'G9' is not a fitting"):
        parse_elevation(
            dict(BATH, tiling={"tile_length": 600, "tile_width": 300, "cut_around": ["G9"]})
        )
    with pytest.raises(ValueError, match="spec.tiling.tile_width: must be positive"):
        parse_elevation(dict(BATH, tiling={"tile_length": 600, "tile_width": 0}))


@pytest.mark.asyncio
async def test_dry_run_draws_nothing_and_the_tiles_land_on_their_layer(client):
    args = {"x": 0, "y": 0, "spec": TILED}
    dry = (
        await client.call_tool("interior_wall_elevation", {**args, "dry_run": True})
    ).structured_content
    assert dry["dry_run"] is True and dry["handles"] == [] and dry["tiling"]["handles"] == []
    assert (await client.call_tool("entity_list", {})).structured_content["result"] == []
    drawn = (await client.call_tool("interior_wall_elevation", args)).structured_content
    assert drawn["tiling"]["takeoff"] == dry["tiling"]["takeoff"]
    assert drawn["tiling"]["cutouts"] == ["D1", "O1", "BLAT"]
    # the wall tiling answers with the floor layout's keys, plus its band
    floor = (
        await client.call_tool(
            "interior_tile_layout",
            {
                "outline": [[0, 0], [1200, 0], [1200, 900], [0, 900]],
                "tile_length": 600,
                "tile_width": 300,
                "dry_run": True,
            },
        )
    ).structured_content
    tile_keys = {
        "layer",
        "start",
        "pattern",
        "cutouts",
        "takeoff",
        "smallest_cut_share",
        "handles",
        "count",
    }
    assert tile_keys <= set(floor) and set(drawn["tiling"]) == tile_keys | {"bottom", "top"}
    assert [r["value"] for r in drawn["dimensions"]] == pytest.approx(
        [r["value"] for r in dry["dimensions"]]
    )
    tiles = (
        await client.call_tool("entity_list", {"layer_filter": "A-ELEV-TILE-E-N"})
    ).structured_content
    assert {row["handle"] for row in tiles["result"]} == set(drawn["tiling"]["handles"])
    assert drawn["tiling"]["handles"]
