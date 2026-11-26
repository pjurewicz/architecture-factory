"""interior_tile_layout: the cut grid and its takeoff, with hand-derived expectations."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.exceptions import ToolError

import server
from engineering.arch.tiles import layout

ROOM = [[0, 0], [4000, 0], [4000, 3000], [0, 3000]]


def test_grid_without_joint_counts_full_and_cut_tiles():
    # 600 x 300 from (0, 0): 6 full columns + a 400 mm strip, 10 rows exactly.
    t = layout(ROOM, length=600, width=300, joint=0)["takeoff"]
    # The 200 mm offcut of a 400 mm strip piece fits no other piece: no reuse.
    assert (t["full"], t["cut"], t["tiles"], t["tiles_no_reuse"]) == (60, 10, 70, 70)
    assert t["floor_area_m2"] == 12.0 and t["tiled_area_m2"] == 12.0
    assert t["order_tiles"] == 77  # ceil(70 x 1.10)


def test_running_bond_reuses_row_end_offcuts():
    # Rows 0, 2, ..: 6 full + one 400 cut; rows 1, 3, ..: a 300 cut, 6 full, a 100 cut.
    # Five 400 cuts leave 197 mm and five 300 cuts 297 mm (3 mm kerf): the five
    # 100 mm pieces come out of those offcuts, so the cuts take 10 tiles, not 15.
    t = layout(ROOM, length=600, width=300, joint=0, pattern="running_bond")["takeoff"]
    assert (t["full"], t["cut"], t["cut_tiles"], t["tiles"]) == (60, 15, 10, 70)
    assert t["tiles_no_reuse"] == 75 and t["order_tiles"] == 77
    off = layout(ROOM, length=600, width=300, joint=0, pattern="running_bond", offcut_reuse=False)
    assert (off["takeoff"]["tiles"], off["takeoff"]["order_tiles"]) == (75, 83)


def test_kerf_can_make_an_offcut_too_short():
    # 1000 x 600 room, 600 x 300 tiles, half bond -> row 0: 600 full + a 400 cut,
    # row 1: a 300 cut + 600 full + a 100 cut. The 100 piece fits the 600 - 400 -
    # kerf offcut at kerf 0 (2 tiles for the cuts); at kerf 250 the offcuts are
    # -50 and 50 mm and it needs its own tile (3).
    room = [[0, 0], [1000, 0], [1000, 600], [0, 600]]
    args = {"length": 600, "width": 300, "joint": 0, "pattern": "running_bond"}
    assert layout(room, kerf=0, **args)["takeoff"]["cut_tiles"] == 2
    assert layout(room, kerf=250, **args)["takeoff"]["cut_tiles"] == 3


def test_a_cutout_is_left_untiled_and_its_area_taken_off():
    bath = {"points": [[0, 0], [900, 0], [900, 1700], [0, 1700]], "label": "bath"}
    r = layout(ROOM, length=600, width=300, joint=3, cutouts=[bath])
    assert r["takeoff"]["floor_area_m2"] == pytest.approx(12.0 - 0.9 * 1.7)
    assert r["cutout_labels"] == ["bath"]
    for piece in r["pieces"]:  # no piece reaches into the bath
        xs = [p[0] for loop in piece.loops for p in loop]
        ys = [p[1] for loop in piece.loops for p in loop]
        assert not (max(xs) <= 900 + 1e-6 and max(ys) <= 1700 + 1e-6)


def test_herringbone_at_45_degrees_on_an_l_room_tiles_the_exact_floor_without_joints():
    # Joint 0: the pieces must sum to the region exactly (no overlap, no gap).
    floor = [[0, 0], [5000, 0], [5000, 2000], [2000, 2000], [2000, 4000], [0, 4000]]
    notch = {"points": [[3000, 0], [4000, 0], [4000, 1500], [3500, 1500], [3500, 500], [3000, 500]]}
    r = layout(
        floor,
        length=600,
        width=100,
        joint=0,
        pattern="herringbone",
        angle_deg=45,
        start=(37, 11),
        cutouts=[notch],
    )
    expected = (5000 * 2000 + 2000 * 2000 - (1000 * 500 + 500 * 1000)) / 1e6
    assert r["takeoff"]["floor_area_m2"] == pytest.approx(expected)
    assert sum(p.area for p in r["pieces"]) / 1e6 == pytest.approx(expected, rel=1e-9)
    t = r["takeoff"]
    # reuse never orders less material than the pieces cover, nor more than no reuse
    assert t["tiles"] * 600 * 100 / 1e6 >= expected
    assert t["full"] < t["tiles"] <= t["tiles_no_reuse"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"pattern": "basket"}, "pattern"),
        ({"pattern": "herringbone", "width": 600}, "longer than it is wide"),
        ({"joint": -1}, "joint"),
        ({"kerf": -1}, "kerf"),
        ({"bond": 1.0}, "bond"),
        ({"outline": [[0, 0], [30, 10], [30, 0], [0, 10], [-5, 5]]}, "cross"),
        ({"length": 1, "width": 1}, "tiles would cover"),
    ],
)
def test_refusals_name_their_cause(kwargs, match):
    args = {"outline": ROOM, "length": 600, "width": 300} | kwargs
    with pytest.raises(ValueError, match=match):
        layout(**args)


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setenv("AUTOCAD_MCP_BACKEND", "ezdxf")
    async with Client(server.mcp) as connected:
        await connected.call_tool("drawing_new", {})
        yield connected


@pytest.mark.asyncio
async def test_the_tool_draws_one_closed_polyline_per_piece_on_the_floor_layer(client):
    args = {"outline": ROOM, "tile_length": 600, "tile_width": 300, "joint": 0}
    dry = (
        await client.call_tool("interior_tile_layout", args | {"dry_run": True})
    ).structured_content
    assert "handles" not in dry and dry["takeoff"]["tiles"] == 70
    drawn = (await client.call_tool("interior_tile_layout", args)).structured_content
    assert drawn["count"] == 70 and drawn["layer"] == "A-FLOR-E-N"
    listed = (
        await client.call_tool("entity_list", {"layer_filter": "A-FLOR-E-N", "limit": 1000})
    ).structured_content
    rows = listed.get("entities", listed.get("result", listed))
    assert len(rows) == 70
    with pytest.raises(ToolError, match="give both or neither"):
        await client.call_tool("interior_tile_layout", args | {"start_x": 5})


@pytest.mark.asyncio
async def test_tiles_laid_after_the_furniture_go_behind_its_wipeout(backend):
    """The pieces are sent to the back, so a bed inserted first still masks them."""
    from engineering.arch.catalogue import insert_item
    from engineering.arch.draw import draw_tile_layout

    bed = await insert_item(backend, "single_bed", at=(600.0, 300.0))
    drawn = await draw_tile_layout(backend, outline=ROOM, length=600, width=300, joint=0)
    order = [e.dxf.handle for e in backend._doc.modelspace().entities_in_redraw_order()]
    assert order.index(bed["handle"]) > max(order.index(h) for h in drawn["handles"])
