"""In-place edit primitives (v1.8.0, plan item 1.8-06).

A polyline's vertices, a label's position and a polyline's arcs change without
the entity being deleted and redrawn: the handle, the layer and the XDATA a
generator wrote stay where they were. That is the whole point - a revision that
deletes and redraws loses every reference anybody kept to the old handle.

The tests run on the real headless engine.
"""

from __future__ import annotations

import math

import pytest
import pytest_asyncio
from fastmcp import Client

import server
from backends.base import UnsupportedCapabilityError

pytestmark = pytest.mark.asyncio

FOUR = [[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0]]
SIX = [[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [20.0, 30.0], [0.0, 20.0], [0.0, 10.0]]
SIX_BULGES = [1.0, 0.0, 0.5, 0.0, 0.0, 0.0]
TAG = {"ACADMCP_MECH": ['{"feature":"","kind":"feature","part":"P","v":1,"view":"front"}']}


def _arc_length(p1, p2, bulge: float) -> float:
    chord = math.dist(p1, p2)
    if bulge == 0.0:
        return chord
    sweep = 4.0 * math.atan(abs(bulge))
    radius = chord / (2.0 * math.sin(sweep / 2.0))
    return radius * sweep


def _perimeter(points, bulges, closed=True) -> float:
    pairs = list(zip(points, points[1:] + (points[:1] if closed else []), strict=False))
    return sum(_arc_length(a, b, bulges[i]) for i, (a, b) in enumerate(pairs))


def _close(a, b, tol=1e-9) -> bool:
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b, strict=True))


# ── headless: entity_set_vertices ────────────────────────────────────────────


async def test_four_to_six_vertices_with_two_bulges_keeps_handle_layer_and_xdata(backend):
    pline = await backend.entity_create_polyline(FOUR, True, layer="GEOMETRY", xdata=TAG)
    updated = await backend.entity_set_vertices(pline.handle, SIX, SIX_BULGES)

    assert updated.handle == pline.handle
    assert updated.layer == "GEOMETRY"
    assert [list(p) for p in updated.properties["points"]] == SIX
    assert updated.properties["bulges"] == SIX_BULGES
    assert updated.properties["closed"] is True
    xdata = await backend.entity_get_xdata(pline.handle, "ACADMCP_MECH")
    assert xdata["xdata"]["ACADMCP_MECH"] == TAG["ACADMCP_MECH"]

    measured = await backend.entity_measure(pline.handle)
    assert math.isclose(measured["perimeter"], _perimeter(SIX, SIX_BULGES), rel_tol=1e-9)
    assert measured["exact"] is True
    got = await backend.entity_get(pline.handle)
    assert math.isclose(got.properties["length"], _perimeter(SIX, SIX_BULGES), rel_tol=1e-6)


async def test_closed_is_applied_only_when_given(backend):
    pline = await backend.entity_create_polyline(FOUR, True)
    kept = await backend.entity_set_vertices(pline.handle, FOUR[:3])
    assert kept.properties["closed"] is True
    opened = await backend.entity_set_vertices(pline.handle, FOUR, closed=False)
    assert opened.properties["closed"] is False


async def test_omitted_bulges_keep_the_arcs_when_the_count_is_unchanged(backend):
    pline = await backend.entity_create_polyline(SIX, True, bulges=SIX_BULGES)
    moved = [[x + 5.0, y] for x, y in SIX]
    same = await backend.entity_set_vertices(pline.handle, moved)
    assert same.properties["bulges"] == SIX_BULGES
    fewer = await backend.entity_set_vertices(pline.handle, FOUR)
    assert fewer.properties["bulges"] == [0.0] * 4


async def test_a_mirrored_polyline_takes_wcs_points_and_wcs_bulges(backend):
    pline = await backend.entity_create_polyline(FOUR, True)
    entity = backend._doc.entitydb[pline.handle]
    entity.dxf.extrusion = (0.0, 0.0, -1.0)

    updated = await backend.entity_set_vertices(pline.handle, SIX, SIX_BULGES)
    assert all(
        _close(got, want) for got, want in zip(updated.properties["points"], SIX, strict=True)
    )
    assert updated.properties["bulges"] == SIX_BULGES
    stored = [(x, y, b) for x, y, *_rest, b in entity.get_points("xyseb")]
    assert math.isclose(stored[1][0], -40.0)  # the frame stores x unnegated: -WCS x
    assert stored[0][2] == -1.0  # a reflection reverses the stored arc's sense


async def test_a_tilted_polyline_is_refused_and_left_alone(backend):
    pline = await backend.entity_create_polyline(FOUR, True)
    entity = backend._doc.entitydb[pline.handle]
    entity.dxf.extrusion = (0.0, 1.0, 1.0)
    before = [tuple(p) for p in entity.get_points("xyb")]
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        await backend.entity_set_vertices(pline.handle, SIX)
    assert excinfo.value.capability == "ocs_tilted_plane"
    assert [tuple(p) for p in entity.get_points("xyb")] == before


async def test_set_vertices_refuses_what_is_not_a_lightweight_polyline(backend):
    circle = await backend.entity_create_circle(0.0, 0.0, 5.0)
    with pytest.raises(RuntimeError, match="LWPOLYLINE"):
        await backend.entity_set_vertices(circle.handle, FOUR)


@pytest.mark.parametrize(
    ("points", "bulges", "match"),
    [
        ([[0.0, 0.0]], None, "at least 2"),
        (FOUR, [0.0, 1.0], "one per vertex"),
        ([[0.0, "a"], [1.0, 1.0]], None, "expected a number"),
        (FOUR, [0.0, 0.0, float("nan"), 0.0], "finite"),
    ],
)
async def test_bad_vertex_lists_are_refused_before_anything_changes(backend, points, bulges, match):
    pline = await backend.entity_create_polyline(FOUR, True)
    with pytest.raises(ValueError, match=match):
        await backend.entity_set_vertices(pline.handle, points, bulges)
    got = await backend.entity_get(pline.handle)
    assert [list(p) for p in got.properties["points"]] == FOUR


async def test_create_polyline_takes_bulges(backend):
    pline = await backend.entity_create_polyline(SIX, True, layer="GEOMETRY", bulges=SIX_BULGES)
    assert pline.properties["bulges"] == SIX_BULGES
    measured = await backend.entity_measure(pline.handle)
    assert math.isclose(measured["perimeter"], _perimeter(SIX, SIX_BULGES), rel_tol=1e-9)


# ── headless: entity_edit_geometry / entity_edit_text gain positions ─────────


async def test_edit_geometry_rewrites_a_polyline_in_place(backend):
    pline = await backend.entity_create_polyline(FOUR, True, xdata=TAG)
    updated = await backend.entity_edit_geometry(pline.handle, points=SIX, bulges=SIX_BULGES)
    assert updated.handle == pline.handle
    assert updated.properties["bulges"] == SIX_BULGES
    only_bulges = await backend.entity_edit_geometry(pline.handle, bulges=[0.0] * 6)
    assert [list(p) for p in only_bulges.properties["points"]] == SIX
    assert only_bulges.properties["bulges"] == [0.0] * 6


async def test_edit_geometry_refuses_mixed_vocabularies(backend):
    pline = await backend.entity_create_polyline(FOUR, True)
    circle = await backend.entity_create_circle(0.0, 0.0, 5.0)
    with pytest.raises(RuntimeError, match="points"):
        await backend.entity_edit_geometry(pline.handle, cx=1.0)
    with pytest.raises(RuntimeError, match="LWPOLYLINE"):
        await backend.entity_edit_geometry(circle.handle, points=FOUR)
    with pytest.raises(RuntimeError, match="TEXT"):
        await backend.entity_edit_geometry(circle.handle, x=1.0)


async def test_text_moves_keep_the_handle(backend):
    text = await backend.entity_create_text("A", 1.0, 2.0, xdata=TAG)
    moved = await backend.entity_edit_text(text.handle, x=50.0, y=60.0)
    assert moved.handle == text.handle
    assert _close(moved.properties["insertion"], [50.0, 60.0])
    again = await backend.entity_edit_geometry(text.handle, y=70.0)
    assert again.handle == text.handle
    assert _close(again.properties["insertion"], [50.0, 70.0])
    assert (await backend.entity_get_xdata(text.handle, "ACADMCP_MECH"))["xdata"]


async def test_an_aligned_text_moves_with_its_alignment_point(backend):
    text = await backend.entity_create_text("A", 0.0, 0.0)
    entity = backend._doc.entitydb[text.handle]
    entity.dxf.halign = 1
    entity.dxf.align_point = (10.0, 0.0, 0.0)
    await backend.entity_edit_text(text.handle, x=100.0, y=5.0)
    assert _close(tuple(entity.dxf.insert)[:2], (100.0, 5.0))
    assert _close(tuple(entity.dxf.align_point)[:2], (110.0, 5.0))


async def test_mtext_moves_keep_the_handle(backend):
    mtext = await backend.entity_create_mtext("note", 0.0, 0.0)
    moved = await backend.entity_edit_text(mtext.handle, x=-5.0, y=12.5, text="moved")
    assert moved.handle == mtext.handle
    assert _close(moved.properties["insertion"], [-5.0, 12.5])
    assert "moved" in moved.properties["text"]


async def test_a_tilted_text_position_is_refused(backend):
    text = await backend.entity_create_text("A", 0.0, 0.0)
    backend._doc.entitydb[text.handle].dxf.extrusion = (1.0, 0.0, 1.0)
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        await backend.entity_edit_text(text.handle, x=5.0)
    assert excinfo.value.capability == "ocs_tilted_plane"


# ── headless: xdata= on the dimension and table contracts ────────────────────


async def test_the_four_dimension_contracts_write_xdata_on_the_dimension(backend):
    made = [
        await backend.dimension_linear(0, 0, 10, 0, 5, 5, xdata=TAG),
        await backend.dimension_aligned(0, 0, 10, 10, 0, 5, xdata=TAG),
        await backend.dimension_radius(0, 0, 10, 0, xdata=TAG),
        await backend.dimension_diameter(-10, 0, 10, 0, xdata=TAG),
    ]
    for info in made:
        entity = backend._doc.entitydb[info.handle]
        assert entity.dxftype() == "DIMENSION"  # the dimension, never its *D block
        assert (await backend.entity_get_xdata(info.handle, "ACADMCP_MECH"))["xdata"] == TAG


async def test_the_table_contract_writes_xdata_on_every_child(backend):
    table = await backend.entity_create_table(0, 0, [["A", "1"]], ["K", "V"], xdata=TAG)
    for handle in table.properties["child_handles"]:
        assert (await backend.entity_get_xdata(handle, "ACADMCP_MECH"))["xdata"] == TAG


async def test_a_bad_xdata_tag_creates_no_dimension(backend):
    before = await backend.entity_count()
    with pytest.raises(ValueError):
        await backend.dimension_linear(0, 0, 10, 0, 5, 5, xdata={"ACADMCP_MECH": []})
    assert await backend.entity_count() == before


# ── the MCP tools carry the new parameters ───────────────────────────────────


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setenv("AUTOCAD_MCP_BACKEND", "ezdxf")
    async with Client(server.mcp) as connected:
        status = (await connected.call_tool("system_status", {})).structured_content or {}
        assert status.get("backend") == "ezdxf"
        await connected.call_tool("drawing_new", {})
        yield connected


async def test_the_tools_edit_in_place(client):
    created = (
        await client.call_tool(
            "entity_create_polyline", {"points": FOUR, "closed": True, "bulges": [0.0, 1.0, 0, 0]}
        )
    ).structured_content
    assert created["properties"]["bulges"] == [0.0, 1.0, 0.0, 0.0]
    edited = (
        await client.call_tool(
            "entity_edit_geometry",
            {"handle": created["handle"], "points": SIX, "bulges": SIX_BULGES},
        )
    ).structured_content
    assert edited["handle"] == created["handle"]
    assert edited["properties"]["bulges"] == SIX_BULGES

    text = (
        await client.call_tool("entity_create_text", {"text": "A", "x": 0, "y": 0})
    ).structured_content
    moved = (
        await client.call_tool("entity_edit_text", {"handle": text["handle"], "x": 9, "y": 8})
    ).structured_content
    assert moved["handle"] == text["handle"]
    assert _close(moved["properties"]["insertion"], [9.0, 8.0])
