"""entity_get returns HATCH loops and full DIMENSION fields (B16).

The gate: a HATCH's loop area equals analysis_measure_entity's area for the
same handle (islands, nesting, bulges). Triage row W3-T14-19 (a tilted INSERT
read as mirrored) is fixed here, with its own test.
"""

from __future__ import annotations

import math

import pytest
import pytest_asyncio
from fastmcp import Client

import server
from backends.ezdxf_backend import EzdxfBackend


def _square(x0, y0, side):
    return [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)]


def _nested_hatch(backend, style):
    hatch = backend._doc.modelspace().add_hatch()
    hatch.dxf.hatch_style = style
    hatch.paths.add_polyline_path(_square(0, 0, 20), is_closed=True, flags=1)
    hatch.paths.add_polyline_path(_square(4, 4, 12), is_closed=True, flags=0)
    hatch.paths.add_polyline_path(_square(8, 8, 4), is_closed=True, flags=0)
    return hatch.dxf.handle


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("style", "name", "area"),
    [(0, "normal", 400.0 - 144.0 + 16.0), (1, "outer", 400.0 - 144.0), (2, "ignore", 400.0)],
)
async def test_hatch_loop_area_equals_measure_entity_with_islands_and_nesting(
    backend, style, name, area
):
    handle = _nested_hatch(backend, style)
    info = await backend.entity_get(handle)
    measured = await backend.entity_measure(handle)
    props = info.properties
    assert props["hatch_style"] == name
    assert [loop["depth"] for loop in props["loops"]] == [0, 1, 2]
    assert props["area"] == pytest.approx(area, abs=1e-9)
    assert props["area"] == pytest.approx(measured["area"], abs=1e-9)
    assert props["area_exact"] is True and measured["exact"] is True


@pytest.mark.asyncio
async def test_a_bulged_hatch_loop_is_exact_where_measure_states_its_flattening(backend):
    hatch = backend._doc.modelspace().add_hatch()
    hatch.paths.add_polyline_path(
        [(0, 0, 0), (20, 0, 1.0), (20, 10, 0), (0, 10, 0)], is_closed=True, flags=1
    )
    hatch.paths.add_edge_path(flags=16).add_arc((5, 5), 2, 0, 360)
    info = await backend.entity_get(hatch.dxf.handle)
    measured = await backend.entity_measure(hatch.dxf.handle)
    expected = 200.0 + 12.5 * math.pi - 4.0 * math.pi
    assert info.properties["area"] == pytest.approx(expected, abs=1e-9)
    assert info.properties["area_exact"] is True
    # analysis_measure_entity flattens Bezier boundaries and says so: it agrees
    # within the ~0.028 % Bezier error it documents, never better.
    assert measured["exact"] is False
    assert measured["area"] == pytest.approx(expected, rel=3e-4)


@pytest.mark.asyncio
async def test_entity_get_reports_every_dimension_field(backend):
    msp = backend._doc.modelspace()
    dim = msp.add_linear_dim(
        base=(0, 5), p1=(0, 0), p2=(42, 0), dimstyle="Standard", override={"dimlfac": 2.0}
    )
    dim.render()
    dim.dimension.dxf.text = "40"
    info = await backend.entity_get(dim.dimension.dxf.handle)
    assert info.properties["dimension"] == {
        "dimtype": "linear",
        "dimtype_code": 0,
        "defpoints": [[0.0, 5.0], [0.0, 0.0], [42.0, 0.0], None, None],
        "measurement": 42.0,
        "text_override": "40",
        "dimstyle": "Standard",
        "dimlfac": 2.0,
        "angle": 0.0,
    }


@pytest.mark.asyncio
async def test_entity_list_rows_stay_light(backend):
    _nested_hatch(backend, 0)
    (row,) = await backend.entity_list(type_filter="HATCH")
    assert "loops" not in row.properties


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setenv("AUTOCAD_MCP_BACKEND", "ezdxf")
    async with Client(server.mcp) as connected:
        status = (await connected.call_tool("system_status", {})).structured_content or {}
        assert status.get("backend") == "ezdxf", "these tests must never touch live AutoCAD"
        await connected.call_tool("drawing_new", {})
        yield connected


@pytest.mark.asyncio
async def test_the_entity_get_tool_carries_the_loops_over_the_wire(client):
    made = await client.call_tool(
        "entity_create_hatch",
        {"pattern": "ANSI31", "boundary_points": [[0, 0], [10, 0], [10, 10], [0, 10]]},
    )
    handle = made.structured_content["handle"]
    got = (await client.call_tool("entity_get", {"handle": handle})).structured_content
    assert got["properties"]["loop_count"] == 1
    assert got["properties"]["area"] == pytest.approx(100.0, abs=1e-9)


# -- triage W3-T14-19: a tilted INSERT with a negative-z normal is not "mirrored" --


@pytest.mark.asyncio
async def test_ezdxf_insert_is_mirrored_only_in_a_flat_frame(backend):
    msp = backend._doc.modelspace()
    backend._doc.blocks.new("B").add_line((0, 0), (1, 0))
    flat = msp.add_blockref("B", (0, 0))
    flat.dxf.extrusion = (0.0, 0.0, -1.0)
    tilted = msp.add_blockref("B", (5, 0))
    tilted.dxf.extrusion = (0.5, 0.0, -math.sqrt(3.0) / 2.0)
    flat_props = (await backend.entity_get(flat.dxf.handle)).properties
    tilted_props = (await backend.entity_get(tilted.dxf.handle)).properties
    assert flat_props["mirrored"] is True
    assert tilted_props["mirrored"] is False
    assert tilted_props["plane_normal"] == pytest.approx([0.5, 0.0, -math.sqrt(3.0) / 2.0])


# -- the capability is declared ------------------------------------------------


def test_hatch_loops_are_declared_headless():
    features = EzdxfBackend().capabilities().features
    assert features["hatch_loops"].supported is True
    assert "hatch_boundaries_not_reported" not in (features["ocs_normalized"].reason or "")
