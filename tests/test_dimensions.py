"""Tests for dimension creation via ezdxf backend."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_dimension_linear(backend):
    info = await backend.dimension_linear(0, 0, 100, 0, 50, 20, rotation=0)
    assert info.handle
    assert info.type == "DIMENSION"


async def test_dimension_linear_tool_places_text_at_the_given_point():
    # the batch spec took text_at from the contract; the tool must take it too
    from fastmcp import Client

    import server

    async with Client(server.mcp) as client:
        await client.call_tool("drawing_new", {})
        args = {"x1": 0, "y1": 0, "x2": 100, "y2": 0, "dim_x": 50, "dim_y": -20}
        made = await client.call_tool("dimension_linear", args | {"text_at": [80, -20]})
        got = await client.call_tool("entity_get", {"handle": made.structured_content["handle"]})
    props = got.structured_content["properties"]
    assert props["text_position"] == pytest.approx([80, -20])
    assert props["defpoint"] == pytest.approx([0, -20])  # the dimension line did not move


async def test_dimension_aligned(backend):
    info = await backend.dimension_aligned(0, 0, 100, 50, 50, 60)
    assert info.handle
    assert info.type == "DIMENSION"


async def test_dimension_angular(backend):
    info = await backend.dimension_angular(0, 0, 100, 0, 0, 100, 50, 50)
    assert info.handle
    assert info.type == "DIMENSION"


async def test_dimension_radius(backend):
    await backend.entity_create_circle(50, 50, 30)
    info = await backend.dimension_radius(50, 50, 80, 50, leader_length=15)
    assert info.handle
    assert info.type == "DIMENSION"


async def test_dimension_diameter(backend):
    info = await backend.dimension_diameter(20, 50, 80, 50, leader_length=10)
    assert info.handle
    assert info.type == "DIMENSION"


# ── the number on the dimension must be the number on the drawing ───────────
#
# `add_diameter_dim`/`add_radius_dim` take `mpoint` as a point ON the circle;
# `leader_length` is where the *text* sits. Passing `centre + (radius + leader)`
# as `mpoint` makes ezdxf measure to that point, so every diameter came out
# `2 x leader_length` too big and every radius `leader_length` too big -- at
# DEFAULT settings, silently, on a drawing whose whole purpose is the number.
# The live COM backend was never wrong here: AddDimDiametric takes the two
# chord points and the leader length as separate arguments.


async def test_a_diameter_dimension_measures_the_diameter(backend):
    """Two ends of a diameter of a true 40 mm circle."""
    info = await backend.dimension_diameter(-20, 0, 20, 0)

    dim = backend._doc.entitydb.get(info.handle)
    assert dim.get_measurement() == pytest.approx(40.0)


async def test_the_text_stands_beyond_the_first_chord_point_as_activex_puts_it(backend):
    """AddDimDiametric(ChordPoint, FarChordPoint, LeaderLength), measured on
    AutoCAD 2026: a 50 mm diameter from (x, 170) to (x, 220) with leader 10
    had its text at (x, 160) - beyond the FIRST point. Both engines follow it."""
    info = await backend.dimension_diameter(0, 25, 0, -25, leader_length=10.0)
    dim = backend._doc.entitydb.get(info.handle)
    text = dim.dxf.text_midpoint
    assert text.y == pytest.approx(35.0)
    assert text.x == pytest.approx(0.0)


async def test_the_leader_length_moves_the_text_not_the_measurement(backend):
    """The regression in one assertion: leader_length must not be measurable."""
    near = await backend.dimension_diameter(-20, 0, 20, 0, leader_length=5.0)
    far = await backend.dimension_diameter(-20, 0, 20, 0, leader_length=40.0)

    measurements = [
        backend._doc.entitydb.get(near.handle).get_measurement(),
        backend._doc.entitydb.get(far.handle).get_measurement(),
    ]
    assert measurements == pytest.approx([40.0, 40.0])


async def test_a_radius_dimension_measures_the_radius(backend):
    info = await backend.dimension_radius(0, 0, 20, 0)

    dim = backend._doc.entitydb.get(info.handle)
    assert dim.get_measurement() == pytest.approx(20.0)


async def test_a_radius_measurement_ignores_the_leader_too(backend):
    near = await backend.dimension_radius(0, 0, 20, 0, leader_length=5.0)
    far = await backend.dimension_radius(0, 0, 20, 0, leader_length=40.0)

    measurements = [
        backend._doc.entitydb.get(near.handle).get_measurement(),
        backend._doc.entitydb.get(far.handle).get_measurement(),
    ]
    assert measurements == pytest.approx([20.0, 20.0])


async def test_an_off_axis_diameter_is_measured_from_its_own_chord(backend):
    """The two points are a diameter at any angle, not just horizontal."""
    info = await backend.dimension_diameter(0, 0, 30, 40, leader_length=12.0)

    dim = backend._doc.entitydb.get(info.handle)
    assert dim.get_measurement() == pytest.approx(50.0)


@pytest.mark.parametrize("side", [1, -1])
async def test_dimension_aligned_line_passes_through_the_given_point(backend, side):
    # P1 -> P2 is 100 long along (0.6, 0.8); its left normal is (-0.8, 0.6). The
    # point 15 to one side of the midpoint puts the dimension line 15 off the
    # segment, not at the point's distance from P1 (hypot(18, 49) = 52.2).
    dim_x, dim_y = 40 - side * 12, 60 + side * 9
    info = await backend.dimension_aligned(10, 20, 70, 100, dim_x, dim_y)
    props = info.properties
    assert props["defpoint"] == pytest.approx([10 - side * 12, 20 + side * 9])
    assert props["offset"] == pytest.approx(side * 15)
    assert props["measurement"] == pytest.approx(100)
    assert props["text"] == "100"


async def test_dimension_linear_reports_the_rounded_text(backend):
    await backend.system_set_variable("DIMDEC", 0)
    info = await backend.dimension_linear(0, 0, 132.5, 0, 50, -20, rotation=0)
    assert info.properties["measurement"] == pytest.approx(132.5)
    assert info.properties["text"] in ("132", "133")  # what the sheet shows, not the value
    assert info.properties["offset"] == pytest.approx(-20)


async def test_the_text_side_follows_the_style_not_the_point_order():
    # round 6: given in reverse, ezdxf put a horizontal dimension's text below its
    # line and a vertical one's to the right; with DIMTAD 1 it stands above /
    # left (reading from the bottom), whichever point comes first
    from fastmcp import Client

    import server

    cases = {
        "vertical": ((6500, 2000), (6500, 3000), (6600, 2500), 90),
        "horizontal": ((2000, 500), (3000, 500), (2500, 600), 0),
        "aligned": ((0, 0), (400, 300), (-60, 80), None),
    }
    async with Client(server.mcp) as client:
        await client.call_tool("drawing_new", {})
        await client.call_tool("drawing_apply_standard", {"standard": "iso"})
        seen = {}
        for name, (a, b, at, rotation) in cases.items():
            for p, q in ((a, b), (b, a)):
                tool = "dimension_aligned" if rotation is None else "dimension_linear"
                args = {
                    "x1": p[0],
                    "y1": p[1],
                    "x2": q[0],
                    "y2": q[1],
                    "dim_x": at[0],
                    "dim_y": at[1],
                }
                if rotation is not None:
                    args["rotation"] = rotation
                made = await client.call_tool(tool, args)
                got = await client.call_tool(
                    "entity_get", {"handle": made.structured_content["handle"]}
                )
                seen.setdefault(name, []).append(
                    got.structured_content["properties"]["text_position"]
                )
    for name, (first, second) in seen.items():
        assert first == pytest.approx(second), name
    assert seen["vertical"][0][0] < 6600  # left of the dimension line
    assert seen["horizontal"][0][1] > 600  # above it
