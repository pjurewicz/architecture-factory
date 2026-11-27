"""The default layer of a create call: explicit layer= wins, semantic tools keep
their own layer, primitives follow $CLAYER, then TEXT for annotation (only when
the drawing has it), else "0"."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastmcp import Client

import server

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setenv("AUTOCAD_MCP_BACKEND", "ezdxf")
    async with Client(server.mcp) as connected:
        yield connected


async def _layer(client, tool, args):
    return (await client.call_tool(tool, args)).structured_content["layer"]


async def _layers(client):
    rows = (await client.call_tool("layer_list", {})).structured_content
    rows = rows.get("result", rows) if isinstance(rows, dict) else rows
    return {row["name"] for row in rows}


LINE = {"x1": 0, "y1": 0, "x2": 10, "y2": 0}
TEXT = {"text": "A", "x": 0, "y": 0}
TABLE = {"x": 0, "y": 50, "rows": [["a", "1"]]}


async def _primitives(client, layout=None):
    extra = {"layout": layout} if layout else {}
    return {
        "line": await _layer(client, "entity_create_line", LINE),
        "circle": await _layer(client, "entity_create_circle", {"cx": 0, "cy": 0, "radius": 1}),
        "text": await _layer(client, "entity_create_text", TEXT),
        "mtext": await _layer(client, "entity_create_mtext", {**TEXT, **extra}),
        "table": await _layer(client, "entity_create_table", {**TABLE, **extra}),
    }


async def test_bootstrapped_drawing_puts_annotation_on_text(client):
    await client.call_tool("drawing_new", {})
    await client.call_tool("layout_create", {"name": "S"})
    expected = {"line": "0", "circle": "0", "text": "TEXT", "mtext": "TEXT", "table": "TEXT"}
    assert await _primitives(client) == expected
    assert await _primitives(client, layout="S") == expected


async def test_bare_drawing_uses_layer_0_and_creates_no_text_layer(client):
    await client.call_tool("drawing_new", {"bootstrap": False})
    assert "TEXT" not in await _layers(client)
    assert set((await _primitives(client)).values()) == {"0"}
    assert "TEXT" not in await _layers(client), "no layer is created unasked"


async def test_current_layer_drives_primitives_not_semantic_tools(client):
    await client.call_tool("drawing_new", {})
    await client.call_tool("layout_create", {"name": "S"})
    await client.call_tool("layer_set_current", {"name": "A-FURN"})
    assert set((await _primitives(client)).values()) == {"A-FURN"}
    assert await _layer(client, "entity_create_mtext", {**TEXT, "layout": "S"}) == "A-FURN"
    dim = {"x1": 0, "y1": 0, "x2": 10, "y2": 0, "dim_x": 5, "dim_y": -5}
    assert await _layer(client, "dimension_linear", dim) == "DIM"
    callout = (
        await client.call_tool("callout", {"target": [0, 0], "landing": [10, 10], "text": "x"})
    ).structured_content
    parts = [
        (await client.call_tool("entity_get", {"handle": h})).structured_content["layer"]
        for h in callout["handles"]
    ]
    assert set(parts) == {"DIM"}, "every callout part, its WIPEOUT mask included"


async def test_explicit_layer_always_wins(client):
    await client.call_tool("drawing_new", {})
    await client.call_tool("layer_set_current", {"name": "A-FURN"})
    for tool, args in (
        ("entity_create_line", LINE),
        ("entity_create_mtext", TEXT),
        ("entity_create_table", TABLE),
        ("dimension_linear", {"x1": 0, "y1": 0, "x2": 10, "y2": 0, "dim_x": 5, "dim_y": -5}),
    ):
        assert await _layer(client, tool, {**args, "layer": "MINE"}) == "MINE", tool


async def test_batch_items_follow_the_same_rule(client):
    await client.call_tool("drawing_new", {})
    items = [{"type": "line", **LINE}, {"type": "mtext", **TEXT}]
    first = (
        await client.call_tool("entity_batch_create", {"entities": items, "verbose": True})
    ).structured_content
    assert [e["layer"] for e in first["entities"]] == ["0", "TEXT"]
    await client.call_tool("layer_set_current", {"name": "A-FURN"})
    second = (
        await client.call_tool("entity_batch_create", {"entities": items, "verbose": True})
    ).structured_content
    assert [e["layer"] for e in second["entities"]] == ["A-FURN", "A-FURN"]
