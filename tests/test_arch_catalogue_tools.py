"""arch_catalogue_list / arch_catalogue_insert through a real MCP client."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.exceptions import ToolError

import server
from engineering.arch.catalogue import SIZE_BASIS
from engineering.arch.layers import ARCH_ROLE_LAYER

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setenv("AUTOCAD_MCP_BACKEND", "ezdxf")
    async with Client(server.mcp) as connected:
        await connected.call_tool("drawing_new", {})
        yield connected


async def test_list_searches_in_turkish_and_states_the_size_basis(client):
    result = (
        await client.call_tool("arch_catalogue_list", {"query": "yatak", "lang": "tr"})
    ).structured_content
    assert result["count"] == 2
    assert [row["name"] for row in result["items"]] == ["single_bed", "double_bed"]
    assert result["items"][0]["label"] == "tek kişilik yatak"
    assert result["size_basis"] == SIZE_BASIS
    assert result["families"] == ["furniture", "sanitary"]
    # round 3: shared values once, only the asked language's label per row
    assert result["frame"]["front_dir"] == [0, 1]
    assert not {"size_basis", "label_en", "label_tr", "origin"} & set(result["items"][0])


async def test_insert_takes_many_items_and_defines_each_block_once(client):
    result = (
        await client.call_tool(
            "arch_catalogue_insert",
            {
                "items": [
                    {"name": "wc", "x": 0.0, "y": 0.0},
                    {"name": "wc", "x": 1000.0, "y": 0.0, "rotation": 180.0},
                    {"name": "wardrob", "x": 0.0, "y": 0.0},
                ]
            },
        )
    ).structured_content
    assert [row["block"] for row in result["inserted"]] == ["ARCH_WC", "ARCH_WC"]
    assert result["defined"] == ["ARCH_WC"]
    assert result["layers_created"] == [ARCH_ROLE_LAYER["sanitary"]]
    (refused,) = result["refused"]
    assert refused["i"] == 2 and "nearest: wardrobe" in refused["message"]


async def test_every_item_refused_is_invalid_args_naming_them(client):
    raw = await client.call_tool(
        "arch_catalogue_insert",
        {"items": [{"name": "wardrob", "x": 0.0, "y": 0.0}]},
        raise_on_error=False,
    )
    assert raw.is_error and raw.structured_content["kind"] == "invalid_args"
    assert "items[0]" in raw.structured_content["message"]
    assert "nearest: wardrobe" in raw.structured_content["message"]
    with pytest.raises(ToolError, match="furniture, sanitary"):
        await client.call_tool("arch_catalogue_list", {"family": "garden"})
