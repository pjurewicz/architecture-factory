"""Tests for layer and block operations via ezdxf backend."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Layer tests
# ---------------------------------------------------------------------------


async def test_layer_create(backend):
    result = await backend.layer_create("WALLS", color=1)
    assert result.name == "WALLS"
    assert result.color == 1


async def test_layer_list(backend):
    await backend.layer_create("A")
    await backend.layer_create("B")
    layers = await backend.layer_list()
    names = [lyr.name for lyr in layers]
    assert "A" in names
    assert "B" in names


async def test_layer_delete(backend):
    await backend.layer_create("TEMP")
    result = await backend.layer_delete("TEMP")
    assert result.get("ok") is True
    layers = await backend.layer_list()
    names = [lyr.name for lyr in layers]
    assert "TEMP" not in names


async def test_layer_set_current(backend):
    await backend.layer_create("CURRENT_TEST")
    result = await backend.layer_set_current("CURRENT_TEST")
    assert result.get("ok") is True


async def test_layer_modify(backend):
    await backend.layer_create("MOD_LAYER", color=1)
    result = await backend.layer_modify("MOD_LAYER", color=3)
    assert result.color == 3


async def test_layer_freeze_thaw(backend):
    await backend.layer_create("FREEZE_TEST")
    result = await backend.layer_freeze("FREEZE_TEST")
    assert result.get("ok") is True
    result = await backend.layer_thaw("FREEZE_TEST")
    assert result.get("ok") is True


async def test_layer_lock_unlock(backend):
    await backend.layer_create("LOCK_TEST")
    result = await backend.layer_lock("LOCK_TEST")
    assert result.get("ok") is True
    result = await backend.layer_unlock("LOCK_TEST")
    assert result.get("ok") is True


async def test_layer_hide_show(backend):
    await backend.layer_create("VIS_TEST")
    result = await backend.layer_hide("VIS_TEST")
    assert result.get("ok") is True
    result = await backend.layer_show("VIS_TEST")
    assert result.get("ok") is True


# ---------------------------------------------------------------------------
# Block tests
# ---------------------------------------------------------------------------


async def test_block_create_and_list(backend):
    line = await backend.entity_create_line(0, 0, 10, 0)
    circle = await backend.entity_create_circle(5, 5, 3)
    result = await backend.block_create_from_entities(
        "MY_BLOCK", [line.handle, circle.handle], base_x=0, base_y=0
    )
    assert result.get("ok") is True
    blocks = await backend.block_list()
    block_names = [b.name for b in blocks]
    assert "MY_BLOCK" in block_names


async def test_block_insert(backend):
    line = await backend.entity_create_line(0, 0, 10, 0)
    await backend.block_create_from_entities("INS_BLOCK", [line.handle])
    result = await backend.block_insert("INS_BLOCK", 50, 50)
    assert result.handle
    assert result.type == "INSERT"


async def test_block_explode(backend):
    line = await backend.entity_create_line(0, 0, 10, 0)
    await backend.block_create_from_entities("EXPLODE_BLK", [line.handle])
    ref = await backend.block_insert("EXPLODE_BLK", 0, 0)
    result = await backend.block_explode(ref.handle)
    assert result.get("ok") is True


async def test_block_get_attributes(backend):
    line = await backend.entity_create_line(0, 0, 10, 0)
    await backend.block_create_from_entities("ATTR_BLK", [line.handle])
    ref = await backend.block_insert("ATTR_BLK", 0, 0)
    attrs = await backend.block_get_attributes(ref.handle)
    assert isinstance(attrs, dict)


async def test_block_insert_keep_text_upright(backend):
    from ezdxf import bbox

    await backend.block_define(
        "SOCKET",
        [
            {"type": "circle", "cx": 0, "cy": 0, "radius": 5},
            {"type": "text", "text": "S", "x": 0, "y": -9, "height": 2},
        ],
        attdefs=[{"tag": "IP", "default": "IP44", "x": 6, "y": 0, "height": 2.5}],
    )

    def attrib(ref):
        return backend._doc.entitydb[ref.handle].attribs[0]

    plain = attrib(await backend.block_insert("SOCKET", 100, 100, rotation=180))
    assert plain.dxf.rotation == pytest.approx(180)

    ref = await backend.block_insert("SOCKET", 100, 100, rotation=180, keep_text_upright=True)
    upright = attrib(ref)
    assert upright.dxf.rotation == pytest.approx(0)
    # Turned about its own centre: the label covers the same place.
    assert bbox.extents([upright]).center.isclose(bbox.extents([plain]).center, abs_tol=1e-6)
    report = ref.properties["text_upright"]
    assert report["flipped"] == ["IP"]
    assert report["not_flipped"] == [
        {"type": "TEXT", "text": "S", "reason": "definition_text_shared"}
    ]

    # (90°, 270°]: 90 reads bottom-to-top and stays, 270 is turned to 90.
    at_90 = await backend.block_insert("SOCKET", 0, 0, rotation=90, keep_text_upright=True)
    assert attrib(at_90).dxf.rotation == pytest.approx(90)
    assert at_90.properties["text_upright"]["flipped"] == []
    at_270 = await backend.block_insert("SOCKET", 0, 0, rotation=270, keep_text_upright=True)
    assert attrib(at_270).dxf.rotation == pytest.approx(90)


@pytest.mark.asyncio
async def test_layer_create_makes_many_layers_in_one_call_all_or_none():
    # round 6: 22 layer_create steps for one sheet
    from fastmcp import Client

    import server

    async with Client(server.mcp) as client:
        await client.call_tool("drawing_new", {"bootstrap": "none"})
        made = await client.call_tool(
            "layer_create",
            {
                "layers": [{"name": "A-WALL"}, {"name": "A-DOOR", "color": 3, "lineweight": 35}],
                "color": 8,
            },
        )
        assert made.structured_content["created"] == ["A-WALL", "A-DOOR"]
        rows = {
            r["name"]: r
            for r in (await client.call_tool("layer_list", {})).structured_content["result"]
        }
        assert rows["A-WALL"]["color"] == 8 and rows["A-DOOR"]["color"] == 3
        assert rows["A-DOOR"]["lineweight"] == 35
        for bad in (
            [{"name": "NEW"}, {"name": "a-wall"}],  # exists (any case)
            [{"name": "NEW"}, {"name": "NEW"}],  # twice
            [{"name": "NEW"}, {"name": "X", "colour": 1}],  # unknown key
        ):
            refused = await client.call_tool("layer_create", {"layers": bad}, raise_on_error=False)
            assert refused.is_error and "nothing created" in refused.content[0].text
        names = (await client.call_tool("layer_list", {})).structured_content["result"]
        assert "NEW" not in {r["name"] for r in names}
        failed = await client.call_tool(
            "layer_create",
            {"layers": [{"name": "OK1"}, {"name": "BAD", "linetype": "NO_SUCH_LT"}]},
            raise_on_error=False,
        )
        names = (await client.call_tool("layer_list", {})).structured_content["result"]
        assert failed.is_error and "OK1" not in {r["name"] for r in names}
