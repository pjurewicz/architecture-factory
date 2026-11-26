"""M5 — `block_create_from_entities` on the headless engine.

The live backend refused this outright ("not supported in COM backend. Use
system_run_command with _BLOCK instead"), which pushed users at the free-text
escape hatch for something ActiveX does directly: `Blocks.Add` makes the
definition and `Document.CopyObjects` moves entities into it. Meanwhile the
tool's own docstring said the opposite — "works by using AutoCAD's BLOCK command
(COM backend only)" — so the one backend that could not do it was named as the
one that could.

Both engines also used to swallow handles they could not resolve: pass five
handles with two typos and you got `entity_count: 3` and no mention of the two.
The count is now accompanied by what was skipped.
"""

from __future__ import annotations

import pytest

from backends.ezdxf_backend import EzdxfBackend

pytestmark = pytest.mark.asyncio


# ── headless ────────────────────────────────────────────────────────────────


async def test_ezdxf_builds_the_definition_from_real_entities(backend):
    line = await backend.entity_create_line(0, 0, 10, 0)
    circle = await backend.entity_create_circle(5, 5, 2)

    result = await backend.block_create_from_entities("BRACKET", [line.handle, circle.handle])

    assert result["ok"] is True
    assert result["entity_count"] == 2
    assert result["skipped"] == []
    block = backend._doc.blocks.get("BRACKET")
    assert sorted(e.dxftype() for e in block) == ["CIRCLE", "LINE"]


async def test_ezdxf_reports_handles_it_could_not_resolve(backend):
    line = await backend.entity_create_line(0, 0, 10, 0)

    result = await backend.block_create_from_entities("PARTIAL", [line.handle, "NOSUCH", "ALSONO"])

    assert result["entity_count"] == 1
    assert result["skipped"] == ["NOSUCH", "ALSONO"], (
        "a count with no mention of what was dropped is the same shape of lie "
        "as the audit reporting no fixes"
    )


async def test_ezdxf_refuses_a_block_with_nothing_in_it(backend):
    with pytest.raises(RuntimeError, match="NOSUCH"):
        await backend.block_create_from_entities("EMPTY", ["NOSUCH"])
    assert "EMPTY" not in backend._doc.blocks, "a failed call must leave no stub behind"


async def test_ezdxf_honours_the_base_point(backend):
    line = await backend.entity_create_line(0, 0, 10, 0)
    await backend.block_create_from_entities("BASED", [line.handle], base_x=3.0, base_y=4.0)

    block = backend._doc.blocks.get("BASED")
    assert tuple(block.block.dxf.base_point)[:2] == (3.0, 4.0)


async def test_ezdxf_still_declares_no_capability_gap_here():
    """The headless engine can do this, so it must not look like a boundary."""
    features = EzdxfBackend().capabilities().features
    assert "block_definition" not in features, (
        "a capability key would imply the engine cannot; it can"
    )
