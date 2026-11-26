"""B15 residue - drawing_plan(persist_requirements=True) and the ACADMCP_REQUIREMENTS record.

The set is written into the drawing (an XRECORD ``CURRENT`` inside the
``ACADMCP_REQUIREMENTS`` dictionary, the layer-states pattern), so a saved and
reopened file carries its requirements.
"""

from __future__ import annotations

import ezdxf
import pytest
from fastmcp import Client

import server
from engineering.verification.requirements import (
    load_requirements,
    parse_requirements,
    store_requirements,
)
from engineering.verification.types import thaw

pytestmark = pytest.mark.asyncio

_ITEMS = [
    {
        "id": "D1",
        "quantity": "diameter",
        "target": {"nominal": 40},
        "features": [{"kind": "feature_tag", "part": "flange", "feature": "bore"}],
        "provenance": "explicit",
        "manufacturing_tolerance": {"fit": "H7", "source": "drawing note 2"},
    },
    {
        "id": "N1",
        "quantity": "count",
        "target": {"exact": 6},
        "features": [{"kind": "predicate", "type": "circle", "radius_min": 9, "radius_max": 11}],
        "provenance": "explicit",
    },
]
_RAW = {"units": "mm", "part_type": "flange", "tolerance_policy": "ISO 2768-m", "items": _ITEMS}


async def test_plan_accepts_the_hash_of_a_typed_preflight(backend):
    preflight = await backend.drawing_preflight("flange", _RAW)
    plan = await backend.drawing_plan(
        "flange",
        requirements=preflight.normalized_spec["requirements"],
        spec_hash=preflight.spec_hash,
    )
    assert plan.preflight_status == "ready"
    expected = parse_requirements({"units": "mm", "items": _ITEMS})
    assert plan.requirement_digest == expected.digest()
    assert plan.requirements_persisted is False
    assert await load_requirements(backend) is None, "nothing is written without the flag"


async def test_a_persisted_set_travels_with_the_saved_file(backend, tmp_path):
    preflight = await backend.drawing_preflight("flange", _RAW)
    assert preflight.ready is True
    before = backend.document_identity()["revision"]
    plan = await backend.drawing_plan(
        "flange",
        requirements=preflight.normalized_spec["requirements"],
        spec_hash=preflight.spec_hash,
        persist_requirements=True,
    )
    expected = parse_requirements({"units": "mm", "items": _ITEMS})
    assert plan.requirements_persisted is True
    assert plan.requirement_digest == expected.digest()
    assert backend.document_identity()["revision"] == before + 1, "the record is content"

    path = tmp_path / "flange.dxf"
    await backend.drawing_save_as(str(path))
    await backend.drawing_open(str(path))
    assert await load_requirements(backend) == expected

    record = ezdxf.readfile(str(path)).rootdict.get("ACADMCP_REQUIREMENTS").get("CURRENT")
    chunks = [tag.value for tag in record.tags if tag.code == 1000]
    assert chunks and all(len(chunk) <= 255 and chunk.isascii() for chunk in chunks)


async def test_storing_again_replaces_the_one_record(backend):
    rs = parse_requirements({"units": "mm", "items": _ITEMS})
    first = await store_requirements(backend, rs)
    second = await store_requirements(backend, rs)
    assert (first["replaced"], second["replaced"]) == (False, True)
    assert (first["digest"], first["items"], first["backend"]) == (rs.digest(), 2, "ezdxf")
    holder = backend._doc.rootdict.get("ACADMCP_REQUIREMENTS")
    assert list(holder.keys()) == ["CURRENT"]


async def test_persisting_nothing_typed_is_refused_by_name(backend):
    with pytest.raises(ValueError, match="nothing to read"):
        await backend.drawing_plan(
            "flange", requirements={"units": "mm"}, persist_requirements=True
        )
    assert backend.get_plan_spec() is None
    assert await load_requirements(backend) is None


async def test_a_legacy_only_set_is_stored_as_given(backend):
    plan = await backend.drawing_plan(
        "bracket",
        requirements={"units": "mm", "dimensions": {"width": 80.0}},
        persist_requirements=True,
    )
    loaded = await load_requirements(backend)
    assert loaded.requirements == () and thaw(loaded.legacy) == {"width": 80.0}
    assert plan.requirement_digest == loaded.digest()


async def test_the_tool_takes_persist_requirements():
    async with Client(server.mcp) as client:
        await client.call_tool("drawing_new", {})
        plan = (
            await client.call_tool(
                "drawing_plan",
                {
                    "intent": "flange",
                    "requirements": {"units": "mm", "items": _ITEMS},
                    "persist_requirements": True,
                },
            )
        ).structured_content
    expected = parse_requirements({"units": "mm", "items": _ITEMS})
    assert plan["requirements_persisted"] is True
    assert plan["requirement_digest"] == expected.digest()
