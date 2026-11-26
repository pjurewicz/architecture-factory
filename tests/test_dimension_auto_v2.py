"""1.8-13: ``dimension_auto`` v2 - dimension a part in one call.

§4.4 H5 of the rival's report: dimensioning a plate took us 89 calls (the
rival 10) because ``dimension_auto`` refused everything but LINE. These tests
hold the planner (``engineering/dimension_auto.py``) per entity type, the
shared ``PremiumContract.dimension_auto`` on the headless engine (it draws
through the existing ``dimension_*`` contracts), the tool's
answer and refusals, ``skip_existing`` (a second run draws nothing),
``omitted`` (never a silent drop), ``group_equal``, the feature records copied
from 1.8-06 tags, and the binding of every drawn dimension by
``drawing_verify_requirements`` without a ``dimension_mismatch``.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastmcp import Client

import server
from benchmarks.call_budget import (
    PLATE_ARCS,
    PLATE_HOLE_RADIUS,
    build_plate,
)
from engineering import dimension_auto as da
from engineering.errors import InvalidArgsError, classify
from engineering.mech.xdata import APP_ID, decode, to_values

pytestmark = pytest.mark.asyncio

BOX = (0.0, 0.0, 120.0, 80.0)


def _plan(infos, **kwargs) -> da.Plan:
    include = da.parse_include(kwargs.pop("include", None))
    sources, omitted = da.normalize(infos)
    plan = da.plan_dimensions(sources, include=include, **kwargs)
    plan.omitted[:0] = omitted
    return plan


def _outside(point, box=BOX) -> bool:
    x0, y0, x1, y1 = box
    return not (x0 < point[0] < x1 and y0 < point[1] < y1)


# -- the headless engine -------------------------------------------------------------


@pytest_asyncio.fixture
async def plate(backend, tmp_path):
    path = build_plate(tmp_path / "plate.dxf")
    await backend.drawing_open(str(path))
    return backend


def _dims(backend) -> list:
    return list(backend._require_doc().modelspace().query("DIMENSION"))


async def test_the_plate_in_one_call_on_the_headless_engine(plate):
    result = await plate.dimension_auto(selector={"layer": "GEOMETRY"})
    answer = result.to_dict()
    dims = _dims(plate)
    assert len(result) == len(answer["created"]) == len(dims) == 12
    kinds = sorted(d.dxf.dimtype & 0x0F for d in dims)
    assert kinds == [0, 0, 0, 0] + [3] * 6 + [4] * 2
    assert all(d.dxf.layer == "DIM" for d in dims)
    assert answer["omitted"] == [] and answer["skipped_existing"] == []
    assert answer["dim_overlap"] == {"checked": True, "issues": []}
    # every diameter's text stands outside the part
    for dim in dims:
        if dim.dxf.dimtype & 0x0F in (3, 4):
            assert _outside(tuple(dim.dxf.text_midpoint)[:2]), dim.dxf.handle


async def test_a_second_run_draws_nothing_and_says_what_it_skipped(plate):
    await plate.dimension_auto(selector={"layer": "GEOMETRY"})
    revision = plate.document_identity()["revision"]
    again = (await plate.dimension_auto(selector={"layer": "GEOMETRY"})).to_dict()
    assert again["created"] == []
    assert len(again["skipped_existing"]) == 12
    assert {row["kind"] for row in again["skipped_existing"]} == {"linear", "diameter", "radius"}
    assert len(_dims(plate)) == 12
    assert plate.document_identity()["revision"] == revision
    redraw = await plate.dimension_auto(selector={"layer": "GEOMETRY"}, skip_existing=False)
    assert len(redraw) == 12, "skip_existing=False draws them again"


async def test_handles_and_selector_are_one_or_the_other(plate):
    for kwargs in ({}, {"handles": ["31"], "selector": {"layer": "GEOMETRY"}}):
        with pytest.raises(InvalidArgsError, match="exactly one of handles") as caught:
            await plate.dimension_auto(**kwargs)
        assert classify(caught.value)["kind"] == "invalid_args"
    assert _dims(plate) == []


async def test_a_dry_run_plans_and_draws_nothing(plate):
    revision = plate.document_identity()["revision"]
    result = await plate.dimension_auto(selector={"layer": "GEOMETRY"}, dry_run=True)
    answer = result.to_dict()
    assert answer["dry_run"] is True and answer["created"] == []
    assert len(answer["planned"]) == 12
    assert _dims(plate) == [] and plate.document_identity()["revision"] == revision


async def test_ellipse_and_spline_in_the_selection_are_omitted_never_dropped(plate):
    doc = plate._require_doc()
    msp = doc.modelspace()
    ellipse = msp.add_ellipse(
        (200, 0), major_axis=(10, 0), ratio=0.5, dxfattribs={"layer": "GEOMETRY"}
    )
    spline = msp.add_spline([(200, 20), (210, 30), (220, 20)], dxfattribs={"layer": "GEOMETRY"})
    answer = (await plate.dimension_auto(selector={"layer": "GEOMETRY"})).to_dict()
    omitted = {row["handle"]: row for row in answer["omitted"]}
    assert set(omitted) == {ellipse.dxf.handle, spline.dxf.handle}
    assert omitted[ellipse.dxf.handle]["type"] == "ELLIPSE"
    assert "spline" in omitted[spline.dxf.handle]["reason"]
    assert len(answer["created"]) >= 12


async def test_group_equal_writes_the_count_into_one_diameter(plate):
    answer = (
        await plate.dimension_auto(
            selector={"layer": "GEOMETRY"}, include=["diameter"], group_equal=True
        )
    ).to_dict()
    (handle,) = answer["created"]
    dim = plate._require_doc().entitydb[handle]
    assert dim.dxf.text == "6× <>"
    assert dim.get_measurement() == pytest.approx(2.0 * PLATE_HOLE_RADIUS)


async def test_the_17_line_run_is_unchanged_and_still_a_list(backend):
    l1 = await backend.entity_create_line(0, 0, 30, 0)
    l2 = await backend.entity_create_line(30, 0, 70, 0)
    dims = await backend.dimension_auto([l1.handle, l2.handle], style="chain")
    assert len(dims) == 2 and dims[0].layer == "DIM"
    assert dims == [dims[0], dims[1]] and isinstance(dims, list)
    assert await backend.dimension_auto([], style="chain") == []


async def test_a_feature_tag_becomes_the_dimensions_feature_record(backend):
    tag = {"v": 1, "kind": "feature", "part": "1A", "feature": "h1", "view": "front"}
    circle = await backend.entity_create_circle(
        10, 10, 3, layer="GEOMETRY", xdata={APP_ID: to_values(tag)}
    )
    plain = await backend.entity_create_circle(40, 10, 3, layer="GEOMETRY")
    result = await backend.dimension_auto(
        handles=[circle.handle, plain.handle], include=["diameter"]
    )
    records = {}
    for info in result:
        raw = await backend.entity_get_xdata(info.handle, APP_ID)
        values = (raw.get("xdata") or {}).get(APP_ID)
        records[info.properties.get("dim_type"), info.handle] = decode(values) if values else None
    tagged = [r for r in records.values() if r]
    assert tagged == [{**tag, "role": "dim"}]
    assert sum(1 for r in records.values() if r is None) == 1


async def test_every_drawn_dimension_binds_without_a_mismatch(plate):
    from engineering.verification.report import verify_requirements
    from engineering.verification.requirements import parse_requirements

    result = await plate.dimension_auto(selector={"layer": "GEOMETRY"})
    items = []
    for info in result:
        dim = plate._require_doc().entitydb[info.handle]
        dimtype = dim.dxf.dimtype & 0x0F
        quantity = {3: "diameter", 4: "radius"}.get(dimtype, "distance")
        nominal = {
            "diameter": 2 * PLATE_HOLE_RADIUS,
            "radius": PLATE_ARCS[0][2],
            "distance": dim.get_measurement(),
        }[quantity]
        items.append(
            {
                "id": f"R{info.handle}",
                "quantity": quantity,
                "target": {"nominal": nominal, "unit": "mm"},
                "features": [{"kind": "dimension", "handle": info.handle}],
                "provenance": "explicit",
            }
        )
    report = await verify_requirements(plate, parse_requirements({"units": "mm", "items": items}))
    assert not [f for f in report["findings"] if f["kind"] == "dimension_mismatch"]
    bound = set()
    for row in report["rows"]:
        assert row["dimension"]["status"] == "bound" and not row["dimension"]["mismatch"]
        bound.update(h for ref in row["dimension"]["refs"] for h in ref["handles"])
        if row["quantity"] != "distance":
            assert row["status"] == "pass", row
    features = {
        e.dxf.handle for e in plate._require_doc().modelspace() if e.dxftype() != "DIMENSION"
    }
    assert bound == features, "every feature of the plate is bound by a dimension"


async def test_no_two_dimensions_occupy_the_same_place(plate):
    from engineering.mech.critique import _duplicate_dimension

    await plate.dimension_auto(selector={"layer": "GEOMETRY"})
    entities = await plate.entity_list(type_filter="DIMENSION", limit=100)
    assert _duplicate_dimension({"entities": entities}) == []


# -- the tool ----------------------------------------------------------------------


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setenv("AUTOCAD_MCP_BACKEND", "ezdxf")
    async with Client(server.mcp) as connected:
        status = (await connected.call_tool("system_status", {})).structured_content or {}
        assert status.get("backend") == "ezdxf", "these tests must never touch live AutoCAD"
        yield connected


async def _call(client, tool, args):
    return (await client.call_tool(tool, args)).structured_content


async def test_the_tool_answers_created_skipped_and_omitted(client, tmp_path):
    path = build_plate(tmp_path / "plate.dxf")
    await _call(client, "drawing_open", {"path": str(path)})
    answer = await _call(client, "dimension_auto", {"selector": {"layer": "GEOMETRY"}})
    assert set(answer) >= {"created", "skipped_existing", "omitted", "dim_overlap"}
    assert len(answer["created"]) == 12
    again = await _call(client, "dimension_auto", {"selector": {"layer": "GEOMETRY"}})
    assert again["created"] == [] and len(again["skipped_existing"]) == 12


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"selector": {"layer": "GEOMETRY", "kind": "x"}},
        {"selector": {"layer": "GEOMETRY"}, "include": ["angle"]},
    ],
)
async def test_the_tool_refuses_by_name(client, args):
    result = await client.call_tool("dimension_auto", args, raise_on_error=False)
    assert result.is_error
    assert (result.structured_content or {}).get("kind") == "invalid_args"


async def test_dimension_auto_is_a_cad_batch_step(client, tmp_path):
    path = build_plate(tmp_path / "plate.dxf")
    await _call(client, "drawing_open", {"path": str(path)})
    batch = await _call(
        client,
        "cad_batch",
        {"steps": [{"tool": "dimension_auto", "args": {"selector": {"layer": "GEOMETRY"}}}]},
    )
    assert batch["ok"] is True, batch
    assert len(batch["results"][0]["result"]["created"]) == 12


async def test_the_schema_offers_the_v2_parameters(full_catalog):
    tools = {t.name: t for t in await server.mcp.list_tools()}
    props = tools["dimension_auto"].parameters["properties"]
    assert {"handles", "selector", "include", "group_equal", "skip_existing", "dry_run"} <= set(
        props
    )
    assert (
        "required" not in tools["dimension_auto"].parameters
        or not tools["dimension_auto"].parameters["required"]
    )
