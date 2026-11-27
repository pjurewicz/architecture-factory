"""``arch_plan_patch``: a patched plan is the patched spec drawn from scratch, apart from handles.

Headless ezdxf. The equivalence is measured on every entity of the drawing,
each reduced to its DXF tags without the handle, the owner and the pointers
(handles, block names of a dimension's or table's anonymous block) and
without XDATA, compared as a multiset; the ``ACADMCP_ARCH`` records are
compared beside it. A patch that redrew too little leaves an old entity behind,
one that redrew wrongly differs from the fresh drawing.
"""

from __future__ import annotations

import copy
from collections import Counter

import pytest
from ezdxf.lldxf.tagwriter import TagCollector

from backends.ezdxf_backend import EzdxfBackend
from engineering.arch.catalogue import insert_item
from engineering.arch.draw import read_plan
from engineering.arch.patch import apply_changes, compact, patch_plan
from engineering.arch.spec import EXAMPLE_SPEC, draw_plan_from_spec
from engineering.changes.preview import StaleRevisionError, entity_fingerprints
from engineering.errors import (
    HandEditedTaggedEntityError,
    InvalidArgsError,
    RevisionRefusedError,
)

#: Tags that name another object rather than describe this one.
_POINTERS = {5, 105, 330, 340, 342, 343, 360}


def _shape(entity, dxfversion) -> tuple:
    collector = TagCollector(dxfversion=dxfversion)
    entity.export_dxf(collector)
    kind = entity.dxftype()
    out = []
    for code, value in collector.tags:
        if code == 1001:
            break
        if code in _POINTERS or (code == 2 and kind in ("DIMENSION", "ACAD_TABLE")):
            continue
        if isinstance(value, float):
            value = round(value, 6)
        elif isinstance(value, tuple):
            value = tuple(round(v, 6) if isinstance(v, float) else v for v in value)
        out.append((code, repr(value)))
    return (kind, tuple(out))


async def _drawing(backend) -> Counter:
    doc = backend._require_doc()
    return Counter(_shape(e, doc.dxfversion) for e in doc.modelspace())


async def _records(backend) -> dict:
    plan = await read_plan(backend)
    return {k: sorted(map(repr, v)) for k, v in plan.items() if k != "carriers"}


async def _fresh(spec) -> EzdxfBackend:
    b = EzdxfBackend()
    await b.connect()
    await b.drawing_new()
    await draw_plan_from_spec(b, spec)
    return b


#: Each edit the task names, as a patch: a moved door, a flipped hinge, a
#: renamed and a renumbered room, a thicker wall, a moved wall end.
EDITS = {
    "move_door": {"openings.D2.offset": 3200},
    "flip_hinge": {"openings.D2.hand": None, "openings.D2.hinge": "end"},
    "rename_room": {"rooms.R2.name": "KITCHEN"},
    "renumber_room": {"rooms.R1.number": "11"},
    "wall_thickness": {"walls.INT.thickness": 150},
    "move_wall_end": {"walls.INT.axis[1]": [5600, 6000], "walls.INT.axis[0]": [5600, 0]},
    "add_and_remove": {
        "openings.W2": None,
        "openings+": {"id": "W3", "wall": "EXT", "kind": "window", "offset": 3000, "width": 900},
    },
}


@pytest.mark.parametrize("name", sorted(EDITS))
@pytest.mark.asyncio
async def test_a_patch_draws_what_the_patched_spec_draws(backend, name):
    await draw_plan_from_spec(backend, EXAMPLE_SPEC)
    unpatched = await _drawing(backend)
    result = await patch_plan(backend, EDITS[name])
    assert result["changed"] is True
    assert await _drawing(backend) != unpatched  # the comparison sees the edit
    patched, _diff = apply_changes(EXAMPLE_SPEC, EDITS[name])
    fresh = await _fresh(patched)
    try:
        assert await _drawing(backend) == await _drawing(fresh)
        assert await _records(backend) == await _records(fresh)
    finally:
        await fresh.disconnect()


@pytest.mark.asyncio
async def test_a_sequence_of_patches_draws_the_final_spec(backend):
    await draw_plan_from_spec(backend, EXAMPLE_SPEC)
    spec = copy.deepcopy(EXAMPLE_SPEC)
    for name in ("wall_thickness", "move_door", "flip_hinge", "rename_room", "move_wall_end"):
        await patch_plan(backend, EDITS[name])
        spec, _ = apply_changes(spec, EDITS[name])
    fresh = await _fresh(spec)
    try:
        assert await _drawing(backend) == await _drawing(fresh)
    finally:
        await fresh.disconnect()


@pytest.mark.asyncio
async def test_only_the_affected_set_is_redrawn(backend):
    drawn = await draw_plan_from_spec(backend, EXAMPLE_SPEC)
    before = {e.handle for e in await backend.entity_list(limit=100000)}
    # a hinge flip: the door's symbol and the door schedule, nothing else
    flip = await patch_plan(backend, EDITS["flip_hinge"])
    assert flip["affected"] == {
        "walls": [],
        "openings": ["D2"],
        "rooms": [],
        "chains": False,
        "schedules": ["doors"],
        "removed": [],
    }
    gone = {h for row in flip["handle_map"] for h in row["old"]}
    after = {e.handle for e in await backend.entity_list(limit=100000)}
    assert before - after == gone
    stair = drawn["stairs"][0]["handles"]
    assert set(stair) <= after
    # a moved interior door: its host wall only; the areas do not move
    move = await patch_plan(backend, EDITS["move_door"])
    assert move["affected"]["walls"] == ["INT"]
    assert move["affected"]["rooms"] == [] and move["affected"]["chains"] is False
    # a thicker interior wall: the T it makes moves the exterior faces too
    thick = await patch_plan(backend, EDITS["wall_thickness"])
    assert thick["affected"]["walls"] == ["EXT", "INT"]
    assert thick["affected"]["rooms"] == ["R1", "R2"]
    assert thick["affected"]["chains"] is True
    rooms = {r["room"]["id"]: r["room"]["area"] for r in thick["rooms"]}
    assert rooms == {"R1": 4800.0 * 5750.0, "R2": 3800.0 * 5750.0}


@pytest.mark.asyncio
async def test_what_the_plan_did_not_draw_keeps_its_handle_and_its_bytes(backend):
    await draw_plan_from_spec(backend, EXAMPLE_SPEC)
    sofa = await insert_item(backend, "sofa_3_seat", at=[1500, 4000])
    note = await backend.entity_create_text("SITE NOTE", 500, 7000, height=125, layer="A-ANNO")
    dim = await backend.dimension_linear(0, 7500, 9000, 7500, 4500, 8000, 0.0, "A-ANNO")
    await backend.layout_create("PLAN-50")
    theirs = [sofa["handle"], note.handle, dim.handle]
    prints = await entity_fingerprints(backend, theirs)
    await patch_plan(backend, EDITS["wall_thickness"])
    assert await entity_fingerprints(backend, theirs) == prints
    assert "PLAN-50" in str((await backend.layout_list())["layouts"])


@pytest.mark.asyncio
async def test_a_hand_edited_plan_entity_refuses_unless_overwrite(backend):
    drawn = await draw_plan_from_spec(backend, EXAMPLE_SPEC)
    face = drawn["drawn"]["walls"][1]["handles"][0]
    await backend.entity_move(face, 10, 0)
    rev = backend.document_identity()["revision"]
    with pytest.raises(HandEditedTaggedEntityError) as caught:
        await patch_plan(backend, EDITS["rename_room"])
    assert face in caught.value.fields()["handles"]
    assert backend.document_identity()["revision"] == rev
    result = await patch_plan(backend, EDITS["rename_room"], overwrite=True)
    assert result["overwritten"][0]["handle"] == face


@pytest.mark.asyncio
async def test_a_saved_and_reopened_plan_patches_without_false_drift(backend, tmp_path):
    await draw_plan_from_spec(backend, EXAMPLE_SPEC)
    path = tmp_path / "plan.dxf"
    await backend.drawing_save(str(path))
    await backend.drawing_open(str(path))
    result = await patch_plan(backend, EDITS["wall_thickness"])
    assert result["overwritten"] == []
    assert result["rev"] == 1


@pytest.mark.asyncio
async def test_dry_run_changes_nothing_and_its_preview_goes_stale(backend):
    await draw_plan_from_spec(backend, EXAMPLE_SPEC)
    count = len(await backend.entity_list(limit=100000))
    dry = await patch_plan(backend, EDITS["move_door"], dry_run=True)
    assert dry["affected"]["walls"] == ["INT"]
    assert len(await backend.entity_list(limit=100000)) == count
    preview = dry["preview"]
    await patch_plan(backend, EDITS["rename_room"])
    with pytest.raises(StaleRevisionError):
        await patch_plan(
            backend,
            EDITS["move_door"],
            expected_revision=preview["revision"],
            preconditions=preview["preconditions"],
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sheet.scale": 100}, r"'sheet' is not patched"),
        ({"openings.D9.offset": 1}, r"no element 'D9'"),
        ({"walls.INT.thickness": -5}, r"walls\[1\]\.thickness"),
        ({"rooms.R1.area": 10}, r"rooms\[0\]\.area"),
    ],
)
@pytest.mark.asyncio
async def test_a_bad_patch_is_refused_by_path_and_writes_nothing(backend, changes, message):
    await draw_plan_from_spec(backend, EXAMPLE_SPEC)
    rev = backend.document_identity()["revision"]
    with pytest.raises(InvalidArgsError, match=message):
        await patch_plan(backend, changes)
    assert backend.document_identity()["revision"] == rev


@pytest.mark.asyncio
async def test_a_drawing_without_a_stored_plan_is_refused(backend):
    with pytest.raises(RevisionRefusedError, match="stores no plan"):
        await patch_plan(backend, EDITS["move_door"])


@pytest.mark.asyncio
async def test_the_default_reply_is_compact(backend):
    await draw_plan_from_spec(backend, EXAMPLE_SPEC)
    reply = compact(await patch_plan(backend, EDITS["rename_room"]))
    assert set(reply) >= {"rev", "affected", "rooms", "warnings", "plan_bbox", "outer_bbox"}
    assert "handle_map" not in reply
    assert reply["rooms"] == [
        {"id": "R2", "number": "02", "name": "KITCHEN", "area_m2": 21.99375, "fits": False}
    ]
