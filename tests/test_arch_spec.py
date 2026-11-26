"""``arch_plan_from_spec``: validation by path, one transaction, the shielded rollback.

Headless ezdxf against a real document. The areas are computed by hand from
``EXAMPLE_SPEC``: the exterior ring runs on axis (0,0)-(9000,6000) at 250 mm,
so its inner faces are x = 125 / 8875 and y = 125 / 5875; the interior wall
runs on x = 5000 at 100 mm, faces x = 4950 / 5050. Room 01 is therefore
(4950 - 125) x (5875 - 125) = 4825 x 5750 = 27 743 750 mm², room 02
(8875 - 5050) x 5750 = 3825 x 5750 = 21 993 750 mm² - the net floor, not the
axis area.
"""

from __future__ import annotations

import copy

import anyio
import pytest

from engineering.arch.spec import EXAMPLE_SPEC, draw_plan_from_spec, validate_spec

ROOM_01 = 4825.0 * 5750.0
ROOM_02 = 3825.0 * 5750.0


def _spec(**over) -> dict:
    spec = copy.deepcopy(EXAMPLE_SPEC)
    spec.update(over)
    return spec


async def _count(backend) -> int:
    return len(await backend.entity_list(limit=5000))


# -- validation, before anything is drawn ---------------------------------------


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"colour": "red"}, r"^spec: unknown key\(s\) colour; keys are sheet"),
        ({"walls": []}, r"^walls: name at least one wall"),
        ({"lang": "de"}, r"^lang: unknown language 'de'; languages are en, tr, pl"),
        ({"sheet": {"scale": 0}}, r"^sheet\.scale: must be a finite number greater than zero"),
        ({"chains": {"sides": ["north"]}}, r"^chains\.sides: \['north'\]"),
        (
            {"schedules": [{"kind": "stairs", "at": [0, 0]}]},
            r"^schedules\[0\]\.kind: 'stairs' is not a schedule",
        ),
    ],
)
@pytest.mark.asyncio
async def test_the_document_is_refused_by_path(backend, over, message):
    with pytest.raises(ValueError, match=message):
        await draw_plan_from_spec(backend, _spec(**over))
    assert await _count(backend) == 0


@pytest.mark.asyncio
async def test_a_bad_wall_is_named_by_its_index(backend):
    spec = _spec()
    spec["walls"][1]["thickness"] = -100
    with pytest.raises(ValueError, match=r"^walls\[1\]\.thickness: must be greater than zero"):
        await draw_plan_from_spec(backend, spec)
    assert await _count(backend) == 0


@pytest.mark.asyncio
async def test_two_overlapping_openings_are_refused_naming_both(backend):
    spec = _spec()
    spec["openings"][0]["offset"] = 1000
    spec["openings"].append(
        {"id": "D9", "wall": "EXT", "kind": "door", "offset": 1500, "width": 900}
    )
    with pytest.raises(ValueError, match=r"openings D1 and D9 overlap on wall EXT"):
        await draw_plan_from_spec(backend, spec)
    assert await _count(backend) == 0


@pytest.mark.asyncio
async def test_a_typed_room_area_is_refused(backend):
    spec = _spec()
    spec["rooms"][0]["area"] = 27.0e6
    with pytest.raises(ValueError, match=r"^rooms\[0\]\.area: an area is measured"):
        await draw_plan_from_spec(backend, spec)
    assert await _count(backend) == 0


@pytest.mark.asyncio
async def test_room_align_and_fit_reach_the_label(backend):
    from engineering.arch.spec import spec_schema

    room_props = spec_schema()["properties"]["rooms"]["items"]["properties"]
    assert {"align", "fit"} <= set(room_props)
    spec = _spec()
    spec["rooms"][0].update(align="center", fit=True)
    result = await draw_plan_from_spec(backend, spec)
    first, second = result["rooms"]
    assert first["align"] == "center" and first["fit"]["name_height_mm"] > 0
    assert second["align"] == "left" and second["fit"] is None


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"align": "right"}, r"^rooms\[0\]\.align: 'right' is not an alignment"),
        ({"fit": "yes"}, r"^rooms\[0\]\.fit: expected true or false"),
    ],
)
@pytest.mark.asyncio
async def test_a_bad_room_label_option_is_refused_by_path(backend, over, message):
    spec = _spec()
    spec["rooms"][0].update(over)
    with pytest.raises(ValueError, match=message):
        await draw_plan_from_spec(backend, spec)
    assert await _count(backend) == 0


def test_validate_spec_turns_dicts_into_the_model():
    norm = validate_spec(EXAMPLE_SPEC)
    assert [w.id for w in norm["walls"]] == ["EXT", "INT"]
    assert [o.id for o in norm["openings"]] == ["D1", "D2", "W1", "W2"]
    assert norm["scale"] == 50.0 and norm["lang"] == "en"
    assert norm["chains"] == ("bottom", "left")
    assert [s["kind"] for s in norm["schedules"]] == ["doors", "windows", "rooms"]


# -- the whole plan --------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_example_plan_draws_in_one_transaction_and_measures_its_rooms(backend):
    from engineering.arch.draw import read_plan

    result = await draw_plan_from_spec(backend, EXAMPLE_SPEC)
    assert result["walls"] == ["EXT", "INT"]
    assert result["openings"] == ["D1", "D2", "W1", "W2"]
    assert result["omitted"] == []
    assert len(result["rooms"]) == 2 and len(result["schedules"]) == 3
    assert result["critique"] == []
    plan = await read_plan(backend)
    areas = {room.number: room.area for room in plan["rooms"]}
    assert areas["01"] == pytest.approx(ROOM_01, abs=0.5)
    assert areas["02"] == pytest.approx(ROOM_02, abs=0.5)
    assert backend.get_plan_spec()["layer_set_id"] == "arch"


@pytest.mark.asyncio
async def test_a_room_outside_every_face_is_labelled_without_an_area(backend):
    from engineering.arch.draw import read_plan

    spec = _spec()
    spec["rooms"][1]["at"] = [20000, 20000]
    result = await draw_plan_from_spec(backend, spec)
    open_room = result["rooms"][1]
    assert open_room["room"]["area"] is None and open_room["loop"] is None
    texts = [(await backend.entity_get(h)).properties for h in open_room["handles"]]
    assert len(texts) == 2, "name and number, no area line"
    assert [w["element"] for w in result["warnings"]] == ["rooms[1]"]
    assert result["warnings"][0]["code"] == "room_open"
    plan = await read_plan(backend)
    assert [room.number for room in plan["rooms"]] == ["01"], "no record for the open room"
    assert plan["rooms"][0].area == pytest.approx(ROOM_01, abs=0.5)


#: A U of 200 mm walls on axis (0,0)-(6000,0)-(6000,4000)-(0,4000), open on x = 0:
#: inner faces y = 100 / 3900 and x = 5900, end caps on x = 0.
U_SPEC = {
    "walls": [{"id": "U", "axis": [[0, 0], [6000, 0], [6000, 4000], [0, 4000]], "thickness": 200}],
    "rooms": [{"id": "R1", "name": "SALON", "at": [3000, 2000]}],
}


@pytest.mark.asyncio
async def test_an_open_room_reports_the_free_ends_of_its_opening(backend):
    result = await draw_plan_from_spec(backend, copy.deepcopy(U_SPEC))
    (warning,) = result["warnings"]
    assert warning["code"] == "room_open"
    ends = {(row["element"], row["end"]): row for row in warning["open_ends"]}
    assert set(ends) == {("U", "start"), ("U", "end")}
    assert ends["U", "start"]["at"] == [0.0, 0.0]
    assert sorted(ends["U", "start"]["faces"]) == [[0.0, -100.0], [0.0, 100.0]]
    assert sorted(ends["U", "end"]["faces"]) == [[0.0, 3900.0], [0.0, 4100.0]]


@pytest.mark.asyncio
async def test_a_room_separator_closes_the_open_room_and_its_area_is_measured(backend):
    from engineering.arch.draw import read_plan
    from engineering.arch.rooms import rooms_detect

    spec = copy.deepcopy(U_SPEC)
    spec["room_separators"] = [{"id": "S1", "points": [[0, 100], [0, 3900]]}]
    result = await draw_plan_from_spec(backend, spec)
    assert result["warnings"] == []
    assert result["rooms"][0]["area_source"] == "walls"
    plan = await read_plan(backend)
    assert [s.id for s in plan["room_separators"]] == ["S1"]
    assert plan["rooms"][0].area == pytest.approx(5900.0 * 3800.0, abs=0.5)
    detected = await rooms_detect(backend)
    assert [(r["area"], r["confidence"]) for r in detected["rooms"]] == [
        (pytest.approx(5900.0 * 3800.0, abs=0.5), 1.0)
    ], "arch_rooms_detect reads the separator as ours"


@pytest.mark.asyncio
async def test_a_room_boundary_gives_the_area_without_reading_the_walls(backend):
    from engineering.arch.draw import read_plan

    spec = copy.deepcopy(U_SPEC)
    spec["rooms"][0]["boundary"] = [[1000, 100], [5900, 100], [5900, 3900], [1000, 3900]]
    result = await draw_plan_from_spec(backend, spec)
    assert result["warnings"] == [] and result["rooms"][0]["area_source"] == "boundary"
    plan = await read_plan(backend)
    assert plan["rooms"][0].area == pytest.approx(4900.0 * 3800.0, abs=0.5)
    assert plan["rooms"][0].boundary[0] == (1000.0, 100.0), "the boundary travels with the record"


def test_a_label_point_outside_its_own_boundary_is_refused_before_drawing():
    spec = copy.deepcopy(U_SPEC)
    spec["rooms"][0]["boundary"] = [[4000, 100], [5900, 100], [5900, 3900], [4000, 3900]]
    with pytest.raises(ValueError, match=r"^rooms\[0\]\.at: .*outside the room's own boundary"):
        validate_spec(spec)


@pytest.mark.asyncio
async def test_a_room_the_drawing_refuses_rolls_the_whole_plan_back(backend):
    spec = _spec()
    spec["rooms"][1]["at"] = list(spec["rooms"][0]["at"])  # the same face twice
    with pytest.raises(ValueError, match=r"^rooms\[1\]: .*already labelled"):
        await draw_plan_from_spec(backend, spec)
    assert await _count(backend) == 0
    begun = await backend.transaction_begin()
    assert begun["ok"] is True, "the rollback must have closed the transaction"
    await backend.transaction_rollback()


@pytest.mark.asyncio
async def test_a_cancelled_request_still_rolls_back(backend, monkeypatch):
    """The MCP SDK cancels a request by cancelling an anyio scope; the rollback
    runs under a shield, so the cancelled plan leaves nothing behind."""
    from engineering.arch import rooms

    async def hang(*_args, **_kwargs):
        await anyio.sleep_forever()

    monkeypatch.setattr(rooms, "label_room", hang)
    with anyio.move_on_after(0.5):
        await draw_plan_from_spec(backend, EXAMPLE_SPEC)
    assert await _count(backend) == 0
    begun = await backend.transaction_begin()
    assert begun["ok"] is True
    await backend.transaction_rollback()


@pytest.mark.asyncio
async def test_the_tool_is_registered_and_refuses_through_tool_error():
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    import server

    async with Client(server.mcp) as client:
        await client.call_tool("drawing_new", {})
        result = (
            await client.call_tool("arch_plan_from_spec", {"spec": EXAMPLE_SPEC})
        ).structured_content
        assert result["critique"] == []
        with pytest.raises(ToolError, match=r"walls: name at least one wall"):
            await client.call_tool("arch_plan_from_spec", {"spec": {"walls": []}})
