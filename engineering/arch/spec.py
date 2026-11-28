"""``arch_plan_from_spec``: a whole plan from one declarative document, in one transaction.

Walls with their hosted openings, stairs, room labels (each area measured off
the walls just drawn), the exterior dimension chains and the schedules — in
that order, because each step reads what the one before it drew: a room label
measures the wall faces, and a schedule reads the ``ACADMCP_ARCH`` records.

Everything that can be refused without a drawing is refused before the
transaction opens, by path (``walls[2].thickness``, ``openings[1]: ...``): the
model, the openings against their walls, the junctions (the wall engine is
pure and is run once in memory for exactly this), every stair's plan symbol,
the rooms (a typed ``area`` included), the chain sides and the schedule kinds.
What only the drawing can refuse - a room point inside a wall body or in a room
already labelled - refuses inside the transaction, and the transaction is
rolled back. A room point in no closed face (an open living room) does not:
it is labelled without an area or a room record, and ``warnings`` says so with
the free wall / separator ends nearest to it (``open_ends``). The engine never
invents the missing boundary: a ``room_separators`` line (zero thickness) or
the room's own ``boundary`` polygon says where the room ends, and the area is
then measured from it.

The rollback is shielded for the reason ``engineering/pid/spec.py`` documents
at length: the MCP SDK cancels a request by cancelling an anyio scope, anyio
cancellation is level-triggered, and a bare ``await`` in the handler would
itself be cancelled before the rollback body ran - leaving half a plan inside
an open transaction.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import anyio

import config
from engineering.arch import draw as _draw
from engineering.arch import rooms as _rooms
from engineering.arch import schedule as _schedule
from engineering.validation import within_budget

from .lang import LANGS
from .layers import ARCH_ROLE_LAYER
from .model import (
    opening_from_dict,
    room_from_dict,
    room_separator_from_dict,
    stair_from_dict,
    validate_openings,
    wall_from_dict,
)
from .openings import resolve_door

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

log = logging.getLogger(__name__)

SPEC_KEYS = (
    "sheet",
    "lang",
    "poche",
    "walls",
    "wall_polygons",
    "openings",
    "stairs",
    "room_separators",
    "rooms",
    "chains",
    "schedules",
)
SHEET_KEYS = ("intent", "sheet_size", "scale")
CHAIN_SIDES = ("bottom", "top", "left", "right")
SCHEDULE_KINDS = ("doors", "windows", "rooms")

#: A two-room plan the tool docstring and the tests use: an entrance door and
#: an interior door, two windows, a straight stair, both rooms labelled, the
#: two exterior chains and all three schedules. The room numbers are "01" /
#: "02", not "R1": the finalize validator reads R-and-a-digit as a radius
#: callout typed as text.
EXAMPLE_SPEC: dict = {
    "sheet": {"intent": "Two-room ground floor plan", "sheet_size": "A3", "scale": 50},
    "lang": "en",
    "walls": [
        {
            "id": "EXT",
            "axis": [[0, 0], [9000, 0], [9000, 6000], [0, 6000]],
            "thickness": 250,
            "closed": True,
            "material": "brick",
        },
        {"id": "INT", "axis": [[5000, 0], [5000, 6000]], "thickness": 100, "material": "aac"},
    ],
    "openings": [
        {"id": "D1", "wall": "EXT", "kind": "door", "offset": 1000, "width": 1000, "tag": "D1"},
        {
            "id": "D2",
            "wall": "INT",
            "kind": "door",
            "offset": 2000,
            "width": 900,
            "hand": "right",
            "tag": "D2",
        },
        {
            "id": "W1",
            "wall": "EXT",
            "kind": "window",
            "offset": 20500,
            "width": 1500,
            "sill": 900,
            "height": 1200,
            "tag": "W1",
        },
        {
            "id": "W2",
            "wall": "EXT",
            "kind": "window",
            "offset": 11000,
            "width": 1200,
            "sill": 900,
            "height": 1200,
            "tag": "W2",
        },
    ],
    "stairs": [
        {
            "id": "S1",
            "start": [6000, 1000],
            "direction_deg": 90,
            "width": 900,
            "risers": 16,
            "riser_height": 175,
            "going": 280,
        }
    ],
    "rooms": [
        {"id": "R1", "name": "LIVING", "number": "01", "at": [2500, 3000]},
        {"id": "R2", "name": "HALL", "number": "02", "at": [7900, 3000]},
    ],
    "chains": {"sides": ["bottom", "left"]},
    "schedules": [
        {"kind": "doors", "at": [10500, 6000]},
        {"kind": "windows", "at": [10500, 4000]},
        {"kind": "rooms", "at": [10500, 2000]},
    ],
}


def _number(value, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: expected a number, got {value!r}") from None
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{where}: must be a finite number greater than zero, got {value!r}")
    return number


def _point(value, where: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{where}: expected [x, y], got {value!r}")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        raise ValueError(f"{where}: expected [x, y], got {value!r}") from None
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{where}: must be two finite numbers, got {value!r}")
    return (x, y)


def _list(spec: dict, key: str) -> list:
    value = spec.get(key)
    if value is None:
        return []
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{key}: expected a list, got {type(value).__name__}")
    # B07: every list of the document is bounded, by path, before anything is built.
    within_budget(len(value), config.settings.max_spec_items, key, "MAX_SPEC_ITEMS")
    return list(value)


#: The label options a ``rooms[]`` item takes beside the Room fields; they shape
#: the label ``label_room`` draws and are not part of the stored room record.
ROOM_LABEL_KEYS = ("align", "fit", "mask")


def _room_item_schema(schema: dict) -> dict:
    props = dict(schema["properties"])
    props["align"] = {"type": "string", "enum": list(_rooms.ALIGNS), "default": "left"}
    props["fit"] = {
        "type": "boolean",
        "default": False,
        "description": "Wrap/shrink the name into the room; each room reports fits.",
    }
    props["mask"] = {"type": "boolean", "default": False}
    return {**schema, "properties": props}


def spec_schema() -> dict:
    """The JSON schema of the whole document ``validate_spec`` reads.

    Assembled from the same tuples the validator checks (``SPEC_KEYS``,
    ``SHEET_KEYS``, ``LANGS``, ``CHAIN_SIDES``, ``SCHEDULE_KINDS``) and each
    item from its builder (``item_schema``), so the advertised syntax is the
    accepted one. Ranges and cross-item rules stay with the validator.
    """
    from .model import Opening, Room, RoomSeparator, Stair, Wall, WallPolygon, item_schema

    point = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    props = {
        "sheet": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "sheet_size": {"type": "string", "default": "A3"},
                "scale": {"type": "number", "default": 50, "description": "50 = 1:50"},
            },
            "additionalProperties": False,
        },
        "lang": {"type": "string", "enum": list(LANGS), "default": "en"},
        "poche": {
            "anyOf": [{"type": "boolean"}, {"const": "solid"}],
            "default": True,
            "description": "true: material hatch; 'solid': solid fill",
        },
        "walls": {"type": "array", "items": item_schema(Wall), "minItems": 1},
        "wall_polygons": {"type": "array", "items": item_schema(WallPolygon)},
        "openings": {"type": "array", "items": item_schema(Opening)},
        "stairs": {"type": "array", "items": item_schema(Stair)},
        "room_separators": {"type": "array", "items": item_schema(RoomSeparator)},
        "rooms": {"type": "array", "items": _room_item_schema(item_schema(Room))},
        "chains": {
            "type": ["object", "boolean", "null"],
            "properties": {
                "sides": {"type": "array", "items": {"type": "string", "enum": list(CHAIN_SIDES)}}
            },
            "additionalProperties": False,
        },
        "schedules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(SCHEDULE_KINDS)},
                    "at": point,
                },
                "required": ["kind", "at"],
                "additionalProperties": False,
            },
        },
    }
    assert tuple(props) == SPEC_KEYS and tuple(props["sheet"]["properties"]) == SHEET_KEYS
    return {
        "type": "object",
        "properties": props,
        "required": ["walls"],
        "additionalProperties": False,
    }


def validate_spec(spec: dict) -> dict:
    """The whole document checked and normalised - nothing is drawn here.

    Returns ``{"sheet", "lang", "scale", "poche", "walls", "openings",
    "stairs", "room_separators", "rooms", "chains", "schedules"}`` with model objects in place of
    dicts (rooms stay requests: ``{"room": dict, "align", "fit", "mask"}``, because
    ``label_room`` takes the request, not a Room).
    """
    from engineering.arch.stairs import stair_prims
    from engineering.arch.walls import wall_geometry

    if not isinstance(spec, dict):
        raise ValueError(f"spec: expected a dict, got {type(spec).__name__}")
    unknown = sorted(set(spec) - set(SPEC_KEYS))
    if unknown:
        raise ValueError(
            f"spec: unknown key(s) {', '.join(unknown)}; keys are {', '.join(SPEC_KEYS)}"
        )
    sheet = spec.get("sheet") or {}
    if not isinstance(sheet, dict):
        raise ValueError(f"sheet: expected a dict, got {type(sheet).__name__}")
    extra = sorted(set(sheet) - set(SHEET_KEYS))
    if extra:
        raise ValueError(
            f"sheet: unknown key(s) {', '.join(extra)}; keys are {', '.join(SHEET_KEYS)}"
        )
    scale = _number(sheet.get("scale", 50), "sheet.scale")
    lang = spec.get("lang", "en")
    if lang not in LANGS:
        raise ValueError(f"lang: unknown language {lang!r}; languages are {', '.join(LANGS)}")
    poche = spec.get("poche", True)
    if not isinstance(poche, bool) and poche != "solid":
        raise ValueError(f"poche: expected true, false or 'solid', got {poche!r}")

    raw_walls = _list(spec, "walls")
    if not raw_walls:
        raise ValueError("walls: name at least one wall - a plan without walls has no rooms")
    walls = [wall_from_dict(item, f"walls[{i}]") for i, item in enumerate(raw_walls)]
    try:
        polygons = _draw.parse_wall_polygons(_list(spec, "wall_polygons"))
    except ValueError as exc:
        raise ValueError(f"wall_{exc}") from exc
    openings = [
        opening_from_dict(item, f"openings[{i}]") for i, item in enumerate(_list(spec, "openings"))
    ]
    try:
        validate_openings(walls, openings)
        geometry = wall_geometry(walls, openings, poche=poche)
    except ValueError as exc:
        raise ValueError(f"walls/openings: {exc}") from exc
    hosts = {wall.id: wall for wall in walls}
    openings = [resolve_door(hosts[op.wall], op, f"openings[{i}]") for i, op in enumerate(openings)]

    stairs = []
    for i, item in enumerate(_list(spec, "stairs")):
        stair = stair_from_dict(item, f"stairs[{i}]")
        try:
            stair_prims(stair, lang=lang, scale=scale)
        except ValueError as exc:
            raise ValueError(f"stairs[{i}]: {exc}") from exc
        stairs.append(stair)

    separators = []
    seen_separators: set[str] = set()
    for i, item in enumerate(_list(spec, "room_separators")):
        separator = room_separator_from_dict(item, f"room_separators[{i}]")
        if separator.id in seen_separators:
            raise ValueError(f"room_separators[{i}].id: {separator.id!r} is used twice")
        seen_separators.add(separator.id)
        separators.append(separator)

    rooms = []
    seen_rooms: set[str] = set()
    for i, item in enumerate(_list(spec, "rooms")):
        path = f"rooms[{i}]"
        fields = (
            {k: v for k, v in item.items() if k not in ROOM_LABEL_KEYS}
            if isinstance(item, dict)
            else item
        )
        room = room_from_dict(fields, path)
        if room.id in seen_rooms:
            raise ValueError(f"{path}.id: {room.id!r} is used twice")
        seen_rooms.add(room.id)
        align = item.get("align", "left")
        if align not in _rooms.ALIGNS:
            raise ValueError(
                f"{path}.align: {align!r} is not an alignment; "
                f"alignments are {', '.join(_rooms.ALIGNS)}"
            )
        fit = item.get("fit", False)
        if not isinstance(fit, bool):
            raise ValueError(f"{path}.fit: expected true or false, got {fit!r}")
        mask = item.get("mask", False)
        if not isinstance(mask, bool):
            raise ValueError(f"{path}.mask: expected true or false, got {mask!r}")
        rooms.append({"room": dict(fields), "align": align, "fit": fit, "mask": mask})

    chains = spec.get("chains")
    sides: tuple[str, ...] | None = None
    if chains not in (None, False):
        if chains is True:
            chains = {}
        if not isinstance(chains, dict) or set(chains) - {"sides"}:
            raise ValueError(f"chains: expected {{sides: [...]}}, got {chains!r}")
        sides = tuple(chains.get("sides") or ("bottom", "left"))
        bad = [side for side in sides if side not in CHAIN_SIDES]
        if bad or not sides:
            raise ValueError(f"chains.sides: {bad or sides!r} - sides are {', '.join(CHAIN_SIDES)}")

    schedules = []
    for i, item in enumerate(_list(spec, "schedules")):
        if not isinstance(item, dict) or set(item) - {"kind", "at"}:
            raise ValueError(f"schedules[{i}]: expected {{kind, at}}, got {item!r}")
        kind = item.get("kind")
        if kind not in SCHEDULE_KINDS:
            raise ValueError(
                f"schedules[{i}].kind: {kind!r} is not a schedule; "
                f"schedules are {', '.join(SCHEDULE_KINDS)}"
            )
        schedules.append({"kind": kind, "at": _point(item.get("at"), f"schedules[{i}].at")})

    return {
        "sheet": {
            "intent": str(sheet.get("intent") or "architectural plan"),
            "sheet_size": str(sheet.get("sheet_size") or "A3"),
        },
        "scale": scale,
        "lang": lang,
        "poche": poche,
        "walls": walls,
        "wall_polygons": polygons,
        "openings": openings,
        "omitted": list(geometry.omitted),
        "stairs": stairs,
        "room_separators": separators,
        "rooms": rooms,
        "chains": sides,
        "schedules": schedules,
    }


async def _extents(backend: AutoCADBackend, handles) -> dict | None:
    """The drawn extents of ``handles`` as ``{min, max}``, read back from the drawing.

    A DIMENSION's box is its rendered ``*D`` block, so the text, the arrows and
    the extension lines count, not only the dimension line.
    """
    boxes = []
    for handle in handles:
        box = (await backend.entity_get(handle)).properties.get("bounding_box")
        if box:
            boxes.append(box)
    if not boxes:
        return None
    return {
        "min": [min(b["min"][0] for b in boxes), min(b["min"][1] for b in boxes)],
        "max": [max(b["max"][0] for b in boxes), max(b["max"][1] for b in boxes)],
    }


async def draw_plan_from_spec(backend: AutoCADBackend, spec: dict) -> dict:
    """Validate, then draw the whole plan inside one transaction.

    Returns ``{"walls", "openings", "stairs", "room_separators", "rooms", "chains", "schedules",
    "plan_bbox", "outer_bbox", "omitted", "drawn", "critique", "warnings", "backend"}``:
    ``plan_bbox`` is the drawn plan (walls, openings, stairs, separators, room labels),
    ``outer_bbox`` the same with the dimension chains' rendered text and lines - where
    section marks, a north arrow or a view title can start; schedules are in neither,
    since their ``at`` is the caller's. The wall and opening ids that
    were drawn (an opening the wall engine could not cut is in ``omitted``, not
    in ``openings``), each drawing step's own result (``drawn`` is
    ``draw_walls``'), and the three ``arch_*`` critique focuses run over the
    committed drawing.
    """
    from engineering.arch.critique import ARCH_FOCUSES, build_index, issues_for

    norm = validate_spec(spec)
    scale = norm["scale"]
    lang = norm["lang"]
    if backend.get_plan_spec() is None:
        await backend.drawing_plan(
            norm["sheet"]["intent"], norm["sheet"]["sheet_size"], 1.0 / scale, "arch"
        )

    begun = await backend.transaction_begin()
    if not isinstance(begun, dict) or not begun.get("ok"):
        # a refusal, not a crash: nothing has been drawn yet
        detail = begun.get("error") if isinstance(begun, dict) else begun
        raise ValueError(
            f"arch_plan_from_spec: could not open a transaction ({detail}); nothing was drawn. "
            "If one is already open, close it with transaction_commit or "
            "transaction_rollback and retry."
        )

    result: dict = {
        "walls": None,
        "wall_polygons": [],
        "stairs": [],
        "room_separators": [],
        "rooms": [],
        "chains": None,
        "schedules": [],
    }
    warnings: list[dict] = []
    try:
        await backend.drawing_apply_iso_layers("arch")
        result["walls"] = await _draw.draw_walls(
            backend, norm["walls"], norm["openings"], poche=norm["poche"], scale=norm["scale"]
        )
        result["wall_polygons"] = await _draw.draw_wall_polygons(
            backend, norm["wall_polygons"], poche=norm["poche"], scale=norm["scale"]
        )
        for i, stair in enumerate(norm["stairs"]):
            try:
                result["stairs"].append(
                    await _draw.draw_stair(backend, stair, lang=lang, scale=scale)
                )
            except ValueError as exc:
                raise ValueError(f"stairs[{i}]: {exc}") from exc
        result["room_separators"] = await _draw.draw_room_separators(
            backend, norm["room_separators"]
        )
        for i, room in enumerate(norm["rooms"]):
            try:
                result["rooms"].append(
                    await _rooms.label_room(
                        backend,
                        room["room"],
                        lang=lang,
                        scale=scale,
                        align=room["align"],
                        fit=room["fit"],
                        mask=room["mask"],
                        open_room=True,
                    )
                )
            except ValueError as exc:
                raise ValueError(f"rooms[{i}]: {exc}") from exc
            warning = result["rooms"][-1].get("warning")
            if warning:
                warnings.append({"element": f"rooms[{i}]", **warning})
        if norm["chains"] is not None:
            result["chains"] = await _draw.draw_chains(
                backend,
                norm["walls"],
                norm["openings"],
                sides=norm["chains"],
                scale=scale,
                polygons=await _draw.read_wall_outlines(backend),
            )
        for i, item in enumerate(norm["schedules"]):
            try:
                result["schedules"].append(
                    await _schedule.draw_schedule(
                        backend, item["kind"], at=item["at"], lang=lang, scale=scale
                    )
                )
            except ValueError as exc:
                raise ValueError(f"schedules[{i}]: {exc}") from exc
        # the spec, each element's handles and their digests: what arch_plan_patch edits
        from engineering.arch.patch import owned_from, write_record

        await write_record(backend, spec, owned_from(result))
    except BaseException as exc:
        cancelled = isinstance(exc, anyio.get_cancelled_exc_class())
        with anyio.CancelScope(shield=True):
            try:
                await backend.transaction_rollback()
            except Exception as rollback_exc:
                if not cancelled:
                    raise
                log.error(
                    "arch_plan_from_spec: rollback after a cancelled request failed (%s: %s); "
                    "the plan may be half-drawn - check system_status and "
                    "transaction_rollback before retrying",
                    type(rollback_exc).__name__,
                    rollback_exc,
                )
        raise
    await backend.transaction_commit()

    index = await build_index(backend)
    critique = [issue.to_dict() for focus in ARCH_FOCUSES for issue in issues_for(focus, index)]
    drawn = result["walls"] or {}
    plan_handles = [
        *drawn.get("handles", ()),
        *(h for item in result["wall_polygons"] for h in item["handles"]),
        *(h for item in result["stairs"] for h in item["handles"]),
        *(item["handle"] for item in result["room_separators"]),
        *(h for item in result["rooms"] for h in item.get("handles", ())),
    ]
    plan_bbox = await _extents(backend, plan_handles)
    outer_bbox = await _extents(
        backend, plan_handles + list((result["chains"] or {}).get("handles", ()))
    )
    omitted = list(drawn.get("omitted", norm["omitted"]))
    skipped = {str(item.get("element")) for item in omitted}
    sheet_scale = (spec.get("sheet") or {}).get("scale")
    linetypes = None
    if sheet_scale:  # round 7: linetypes plot at their pattern size; LTSCALE stays 1
        linetypes = await backend.sheet_linetype_scale(float(sheet_scale))
    return {
        "walls": [wall.id for wall in norm["walls"]],
        "wall_polygons": [polygon.id for polygon in norm["wall_polygons"]],
        "openings": [o.id for o in norm["openings"] if o.id not in skipped],
        "stairs": result["stairs"],
        "room_separators": result["room_separators"],
        "rooms": result["rooms"],
        "chains": result["chains"],
        "schedules": result["schedules"],
        "plan_bbox": plan_bbox,
        "outer_bbox": outer_bbox,
        "omitted": omitted,
        "drawn": drawn,
        "critique": critique,
        "warnings": warnings,
        # round 4: what a caller restyles (poche="solid" fills A-WALL-H-N, not
        # the outline layer) and what `sheet` did - no layout is made
        "layers": {"walls": ARCH_ROLE_LAYER["wall"], "poche": ARCH_ROLE_LAYER["wall_hatch"]},
        "sheet": {
            **norm["sheet"],
            "used_for": "lettering scale, critique, CANNOSCALE; no layout is made",
        },
        **({"linetype_scale": linetypes} if linetypes else {}),
        "backend": backend.name,
    }
