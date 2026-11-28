"""The async layer of the architectural track - the only module here that touches a backend.

Everything else under `engineering/arch/` is pure; this module turns its
primitives into entities through the generic contract members and writes the plan onto the drawing:

* each wall's ``ACADMCP_ARCH`` record on its first face line, each opening's
  on its door leaf or its window frame, each stair's on its first riser line;
* next to each record, under ``ACADMCP_ARCH_H``, the handles of every entity
  that element owns - faces, jambs, poche, the symbol - so a later call can
  take exactly those entities back out and redraw them.

`read_plan` reads the records back. `add_walls` and `add_opening` build on
it: they compute the whole network before and after the change, redraw only
the walls whose outline the change touches, and say which.

Every refusal fires before the first entity is created or erased.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

from engineering.arch.dimension import chain_text_positions, exterior_chains
from engineering.arch.lang import tag_prefix
from engineering.arch.layers import ARCH_LAYERS, ARCH_ROLE_LAYER
from engineering.arch.model import (
    APP_ID,
    Opening,
    Room,
    RoomSeparator,
    Stair,
    Wall,
    WallPolygon,
    decode,
    from_payload,
    opening_from_dict,
    resolve_corner,
    stair_from_dict,
    to_payload,
    to_values,
    wall_from_dict,
    wall_polygon_from_dict,
)
from engineering.arch.openings import opening_prims, resolve_door
from engineering.arch.stairs import blondel, stair_prims
from engineering.arch.walls import WallGeometry, wall_geometry_by_wall
from engineering.mech.draw import draw_prims
from engineering.mech.primitives import HatchArea, Poly

log = logging.getLogger(__name__)

#: The application that carries an element's owned handles beside its record.
HANDLES_APP = "ACADMCP_ARCH_H"

#: The layers `read_plan` scans for records - never the poche or the dimensions.
#: A cased opening has no leaf: its record sits on its head line, on the overhead layer.
RECORD_ROLES: tuple[str, ...] = (
    "wall",
    "wall_removed",
    "door",
    "window",
    "stair",
    "room",
    "overhead",
    "room_separator",
)

#: Model class -> the record kind, read off the object rather than the payload,
#: so the reader does not depend on how a payload spells its own kind.
_KIND = {
    Wall: "wall",
    Opening: "opening",
    Stair: "stair",
    Room: "room",
    WallPolygon: "wall_polygon",
    RoomSeparator: "room_separator",
}

#: Layers whose name starts with this are wall layers: a closed straight-edged
#: polyline on one of them is a wall outline, ours or the drafter's own.
WALL_LAYER_PREFIX = "A-WALL"


# -- layers -----------------------------------------------------------------------


async def ensure_arch_layers(backend, names) -> tuple[str, ...]:
    """Create the named layers the drawing lacks, with their ``ARCH_LAYERS`` row.

    AutoCAD refuses ``entity.Layer = name`` on a missing layer (measured on
    AutoCAD 2026), so every draw path ensures its layers first.
    """
    rows = {row[0]: row for row in ARCH_LAYERS}
    existing = {layer.name.lower() for layer in await backend.layer_list()}
    created: list[str] = []
    for name in dict.fromkeys(names):
        if name.lower() in existing:
            continue
        _name, color, linetype, lineweight, _description = rows.get(
            name, (name, 7, "Continuous", 0.25, "")
        )
        await backend.layer_create(name=name, color=color, linetype=linetype, lineweight=lineweight)
        created.append(name)
    return tuple(created)


# -- the underlay of an installation sheet ------------------------------------------

#: The layer state holding the architecture layers' own colour and weight while
#: the underlay is on (an ``ACADMCP_LAYERSTATES`` entry, listed by layer_state_list).
UNDERLAY_STATE = "ARCH_UNDERLAY_BEFORE"
UNDERLAY_HATCH_COLOR = 253  # light grey: a poche stays readable without shouting


def _underlay_layers(doc) -> list[str]:
    """The architecture layers: every ``A-*`` but the installation layers, any
    layer carrying an installation record, the symbol layer and any layer
    with the sheet's legend / title block (round 7: A-SYMB-T-N greyed the
    legend, drawn before or after), and the construction scratch."""
    from engineering.annotation_record import read_annotation
    from engineering.arch.services import SERVICE_ROLES, read_records

    keep = {ARCH_ROLE_LAYER[r].casefold() for r in (*SERVICE_ROLES, "symbol")}
    keep |= {e.dxf.layer.casefold() for e, _p in read_records(doc)}
    sheet_tools = {"arch_legend", "arch_title_block"}
    for space in (doc.modelspace(), *(doc.layouts.get(n) for n in doc.layouts.names())):
        for e in space:
            if (read_annotation(e) or {}).get("tool") in sheet_tools:
                keep.add(e.dxf.layer.casefold())
    return [
        lyr.dxf.name
        for lyr in doc.layers
        if lyr.dxf.name.casefold().startswith("a-")
        and lyr.dxf.name.casefold() not in keep
        and "CONST" not in lyr.dxf.name.upper()
    ]


async def arch_underlay(backend, *, on: bool = True, color: int = 8, lineweight: float = 0.18):
    """Turn the architecture into a grey, thin background for an installation
    sheet, or put its own colours and weights back (``on=False``).

    Every architecture layer (:func:`_underlay_layers`) takes ``color`` and
    ``lineweight``; a hatch layer (``-H-`` presentation field) takes light grey
    :data:`UNDERLAY_HATCH_COLOR` and keeps a thinner weight of its own. The
    installation layers are never touched. The layers' own values are saved
    once, in layer state :data:`UNDERLAY_STATE`, and ``on=False`` restores
    exactly those layers and deletes the state - a second ``on=True`` re-greys
    with new values without losing the originals.
    """
    from engineering.environment.layer_states import DICT_NAME, decode_state
    from engineering.errors import InvalidArgsError
    from engineering.plan_spec import ISO_128_LINEWEIGHTS_MM

    if isinstance(color, bool) or not isinstance(color, int) or not 1 <= color <= 255:
        raise InvalidArgsError(f"invalid_args: color: an ACI 1-255, got {color!r}.")
    if lineweight not in ISO_128_LINEWEIGHTS_MM:
        raise InvalidArgsError(
            f"invalid_args: lineweight: an ISO 128 weight "
            f"{', '.join(f'{w:g}' for w in ISO_128_LINEWEIGHTS_MM)} mm, got {lineweight!r}."
        )
    saved = {s["name"].upper() for s in await backend.layer_state_list()}
    if not on:
        if UNDERLAY_STATE not in saved:
            raise InvalidArgsError(
                "invalid_args: on: the underlay is not on (no layer state "
                f"{UNDERLAY_STATE}); nothing to restore."
            )

        def _restore():
            doc = backend._require_doc()
            record = doc.rootdict.get(DICT_NAME).get(UNDERLAY_STATE)
            state = decode_state([t.value for t in record.tags if t.code == 1000])
            done = []
            for name in _underlay_layers(doc):
                snap = state["layers"].get(name)
                if snap is None:  # a layer made since: nothing of its own to restore
                    continue
                lyr = doc.layers.get(name)
                lyr.color = abs(int(snap["color"])) or 7
                lyr.dxf.lineweight = int(snap["lineweight"])
                done.append(name)
            backend._mark_dirty()
            return done

        layers = await backend._async(_restore)
        await backend.layer_state_delete(UNDERLAY_STATE)
        return {"ok": True, "on": False, "layers": layers, "backend": backend.name}
    if UNDERLAY_STATE not in saved:
        await backend.layer_state_save(UNDERLAY_STATE, "arch_underlay: the layers before")

    def _apply():
        doc = backend._require_doc()
        weight = round(lineweight * 100)
        out = []
        for name in _underlay_layers(doc):
            lyr = doc.layers.get(name)
            hatch = "-H-" in name.upper()
            lyr.color = UNDERLAY_HATCH_COLOR if hatch else color
            own = int(lyr.dxf.get("lineweight", -3))
            lyr.dxf.lineweight = min(own, weight) if hatch and own > 0 else weight
            out.append(name)
        backend._mark_dirty()
        return out

    layers = await backend._async(_apply)
    return {
        "ok": True,
        "on": True,
        "layers": layers,
        "color": color,
        "hatch_color": UNDERLAY_HATCH_COLOR,
        "lineweight": lineweight,
        "state": UNDERLAY_STATE,
        "backend": backend.name,
    }


# -- records --------------------------------------------------------------------


async def _write_record(backend, handle: str, element, owned: list[str]) -> None:
    await backend.entity_set_xdata(handle, APP_ID, to_values(to_payload(element)))
    await backend.entity_set_xdata(handle, HANDLES_APP, list(owned))


async def read_plan(backend) -> dict:
    """Every ``ACADMCP_ARCH`` record on the drawing, rebuilt into model objects.

    ``carriers`` maps ``"<kind>:<id>"`` to the entity that carries the record.
    Reads only; the drawing is never modified.
    """
    plan: dict = {
        "walls": [],
        "openings": [],
        "stairs": [],
        "rooms": [],
        "wall_polygons": [],
        "room_separators": [],
        "carriers": {},
    }
    for role in RECORD_ROLES:
        for info in await backend.entity_list(layer_filter=ARCH_ROLE_LAYER[role], limit=100000):
            raw = await backend.entity_get_xdata(info.handle, APP_ID)
            values = (raw.get("xdata") or {}).get(APP_ID) or []
            if not values:
                continue
            element = from_payload(decode(values))
            kind = _KIND[type(element)]
            plan[f"{kind}s"].append(element)
            plan["carriers"][f"{kind}:{element.id}"] = info.handle
    return plan


async def _owned(backend, carrier: str) -> list[str]:
    raw = await backend.entity_get_xdata(carrier, HANDLES_APP)
    return [str(h) for h in ((raw.get("xdata") or {}).get(HANDLES_APP) or [])]


async def _erase(backend, plan: dict, wall_id: str) -> list[str]:
    """Take a wall, its openings and their poche back out. Returns handles already gone."""
    carriers = [plan["carriers"][f"wall:{wall_id}"]]
    carriers += [
        plan["carriers"][f"opening:{op.id}"] for op in plan["openings"] if op.wall == wall_id
    ]
    missing: list[str] = []
    for carrier in carriers:
        for handle in await _owned(backend, carrier) or [carrier]:
            try:
                await backend.entity_delete(handle)
            except Exception as exc:  # erased by hand since it was drawn: nothing to take back
                log.info("arch: %s was already gone (%s)", handle, exc)
                missing.append(handle)
    return missing


# -- walls ------------------------------------------------------------------------


def _signature(geometry: WallGeometry) -> tuple:
    """What a wall looks like, rounded - equal signatures mean nothing to redraw."""

    def seg(line):
        a = (round(line.p1[0], 6), round(line.p1[1], 6))
        b = (round(line.p2[0], 6), round(line.p2[1], 6))
        return tuple(sorted((a, b)))

    def ring(points):
        pts = [(round(x, 6), round(y, 6)) for x, y in points]
        k = pts.index(min(pts))
        return tuple(pts[k:] + pts[:k])

    return (
        frozenset(seg(line) for line in geometry.faces),
        frozenset(seg(line) for line in geometry.jambs),
        frozenset(seg(line) for line in geometry.seams),
        frozenset(seg(line) for line in geometry.thresholds),  # a glass wall's only lines
        frozenset(ring(region) for region in geometry.regions),
        frozenset(entry["element"] for entry in geometry.omitted),
    )


async def _draw_wall(backend, wall, geometry: WallGeometry, openings, scale=50) -> dict:
    """One wall's faces, jambs, seams, poche and hosted symbols, with their records.

    The poche's pattern scale follows the plot scale ``scale`` (50 = 1:50).
    """
    skipped = {entry["element"] for entry in geometry.omitted}
    cuts = {cut["element"]: cut for cut in geometry.cuts}
    # Records are stored resolved: swing_towards turned into swing, hinge beside hand.
    resolved = [resolve_door(wall, op) for op in openings if op.id not in skipped]
    symbols = [(op, opening_prims(wall, op, cuts.get(op.id))) for op in resolved]
    glazing = [p for cut in geometry.cuts if cut.get("glazing") for p in _glazing(wall, cut)]
    # Faces first, so owned[0] - the record carrier - is the wall's first face line
    # (a glass wall's first jamb or glazing line); the poche follows on the
    # wall-hatch layer, in each material's pattern.
    drawn = await draw_prims(
        backend,
        [*geometry.faces, *geometry.jambs, *geometry.seams, *glazing, *geometry.hatches],
        role_layer=ARCH_ROLE_LAYER,
        plot_scale=scale,
    )
    owned = list(drawn["handles"])
    done = [await _draw_symbol(backend, op, prims, wall, scale) for op, prims in symbols]
    await _write_record(backend, owned[0], wall, owned)
    return {
        "wall": {"id": wall.id, "record": owned[0], "handles": owned},
        "openings": done,
        "omitted": list(geometry.omitted),
    }


def _glazing(wall, cut: dict) -> tuple:
    """A glass wall's glazed gap, drawn as the curtain-wall symbol of its cut."""
    width = max(math.dist(*cut["left"]), math.dist(*cut["right"]))
    op = Opening(cut["element"], wall.id, "curtain_wall", 0.0, width)
    return opening_prims(wall, op, cut)


async def _draw_symbol(backend, op: Opening, prims, wall=None, scale=50) -> dict:
    """One opening's symbol with its record on the first entity; its ``label``
    (round 7) lettered beside it, outside the swing, along the wall."""
    if op.label and wall is not None:
        prims = (*prims, _label_prim(wall, op, scale))
    placed = await draw_prims(
        backend, prims, role_layer=ARCH_ROLE_LAYER, annotation="arch_plan_from_spec"
    )
    await _write_record(backend, placed["handles"][0], op, placed["handles"])
    return {
        "id": op.id,
        "tag": op.tag,
        "record": placed["handles"][0],
        "handles": placed["handles"],
    }


async def draw_opening(backend, wall, geometry: WallGeometry, op: Opening, scale=50) -> dict:
    """Redraw one opening's symbol in a wall whose cut did not move (a hinge flip)."""
    cut = next((c for c in geometry.cuts if c["element"] == op.id), None)
    op = resolve_door(wall, op)
    return await _draw_symbol(backend, op, opening_prims(wall, op, cut), wall, scale)


#: An opening label's paper height and its clearance from the wall face (mm).
LABEL_HEIGHT, LABEL_GAP = 2.0, 1.5


def _label_prim(wall, op: Opening, scale: float):
    """The label's TEXT: centred on the opening's middle, beyond the face
    opposite the door's swing (the outer face for a window), turned along the
    wall and kept readable (never upside down)."""
    from engineering.arch.model import segment_of
    from engineering.arch.walls import face_offsets, wall_segments
    from engineering.mech.primitives import Text

    index, along = segment_of(wall, float(op.offset) + float(op.width) / 2.0)
    (ax, ay), (bx, by) = wall_segments(wall)[index]
    length = math.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / length, (by - ay) / length
    left, right = face_offsets(wall)
    h, gap = LABEL_HEIGHT * scale, LABEL_GAP * scale
    side = -1.0 if op.swing == "in" else 1.0  # away from the swing (in = the axis's left)
    reach = (right - gap - h / 2.0) if side < 0 else (left + gap + h / 2.0)
    mx, my = ax + ux * along - uy * reach, ay + uy * along + ux * reach
    angle = math.degrees(math.atan2(uy, ux))
    if angle > 90.0 + 1e-9 or angle <= -90.0 + 1e-9:
        angle -= 180.0 if angle > 0 else -180.0
    return Text((mx, my), op.label, h, angle, "symbol", "middle_center")


def _collect(results: list[dict]) -> dict:
    return {
        "walls": [r["wall"] for r in results],
        "openings": [op for r in results for op in r["openings"]],
        "handles": [
            h
            for r in results
            for h in (*r["wall"]["handles"], *(x for op in r["openings"] for x in op["handles"]))
        ],
        "omitted": [entry for r in results for entry in r["omitted"]],
    }


async def draw_walls(backend, walls, openings=(), *, poche=True, scale=50) -> dict:
    """Draw a wall network and its openings, writing every record. Refusals first."""
    walls = list(walls)
    openings = list(openings)
    geometry = wall_geometry_by_wall(walls, openings, poche=poche)
    for wall in walls:  # every symbol is validated before the first entity
        for op in openings:
            if op.wall == wall.id:
                opening_prims(wall, op)
    results = [
        await _draw_wall(
            backend,
            wall,
            geometry[wall.id],
            [op for op in openings if op.wall == wall.id],
            scale,
        )
        for wall in walls
    ]
    return {**_collect(results), "backend": backend.name}


# -- wall polygons ----------------------------------------------------------------


def wall_polygon_prims(polygon: WallPolygon, *, poche: bool = True) -> tuple:
    """The outline as one closed polyline on the wall layer, then its poche."""
    role = "wall_removed" if polygon.material == "demolition" else "wall"
    outline = Poly(tuple(polygon.points), True, role)
    if not poche:
        return (outline,)
    material = "solid" if poche == "solid" else polygon.material or "generic"
    return (outline, HatchArea((tuple(polygon.points),), material))


async def draw_wall_polygons(backend, polygons, *, poche: bool = True, scale=50) -> list[dict]:
    """Each outline and its poche, with the record on the polyline."""
    out: list[dict] = []
    for polygon in polygons:
        drawn = await draw_prims(
            backend,
            wall_polygon_prims(polygon, poche=poche),
            role_layer=ARCH_ROLE_LAYER,
            plot_scale=scale,
        )
        await _write_record(backend, drawn["handles"][0], polygon, drawn["handles"])
        out.append(
            {"id": polygon.id, "record": drawn["handles"][0], "handles": list(drawn["handles"])}
        )
    return out


def room_separator_prims(separator: RoomSeparator) -> tuple:
    """The separation line as one open polyline: zero thickness, no poche."""
    return (Poly(tuple(separator.points), False, "room_separator"),)


async def draw_room_separators(backend, separators) -> list[dict]:
    """Each separation line, with its record on the polyline."""
    out: list[dict] = []
    for separator in separators:
        drawn = await draw_prims(
            backend, room_separator_prims(separator), role_layer=ARCH_ROLE_LAYER
        )
        await _write_record(backend, drawn["handles"][0], separator, drawn["handles"])
        out.append({"id": separator.id, "handle": drawn["handles"][0]})
    return out


async def add_room_separator(backend, separator: dict) -> dict:
    """Draw one room separation line on a plan, with its record; ``id`` defaults to ``S<n>``.

    It bounds rooms from then on (`arch_room`, `arch_rooms_detect`, the
    critique). A label already standing without an area (``room_open``) is not
    relabelled: delete it and label the room again to measure it. Returns
    ``open_ends``, the free wall and separator ends left on the plan.
    """
    from engineering.arch.critique import open_ends  # lazy: critique imports the QA scan
    from engineering.arch.model import room_separator_from_dict

    if not isinstance(separator, dict):
        raise ValueError(f"separator: expected a dict, got {type(separator).__name__}")
    plan = await read_plan(backend)
    taken = {sep.id for sep in plan["room_separators"]}
    data = dict(separator)
    if not data.get("id"):
        number = len(taken) + 1
        while f"S{number}" in taken:
            number += 1
        data["id"] = f"S{number}"
    sep = room_separator_from_dict(data, "separator")
    if sep.id in taken:
        raise ValueError(f"separator.id: {sep.id!r} is already drawn; choose a new id")
    (drawn,) = await draw_room_separators(backend, [sep])
    return {
        "ok": True,
        "id": sep.id,
        "handle": drawn["handle"],
        "layer": ARCH_ROLE_LAYER["room_separator"],
        "points": [[float(x), float(y)] for x, y in sep.points],
        "open_ends": open_ends(plan["walls"], [*plan["room_separators"], sep]),
        "backend": backend.name,
    }


def parse_wall_polygons(polygons, taken=()) -> list[WallPolygon]:
    """Validated polygons, refusing an id used twice or already on the drawing."""
    if isinstance(polygons, (dict, str)) or not isinstance(polygons, (list, tuple)):
        raise ValueError("polygons: give a list of {id, points, material}")
    out: list[WallPolygon] = []
    seen = set(taken)
    for i, item in enumerate(polygons):
        polygon = wall_polygon_from_dict(item, f"polygons[{i}]")
        if polygon.id in seen:
            raise ValueError(f"polygons[{i}].id: {polygon.id!r} is already used; choose a new id")
        seen.add(polygon.id)
        out.append(polygon)
    return out


async def read_wall_outlines(backend, plan: dict | None = None) -> list[tuple]:
    """Every wall outline on the drawing as a tuple of points.

    The ``wall_polygon`` records first, then every closed polyline without a
    bulge on a layer named ``A-WALL*`` that carries no record - a bay, a
    column or a wall imported from another model, drawn by hand. Reads only.
    """
    from engineering.sheet.bom import all_entities  # lazy: sheet imports stay out of arch

    plan = plan if plan is not None else await read_plan(backend)
    outlines = [tuple(polygon.points) for polygon in plan["wall_polygons"]]
    carriers = set(plan["carriers"].values())
    hatch_layer = ARCH_ROLE_LAYER["wall_hatch"].casefold()
    for layer in await backend.layer_list():
        name = layer.name.casefold()
        if not name.startswith(WALL_LAYER_PREFIX.casefold()) or name == hatch_layer:
            continue
        for entity in await all_entities(backend, layer=layer.name):
            props = entity.properties or {}
            if entity.handle in carriers or str(entity.type).upper() not in (
                "LWPOLYLINE",
                "POLYLINE",
            ):
                continue
            points = [tuple(float(v) for v in p[:2]) for p in props.get("points") or ()]
            bulges = [float(b) for b in props.get("bulges") or ()]
            if props.get("closed") and len(points) >= 3 and not any(abs(b) > 1e-12 for b in bulges):
                outlines.append(tuple(points))
    return outlines


async def add_walls(backend, walls, *, poche=True, polygons=(), scale=50) -> dict:
    """Add walls and wall polygons to the plan, redrawing the drawn walls the walls join."""
    if isinstance(walls, dict) or not isinstance(walls, (list, tuple)):
        raise ValueError("walls: give a list of walls {id, axis, thickness, ...}")
    if not walls and not polygons:
        raise ValueError(
            "walls: give at least one wall {id, axis, thickness, ...} or one polygon "
            "{id, points, material}"
        )
    new = [wall_from_dict(item, f"walls[{i}]") for i, item in enumerate(walls)]
    plan = await read_plan(backend)
    outlines = parse_wall_polygons(
        polygons, taken=[polygon.id for polygon in plan["wall_polygons"]]
    )
    if not new:
        drawn_polygons = await draw_wall_polygons(backend, outlines, poche=poche, scale=scale)
        return {
            "walls": [],
            "openings": [],
            "polygons": drawn_polygons,
            "handles": [h for item in drawn_polygons for h in item["handles"]],
            "omitted": [],
            "added": [polygon.id for polygon in outlines],
            "redrawn": [],
            "already_erased": [],
            "backend": backend.name,
        }
    drawn = {wall.id for wall in plan["walls"]}
    for i, wall in enumerate(new):
        if wall.id in drawn:
            raise ValueError(
                f"walls[{i}].id: {wall.id!r} is already drawn; choose a new id "
                f"(drawn: {', '.join(sorted(drawn))})"
            )
    before = (
        wall_geometry_by_wall(plan["walls"], plan["openings"], poche=poche) if plan["walls"] else {}
    )
    network = [*plan["walls"], *new]
    after = wall_geometry_by_wall(network, plan["openings"], poche=poche)
    changed = [
        wall.id
        for wall in plan["walls"]
        if _signature(before[wall.id]) != _signature(after[wall.id])
    ]
    for wall_id in changed:
        was = {entry["element"] for entry in before[wall_id].omitted}
        for entry in after[wall_id].omitted:
            if entry["element"] not in was:
                raise ValueError(
                    f"the new walls would leave opening {entry['element']!r} of wall "
                    f"{wall_id!r} uncuttable ({entry['reason']}); move the opening or the wall"
                )
    by_id = {wall.id: wall for wall in network}
    targets = [*changed, *(wall.id for wall in new)]
    for wall_id in targets:
        for op in plan["openings"]:
            if op.wall == wall_id:
                opening_prims(by_id[wall_id], op)

    missing: list[str] = []
    for wall_id in changed:
        missing += await _erase(backend, plan, wall_id)
    results = [
        await _draw_wall(
            backend,
            by_id[wall_id],
            after[wall_id],
            [op for op in plan["openings"] if op.wall == wall_id],
            scale,
        )
        for wall_id in targets
    ]
    drawn_polygons = await draw_wall_polygons(backend, outlines, poche=poche, scale=scale)
    collected = _collect(results)
    return {
        **collected,
        "polygons": drawn_polygons,
        "handles": [
            *collected["handles"],
            *(h for item in drawn_polygons for h in item["handles"]),
        ],
        "added": [wall.id for wall in new] + [polygon.id for polygon in outlines],
        "redrawn": changed,
        "already_erased": missing,
        "backend": backend.name,
    }


# -- openings ---------------------------------------------------------------------


def assign_tags(openings, existing=(), *, lang: str = "en") -> tuple[Opening, ...]:
    """Give every untagged opening the next free tag of its kind: D1, W2 (EN) / K1, P2 (TR)."""
    used = {op.tag for op in (*existing, *openings) if op.tag}
    out: list[Opening] = []
    for op in openings:
        if op.tag:
            out.append(op)
            continue
        prefix = tag_prefix(op.kind, lang)
        number = 1
        while f"{prefix}{number}" in used:
            number += 1
        used.add(f"{prefix}{number}")
        out.append(replace(op, tag=f"{prefix}{number}"))
    return tuple(out)


async def add_opening(
    backend, opening: dict, *, lang: str = "en", poche: bool = True, scale=50
) -> dict:
    """Host a door or a window in a drawn wall, found by id, and redraw that wall cut."""
    if not isinstance(opening, dict):
        raise ValueError(f"opening: expected a dict, got {type(opening).__name__}")
    raw = dict(opening)
    auto_id = not raw.get("id")
    if auto_id:
        raw["id"] = "__new__"
    op = opening_from_dict(raw, "opening")
    plan = await read_plan(backend)
    walls = {wall.id: wall for wall in plan["walls"]}
    if op.wall not in walls:
        names = ", ".join(sorted(walls)) or "none - draw one with arch_wall"
        raise ValueError(f"opening.wall: {op.wall!r} is not a wall on this drawing; walls: {names}")
    op = resolve_door(walls[op.wall], resolve_corner(walls[op.wall], op), "opening")
    (op,) = assign_tags([op], plan["openings"], lang=lang)
    if auto_id:
        op = replace(op, id=op.tag)
    if op.id in {other.id for other in plan["openings"]}:
        raise ValueError(f"opening.id: {op.id!r} is already drawn; choose a new id")
    hosted = [other for other in plan["openings"] if other.wall == op.wall] + [op]
    geometry = wall_geometry_by_wall(plan["walls"], [*plan["openings"], op], poche=poche)
    for entry in geometry[op.wall].omitted:
        if entry["element"] == op.id:
            raise ValueError(f"opening {op.id!r} cannot be cut: {entry['reason']}")
    wall = walls[op.wall]
    for other in hosted:
        opening_prims(wall, other)
    missing = await _erase(backend, plan, op.wall)
    result = await _draw_wall(backend, wall, geometry[op.wall], hosted, scale)
    return {
        **_collect([result]),
        "opening": op.id,
        "tag": op.tag,
        "redrawn": [op.wall],
        "already_erased": missing,
        "backend": backend.name,
    }


# -- stairs -----------------------------------------------------------------------


async def draw_stair(backend, stair: Stair, *, lang: str = "en", scale=50) -> dict:
    """A stair in plan with its record; the Blondel value is reported, never enforced."""
    prims = stair_prims(stair, lang=lang, scale=scale)
    check = blondel(stair.riser_height, stair.going)
    drawn = await draw_prims(backend, prims, role_layer=ARCH_ROLE_LAYER)
    await _write_record(backend, drawn["handles"][0], stair, drawn["handles"])
    flights = 1 if stair.kind == "straight" else 2
    return {
        "stair": stair.id,
        "record": drawn["handles"][0],
        "handles": drawn["handles"],
        "risers": stair.risers,
        "treads": stair.risers - flights,
        "blondel": check,
        "backend": backend.name,
    }


async def add_stair(backend, stair: dict, *, lang: str = "en", scale=50) -> dict:
    """`draw_stair` for a stair given as a dict, refusing an id already on the drawing."""
    if not isinstance(stair, dict):
        raise ValueError(f"stair: expected a dict, got {type(stair).__name__}")
    model = stair_from_dict(stair, "stair")
    stair_prims(model, lang=lang, scale=scale)
    plan = await read_plan(backend)
    if model.id in {other.id for other in plan["stairs"]}:
        raise ValueError(f"stair.id: {model.id!r} is already drawn; choose a new id")
    return await draw_stair(backend, model, lang=lang, scale=scale)


# -- dimensions ---------------------------------------------------------------------


async def draw_chains(
    backend,
    walls,
    openings,
    *,
    sides=("bottom", "left"),
    scale=50,
    first_offset=None,
    step=None,
    polygons=(),
) -> dict:
    """The exterior chains as real linear DIMENSIONs on the dimension layer.

    Their text, arrows and gaps follow the plot scale like every other plan
    annotation: DIMSCALE is ``scale`` while the chains are drawn, so each chain
    dimension carries it, and the drawing's own DIMSCALE is put back afterwards.
    Drawn at the style's DIMSCALE 1, a 1:50 chain's 2.5 text was 2.5 drawing
    units - 0.05 mm on the sheet, which is no text at all.
    """
    intents = exterior_chains(
        walls,
        openings,
        sides=sides,
        first_offset=first_offset,
        step=step,
        scale=scale,
        polygons=polygons,
    )
    handles, rows = await draw_dimension_intents(backend, intents, scale=scale)
    return {
        "count": len(handles),
        "handles": handles,
        "dimensions": rows,
        "dimscale": float(scale),
        "backend": backend.name,
    }


def dimstyle_for(doc, wanted: str | None) -> str | None:
    """The dimension style ``wanted`` names: None (the current one), ``cm`` (the
    style made from the ``arch-cm`` preset - DIMLFAC 0.1, DIMRND 0.5, DIMDEC 1 -
    the current one first) or a style name; refused with the styles there are."""
    if wanted is None:
        return None
    styles = {s.dxf.name: s for s in doc.dimstyles}
    if wanted.lower() == "cm":
        from engineering.standards.dimstyles import ARCH_CM

        def cm(s):
            return all(
                math.isclose(float(s.dxf.get(k.lower(), 0) or 0), float(ARCH_CM[k]))
                for k in ("DIMLFAC", "DIMRND", "DIMDEC")
            )

        found = sorted(name for name, s in styles.items() if cm(s))
        current = doc.header.get("$DIMSTYLE", "")
        if current in found:
            return current
        if len(found) == 1:
            return found[0]
        raise ValueError(
            "spec.dimensions: cm needs one dimension style from the arch-cm preset; "
            + (f"several are ({', '.join(found)}): name one" if found else
               "there is none: dimstyle_create(name, preset='arch-cm') first")
        )  # fmt: skip
    match = next((n for n in styles if n.lower() == wanted.lower()), None)
    if match is None:
        raise ValueError(
            f"spec.dimensions: {wanted!r} is not none, cm or a dimension style; the styles "
            f"are {', '.join(sorted(styles))}"
        )
    return match


async def draw_dimension_intents(
    backend, intents, *, scale=50, dimstyle: str | None = None
) -> tuple[list[str], list[dict]]:
    """Linear ``DimIntent`` rows (``feature`` = ``"<side>:<row>"``) as real DIMENSIONs.

    Drawn on the dimension layer at DIMSCALE = ``scale``, the drawing's own
    DIMSCALE put back afterwards; the texts are placed by
    ``chain_text_positions``. Shared by the plan chains and the interior
    elevations. Returns the handles and one row per dimension. ``dimstyle``
    draws them with that style, the current style and every ``$DIM*`` header
    variable put back afterwards.
    """
    if dimstyle is None:
        return await _draw_dimension_intents(backend, intents, scale=scale)

    def header(saved=None):
        h = backend._require_doc().header
        if saved is None:
            return {k: h.get(k) for k in h.varnames() if k.startswith("$DIM")}
        for k, v in saved.items():
            h[k] = v
        return saved

    saved = await backend._async(header, readonly=True)
    await backend.dimstyle_set_current(dimstyle)
    try:
        return await _draw_dimension_intents(backend, intents, scale=scale)
    finally:
        await backend._async(lambda: header(saved))


async def _draw_dimension_intents(backend, intents, *, scale=50):
    layer = ARCH_ROLE_LAYER["dim"]
    await ensure_arch_layers(backend, [layer])
    factor = float(scale)
    previous = await backend.system_get_variable("DIMSCALE")
    previous = 1.0 if previous is None else float(previous)
    handles: list[str] = []
    rows: list[dict] = []
    if abs(previous - factor) > 1e-9:
        await backend.system_set_variable("DIMSCALE", factor)
    try:
        style = {
            name: await backend.system_get_variable(name.upper())
            for name in (
                "dimtxt",
                "dimgap",
                "dimtad",
                "dimdec",
                "dimzin",
                "dimdsep",
                "dimrnd",
                "dimpost",
                "dimlfac",
            )
        }
        texts = chain_text_positions(
            intents,
            text_height=float(style["dimtxt"] or 2.5) * factor,
            gap=float(style["dimgap"] or 0.0) * factor,
            tad=int(style["dimtad"] or 0),
            style_vars={k: v for k, v in style.items() if v is not None},
        )
        for intent, text_at in zip(intents, texts, strict=True):
            # the side names the direction: a chain's two points may lie on different
            # features (a wall face and a bay vertex), so their offset says nothing
            horizontal = intent.feature.split(":")[0] in ("bottom", "top")
            info = await backend.dimension_linear(
                intent.p1[0],
                intent.p1[1],
                intent.p2[0],
                intent.p2[1],
                intent.text_at[0],
                intent.text_at[1],
                0.0 if horizontal else 90.0,
                layer,
                text_at=text_at,
            )
            handles.append(info.handle)
            rows.append(
                {
                    "row": intent.feature,
                    "from": list(intent.p1),
                    "to": list(intent.p2),
                    "value": abs(
                        intent.p2[0 if horizontal else 1] - intent.p1[0 if horizontal else 1]
                    ),
                    "text_at": list(text_at),
                }
            )
    finally:
        if abs(previous - factor) > 1e-9:
            await backend.system_set_variable("DIMSCALE", previous)
    return handles, rows


# -- interior elevations -------------------------------------------------------------


async def draw_interior_elevation(
    backend,
    spec: dict,
    *,
    at,
    lang: str = "en",
    scale=20,
    from_plan: dict | None = None,
    dry_run: bool = False,
) -> dict:
    """An interior wall elevation (``engineering/arch/elevation.py``) in one transaction.

    With ``from_plan`` (``{wall, segment, side, at, heights}``) the length and
    the openings are read from the plan on the drawing; ``spec`` then must not
    type them. ``spec.tiling`` lays tiles over the wall with the same engine
    as ``arch_floor_zone``'s ``lay`` and reports its takeoff. Everything is
    validated before the transaction opens; ``dry_run`` returns the levels,
    the dimension rows and the takeoff and draws nothing.
    """
    from engineering.arch import elevation as elev
    from engineering.mech.draw import guarded_transaction
    from engineering.mech.primitives import translate

    spec = dict(spec or {})
    ox, oy = (float(at[0]), float(at[1]))
    source = "typed"
    stretch = None
    if from_plan is not None:
        fp = elev._known(from_plan, "from_plan", ("wall", "segment", "side", "at", "heights"))
        typed = sorted({"length", "openings"} & set(spec))
        if typed:
            raise ValueError(
                f"spec.{typed[0]}: read from the plan when from_plan is given; do not type it"
            )
        wall_id = str(elev._required(fp, "wall", "from_plan"))
        plan = await read_plan(backend)
        walls = {w.id: w for w in plan["walls"]}
        if wall_id not in walls:
            raise ValueError(
                f"from_plan.wall: {wall_id!r} is not drawn; the drawn walls are "
                f"{', '.join(sorted(walls)) or 'none'}"
            )
        segment = fp.get("segment", 0)
        if isinstance(segment, bool) or not isinstance(segment, int):
            raise ValueError(f"from_plan.segment: expected a whole number, got {segment!r}")
        geometry = wall_geometry_by_wall(plan["walls"], plan["openings"], poche=False)
        stretch = elev.plan_wall_stretch(
            walls[wall_id],
            plan["openings"],
            geometry[wall_id].faces,
            segment=segment,
            side=fp.get("side", "left"),
            at=fp.get("at"),
        )
        spec["length"] = stretch["length"]
        spec["openings"] = elev.plan_openings(stretch["openings"], fp.get("heights"))
        source = "plan"
    model = elev.parse_elevation(spec, lang=lang)
    prims = translate(elev.elevation_prims(model, lang=lang, scale=scale), ox, oy)
    intents = elev.shift_intents(elev.elevation_dimensions(model, scale=scale), ox, oy)
    fills = [elev.zone_fill(model, i, scale=scale) for i in range(len(model.zones))]
    style = None
    if intents:
        style = dimstyle_for(backend._require_doc(), model.dimensions)

    async def job():
        drawn = await draw_prims(
            backend, prims, role_layer=ARCH_ROLE_LAYER, annotation="interior_wall_elevation"
        )
        zone_handles = await draw_zone_fills(backend, fills, offset=(ox, oy)) if fills else []
        dims, rows = await draw_dimension_intents(backend, intents, scale=scale, dimstyle=style)
        return drawn, zone_handles, dims, rows

    if dry_run:
        drawn, zone_handles, dims = {"handles": [], "counts": {}}, [{} for _ in fills], []
        rows = [
            {
                "row": i.feature,
                "from": list(i.p1),
                "to": list(i.p2),
                "value": abs(i.p2[0] - i.p1[0]) + abs(i.p2[1] - i.p1[1]),
            }
            for i in intents
        ]
    else:
        drawn, zone_handles, dims, rows = await guarded_transaction(
            backend, job, what="interior_wall_elevation", done="drawn"
        )
    zone_rows = [zone_summary(f, h) for f, h in zip(fills, zone_handles, strict=True)]
    zone_layers = {row["layer"] for row in zone_rows}
    result = {
        "dry_run": bool(dry_run),
        "ok": True,
        "source": source,
        "at": [ox, oy],
        "length": model.length,
        "height": model.height,
        "scale": float(scale),
        "levels": elev.elevation_levels(model),
        "openings": [
            {"id": o.id, "kind": o.kind, "x": o.x, "width": o.width, "sill": o.sill, "head": o.head}
            for o in model.openings
        ],
        "dimensions": rows,
        **({"dimstyle": style} if style else {}),
        "handles": drawn["handles"],
        "dimension_handles": dims,
        "counts": {
            **drawn["counts"],
            "dimension": len(dims),
            "zone_part": sum(len(h.get("all", ())) for h in zone_handles),
        },
        "layers": sorted(
            {ARCH_ROLE_LAYER[r] for r in ("elev_outline", "elev_fitting", "elev_text", "symbol")}
            | ({ARCH_ROLE_LAYER["dim"]} if intents else set())
            | ({ARCH_ROLE_LAYER["elev_service"]} if model.services else set())
            | zone_layers
        ),
        "backend": backend.name,
    }
    if zone_rows:
        result["zones"] = zone_rows
        warnings = [w for f in fills for w in f["warnings"]]
        if warnings:
            result["warnings"] = warnings
    if stretch is not None:
        result["plan"] = {
            "wall": from_plan["wall"],
            "segment": from_plan.get("segment", 0),
            "side": from_plan.get("side", "left"),
            "face_from": stretch["s0"],
            "face_to": stretch["s1"],
            "mirrored": stretch["reverse"],
        }
    return result


# -- wall finish zones ---------------------------------------------------------------


def zone_summary(fill: dict, handles: dict) -> dict:
    """A zone's result row, drawn or dry run alike: measured area, fills,
    the lay takeoff (``start`` elevation-local), what cuts it and its handles."""
    zone = fill["zone"]
    row = {
        "id": zone.id,
        **({"label": zone.label} if zone.label else {}),
        "layer": zone.layer or ARCH_ROLE_LAYER["elev_tile"],
        "area_m2": round(fill["area"] / 1e6, 4),
        "cutouts": fill["cutouts"],
    }
    for key in ("pattern", "joints"):
        if key in fill:
            row[key] = fill[key]
    if "lay" in fill:
        laid, args = fill["lay"], fill["lay"]["args"]
        cut = [p for p in laid["pieces"] if not p.full]
        tile_area = float(args["length"]) * float(args["width"])
        row["lay"] = {
            "pattern": args["pattern"],
            "start": list(laid["origin"]),
            "takeoff": laid["takeoff"],
            "smallest_cut_share": round(min(p.area for p in cut) / tile_area, 4) if cut else None,
        }
    row["handles"] = {k: v for k, v in handles.items() if k != "all"}
    return row


async def draw_zone_fills(backend, fills, *, offset=(0.0, 0.0)) -> list[dict]:
    """Every zone's fills (``arch_floor_zone``'s engine: pattern and joints
    HATCHes, laid pieces, edge) at ``offset``, each part tagged ``ACADMCP_ANNOT``
    with the zone's label as its ``legend`` (``arch_legend``'s row) and the
    zone as its ``set``; sent to the back of the draw order. Inside the caller's
    transaction. One handles dict per zone, ``all`` listing every part."""
    from backends.xdata_specs import encode_values
    from engineering.annotation_record import APP_ID, annotation_values
    from engineering.arch.floor import fill_hatches

    dx, dy = float(offset[0]), float(offset[1])
    await ensure_arch_layers(
        backend, sorted({ARCH_ROLE_LAYER["elev_tile"]} - {f["zone"].layer for f in fills})
    )
    for f in fills:
        if f["zone"].layer:
            from engineering.arch.catalogue import _ensure_layer

            await _ensure_layer(backend, f["zone"].layer)

    def move(loop):
        return [(x + dx, y + dy) for x, y in loop]

    def _draw():
        doc = backend._require_doc()
        msp = doc.modelspace()
        if APP_ID not in doc.appids:
            doc.appids.add(APP_ID)
        out = []
        for f in fills:
            zone, fill = f["zone"], f["zone"].fill
            attribs = {"layer": zone.layer or ARCH_ROLE_LAYER["elev_tile"]}
            if zone.color is not None:
                attribs["color"] = zone.color
            loops = [move(loop) for loop in f["loops"]]
            start = fill["start"] or (zone.x, zone.bottom)
            handles = fill_hatches(
                msp,
                loops,
                attribs,
                pattern=fill["pattern"],
                pscale=f.get("pattern", {}).get("hatch_scale"),
                angle=fill["angle"],
                joints=fill["joints"],
                origin=(start[0] + dx, start[1] + dy),
            )
            lay_spec = None
            if "lay" in f:
                args = f["lay"]["args"]
                lay_spec = {  # what arch_legend's sample redraws
                    "pattern": args["pattern"], "length": float(args["length"]),
                    "width": float(args["width"]), "gap": float(args["joint"]),
                    "angle": float(args["angle_deg"]), "bond": float(args["bond"]),
                    "cut": float(args["cut_deg"]),
                }  # fmt: skip
                handles["pieces"] = [
                    msp.add_lwpolyline(move(loop), close=True, dxfattribs=attribs).dxf.handle
                    for piece in f["lay"]["pieces"]
                    for loop in piece.loops
                ]
            if zone.edge:
                handles["edge"] = [
                    msp.add_lwpolyline(loop, close=True, dxfattribs=attribs).dxf.handle
                    for loop in loops
                ]
            parts = [
                (h, key) for key, v in handles.items() for h in (v if isinstance(v, list) else [v])
            ]
            group = parts[0][0]  # one zone, one legend row: its parts share a set
            for h, key in parts:
                tags = annotation_values(
                    "interior_wall_elevation", h, f"zone:{group}", zone.label,
                    lay_spec if key == "pieces" else None,
                )  # fmt: skip
                doc.entitydb[h].set_xdata(APP_ID, encode_values(tags))
            out.append({**handles, "all": [h for h, _ in parts]})
        backend._mark_dirty()
        return out

    out = await backend._async(_draw)
    drawn = [h for row in out for h in row["all"]]
    if drawn:
        await backend.draw_order(drawn, "send_to_back")
    return out
