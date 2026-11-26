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
    stair_from_dict,
    to_payload,
    to_values,
    wall_from_dict,
    wall_polygon_from_dict,
)
from engineering.arch.openings import opening_prims
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
        frozenset(ring(region) for region in geometry.regions),
        frozenset(entry["element"] for entry in geometry.omitted),
    )


async def _draw_wall(backend, wall, geometry: WallGeometry, openings, scale=50) -> dict:
    """One wall's faces, jambs, seams, poche and hosted symbols, with their records.

    The poche's pattern scale follows the plot scale ``scale`` (50 = 1:50).
    """
    skipped = {entry["element"] for entry in geometry.omitted}
    cuts = {cut["element"]: cut for cut in geometry.cuts}
    symbols = [
        (op, opening_prims(wall, op, cuts.get(op.id))) for op in openings if op.id not in skipped
    ]
    # Faces first, so owned[0] - the record carrier - is the wall's first face line;
    # the poche follows on the wall-hatch layer, in each material's pattern.
    drawn = await draw_prims(
        backend,
        [*geometry.faces, *geometry.jambs, *geometry.seams, *geometry.hatches],
        role_layer=ARCH_ROLE_LAYER,
        plot_scale=scale,
    )
    owned = list(drawn["handles"])
    done: list[dict] = []
    for op, prims in symbols:
        placed = await draw_prims(backend, prims, role_layer=ARCH_ROLE_LAYER)
        await _write_record(backend, placed["handles"][0], op, placed["handles"])
        done.append(
            {
                "id": op.id,
                "tag": op.tag,
                "record": placed["handles"][0],
                "handles": placed["handles"],
            }
        )
    await _write_record(backend, owned[0], wall, owned)
    return {
        "wall": {"id": wall.id, "record": owned[0], "handles": owned},
        "openings": done,
        "omitted": list(geometry.omitted),
    }


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
    outline = Poly(tuple(polygon.points), True, "wall")
    if not poche:
        return (outline,)
    return (outline, HatchArea((tuple(polygon.points),), polygon.material or "generic"))


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
    return {
        "count": len(handles),
        "handles": handles,
        "dimensions": rows,
        "dimscale": factor,
        "backend": backend.name,
    }


# -- floor tiles ----------------------------------------------------------------------


async def draw_tile_layout(backend, *, dry_run: bool = False, **spec) -> dict:
    """`engineering.arch.tiles.layout` drawn: one closed polyline per tile piece on the floor layer.

    A full tile is its four corners; a cut piece is each closed loop of its
    boundary (a cutout wholly inside one tile is a second loop). Refusals fire
    before the first entity; ``dry_run`` computes the takeoff and draws nothing.
    """
    from engineering.arch.tiles import layout

    result = layout(**spec)
    layer = ARCH_ROLE_LAYER["floor"]
    pieces = result["pieces"]
    out = {
        "ok": True,
        "dry_run": bool(dry_run),
        "layer": layer,
        "start": list(result["origin"]),
        "pattern": spec.get("pattern", "grid"),
        "cutouts": result["cutout_labels"],
        "takeoff": result["takeoff"],
        "smallest_cut_share": (
            round(min(p.area for p in pieces if not p.full) / (spec["length"] * spec["width"]), 4)
            if any(not p.full for p in pieces)
            else None
        ),
        "backend": backend.name,
    }
    if dry_run:
        return out
    await ensure_arch_layers(backend, [layer])
    specs = [
        {"type": "polyline", "points": [list(p) for p in loop], "closed": True, "layer": layer}
        for piece in pieces
        for loop in piece.loops
    ]
    drawn = await backend.entity_batch_create(specs)
    if drawn["errors"]:
        handles = [row["handle"] for row in drawn["entities"]]
        if handles:
            await backend.entity_delete_many(handles)
        raise ValueError(
            f"tile layout: {len(drawn['errors'])} piece(s) failed to draw: "
            f"{drawn['errors'][:3]}; nothing was kept"
        )
    out["handles"] = [row["handle"] for row in drawn["entities"]]
    out["count"] = len(out["handles"])
    return out
