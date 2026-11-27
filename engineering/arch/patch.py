"""``arch_plan_patch``: change a drawn plan by element id and redraw only what moved (T-11).

``arch_plan_from_spec`` stores the plan it drew in the drawing: the caller's
spec as given, the handles each element owns and a geometry digest of each of
them, as JSON in the ``PLAN`` XRECORD of the ``ACADMCP_ARCH_PLAN`` dictionary
(the storage pattern of ``engineering/mech/history.py``). The ``ACADMCP_ARCH``
records alone are not enough to re-resolve the plan: they lack the scale, the
language, the poche flag, the chain sides, the schedules and their places, each
room label's ``align`` / ``fit`` / ``mask`` and an open room's label (which
carries no record at all). A patch is then the ``mech_part_revise`` design on
the plan model:

1. The changes are path edits on the stored spec, by id: ``walls.<id>.<key>``,
   ``openings.<id>.<key>``, ``rooms.<id>.<key>`` (a ``[i]`` index reaches into
   a list such as ``axis[1]``; ``null`` on a key removes it, so its default
   applies), ``<list>.<id>`` (``null`` removes the element, an object replaces
   it) and ``<list>+`` (append one object or a list of them). The new spec is
   validated by ``validate_spec``, by path, before anything is touched.
2. **The affected set** comes from the pure engines, old spec against new:

   * a wall is redrawn when its record changed or the junction engine's
     outline of it (``draw._signature``: faces, jambs, seams, poche regions,
     omitted openings) changed - a thickness change moves the faces of every
     wall that meets it, a moved door moves its host's jambs;
   * an opening is redrawn with its host wall, or alone (its symbol only)
     when only its own record changed and its cut did not - a hinge flip;
   * a room label is redrawn when its request or its area changed, or - for a
     ``fit`` or ``mask`` label, which the outline shapes - its outline; a plain
     label whose room changed shape but not area keeps its texts and has its
     fit (``room_label_overflow``) measured again against the new face
     (``rooms_rechecked``); the face is found as ``label_room`` finds it, on
     the wall layers' other entities plus each plan's wall lines, before and
     after;
   * the exterior chains are redrawn when ``exterior_chains`` gives other
     intents, a schedule when ``schedule_rows`` gives other rows.

   Everything else the plan drew keeps its handle, and nothing the plan did
   not draw (furniture, notes, user dimensions, layouts) is read or written.
3. **Hand edits first**: every wall and opening entity (what the affected set
   is computed from) and every entity the patch would erase is compared with
   its stored digest; a missing or edited one refuses the call
   (``hand_edited_tagged_entity``, nothing written) unless ``overwrite=True``.
4. In one transaction: the affected elements' entities are erased, drawn again
   by the ``arch_plan_from_spec`` steps, the record is rewritten with the new
   spec, handles and digests, and the change is appended to its history.

Patching gives the arch geometry drawing the patched spec from scratch gives,
apart from handles (``tests/test_arch_patch.py``). ``dry_run=True`` returns the
affected set and a preview whose preconditions are the fingerprints of every
entity the apply would erase. The sheet, language, poche, stairs, wall polygons,
chains and schedules are not patched here: draw the plan again.

A room separator (``room_separators.<id>``) is addressed by its id, which
``room_separator_from_dict`` already requires and ``validate_spec`` keeps
unique, so an address is never ambiguous. It changes no wall and no chain
(``exterior_chains`` reads walls, openings and the ``A-WALL*`` outlines, not
the separator layer); it changes room faces, so a separator that is added,
removed or moved is drawn again on its own and the rooms are judged as above,
the separators of each version in place of the plan's drawn ones.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ezdxf.lldxf.tagwriter import TagCollector

from engineering.errors import HandEditedTaggedEntityError, InvalidArgsError, RevisionRefusedError

DICT_NAME = "ACADMCP_ARCH_PLAN"
RECORD_NAME = "PLAN"
FORMAT_VERSION = 1
CHUNK_SIZE = 255
MAX_ENTRIES = 200
#: The spec lists a patch edits, by element id.
PATCH_LISTS = ("walls", "openings", "rooms", "room_separators")
REFUSAL_ROWS = 20

_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[\d+\])*)$")
_INDEX = re.compile(r"\[(\d+)\]")


# -- the stored plan ----------------------------------------------------------------


def _stable_digest(entity, dxfversion: str) -> str:
    """The entity's DXF tags without its XDATA (the records a patch rewrites) and
    without a tag that only spells its attribute's default: a save and a reopen
    drop those (a HATCH's ``62 256``, a TEXT's ``7 Standard``), the geometry stays."""
    from engineering.changes.preview import _content_tags

    collector = TagCollector(dxfversion=dxfversion)
    entity.export_dxf(collector)
    defaults = set()
    for attrib in entity.DXFATTRIBS._attribs.values():
        try:
            if attrib.default is not None:
                defaults.add((attrib.code, attrib.default))
        except TypeError:  # an unhashable default
            pass
    digest = hashlib.sha256()
    for code, value in _content_tags(collector.tags):
        if code == 1001:
            break
        try:
            if (code, value) in defaults:
                continue
        except TypeError:
            pass
        digest.update(f"{code}\x1f{value!r}\x1e".encode())
    return digest.hexdigest()[:16]


async def digests(backend, handles) -> dict[str, str | None]:
    """``{handle: digest}``; ``None`` for a handle the document no longer holds."""
    wanted = list(dict.fromkeys(str(h).upper() for h in handles))

    def job(_backend):
        doc = _backend._require_doc()
        out: dict[str, str | None] = {}
        for handle in wanted:
            entity = doc.entitydb.get(handle)
            alive = entity is not None and entity.is_alive
            out[handle] = _stable_digest(entity, doc.dxfversion) if alive else None
        return out

    return await backend._async(job, backend, readonly=True) if wanted else {}


def owned_from(result: dict) -> dict[str, list[str]]:
    """``"<kind>:<id>"`` -> the handles that element owns, from the drawing steps' results."""
    owned: dict[str, list[str]] = {}
    drawn = result.get("walls") or {}
    for row in drawn.get("walls", ()):
        owned[f"wall:{row['id']}"] = list(row["handles"])
    for row in drawn.get("openings", ()):
        owned[f"opening:{row['id']}"] = list(row["handles"])
    for row in result.get("wall_polygons") or ():
        owned[f"wall_polygon:{row['id']}"] = list(row["handles"])
    for row in result.get("stairs") or ():
        owned[f"stair:{row['stair']}"] = list(row["handles"])
    for row in result.get("room_separators") or ():
        owned[f"room_separator:{row['id']}"] = [row["handle"]]
    for row in result.get("rooms") or ():
        if "room" in row:
            owned[f"room:{row['room']['id']}"] = _room_handles(row)
    if result.get("chains"):
        owned["chains"] = list(result["chains"]["handles"])
    for i, row in enumerate(result.get("schedules") or ()):
        owned[f"schedule:{i}"] = _schedule_handles(row)
    return owned


def _room_handles(row: dict) -> list[str]:
    return [*row.get("handles", ()), *((row.get("mask") or {}).get("handles") or ())]


def _schedule_handles(row: dict) -> list[str]:
    return list(dict.fromkeys([row["handle"], *(row.get("child_handles") or ())]))


async def write_record(backend, spec: dict, owned: dict, *, history=None, dropped=0) -> None:
    """Store the plan: its spec, each element's handles and their digests."""
    handles = [h for group in owned.values() for h in group]
    payload = {
        "format": FORMAT_VERSION,
        "spec": spec,
        "owned": owned,
        "digests": await digests(backend, handles),
        "history": list(history or []),
        "dropped": int(dropped),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    def store(_backend):
        doc = _backend._require_doc()
        holder = doc.rootdict.get_required_dict(DICT_NAME)
        if RECORD_NAME in holder:
            holder.remove(RECORD_NAME)
        holder.add_xrecord(RECORD_NAME).extend([(1000, chunk) for chunk in chunks])
        _backend._mark_dirty()

    await backend._async(store, backend)


async def load_record(backend) -> dict | None:
    """The stored plan, or None when the drawing carries none."""

    def read(_backend):
        doc = _backend._require_doc()
        if DICT_NAME not in doc.rootdict:
            return None
        holder = doc.rootdict.get(DICT_NAME)
        if RECORD_NAME not in holder:
            return None
        return [str(tag.value) for tag in holder.get(RECORD_NAME).tags if tag.code == 1000]

    chunks = await backend._async(read, backend, readonly=True)
    if chunks is None:
        return None
    try:
        payload = json.loads("".join(chunks))
    except ValueError as exc:
        raise ValueError(f"{DICT_NAME}: the stored plan is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_VERSION:
        raise ValueError(f"{DICT_NAME}: not a format {FORMAT_VERSION} plan record")
    return payload


# -- path edits ---------------------------------------------------------------------


def _bad(path: str, why: str) -> InvalidArgsError:
    return InvalidArgsError(f"invalid_args: changes[{path!r}]: {why}")


def _ids(items) -> list[str]:
    return [str(item.get("id")) for item in items if isinstance(item, dict)]


def _item_index(items: list, element: str, path: str) -> int:
    for index, item in enumerate(items):
        if isinstance(item, dict) and str(item.get("id")) == element:
            return index
    raise _bad(path, f"no element {element!r}; the ids are {', '.join(_ids(items)) or 'none'}")


def apply_changes(spec: dict, changes: Any) -> tuple[dict, dict]:
    """``(new spec, {path: [old, new]})``: ``changes`` applied in order (module docstring)."""
    if not isinstance(changes, dict) or not changes:
        raise InvalidArgsError(
            "invalid_args: changes: expected a non-empty object of {path: value}, e.g. "
            '{"openings.D1.offset": 1200, "walls.INT.thickness": 120, "rooms.R2.name": '
            '"KITCHEN", "openings.D2": null, "openings+": {"id": "D3", ...}}'
        )
    new = copy.deepcopy(spec)
    diff: dict[str, list] = {}
    for raw_path, value in changes.items():
        path = str(raw_path).strip() if isinstance(raw_path, str) else repr(raw_path)
        parts = path.split(".")
        head = parts[0]
        append = head.endswith("+")
        name = head[:-1] if append else head
        if name not in PATCH_LISTS:
            raise _bad(
                path,
                f"a patch edits {', '.join(PATCH_LISTS)} by id; {name!r} is not patched - "
                "draw the plan again with arch_plan_from_spec",
            )
        items = new.setdefault(name, [])
        if append:
            if len(parts) != 1:
                raise _bad(path, f"'{name}+' appends and takes no path after it")
            added = value if isinstance(value, list) else [value]
            if not added or not all(isinstance(item, dict) for item in added):
                raise _bad(path, f"expected an object or a list of objects, got {value!r}")
            for item in added:
                items.append(copy.deepcopy(item))
                diff[f"{name}.{item.get('id')}"] = [None, copy.deepcopy(item)]
            continue
        if len(parts) < 2 or not parts[1]:
            raise _bad(path, f"name the element: '{name}.<id>' or '{name}.<id>.<key>'")
        index = _item_index(items, parts[1], path)
        if len(parts) == 2:
            before = copy.deepcopy(items[index])
            if value is None:
                del items[index]
            elif isinstance(value, dict):
                items[index] = copy.deepcopy(value)
            else:
                raise _bad(
                    path, f"an element is replaced by an object or removed by null, not {value!r}"
                )
            diff[path] = [before, copy.deepcopy(value)]
            continue
        node: Any = items[index]
        tokens: list = []
        for part in parts[2:]:
            match = _KEY.match(part)
            if match is None:
                raise _bad(path, f"{part!r} is not a key or a key[index]")
            tokens.append(match.group(1))
            tokens.extend(int(i) for i in _INDEX.findall(match.group(2)))
        for token in tokens[:-1]:
            if isinstance(token, int):
                if not isinstance(node, list) or not 0 <= token < len(node):
                    raise _bad(path, f"[{token}] is not an index of {node!r}")
            elif not isinstance(node, dict) or token not in node:
                raise _bad(path, f"no key {token!r} here")
            node = node[token]
        last = tokens[-1]
        if isinstance(last, int):
            if not isinstance(node, list) or not 0 <= last < len(node):
                raise _bad(path, f"[{last}] is not an index of {node!r}")
            before = copy.deepcopy(node[last])
            if value is None:
                del node[last]
            else:
                node[last] = copy.deepcopy(value)
        else:
            if not isinstance(node, dict):
                raise _bad(path, f"{last!r} names a key, and this is not an object")
            before = copy.deepcopy(node.get(last))
            if value is None:
                node.pop(last, None)
            else:
                node[last] = copy.deepcopy(value)
        diff[path] = [before, copy.deepcopy(value)]
    return new, diff


# -- the affected set ---------------------------------------------------------------


def _round_loop(loop) -> tuple:
    return tuple((round(x, 6), round(y, 6)) for x, y in loop)


def _face_key(face) -> tuple | None:
    if face is None:
        return None
    return (
        round(face.area, 3),
        _round_loop(face.loop),
        tuple(sorted(_round_loop(h) for h in face.holes)),
    )


def _room_faces(norm: dict, geometry: dict, foreign: list) -> dict[str, tuple]:
    """``room id -> (area key, outline key, area or None, face)``, found the way
    ``label_room`` finds them."""
    from engineering.arch.faces import DEFAULT_TOL, face_containing, planar_faces
    from engineering.arch.layers import ARCH_ROLE_LAYER
    from engineering.arch.rooms import opening_closures
    from engineering.measure import polygon_area_perimeter

    layer = ARCH_ROLE_LAYER["wall"]
    lines = [
        (line.p1, line.p2, layer)
        for g in geometry.values()
        for line in (*g.faces, *g.jambs, *g.seams)
    ]
    separator_layer = ARCH_ROLE_LAYER["room_separator"]
    lines += [
        (sep.points[i - 1], sep.points[i], separator_layer)
        for sep in norm["room_separators"]
        for i in range(1, len(sep.points))
    ]
    closures, _omitted = opening_closures(norm["walls"], norm["openings"])
    faces = planar_faces([*foreign, *lines, *((p, q, layer) for p, q in closures)], tol=DEFAULT_TOL)
    out: dict[str, tuple] = {}
    for request in norm["rooms"]:
        room = request["room"]
        if room.get("boundary") is not None:
            pts = [(float(x), float(y), 0.0) for x, y in room["boundary"]]
            area, _ = polygon_area_perimeter(pts, True)
            out[str(room["id"])] = (("boundary",), ("boundary",), area, None)
            continue
        face = face_containing(faces, (float(room["at"][0]), float(room["at"][1])))
        area = None if face is None else face.area
        key = None if face is None else round(face.area, 3)
        out[str(room["id"])] = (key, _face_key(face), area, face)
    return out


def _schedule_plan(norm: dict, geometry: dict, faces: dict) -> dict:
    """The records a schedule reads, as the drawing will hold them."""
    from engineering.arch.model import room_from_dict

    skipped = {entry["element"] for g in geometry.values() for entry in g.omitted}
    rooms = []
    for request in norm["rooms"]:
        room = request["room"]
        area = faces[str(room["id"])][2]
        if area is not None:
            rooms.append(replace(room_from_dict(dict(room), "room"), area=area))
    return {
        "walls": list(norm["walls"]),
        "openings": [op for op in norm["openings"] if op.id not in skipped],
        "rooms": rooms,
    }


def affected(old: dict, new: dict, foreign: list, outlines) -> dict:
    """What a patch redraws, from the pure engines: see the module docstring."""
    from engineering.arch.dimension import exterior_chains
    from engineering.arch.draw import _signature
    from engineering.arch.schedule import schedule_rows
    from engineering.arch.walls import wall_geometry_by_wall

    og = wall_geometry_by_wall(old["walls"], old["openings"], poche=old["poche"])
    ng = wall_geometry_by_wall(new["walls"], new["openings"], poche=new["poche"])
    ow = {w.id: w for w in old["walls"]}
    walls = [
        w.id
        for w in new["walls"]
        if w.id not in ow or ow[w.id] != w or _signature(og[w.id]) != _signature(ng[w.id])
    ]
    oo = {o.id: o for o in old["openings"]}
    ocut = {c["element"]: c for g in og.values() for c in g.cuts}
    ncut = {c["element"]: c for g in ng.values() for c in g.cuts}
    symbols = [
        o.id
        for o in new["openings"]
        if o.wall not in walls and (oo.get(o.id) != o or ocut.get(o.id) != ncut.get(o.id))
    ]
    with_walls = [o.id for o in new["openings"] if o.wall in walls]

    old_faces = _room_faces(old, og, foreign)
    new_faces = _room_faces(new, ng, foreign)
    orooms = {str(r["room"]["id"]): r for r in old["rooms"]}
    rooms: list[str] = []
    rechecked: list[str] = []
    for request in new["rooms"]:
        rid = str(request["room"]["id"])
        was, now = old_faces.get(rid), new_faces[rid]
        if orooms.get(rid) != request or was[0] != now[0]:
            rooms.append(rid)  # the request or the area moved: the texts change
        elif was[1] != now[1]:
            # same area, another outline: a fitted or masked label is shaped by
            # the outline and is drawn again; a plain one keeps its texts and
            # only its fit is measured again against the new face
            (rooms if request["fit"] or request["mask"] else rechecked).append(rid)

    def chains(norm, geometry):
        if norm["chains"] is None:
            return None
        return exterior_chains(
            norm["walls"],
            norm["openings"],
            sides=norm["chains"],
            scale=norm["scale"],
            polygons=outlines,
        )

    old_plan = _schedule_plan(old, og, old_faces)
    new_plan = _schedule_plan(new, ng, new_faces)
    schedules = [
        i
        for i, item in enumerate(new["schedules"])
        if schedule_rows(item["kind"], old_plan, lang=old["lang"])
        != schedule_rows(item["kind"], new_plan, lang=new["lang"])
    ]
    nw = {w.id for w in new["walls"]}
    no = {o.id for o in new["openings"]}
    nr = {str(r["room"]["id"]) for r in new["rooms"]}
    osep = {sep.id: sep for sep in old["room_separators"]}
    nsep = {sep.id for sep in new["room_separators"]}
    separators = [sep.id for sep in new["room_separators"] if osep.get(sep.id) != sep]
    return {
        "walls": walls,
        "openings": [*with_walls, *symbols],
        "room_separators": separators,
        "symbols_only": symbols,
        "rooms": rooms,
        "rechecked": rechecked,
        "new_faces": new_faces,
        "chains": chains(old, og) != chains(new, ng),
        "schedules": schedules,
        "removed": [
            *(f"wall:{w}" for w in ow if w not in nw),
            *(f"opening:{o}" for o in oo if o not in no),
            *(f"room:{r}" for r in orooms if r not in nr),
            *(f"room_separator:{s}" for s in osep if s not in nsep),
        ],
        "geometry": ng,
    }


def erased_keys(old: dict, plan: dict) -> list[str]:
    """The element keys whose entities the patch takes out."""
    walls = set(plan["walls"])
    keys = [f"wall:{w}" for w in plan["walls"]]
    keys += [f"opening:{o.id}" for o in old["openings"] if o.wall in walls]
    keys += [f"opening:{o}" for o in plan["symbols_only"]]
    keys += [f"room:{r}" for r in plan["rooms"]]
    keys += [f"room_separator:{s}" for s in plan["room_separators"]]
    keys += plan["removed"]
    if plan["chains"]:
        keys.append("chains")
    keys += [f"schedule:{i}" for i in plan["schedules"]]
    return list(dict.fromkeys(keys))


def _drift(record: dict, current: dict[str, str | None], keys) -> list[dict]:
    rows = []
    for key in keys:
        for handle in record["owned"].get(key, ()):
            now = current.get(handle)
            if now is None:
                rows.append({"element": key, "handle": handle, "status": "missing"})
            elif now != record["digests"].get(handle):
                rows.append({"element": key, "handle": handle, "status": "edited"})
    return rows


# -- the apply ----------------------------------------------------------------------


async def patch_plan(
    backend,
    changes: Any,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    expected_revision: int | None = None,
    preconditions: list | None = None,
) -> dict:
    """Patch the stored plan by id and redraw its affected set in one transaction."""
    from backends.operations import CURRENT_OPERATION
    from engineering.arch import draw as _draw
    from engineering.arch import rooms as _rooms
    from engineering.arch import schedule as _schedule
    from engineering.arch.model import room_from_dict
    from engineering.arch.spec import _extents, validate_spec
    from engineering.changes.preview import (
        check_preconditions,
        current_revision,
        make_preview,
        parse_preconditions,
    )
    from engineering.mech.draw import guarded_transaction

    rows = parse_preconditions(preconditions)
    record = await load_record(backend)
    if record is None:
        raise RevisionRefusedError(
            "arch_plan_patch: this drawing stores no plan; draw it with arch_plan_from_spec, "
            "which stores the spec a patch edits. Nothing was changed.",
            reason="no_plan_record",
        )
    old_spec = record["spec"]
    new_spec, diff = apply_changes(old_spec, changes)
    old = validate_spec(old_spec)
    try:
        new = validate_spec(new_spec)
    except ValueError as exc:
        raise InvalidArgsError(
            f"invalid_args: changes [{', '.join(diff)}]: the patched plan is refused - {exc}. "
            "Nothing was changed."
        ) from exc

    if not dry_run and (expected_revision is not None or rows):
        await check_preconditions(backend, expected_revision=expected_revision, preconditions=rows)
    start = await current_revision(backend)
    # the plan's own walls and separators are computed per version, not read
    wall_handles = [
        h
        for key, group in record["owned"].items()
        if key.split(":")[0] in ("wall", "room_separator")
        for h in group
    ]
    layers = await _rooms.wall_layers(backend)
    foreign, _skipped = await _rooms.read_segments(backend, layers, exclude=frozenset(wall_handles))
    outlines = await _draw.read_wall_outlines(backend)
    plan = affected(old, new, foreign, outlines)
    keys = erased_keys(old, plan)
    checked = list(
        dict.fromkeys(
            [k for k in record["owned"] if k.split(":")[0] in ("wall", "opening", "room_separator")]
            + keys
        )
    )
    handles = [h for k in checked for h in record["owned"].get(k, ())]
    drift = _drift(record, await digests(backend, handles), checked)
    if drift and not overwrite:
        shown = ", ".join(row["handle"] for row in drift[:10])
        raise HandEditedTaggedEntityError(
            f"hand_edited_tagged_entity: {len(drift)} plan entit"
            f"{'y' if len(drift) == 1 else 'ies'} no longer match what arch_plan_from_spec "
            f"drew ({', '.join(sorted({r['status'] for r in drift}))}): {shown}; nothing was "
            "changed. Undo the edit, or pass overwrite=True to replace it with the plan.",
            handles=[row["handle"] for row in drift],
            drift=drift[:REFUSAL_ROWS],
        )
    erase = [h for k in keys for h in record["owned"].get(k, ())]
    summary = {
        "walls": plan["walls"],
        "openings": plan["openings"],
        "room_separators": plan["room_separators"],
        "rooms": plan["rooms"],
        "rooms_rechecked": plan["rechecked"],
        "chains": plan["chains"],
        "schedules": [new["schedules"][i]["kind"] for i in plan["schedules"]],
        "removed": plan["removed"],
    }
    changed = new_spec != old_spec
    kept = sum(len(g) for k, g in record["owned"].items() if k not in keys)
    if dry_run:
        preview = await make_preview(
            backend, handles=erase, planned=[{"changes": diff, "affected": summary}]
        )
        return {
            "dry_run": True,
            "changed": changed,
            "changes": diff,
            "affected": summary,
            "erase": len(erase),
            "kept": kept,
            "overwritten": drift[:REFUSAL_ROWS],
            "preview": preview.to_dict(),
        }
    if not changed:
        return {
            "changed": False,
            "changes": diff,
            "affected": summary,
            "revision": {"checked": start, "after": start},
        }

    geometry = plan["geometry"]
    walls = {w.id: w for w in new["walls"]}
    operation_id = CURRENT_OPERATION.get()
    outcome: dict = {"rooms": [], "warnings": []}

    async def job() -> None:
        for handle in erase:
            try:
                await backend.entity_delete(handle)
            except Exception:  # erased by hand (overwrite=True): nothing to take back
                pass
        owned = {k: v for k, v in record["owned"].items() if k not in keys}
        await backend.drawing_apply_iso_layers("arch")
        for wall in new["walls"]:
            if wall.id not in plan["walls"]:
                continue
            hosted = [o for o in new["openings"] if o.wall == wall.id]
            done = await _draw._draw_wall(backend, wall, geometry[wall.id], hosted, new["scale"])
            owned[f"wall:{wall.id}"] = list(done["wall"]["handles"])
            for row in done["openings"]:
                owned[f"opening:{row['id']}"] = list(row["handles"])
        for op in new["openings"]:
            if op.id in plan["symbols_only"]:
                done = await _draw.draw_opening(backend, walls[op.wall], geometry[op.wall], op)
                owned[f"opening:{op.id}"] = list(done["handles"])
        redrawn = [sep for sep in new["room_separators"] if sep.id in plan["room_separators"]]
        for row in await _draw.draw_room_separators(backend, redrawn):
            owned[f"room_separator:{row['id']}"] = [row["handle"]]
        for i, request in enumerate(new["rooms"]):
            room_id = str(request["room"]["id"])
            if room_id not in plan["rooms"]:
                continue
            try:
                done = await _rooms.label_room(
                    backend,
                    request["room"],
                    lang=new["lang"],
                    scale=new["scale"],
                    align=request["align"],
                    fit=request["fit"],
                    mask=request["mask"],
                    open_room=True,
                )
            except ValueError as exc:
                raise InvalidArgsError(f"invalid_args: rooms[{i}]: {exc}") from exc
            owned[f"room:{room_id}"] = _room_handles(done)
            outcome["rooms"].append(done)
            if done.get("warning"):
                outcome["warnings"].append((i, {"element": f"rooms[{i}]", **done["warning"]}))
        if plan["chains"]:
            done = await _draw.draw_chains(
                backend,
                new["walls"],
                new["openings"],
                sides=new["chains"],
                scale=new["scale"],
                polygons=await _draw.read_wall_outlines(backend),
            )
            owned["chains"] = list(done["handles"])
            outcome["chains"] = done
        for i in plan["schedules"]:
            item = new["schedules"][i]
            done = await _schedule.draw_schedule(
                backend, item["kind"], at=item["at"], lang=new["lang"], scale=new["scale"]
            )
            owned[f"schedule:{i}"] = _schedule_handles(done)
        history = list(record.get("history") or [])
        last = max((int(e.get("rev", 0)) for e in history), default=0)
        entry = {
            "rev": last + 1,
            "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "operation_id": operation_id,
            "changes": diff,
        }
        history.append(entry)
        dropped = int(record.get("dropped") or 0) + max(0, len(history) - MAX_ENTRIES)
        await write_record(
            backend, new_spec, owned, history=history[-MAX_ENTRIES:], dropped=dropped
        )
        outcome.update(owned=owned, entry=entry)

    await guarded_transaction(backend, job, what="arch_plan_patch", done="changed")

    # a plain label in a room that changed shape, not area: its texts stay, its fit
    # is measured again against the new face, as label_room measures it
    for i, request in enumerate(new["rooms"]):
        room_id = str(request["room"]["id"])
        if room_id not in plan["rechecked"]:
            continue
        _key, _outline, area, face = plan["new_faces"][room_id]
        measured = replace(room_from_dict(dict(request["room"]), "room"), area=area)
        outside = await _rooms.plain_label_outside(
            backend, measured, face, lang=new["lang"], scale=new["scale"], align=request["align"]
        )
        row = {
            "room": {
                "id": measured.id,
                "name": measured.name,
                "number": measured.number,
                "area": area,
                "area_m2": area / 1.0e6,
            },
            "fits": not outside,
            "rechecked": True,
        }
        if outside:
            row["warning"] = _rooms.overflow_warning(measured, outside, fit=False)
            outcome["warnings"].append((i, {"element": f"rooms[{i}]", **row["warning"]}))
        outcome["rooms"].append(row)
    order = {str(r["room"]["id"]): i for i, r in enumerate(new["rooms"])}
    outcome["rooms"].sort(key=lambda row: order[row["room"]["id"]])

    from engineering.arch.critique import ARCH_FOCUSES, build_index, issues_for

    index = await build_index(backend)
    critique = [issue.to_dict() for focus in ARCH_FOCUSES for issue in issues_for(focus, index)]
    owned = outcome["owned"]
    plan_handles = [
        h for k, g in owned.items() if k != "chains" and not k.startswith("schedule:") for h in g
    ]
    handle_map = [
        {"element": k, "old": record["owned"].get(k, []), "new": owned.get(k, [])} for k in keys
    ]
    handle_map += [
        {"element": k, "old": [], "new": owned[k]}
        for k in owned
        if k not in record["owned"] and k not in keys
    ]
    return {
        "changed": True,
        "rev": outcome["entry"]["rev"],
        "changes": diff,
        "affected": summary,
        "kept": kept,
        "handle_map": handle_map,
        "rooms": outcome["rooms"],
        "warnings": [w for _i, w in sorted(outcome["warnings"], key=lambda item: item[0])],
        "omitted": [entry for g in geometry.values() for entry in g.omitted],
        "plan_bbox": await _extents(backend, plan_handles),
        "outer_bbox": await _extents(backend, plan_handles + owned.get("chains", [])),
        "overwritten": drift[:REFUSAL_ROWS],
        "critique": critique,
        "revision": {"checked": start, "after": await current_revision(backend)},
        "backend": backend.name,
    }


def compact(result: dict) -> dict:
    """The T-09 shape: what a caller acts on, not the drawing's own lists."""
    if not result.get("changed") or result.get("dry_run"):
        return result
    out = {
        key: result[key] for key in ("rev", "changes", "affected", "kept", "warnings", "omitted")
    }
    out["rooms"] = [
        {
            **{key: item["room"][key] for key in ("id", "number", "name", "area_m2")},
            "fits": item.get("fits"),
        }
        for item in result["rooms"]
    ]
    out["plan_bbox"] = result["plan_bbox"]
    out["outer_bbox"] = result["outer_bbox"]
    if result.get("overwritten"):
        out["overwritten"] = result["overwritten"]
    if result.get("critique"):
        out["critique"] = result["critique"]
    return out
