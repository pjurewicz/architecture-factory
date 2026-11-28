"""A floor zone: a room or an outline minus what stands on it, hatched, its area measured.

The zone is the region :mod:`engineering.arch.tiles` cuts tiles to - the
outline (a room's wall face, its explicit boundary, or a typed outline) minus
cutouts - and its area is that region's exact area, the same Green's sum
``interior_tile_layout`` reports: measured from the geometry, never typed.

Cutouts are the face's own holes (a column), the footprints of catalogue
items selected by class or name (``fitted`` - joinery, appliances, sanitary
ware, :data:`engineering.arch.catalogue.FITTED` - by default, so underfloor
heating stops at the kitchen run but runs under the bed) and any entity
given by handle. An INSERT's footprint is its WIPEOUT mask when its block has
one, else its block extents turned with it; a closed polyline or a hatch is
its outline.

The fill is one HATCH on ``A-FLOR-E-N``: a predefined pattern at ``scale`` x
``pattern_scale`` (so it prints at its .pat size at 1:scale), or ``joints``
lines at a real spacing (a board or tile division). It carries an
``ACADMCP_ANNOT`` record whose ``legend`` is the zone's label, which is how
``arch_legend`` names the row, and its ``set`` the zone name: a rerun with the
same name replaces the zone. It goes to the back of the draw order, so the
masks of furniture inserted before it still cover it.
"""

from __future__ import annotations

import difflib
import math

from engineering.arch.tiles import Region, _inside, clip, polygon, signed_area

TOOL = "arch_floor_zone"
CLASSES = ("fitted", "movable")
#: Vertices of a circle's footprint polygon.
CIRCLE_SIDES = 32


def _footprints(entity) -> list[list[tuple[float, float]]]:
    """The polygons an entity covers in plan, WCS; [] for one that covers none."""
    kind = entity.dxftype()
    if kind == "INSERT":
        masks = [
            [(v.x, v.y) for v in e.boundary_path_wcs()]
            for e in entity.virtual_entities()
            if e.dxftype() == "WIPEOUT"
        ]
        if masks:
            return masks
        from ezdxf import bbox

        block = entity.doc.blocks.get(entity.dxf.name)
        box = bbox.extents(block) if block is not None else None
        if box is None or not box.has_data:
            return []
        m = entity.matrix44()
        (x0, y0, _), (x1, y1, _) = box.extmin, box.extmax
        corners = [m.transform((x, y, 0.0)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        return [[(c.x, c.y) for c in corners]]
    if kind == "LWPOLYLINE" and entity.closed:
        return [[(p[0], p[1]) for p in entity.get_points("xy")]]
    if kind == "CIRCLE":
        (cx, cy), r = (entity.dxf.center.x, entity.dxf.center.y), entity.dxf.radius
        return [
            [
                (cx + r * math.cos(2 * math.pi * i / CIRCLE_SIDES), cy + r * math.sin(2 * math.pi * i / CIRCLE_SIDES))
                for i in range(CIRCLE_SIDES)
            ]
        ]  # fmt: skip
    if kind == "HATCH":
        return [
            [(v[0], v[1]) for v in path.vertices]
            for path in entity.paths
            if hasattr(path, "vertices") and len(path.vertices) >= 3
        ]
    return []


def _touches(poly, outline) -> bool:
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    return any(_inside(p, outline) for p in [*poly, (cx, cy)]) or any(
        _inside(p, poly) for p in outline
    )


def check_minus(minus) -> list[str]:
    from engineering.arch.catalogue import CATALOGUE_NAMES

    if isinstance(minus, str) or not isinstance(minus, (list, tuple)):
        raise ValueError("minus: give a list of fitted | movable | catalogue item names")
    names = (*CLASSES, *CATALOGUE_NAMES)
    for i, entry in enumerate(minus):
        if entry not in names:
            near = difflib.get_close_matches(str(entry), names, n=3, cutoff=0.5)
            hint = f"; nearest: {', '.join(near)}" if near else ""
            raise ValueError(
                f"minus[{i}]: {entry!r} is not fitted, movable or a catalogue item{hint}"
            )
    return list(minus)


async def _outline(backend, room, outline):
    """``(outline, holes, source)`` of the zone, refused by name."""
    if (room is None) == (outline is None):
        raise ValueError("room / outline: give exactly one (a room id or [x, y], or points)")
    if outline is not None:
        return polygon(outline, "outline"), [], "outline"
    from engineering.arch.faces import face_containing
    from engineering.arch.rooms import plan_faces, read_records

    records = await read_records(backend)
    if isinstance(room, str):
        found = [r for r in records["rooms"] if r.id == room]
        if not found:
            ids = sorted(r.id for r in records["rooms"])
            raise ValueError(f"room: no room {room!r} on this drawing; rooms are {ids}")
        if found[0].boundary:
            return polygon(found[0].boundary, "room.boundary"), [], "room_boundary"
        at = found[0].at
    elif isinstance(room, (list, tuple)) and len(room) == 2:
        at = (float(room[0]), float(room[1]))
    else:
        raise ValueError(f"room: expected a room id or [x, y], got {room!r}")
    faces, layers, *_ = await plan_faces(backend, records)
    face = face_containing(faces, at)
    if face is None:
        raise ValueError(
            f"room: ({at[0]:g}, {at[1]:g}) lies in no closed face of {', '.join(layers)}; "
            "give the outline instead"
        )
    return polygon(face.loop, "room"), [polygon(h, "room.hole") for h in face.holes], "walls"


async def floor_zone(
    backend,
    *,
    room=None,
    outline=None,
    minus=("fitted",),
    minus_handles=(),
    pattern: str = "ANSI37",
    joints=None,
    angle: float = 0.0,
    start=None,
    scale: float = 50.0,
    pattern_scale: float = 1.0,
    label: str | None = None,
    name: str | None = None,
    layer: str | None = None,
    color: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Measure, and unless ``dry_run`` draw, one floor zone; see the module docstring."""
    from backends.ezdxf_backend import _hatch_pattern
    from engineering.arch.catalogue import FITTED, item_of_block
    from engineering.arch.layers import ARCH_ROLE_LAYER

    minus = check_minus(minus)
    if not (scale > 0 and pattern_scale > 0):
        raise ValueError("scale / pattern_scale: must be > 0")
    if joints is not None:
        if not isinstance(joints, (list, tuple)) or not 1 <= len(joints) <= 2:
            raise ValueError("joints: give [spacing] or [along, across] in mm")
        joints = [float(v) for v in joints]
        if min(joints) <= 0:
            raise ValueError("joints: spacings must be > 0 mm")
        joints = joints * 2 if len(joints) == 1 else joints
    else:
        pattern = _hatch_pattern(pattern)
    floor, holes, source = await _outline(backend, room, outline)
    layer = layer or ARCH_ROLE_LAYER["floor"]

    def _read():
        doc = backend._require_doc()
        msp = doc.modelspace()
        picked = []
        for ref in msp.query("INSERT"):
            item = item_of_block(ref.dxf.name)
            if item is None:
                continue
            fitted = item.name in FITTED
            if (
                ("fitted" in minus and fitted)
                or ("movable" in minus and not fitted)
                or item.name in minus
            ):
                picked.append((ref, item.name))
        for h in minus_handles:
            entity = doc.entitydb.get(str(h))
            if entity is None or not entity.is_alive:
                raise ValueError(f"minus_handles: no entity {h!r} on this drawing")
            picked.append((entity, None))
        out = []
        for entity, item in picked:
            for poly in _footprints(entity):
                if len(poly) >= 3 and _touches(poly, floor):
                    out.append((entity.dxf.handle, item, poly))
        return out

    cut = await backend._async(_read, readonly=True)
    cutouts = [*holes]
    excluded = []
    for handle, item, poly in cut:
        try:
            cutouts.append(polygon(poly, f"footprint {handle}"))
        except ValueError as exc:
            raise ValueError(
                f"minus: {handle}'s footprint is not a simple polygon ({exc})"
            ) from None
        excluded.append({"handle": handle, **({"item": item} if item else {})})
    region = Region.of(floor, cutouts)
    xs, ys = [p[0] for p in floor], [p[1] for p in floor]
    pad = 1.0 + max(max(xs) - min(xs), max(ys) - min(ys))
    cover = (
        (min(xs) - pad, min(ys) - pad),
        (max(xs) + pad, min(ys) - pad),
        (max(xs) + pad, max(ys) + pad),
        (min(xs) - pad, max(ys) + pad),
    )
    area, loops = clip(cover, region)
    if area <= 0 or not loops:
        raise ValueError("minus: nothing of the floor is left once the cutouts are taken out")
    result = {
        "ok": True,
        "dry_run": bool(dry_run),
        "area_m2": round(area / 1e6, 4),
        "outline_area_m2": round(signed_area(floor) / 1e6, 4),
        "area_source": source,
        "excluded": excluded,
        "loops": len(loops),
        "layer": layer,
        "outline": [list(p) for p in floor],
        "cutouts": [[list(p) for p in c] for c in cutouts],
    }
    if label:
        result["legend"] = label
    else:
        result["warnings"] = [
            {"code": "no_label", "message": "no label: arch_legend shows the pattern name"}
        ]
    if dry_run:
        return result

    from ezdxf.lldxf import const as dxfconst

    from engineering.annotation_record import replace_set, tag_annotation
    from engineering.arch.catalogue import _ensure_layer
    from engineering.mech.draw import guarded_transaction

    a = math.radians(angle)
    origin = tuple(start) if start is not None else (min(xs), min(ys))

    def _draw():
        msp = backend._require_doc().modelspace()
        hatch = msp.add_hatch(color=256, dxfattribs={"layer": layer})
        if joints is not None:
            along, across = joints
            hatch.set_pattern_fill(
                "_USER",
                color=256,
                pattern_type=dxfconst.HATCH_TYPE_USER_DEFINED,
                definition=[
                    [angle, origin, (-across * math.sin(a), across * math.cos(a)), []],
                    [angle + 90.0, origin, (along * math.cos(a), along * math.sin(a)), []],
                ],
            )
        elif pattern == "SOLID":
            hatch.set_solid_fill(color=256)
        else:
            hatch.set_pattern_fill(
                pattern, color=256, scale=scale * pattern_scale, angle=float(angle)
            )
        for loop in loops:
            flag = 1 if signed_area(loop) > 0 else 0
            hatch.paths.add_polyline_path(list(loop), is_closed=True, flags=flag)
        if color is not None:
            hatch.dxf.color = int(color)
        backend._mark_dirty()
        return hatch.dxf.handle

    async def job():
        replaced = await replace_set(backend, None, TOOL, name) if name else 0
        await _ensure_layer(backend, layer)
        handle = await backend._async(_draw)
        await tag_annotation(backend, handle, TOOL, set_name=name, legend=label)
        await backend.draw_order([handle], "send_to_back")
        return handle, replaced

    handle, replaced = await guarded_transaction(backend, job, what=TOOL, done="drawn")
    return {**result, "handle": handle, "replaced": replaced}
