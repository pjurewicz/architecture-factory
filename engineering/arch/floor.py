"""A floor zone: a room or an outline minus what stands on it, hatched, its area measured.

The zone is the region :mod:`engineering.arch.tiles` cuts tiles to - the
outline (a room's wall face, its explicit boundary, or a typed outline) minus
cutouts - and its area is that region's exact area, the same Green's sum the
laid pieces add up to: measured from the geometry, never typed.

Cutouts are the face's own holes (a column), the footprints of catalogue
items selected by class or name (``fitted`` - joinery, appliances, sanitary
ware, :data:`engineering.arch.catalogue.FITTED` - by default, so underfloor
heating stops at the kitchen run but runs under the bed) and any entity
given by handle. An INSERT's footprint is its WIPEOUT mask when its block has
one, else its block extents turned with it; a closed polyline or a hatch is
its outline.

The zone is drawn on ``A-FLOR-E-N`` as up to three parts, in this order: a
predefined ``pattern`` HATCH, then its joints - a ``joints`` HATCH (``_USER``
lines at a real spacing - a board or tile division) or, with ``lay``, every
tile or plank of a laying pattern cut to the zone as closed polylines
(:func:`engineering.arch.tiles.layout`: grid, running bond, herringbone,
chevron, from the plank's size, ``angle`` and ``start``, with the takeoff) -
and, with ``edge``, the zone's loops as closed polylines - a tile with dots,
its joints and its outline in one call.
A pattern's size follows acadiso.pat: the ``AR-*`` patterns are drawn in
building millimetres (a brick of AR-HBONE is 101.6 x 304.8 mm), so their hatch
scale is ``pattern_scale`` alone; every other pattern is a sheet pattern drawn
at ``scale`` x ``pattern_scale``, so it prints at its .pat size at 1:scale.
Each fill reports its period in mm (:func:`fill_period`) and how often it
repeats across the zone, and warns when the zone holds less than
:data:`MIN_REPEATS` periods - a pattern that plots as a stray stroke or a few
dots says so instead of passing silently.

``skirting`` (``applied`` / ``flush`` / ``none`` for the whole zone, or per
wall id) runs one polyline per stretch along the room's faces, holes included,
on ``A-FLOR-SKRT-E-N`` / ``A-FLOR-SKRF-E-N`` (ACAD_ISO02W100 at the 0.25 pen),
:data:`SKIRTING_GAP_PAPER_MM` inside the face so the two plot apart. It stops
at every door and at an opening whose sill is below :data:`SKIRTING_TOP` (read
from the plan's records), behind the fitted items the zone cuts out, and at a
room separation line. Each stretch's length is measured on the wall face, in
metres; the gap, the nominal top and the layer colours are a repository
convention, not a standard. Its record's ``legend`` is ``skirting_<type>``,
which ``arch_legend`` letters in its ``lang`` (:data:`SKIRTING_LABELS`).
``pattern="none"`` draws no hatch (skirting or edge only).

Every part carries an ``ACADMCP_ANNOT`` record whose ``legend`` is the zone's
label, which is how ``arch_legend`` names the row, and its ``set`` the zone
name: a rerun with the same name replaces the zone. The parts go to the back
of the draw order, so the masks of furniture inserted before them still cover
them.
"""

from __future__ import annotations

import difflib
import math

from engineering.arch.tiles import Region, _inside, clip, polygon, signed_area

TOOL = "arch_floor_zone"
#: ``tiles.layout``'s argument names in its refusals -> this tool's.
_LAY_NAMES = (
    ("tile_length", "plank_length"),
    ("tile_width", "plank_width"),
    ("joint:", "gap:"),
    ("angle_deg", "angle"),
    ("cut_deg", "cut_angle"),
    ("pattern:", "lay:"),
)
CLASSES = ("fitted", "movable")
#: Vertices of a circle's footprint polygon.
CIRCLE_SIDES = 32
#: Fewer periods than this across the zone's narrower side is a warning.
MIN_REPEATS = 3


def hatch_scale(pattern: str, scale: float, pattern_scale: float) -> float:
    """The HATCH scale of ``pattern``: AR-* are building mm, the rest sheet patterns."""
    return pattern_scale if pattern.startswith("AR-") else scale * pattern_scale


def fill_period(definition, hatch_scale: float = 1.0) -> float:
    """The period of a hatch definition in drawing units: over its line families,
    the longest of a family's spacing and its dash cycle, times ``hatch_scale``.

    ``definition`` is ezdxf's ``[angle, base, offset, dashes]`` per family, the
    offset in the pattern's frame (as :data:`ezdxf.tools.pattern.ISO_PATTERN`).
    """
    period = 0.0
    for angle, _base, (ox, oy), dashes in definition:
        a = math.radians(angle)
        spacing = abs(-ox * math.sin(a) + oy * math.cos(a))
        period = max(period, spacing, sum(abs(d) for d in dashes))
    return period * hatch_scale


def _fill_report(kind: str, period: float, side: float) -> tuple[dict, dict | None]:
    """``(row, warning)`` for one fill of period ``period`` across ``side`` mm."""
    repeats = side / period
    row = {"period_mm": round(period, 1), "repeats": round(repeats, 1)}
    if repeats >= MIN_REPEATS:
        return row, None
    code = "fill_period_exceeds_zone" if repeats < 1 else "fill_few_repeats"
    return row, {
        "code": code,
        "message": (
            f"{kind}: period {period:.0f} mm against the zone's {side:.0f} mm side - "
            f"{repeats:.1f} repeats (fewer than {MIN_REPEATS}); it plots as a few strokes "
            "or dots: lower pattern_scale (or the joint spacing)"
        ),
    }


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


# -- skirting ---------------------------------------------------------------------------

#: ``skirting`` types; ``none`` leaves a wall bare.
SKIRTING_TYPES = ("applied", "flush", "none")
#: Nominal skirting top, mm: an opening whose sill is lower interrupts the skirting.
SKIRTING_TOP = 80.0
#: Paper mm between the wall face and the skirting line, so they plot apart.
SKIRTING_GAP_PAPER_MM = 0.5
#: The record's ``legend`` key per type; ``arch_legend`` letters it in its ``lang``.
SKIRTING_LABELS = {
    "skirting_applied": {
        "en": "applied skirting",
        "pl": "listwa przypodłogowa nakładana",
        "tr": "yüzeye monte süpürgelik",
    },
    "skirting_flush": {
        "en": "flush skirting (shadow gap)",
        "pl": "cokół szczelinowy (podcięcie)",
        "tr": "gömme süpürgelik (gölge derz)",
    },
}
_ON = 1.0  # mm: an edge on a face line or an opening's closure
_PROBE = 5.0  # mm into the room: where a fitted item is tested against an edge
_T = 1e-6


def check_skirting(value, wall_ids) -> dict | None:
    """``{"default": type, <wall id>: type}``, refused by path."""
    if value is None:
        return None
    if isinstance(value, str):
        value = {"default": value}
    if not isinstance(value, dict) or not value:
        raise ValueError("skirting: give applied | flush | none, or {default, <wall id>: type}")
    out = {}
    for key, kind in value.items():
        if kind not in SKIRTING_TYPES:
            raise ValueError(f"skirting.{key}: {kind!r}; types are {', '.join(SKIRTING_TYPES)}")
        if key != "default" and key not in wall_ids:
            raise ValueError(
                f"skirting.{key}: no wall {key!r} bounds this zone's plan; "
                f"walls are {sorted(wall_ids)}"
            )
        out[str(key)] = kind
    out.setdefault("default", "applied")
    return out


def _span(a, b, p, q):
    """``(t0, t1)`` of ``pq`` projected on ``ab`` when ``pq`` lies on its line, else None."""
    length = math.dist(a, b)
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    if any(abs((r[0] - a[0]) * uy - (r[1] - a[1]) * ux) > _ON for r in (p, q)):
        return None
    t0, t1 = sorted(((r[0] - a[0]) * ux + (r[1] - a[1]) * uy) / length for r in (p, q))
    return t0, t1


def _cross_t(p, q, c, d):
    rx, ry, sx, sy = q[0] - p[0], q[1] - p[1], d[0] - c[0], d[1] - c[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-12:
        return None
    t = ((c[0] - p[0]) * sy - (c[1] - p[1]) * sx) / den
    u = ((c[0] - p[0]) * ry - (c[1] - p[1]) * rx) / den
    return t if 0.0 < t < 1.0 and 0.0 <= u <= 1.0 else None


def _covered(a, b, n, polys) -> list[tuple[float, float]]:
    """Where a footprint stands on edge ``ab``: tested ``_PROBE`` mm into the room."""
    pa = (a[0] + n[0] * _PROBE, a[1] + n[1] * _PROBE)
    pb = (b[0] + n[0] * _PROBE, b[1] + n[1] * _PROBE)
    out = []
    for poly in polys:
        ts = {0.0, 1.0}
        for i, c in enumerate(poly):
            t = _cross_t(pa, pb, c, poly[(i + 1) % len(poly)])
            if t is not None:
                ts.add(t)
        ts = sorted(ts)
        for t0, t1 in zip(ts, ts[1:], strict=False):
            m = (t0 + t1) / 2.0
            if _inside((pa[0] + (pb[0] - pa[0]) * m, pa[1] + (pb[1] - pa[1]) * m), poly):
                out.append((t0, t1))
    return out


def _keep(cuts, length) -> list[tuple[float, float]]:
    """``[0, 1]`` minus ``cuts``, pieces under ``_ON`` mm dropped."""
    kept, at = [], 0.0
    for t0, t1 in sorted((max(0.0, a), min(1.0, b)) for a, b in cuts if b > 0 and a < 1):
        if t0 > at:
            kept.append((at, t0))
        at = max(at, t1)
    if at < 1.0:
        kept.append((at, 1.0))
    return [(t0, t1) for t0, t1 in kept if (t1 - t0) * length >= _ON]


def _face_lines(records):
    """``[(a, b, wall id, reach)]``: both face lines of every wall axis segment."""
    from engineering.arch.model import axis_segments
    from engineering.arch.rooms import _face_offsets

    out = []
    for wall in records["walls"]:
        for p0, p1 in axis_segments(wall):
            length = math.dist(p0, p1)
            nx, ny = -(p1[1] - p0[1]) / length, (p1[0] - p0[0]) / length
            for d in _face_offsets(wall):
                a = (p0[0] + nx * d, p0[1] + ny * d)
                b = (p1[0] + nx * d, p1[1] + ny * d)
                out.append((a, b, wall.id, float(wall.thickness) / length))
    for sep in records["room_separators"]:
        pts = [tuple(p) for p in sep.points]
        out += [(p, q, None, 0.0) for p, q in zip(pts, pts[1:], strict=False)]
    return out


def _wall_of(a, b, lines):
    """``(found, wall id)`` of the face line edge ``ab`` lies on; a separator's id is None."""
    for c, d, wall, reach in lines:
        span = _span(c, d, a, b)
        if span and span[1] > -reach + _T and span[0] < 1.0 + reach - _T:
            return True, wall
    return False, None


def _breaks(records) -> list[tuple]:
    """The face closures of every opening the skirting stops at: doors and openings
    whose sill is below :data:`SKIRTING_TOP` (an unknown sill does not stop it)."""
    from engineering.arch.model import DOOR_KINDS
    from engineering.arch.rooms import opening_closures

    out = []
    for op in records["openings"]:
        low = op.kind == "curtain_wall" or (op.sill is not None and op.sill < SKIRTING_TOP)
        if op.kind in DOOR_KINDS or low:
            closures, _ = opening_closures(records["walls"], [op])
            out += [(op.id, c, d) for c, d in closures]
    return out


def _miter(v, n1, n2, gap):
    dot = n1[0] * n2[0] + n1[1] * n2[1]
    if dot < -0.99:
        return (v[0] + n1[0] * gap, v[1] + n1[1] * gap)
    k = gap / (1.0 + dot)
    return (v[0] + (n1[0] + n2[0]) * k, v[1] + (n1[1] + n2[1]) * k)


def skirting_runs(loops, records, types, fitted, gap) -> tuple[list[dict], list[str]]:
    """``(pieces, openings it stops at)`` along the room's loops (outer, then holes).

    A piece is ``{type, walls, length_m, points, closed}``: its length is measured
    on the wall face, its points the line ``gap`` inside it.
    """
    lines = _face_lines(records) if records else []
    breaks = _breaks(records) if records else []
    stopped: set[str] = set()
    pieces = []
    for index, loop in enumerate(loops):
        side = 1.0 if index == 0 else -1.0  # the room: left of the outer loop, right of a hole
        n_edges = len(loop)
        runs = []
        for i in range(n_edges):
            a, b = loop[i], loop[(i + 1) % n_edges]
            length = math.dist(a, b)
            if length < _ON:
                continue
            ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
            n = (-uy * side, ux * side)
            found, wall = _wall_of(a, b, lines)
            kind = "none" if found and wall is None else types.get(wall, types["default"])
            if kind == "none":
                continue
            cuts = _covered(a, b, n, fitted)
            for op_id, c, d in breaks:
                span = _span(a, b, c, d)
                if span and span[1] > _T and span[0] < 1.0 - _T:
                    cuts.append(span)
                    stopped.add(op_id)
            for t0, t1 in _keep(cuts, length):
                runs.append((i, t0, t1, kind, wall, a, b, n, length))
        if not runs:
            continue

        def joins(r, s, n=n_edges):
            return s[3] == r[3] and r[2] > 1 - _T and s[1] < _T and s[0] == (r[0] + 1) % n

        closed = all(joins(runs[k - 1], runs[k]) for k in range(len(runs)))
        start = (
            0 if closed else next(k for k in range(len(runs)) if not joins(runs[k - 1], runs[k]))
        )
        groups: list[list] = []
        for r in runs[start:] + runs[:start]:
            if groups and joins(groups[-1][-1], r) and not (closed and r is runs[start]):
                groups[-1].append(r)
            else:
                groups.append([r])
        for group in groups:

            def at(r, t):
                return (r[5][0] + (r[6][0] - r[5][0]) * t, r[5][1] + (r[6][1] - r[5][1]) * t)

            first, last = group[0], group[-1]
            pts = [] if closed else [_miter(at(first, first[1]), first[7], first[7], gap)]
            pairs = list(zip(group, group[1:], strict=False)) + ([(last, first)] if closed else [])
            pts += [_miter(r[6], r[7], s[7], gap) for r, s in pairs]
            if not closed:
                pts.append(_miter(at(last, last[2]), last[7], last[7], gap))
            walls = list(dict.fromkeys(r[4] for r in group if r[4]))
            pieces.append(
                {
                    "type": first[3],
                    "walls": walls,
                    "length_m": round(sum((r[2] - r[1]) * r[8] for r in group) / 1000.0, 3),
                    "points": [[round(x, 3), round(y, 3)] for x, y in pts],
                    "closed": closed,
                }
            )
    return pieces, sorted(stopped)


async def _outline(backend, room, outline):
    """``(outline, holes, source, records)`` of the zone, refused by name; ``records``
    are the plan's (None for a typed outline)."""
    if (room is None) == (outline is None):
        raise ValueError("room / outline: give exactly one (a room id or [x, y], or points)")
    if outline is not None:
        return polygon(outline, "outline"), [], "outline", None
    from engineering.arch.faces import face_containing
    from engineering.arch.rooms import plan_faces, read_records

    records = await read_records(backend)
    if isinstance(room, str):
        found = [r for r in records["rooms"] if r.id == room]
        if not found:
            ids = sorted(r.id for r in records["rooms"])
            raise ValueError(f"room: no room {room!r} on this drawing; rooms are {ids}")
        if found[0].boundary:
            return polygon(found[0].boundary, "room.boundary"), [], "room_boundary", records
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
    holes = [polygon(h, "room.hole") for h in face.holes]
    return polygon(face.loop, "room"), holes, "walls", records


async def floor_zone(
    backend,
    *,
    room=None,
    outline=None,
    minus=("fitted",),
    minus_handles=(),
    pattern: str | None = None,
    joints=None,
    angle: float = 0.0,
    start=None,
    scale: float = 50.0,
    pattern_scale: float = 1.0,
    label: str | None = None,
    name: str | None = None,
    layer: str | None = None,
    color: int | None = None,
    edge: bool = False,
    skirting=None,
    lay: str | None = None,
    plank=None,
    gap: float = 3.0,
    bond: float = 0.5,
    cut_angle: float = 45.0,
    waste_pct: float = 10.0,
    kerf: float = 3.0,
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
    if lay is not None and joints is not None:
        raise ValueError("joints: lay draws the joints of its pieces; give one of lay / joints")
    if lay is not None and (
        not isinstance(plank, (list, tuple)) or len(plank) != 2 or None in plank
    ):
        raise ValueError("plank_length / plank_width: lay needs both, mm")
    no_fill = isinstance(pattern, str) and pattern.strip().lower() == "none"
    if no_fill:
        pattern = None
    elif pattern is not None or (joints is None and lay is None):
        pattern = _hatch_pattern(pattern or "ANSI37")
    floor, holes, source, records = await _outline(backend, room, outline)
    walls = {w.id for w in records["walls"]} if records else set()
    types = check_skirting(skirting, walls)
    if pattern is None and joints is None and lay is None and not edge and types is None:
        raise ValueError("pattern: 'none' with no joints, edge or skirting draws nothing")
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
    }
    from ezdxf.tools.pattern import ISO_PATTERN

    side = min(max(xs) - min(xs), max(ys) - min(ys))
    warnings = []
    if pattern not in (None, "SOLID"):
        pscale = hatch_scale(pattern, scale, pattern_scale)
        row, warn = _fill_report(pattern, fill_period(ISO_PATTERN[pattern], pscale), side)
        result["pattern"] = {"name": pattern, "hatch_scale": pscale, **row}
        warnings += [warn] if warn else []
    planks = []  # the laid pieces
    if lay is not None:
        from engineering.arch.tiles import layout

        try:
            laid = layout(
                floor,
                length=plank[0],
                width=plank[1],
                pattern=lay,
                joint=gap,
                start=start,
                angle_deg=angle,
                bond=bond,
                cut_deg=cut_angle,
                cutouts=[{"points": c} for c in cutouts],
                waste_pct=waste_pct,
                kerf=kerf,
            )
        except ValueError as exc:  # the engine names its own arguments; name ours
            text = str(exc)
            for old, new in _LAY_NAMES:
                text = text.replace(old, new)
            raise ValueError(text) from None
        planks = laid["pieces"]
        cut_shares = [p.area / (plank[0] * plank[1]) for p in planks if not p.full]
        result["lay"] = {
            "pattern": lay,
            "plank": [float(plank[0]), float(plank[1])],
            "gap": float(gap),
            "start": list(laid["origin"]),
            "angle": float(angle),
            **({"cut_angle": float(cut_angle)} if lay == "chevron" else {}),
            "takeoff": laid["takeoff"],
            "smallest_cut_share": round(min(cut_shares), 4) if cut_shares else None,
        }
    if joints is not None:
        row, warn = _fill_report("joints", max(joints), side)
        result["joints"] = {"spacing_mm": joints, **row}
        warnings += [warn] if warn else []
    if types is not None:
        fitted = cutouts[len(holes) :]
        pieces, stopped = skirting_runs(
            [floor, *holes], records, types, fitted, SKIRTING_GAP_PAPER_MM * scale
        )
        totals = {
            f"{kind}_m": round(sum(p["length_m"] for p in pieces if p["type"] == kind), 2)
            for kind in ("applied", "flush")
        }
        result["skirting"] = {**totals, "stops_at": stopped, "pieces": pieces}
        if records is None:
            warnings.append(
                {
                    "code": "skirting_openings_unknown",
                    "message": "outline: no plan records, so the skirting runs past every door",
                }
            )
    if label:
        result["legend"] = label
    elif pattern is not None or joints is not None:
        warnings.append(
            {"code": "no_label", "message": "no label: arch_legend shows the pattern name"}
        )
    if warnings:
        result["warnings"] = warnings
    if dry_run:
        return result

    from ezdxf.lldxf import const as dxfconst

    from backends.xdata_specs import encode_values
    from engineering.annotation_record import APP_ID, annotation_values, replace_set
    from engineering.arch.catalogue import _ensure_layer
    from engineering.arch.services import ISO_PEN
    from engineering.mech.draw import guarded_transaction

    a = math.radians(angle)
    origin = tuple(start) if start is not None else (min(xs), min(ys))
    lay_spec = lay and {  # what arch_legend's sample redraws
        "pattern": lay, "length": float(plank[0]), "width": float(plank[1]), "gap": float(gap),
        "angle": float(angle), "bond": float(bond), "cut": float(cut_angle),
    }  # fmt: skip

    def _draw():
        doc = backend._require_doc()
        msp = doc.modelspace()
        attribs = {"layer": layer, **({"color": int(color)} if color is not None else {})}

        def _hatch():
            hatch = msp.add_hatch(color=256, dxfattribs=attribs)
            for loop in loops:
                flag = 1 if signed_area(loop) > 0 else 0
                hatch.paths.add_polyline_path(list(loop), is_closed=True, flags=flag)
            return hatch

        handles = {}
        if pattern == "SOLID":
            hatch = _hatch()
            hatch.set_solid_fill(color=hatch.dxf.color)
            handles["pattern"] = hatch.dxf.handle
        elif pattern is not None:
            hatch = _hatch()
            hatch.set_pattern_fill(
                pattern,
                color=hatch.dxf.color,
                scale=hatch_scale(pattern, scale, pattern_scale),
                angle=float(angle),
            )
            handles["pattern"] = hatch.dxf.handle
        if joints is not None:
            along, across = joints
            hatch = _hatch()
            hatch.set_pattern_fill(
                "_USER",
                color=hatch.dxf.color,
                pattern_type=dxfconst.HATCH_TYPE_USER_DEFINED,
                definition=[
                    [angle, origin, (-across * math.sin(a), across * math.cos(a)), []],
                    [angle + 90.0, origin, (along * math.cos(a), along * math.sin(a)), []],
                ],
            )
            handles["joints"] = hatch.dxf.handle
        if planks:
            handles["pieces"] = [
                msp.add_lwpolyline(loop, close=True, dxfattribs=attribs).dxf.handle
                for piece in planks
                for loop in piece.loops
            ]
        if edge:
            handles["edge"] = [
                msp.add_lwpolyline(loop, close=True, dxfattribs=attribs).dxf.handle
                for loop in loops
            ]
        skirts = []
        for piece in result.get("skirting", {}).get("pieces", ()):
            skirt_layer = ARCH_ROLE_LAYER[f"skirting_{piece['type']}"]
            dashed = doc_linetype(skirt_layer).upper().startswith("ACAD_ISO")
            line = msp.add_lwpolyline(
                [tuple(p) for p in piece["points"]],
                close=piece["closed"],
                dxfattribs={"layer": skirt_layer, **({"ltscale": ISO_PEN} if dashed else {})},
            )
            skirts.append((line.dxf.handle, f"skirting_{piece['type']}"))
        if skirts:
            handles["skirting"] = [h for h, _ in skirts]
        if APP_ID not in doc.appids:
            doc.appids.add(APP_ID)
        tagged = [
            (h, label, lay_spec if key == "pieces" else None)
            for key, v in handles.items()
            if key != "skirting"
            for h in (v if isinstance(v, list) else [v])
        ]
        for handle, legend, spec in [*tagged, *((h, k, None) for h, k in skirts)]:
            tags = encode_values(annotation_values(TOOL, handle, name, legend, spec))
            doc.entitydb[handle].set_xdata(APP_ID, tags)
        backend._mark_dirty()
        return handles, skirts

    def doc_linetype(name):
        return backend._require_doc().layers.get(name).dxf.linetype

    async def job():
        replaced = await replace_set(backend, None, TOOL, name) if name else 0
        await _ensure_layer(backend, layer)
        for kind in {p["type"] for p in result.get("skirting", {}).get("pieces", ())}:
            await _ensure_layer(backend, ARCH_ROLE_LAYER[f"skirting_{kind}"])
        handles, skirts = await backend._async(_draw)
        drawn = [
            h
            for key, v in handles.items()
            if key != "skirting"
            for h in (v if isinstance(v, list) else [v])
        ]
        if drawn:
            await backend.draw_order(drawn, "send_to_back")
        return handles, replaced

    handles, replaced = await guarded_transaction(backend, job, what=TOOL, done="drawn")
    return {**result, "handles": handles, "replaced": replaced}
