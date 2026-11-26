"""Interior wall elevations (``interior_wall_elevation``): one wall seen from the room.

An elevation is drawn in its own frame - x along the wall as the viewer sees
it, from the left end of the wall, y up from the finished floor - and placed on
the drawing with its floor-left corner at ``at``. It carries:

* the **outline**: floor, ceiling and both ends of the wall (``length`` x
  ``height``), with the floor and ceiling **level marks** on the left
  (``floor_level`` in metres, the ceiling at ``floor_level + height``);
* the **openings** (doors from the floor to their head, windows from their sill
  to their head), as frames; a door given a ``hinge`` gets the opening triangle
  whose apex is on the hinge side;
* the **fittings**: ``skirting`` (along the whole wall by default, interrupted
  by every opening that comes down into it), ``worktop``, ``base_cabinet``,
  ``wall_cabinet``, ``mirror``, ``panel``, ``socket`` and ``switch`` (a box of
  ``size`` x ``size`` per gang around its centre);
* the **service axes**: a centre-line cross at each connection point (water,
  waste, gas, power, data, heating), with its short label;
* the **dimensions**, as real DIMENSIONs: on the right a chain of every level
  the elements define (skirting top, worktop top, cabinet bottoms and tops,
  mirror, sills and heads, socket and service centres) and the floor-to-ceiling
  overall; below, a chain of the jambs, the socket and switch centres and the
  service axes, and the overall length. A chain that would only restate the
  overall one is not drawn (ISO 129-1: each measurement once).

The wall can be typed (``length`` and ``openings``) or read from the plan: a
wall drawn with ``arch_wall`` / ``arch_opening``, one of its axis segments and
the side it is seen from. Then the length is the clear face between the
junctions on that side, measured off the wall engine's own faces, and the
openings are the ones hosted there, mirrored as the viewer sees them - never
re-typed.

Sizes on paper are millimetres times ``scale`` (the plot-scale denominator: 20
is 1:20) - text, level marks, the service cross and the row spacing. Those
paper sizes, the 40 mm worktop thickness and the 80 mm socket box are this
module's drafting defaults, declared as such; no standard here fixes them, and
no mounting height is ever assumed: every height comes from the caller or the
plan.

Every refusal fires before anything is drawn and names its path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engineering.arch.dimension import FIRST_OFFSET_PAPER_MM, STEP_PAPER_MM
from engineering.arch.lang import vocab
from engineering.arch.model import WINDOW_KINDS
from engineering.arch.symbols import LEVEL_LINE, LEVEL_TRIANGLE, symbol_prims
from engineering.mech.primitives import Circle, DimIntent, Line, Poly, Prim, Pt, Text
from engineering.sheet.frames import centred_text

OPENING_KINDS: tuple[str, ...] = ("door", "window")
HINGES: tuple[str, ...] = ("left", "right")
FITTING_KINDS: tuple[str, ...] = (
    "skirting",
    "worktop",
    "base_cabinet",
    "wall_cabinet",
    "mirror",
    "panel",
    "socket",
    "switch",
)
SERVICE_KINDS: tuple[str, ...] = (
    "cold_water",
    "hot_water",
    "waste",
    "gas",
    "power",
    "data",
    "heating",
    "generic",
)
SIDES: tuple[str, ...] = ("left", "right")

# -- declared drafting defaults ------------------------------------------------

#: Paper millimetres (times ``scale``).
TEXT_PAPER_MM = 2.5
TITLE_PAPER_MM = 5.0
SERVICE_ARM_PAPER_MM = 5.0
SERVICE_RADIUS_PAPER_MM = 1.5
FRAME_PAPER_MM = 1.0
LEVEL_GAP_PAPER_MM = 2.0
#: Drawing millimetres: a nominal box, not a product size.
WORKTOP_THICKNESS = 40.0
DEVICE_SIZE = 80.0

#: Two levels closer than this are one level (mm).
_LEVEL_TOL = 0.5
_EPS = 1e-6

#: The service label as written on a drawing in each language.
SERVICE_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "cold_water": "CW",
        "hot_water": "HW",
        "waste": "W",
        "gas": "G",
        "power": "E",
        "data": "D",
        "heating": "H",
        "generic": "S",
    },
    "pl": {
        "cold_water": "ZW",
        "hot_water": "CWU",
        "waste": "K",
        "gas": "G",
        "power": "E",
        "data": "T",
        "heating": "CO",
        "generic": "P",
    },
    "tr": {
        "cold_water": "SS",
        "hot_water": "SC",
        "waste": "PS",
        "gas": "G",
        "power": "E",
        "data": "D",
        "heating": "IS",
        "generic": "T",
    },
}


@dataclass(frozen=True)
class ElevOpening:
    id: str
    kind: str  # door | window
    x: float
    width: float
    sill: float  # 0 for a door
    head: float
    hinge: str | None = None


@dataclass(frozen=True)
class Fitting:
    id: str
    kind: str
    x: float
    width: float
    bottom: float
    top: float
    label: str | None = None
    gangs: int = 1
    #: socket / switch: the centre height; the level the dimension chain reads
    centre: float | None = None


@dataclass(frozen=True)
class Service:
    id: str
    kind: str
    x: float
    y: float
    label: str


@dataclass(frozen=True)
class Tiling:
    """Wall tiling: ``tiles.layout`` arguments over the band ``bottom``..``top``.

    ``cut_around`` are the ids the tiles stop at: every opening and, unless the
    caller names them, every worktop, cabinet and panel (a mirror, a socket and
    a switch sit on the tiles; a skirting is below ``bottom`` when the caller
    starts the band on it).
    """

    bottom: float
    top: float
    cut_around: tuple[str, ...]
    layout: dict


@dataclass(frozen=True)
class Elevation:
    length: float
    height: float
    floor_level: float
    openings: tuple[ElevOpening, ...]
    fittings: tuple[Fitting, ...]
    services: tuple[Service, ...]
    title: str | None = None
    tiling: Tiling | None = None


# -- field readers ---------------------------------------------------------------


def _number(value, where: str, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: expected a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{where}: must be finite, got {value!r}")
    if minimum is not None and (number <= minimum if strict else number < minimum):
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{where}: must be {relation} {minimum:g}, got {number:g}")
    return number


def _known(d, where: str, allowed) -> dict:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected a dict, got {type(d).__name__}")
    unknown = sorted(set(d) - set(allowed))
    if unknown:
        raise ValueError(
            f"{where}: unknown field(s) {', '.join(unknown)}; fields are {', '.join(allowed)}"
        )
    return d


def _required(d: dict, key: str, where: str):
    if d.get(key) is None:
        raise ValueError(f"{where}.{key}: required")
    return d[key]


def _choice(value, where: str, allowed) -> str:
    if not isinstance(value, str) or value.strip().lower() not in allowed:
        raise ValueError(f"{where}: {value!r} is not one of {', '.join(allowed)}")
    return value.strip().lower()


def _label(value, where: str) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        raise ValueError(f"{where}: must not be empty")
    return text


def _list(value, where: str) -> list:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{where}: expected a list, got {type(value).__name__}")
    return list(value)


# -- the model -------------------------------------------------------------------

_OPENING_FIELDS = ("id", "kind", "x", "width", "height", "sill", "hinge")
_FITTING_FIELDS = (
    "id",
    "kind",
    "x",
    "width",
    "bottom",
    "top",
    "label",
    "thickness",
    "y",
    "size",
    "gangs",
)
_SERVICE_FIELDS = ("id", "kind", "x", "y", "label")
SPEC_FIELDS = (
    "length",
    "height",
    "floor_level",
    "openings",
    "fittings",
    "services",
    "title",
    "tiling",
)
#: ``spec.tiling`` fields: the band, what it stops at, and ``interior_tile_layout``'s own
#: arguments under the same names (``start_x`` / ``start_y`` elevation-local).
TILING_FIELDS = (
    "bottom",
    "top",
    "cut_around",
    "tile_length",
    "tile_width",
    "pattern",
    "joint",
    "start_x",
    "start_y",
    "angle_deg",
    "bond",
    "waste_pct",
    "offcut_reuse",
    "kerf",
)
#: Fittings the tiles stop at unless ``cut_around`` names others.
TILE_AROUND_KINDS: tuple[str, ...] = ("worktop", "base_cabinet", "wall_cabinet", "panel")


def _opening(d, where: str, length: float, height: float, index: int) -> ElevOpening:
    _known(d, where, _OPENING_FIELDS)
    kind = _choice(_required(d, "kind", where), f"{where}.kind", OPENING_KINDS)
    x = _number(_required(d, "x", where), f"{where}.x", minimum=0.0)
    width = _number(_required(d, "width", where), f"{where}.width", minimum=0.0, strict=True)
    tall = _number(_required(d, "height", where), f"{where}.height", minimum=0.0, strict=True)
    if kind == "door":
        if d.get("sill") not in (None, 0, 0.0):
            raise ValueError(f"{where}.sill: a door has no sill; it stands on the floor")
        sill = 0.0
    else:
        sill = _number(_required(d, "sill", where), f"{where}.sill", minimum=0.0)
    hinge = d.get("hinge")
    if hinge is not None:
        if kind != "door":
            raise ValueError(f"{where}.hinge: only a door has a hinge side")
        hinge = _choice(hinge, f"{where}.hinge", HINGES)
    if x + width > length + _EPS:
        raise ValueError(
            f"{where}: runs to {x + width:g}, past the end of the {length:g} long wall"
        )
    if sill + tall > height + _EPS:
        raise ValueError(f"{where}: its head at {sill + tall:g} is above the {height:g} ceiling")
    oid = _label(d.get("id"), f"{where}.id") or f"{'D' if kind == 'door' else 'W'}{index + 1}"
    return ElevOpening(oid, kind, x, width, sill, sill + tall, hinge)


def _fitting(d, where: str, length: float, height: float, index: int) -> Fitting:
    _known(d, where, _FITTING_FIELDS)
    kind = _choice(_required(d, "kind", where), f"{where}.kind", FITTING_KINDS)
    allowed = {
        "skirting": {"x", "width", "top"},
        "worktop": {"x", "width", "top", "thickness"},
        "socket": {"x", "y", "size", "gangs"},
        "switch": {"x", "y", "size", "gangs"},
    }.get(kind, {"x", "width", "bottom", "top"}) | {"id", "kind", "label"}
    extra = sorted(set(d) - allowed)
    if extra:
        raise ValueError(
            f"{where}: a {kind} does not take {', '.join(extra)}; it takes "
            f"{', '.join(sorted(allowed))}"
        )
    label = _label(d.get("label"), f"{where}.label")
    gangs = 1
    centre = None
    if kind == "skirting":
        x = _number(d.get("x", 0.0), f"{where}.x", minimum=0.0)
        width = _number(d.get("width", length - x), f"{where}.width", minimum=0.0, strict=True)
        bottom = 0.0
        top = _number(_required(d, "top", where), f"{where}.top", minimum=0.0, strict=True)
    elif kind == "worktop":
        x = _number(_required(d, "x", where), f"{where}.x", minimum=0.0)
        width = _number(_required(d, "width", where), f"{where}.width", minimum=0.0, strict=True)
        top = _number(_required(d, "top", where), f"{where}.top", minimum=0.0, strict=True)
        thickness = _number(
            d.get("thickness", WORKTOP_THICKNESS), f"{where}.thickness", minimum=0.0, strict=True
        )
        bottom = top - thickness
        if bottom < -_EPS:
            raise ValueError(
                f"{where}: a {thickness:g} thick worktop cannot have its top at {top:g}"
            )
    elif kind in ("socket", "switch"):
        cx = _number(_required(d, "x", where), f"{where}.x", minimum=0.0)
        centre = _number(_required(d, "y", where), f"{where}.y", minimum=0.0)
        size = _number(d.get("size", DEVICE_SIZE), f"{where}.size", minimum=0.0, strict=True)
        raw = d.get("gangs", 1)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError(f"{where}.gangs: expected a whole number of at least 1, got {raw!r}")
        gangs = raw
        width = size * gangs
        x = cx - width / 2.0
        bottom, top = centre - size / 2.0, centre + size / 2.0
        if x < -_EPS or bottom < -_EPS:
            raise ValueError(f"{where}: its {width:g} x {size:g} box runs past the wall's edge")
    else:
        x = _number(_required(d, "x", where), f"{where}.x", minimum=0.0)
        width = _number(_required(d, "width", where), f"{where}.width", minimum=0.0, strict=True)
        bottom = _number(_required(d, "bottom", where), f"{where}.bottom", minimum=0.0)
        top = _number(_required(d, "top", where), f"{where}.top", minimum=0.0)
        if top <= bottom + _EPS:
            raise ValueError(f"{where}: top {top:g} must be above bottom {bottom:g}")
    if x + width > length + _EPS:
        raise ValueError(
            f"{where}: runs to {x + width:g}, past the end of the {length:g} long wall"
        )
    if top > height + _EPS:
        raise ValueError(f"{where}: its top at {top:g} is above the {height:g} ceiling")
    fid = _label(d.get("id"), f"{where}.id") or f"{kind}{index + 1}"
    return Fitting(fid, kind, x, width, bottom, top, label, gangs, centre)


def _service(d, where: str, length: float, height: float, index: int, lang: str) -> Service:
    _known(d, where, _SERVICE_FIELDS)
    kind = _choice(_required(d, "kind", where), f"{where}.kind", SERVICE_KINDS)
    x = _number(_required(d, "x", where), f"{where}.x", minimum=0.0)
    y = _number(_required(d, "y", where), f"{where}.y", minimum=0.0)
    if x > length + _EPS or y > height + _EPS:
        raise ValueError(f"{where}: ({x:g}, {y:g}) lies outside the {length:g} x {height:g} wall")
    label = _label(d.get("label"), f"{where}.label") or SERVICE_LABELS[lang][kind]
    sid = _label(d.get("id"), f"{where}.id") or f"{kind}{index + 1}"
    return Service(sid, kind, x, y, label)


def _box(item) -> tuple[float, float, float, float]:
    if isinstance(item, ElevOpening):
        return (item.x, item.x + item.width, item.sill, item.head)
    return (item.x, item.x + item.width, item.bottom, item.top)


def _overlap(a, b) -> bool:
    return a[0] < b[1] - _EPS and b[0] < a[1] - _EPS and a[2] < b[3] - _EPS and b[2] < a[3] - _EPS


def parse_elevation(spec, *, lang: str = "en") -> Elevation:
    """The elevation model, validated by path; nothing is drawn here.

    Refused: an unknown field, a non-positive length or height, an opening,
    fitting or service outside the wall, two openings side by side over the
    same stretch, a fitting or a service inside an opening (both ids named),
    a duplicate id, an unknown kind.
    """
    vocab(lang)
    _known(spec, "spec", SPEC_FIELDS)
    length = _number(_required(spec, "length", "spec"), "spec.length", minimum=0.0, strict=True)
    height = _number(_required(spec, "height", "spec"), "spec.height", minimum=0.0, strict=True)
    floor_level = _number(spec.get("floor_level", 0.0), "spec.floor_level")
    openings = tuple(
        _opening(d, f"openings[{i}]", length, height, i)
        for i, d in enumerate(_list(spec.get("openings"), "spec.openings"))
    )
    fittings = tuple(
        _fitting(d, f"fittings[{i}]", length, height, i)
        for i, d in enumerate(_list(spec.get("fittings"), "spec.fittings"))
    )
    services = tuple(
        _service(d, f"services[{i}]", length, height, i, lang)
        for i, d in enumerate(_list(spec.get("services"), "spec.services"))
    )
    seen: dict[str, str] = {}
    for group, items in (("openings", openings), ("fittings", fittings), ("services", services)):
        for i, item in enumerate(items):
            if item.id in seen:
                raise ValueError(f"{group}[{i}].id: {item.id!r} is already used by {seen[item.id]}")
            seen[item.id] = f"{group}[{i}]"
    for i, a in enumerate(openings):
        for b in openings[i + 1 :]:
            if a.x < b.x + b.width - _EPS and b.x < a.x + a.width - _EPS:
                raise ValueError(f"openings: {a.id} and {b.id} overlap along the wall")
    solid = [f for f in fittings if f.kind != "skirting"]
    for i, a in enumerate(solid):
        for b in solid[i + 1 :]:
            if _overlap(_box(a), _box(b)):
                raise ValueError(f"fittings: {a.id} and {b.id} overlap")
    for opening in openings:
        box = _box(opening)
        for fitting in fittings:
            if fitting.kind != "skirting" and _overlap(box, _box(fitting)):
                raise ValueError(f"fittings: {fitting.id} lies inside opening {opening.id}")
        for service in services:
            if box[0] < service.x < box[1] and box[2] < service.y < box[3]:
                raise ValueError(f"services: {service.id} lies inside opening {opening.id}")
    title = _label(spec.get("title"), "spec.title")
    tiling = None
    if spec.get("tiling") is not None:
        tiling = _tiling(spec["tiling"], length, height, openings, fittings)
    return Elevation(length, height, floor_level, openings, fittings, services, title, tiling)


def _tiling(d, length: float, height: float, openings, fittings) -> Tiling:
    where = "spec.tiling"
    _known(d, where, TILING_FIELDS)
    bottom = _number(d.get("bottom", 0.0), f"{where}.bottom", minimum=0.0)
    top = _number(d.get("top", height), f"{where}.top", minimum=0.0)
    if top > height + _EPS:
        raise ValueError(f"{where}.top: {top:g} is above the {height:g} ceiling")
    if top <= bottom + _EPS:
        raise ValueError(f"{where}: top {top:g} must be above bottom {bottom:g}")
    ids = {f.id: f for f in fittings}
    if d.get("cut_around") is None:
        around = [f.id for f in fittings if f.kind in TILE_AROUND_KINDS]
    else:
        around = []
        for i, name in enumerate(_list(d["cut_around"], f"{where}.cut_around")):
            if name not in ids:
                raise ValueError(
                    f"{where}.cut_around[{i}]: {name!r} is not a fitting; the fittings are "
                    f"{', '.join(ids) or 'none'} (openings are always cut around)"
                )
            around.append(name)
    for key in ("tile_length", "tile_width"):
        _required(d, key, where)
    args = {k: d[k] for k in TILING_FIELDS[3:] if k in d and k not in ("start_x", "start_y")}
    args["length"] = args.pop("tile_length")
    args["width"] = args.pop("tile_width")
    if ("start_x" in d) != ("start_y" in d):
        raise ValueError(f"{where}.start_x / start_y: give both or neither")
    if "start_x" in d:
        args["start"] = (d["start_x"], d["start_y"])
    band = (0.0, length, bottom, top)
    cutouts = []
    for item in (*openings, *(ids[name] for name in around)):
        b = _box(item)
        x0, x1 = max(b[0], band[0]), min(b[1], band[1])
        y0, y1 = max(b[2], band[2]), min(b[3], band[3])
        if x1 - x0 > _EPS and y1 - y0 > _EPS:
            cutouts.append({"points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "label": item.id})
    args["cutouts"] = cutouts
    args["outline"] = [[0.0, bottom], [length, bottom], [length, top], [0.0, top]]
    tiling = Tiling(bottom, top, tuple(around), args)
    tile_pieces(tiling)  # refuses a bad size, pattern or count now, before anything is drawn
    return tiling


def tile_pieces(tiling: Tiling) -> dict:
    """``engineering.arch.tiles.layout`` over the band; refusals named under ``spec.tiling``."""
    from engineering.arch.tiles import layout

    args = dict(tiling.layout)
    outline = args.pop("outline")
    try:
        return layout(outline, **args)
    except ValueError as exc:
        raise ValueError(f"spec.tiling.{exc}") from None


# -- from the plan ---------------------------------------------------------------


def plan_wall_stretch(wall, openings, faces, *, segment: int, side: str, at=None) -> dict:
    """The clear face of one axis segment of a drawn wall, seen from ``side``.

    ``faces`` are the wall engine's face lines for this wall
    (``wall_geometry_by_wall(...)[wall.id].faces``). The face on ``side``
    (looking along the axis) is cut by junctions and by openings; the stretch
    is the run of face pieces joined only across this segment's openings that
    contains ``at`` (a distance from the segment start; default its middle).

    Returns ``{length, s0, s1, reverse, openings}``: ``openings`` are the
    hosted openings on the stretch as ``(Opening, x)`` with ``x`` the near
    jamb in elevation coordinates. Seen from the left of the axis the viewer
    faces the right normal and the axis runs right to left, so the elevation is
    mirrored (``reverse``).
    """
    from engineering.arch.model import axis_segments, segment_of
    from engineering.arch.walls import face_offsets

    segs = axis_segments(wall)
    if not 0 <= segment < len(segs):
        raise ValueError(
            f"from_plan.segment: wall {wall.id!r} has segments 0..{len(segs) - 1}, got {segment}"
        )
    side = _choice(side, "from_plan.side", SIDES)
    a, b = segs[segment]
    seg_len = math.dist(a, b)
    ux, uy = (b[0] - a[0]) / seg_len, (b[1] - a[1]) / seg_len
    nx, ny = -uy, ux
    offset = face_offsets(wall)[0 if side == "left" else 1]
    tol = 1.0
    pieces: list[tuple[float, float]] = []
    for line in faces:
        projs = []
        for p in (line.p1, line.p2):
            dx, dy = p[0] - a[0], p[1] - a[1]
            projs.append((dx * ux + dy * uy, dx * nx + dy * ny))
        if all(abs(q[1] - offset) <= tol for q in projs):
            lo, hi = sorted(q[0] for q in projs)
            if hi - lo > tol:
                pieces.append((lo, hi))
    hosted = []
    for op in openings:
        if op.wall != wall.id:
            continue
        index, t = segment_of(wall, op.offset)
        if index == segment:
            hosted.append((op, t, t + op.width))
    spans = sorted(pieces + [(t0, t1) for _op, t0, t1 in hosted])
    if not pieces:
        raise ValueError(
            f"from_plan: wall {wall.id!r} segment {segment} has no face on its {side} side"
        )
    merged: list[list[float]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    where = seg_len / 2.0 if at is None else _number(at, "from_plan.at")
    chosen = next((m for m in merged if m[0] - tol <= where <= m[1] + tol), None)
    if chosen is None:
        stretches = ", ".join(f"{lo:g}..{hi:g}" for lo, hi in merged)
        raise ValueError(
            f"from_plan.at: {where:g} is on no clear stretch of that face; the stretches "
            f"(distance from the segment start) are {stretches}"
        )
    s0, s1 = chosen
    reverse = side == "left"
    placed = []
    for op, t0, t1 in hosted:
        if t0 >= s0 - tol and t1 <= s1 + tol:
            placed.append((op, (s1 - t1) if reverse else (t0 - s0)))
    return {"length": s1 - s0, "s0": s0, "s1": s1, "reverse": reverse, "openings": placed}


def plan_openings(placed, heights) -> list[dict]:
    """Elevation openings from plan openings: kind, x, width, and the heights.

    A plan opening's ``height`` / ``sill`` are schedule fields and may be
    unset; ``heights`` (``{opening id: {height, sill}}``) fills or overrides
    them. A door without a height, or a window without a height or a sill, is
    refused by id: a height is never assumed.
    """
    heights = _known(heights or {}, "from_plan.heights", [op.id for op, _x in placed])
    out = []
    for op, x in placed:
        given = _known(heights.get(op.id) or {}, f"from_plan.heights.{op.id}", ("height", "sill"))
        kind = "window" if op.kind in WINDOW_KINDS else "door"
        tall = given.get("height", op.height)
        sill = given.get("sill", op.sill)
        if tall is None:
            raise ValueError(
                f"from_plan: opening {op.id} has no height on the plan; give "
                f"from_plan.heights.{op.id}.height"
            )
        row = {"id": op.id, "kind": kind, "x": x, "width": op.width, "height": tall}
        if kind == "window":
            if sill is None:
                raise ValueError(
                    f"from_plan: window {op.id} has no sill on the plan; give "
                    f"from_plan.heights.{op.id}.sill"
                )
            row["sill"] = sill
        out.append(row)
    return out


# -- levels and dimensions -----------------------------------------------------------


def elevation_levels(model: Elevation) -> list[dict]:
    """Every height the elements define, floor and ceiling included, with what sets it."""
    found: list[tuple[float, str]] = [(0.0, "floor"), (model.height, "ceiling")]
    for op in model.openings:
        if op.kind == "window":
            found.append((op.sill, f"{op.id}.sill"))
        found.append((op.head, f"{op.id}.head"))
    for f in model.fittings:
        if f.centre is not None:
            found.append((f.centre, f"{f.id}.centre"))
            continue
        if f.bottom > _LEVEL_TOL and f.kind != "worktop":
            found.append((f.bottom, f"{f.id}.bottom"))
        found.append((f.top, f"{f.id}.top"))
    for s in model.services:
        found.append((s.y, f"{s.id}.axis"))
    if model.tiling is not None:
        found += [(model.tiling.bottom, "tiling.bottom"), (model.tiling.top, "tiling.top")]
    levels: list[dict] = []
    for value, source in sorted(found, key=lambda pair: pair[0]):
        if levels and value - levels[-1]["level"] <= _LEVEL_TOL:
            levels[-1]["from"].append(source)
        else:
            levels.append({"level": value, "from": [source]})
    return levels


def _positions(model: Elevation) -> list[float]:
    xs = [0.0, model.length]
    for op in model.openings:
        xs += [op.x, op.x + op.width]
    for f in model.fittings:
        if f.kind in ("socket", "switch"):
            xs.append(f.x + f.width / 2.0)
    xs += [s.x for s in model.services]
    out: list[float] = []
    for x in sorted(xs):
        if not out or x - out[-1] > _LEVEL_TOL:
            out.append(x)
    return out


def elevation_dimensions(model: Elevation, *, scale: float) -> tuple[DimIntent, ...]:
    """The height chain and overall on the right, the position chain and overall below.

    Points sit on the outline (the right edge for heights, the floor for
    positions), as the plan chains sit on the wall faces. A chain that equals
    its overall row is not emitted.
    """
    k = _number(scale, "scale", minimum=0.0, strict=True)
    first, step = FIRST_OFFSET_PAPER_MM * k, STEP_PAPER_MM * k
    L, H = model.length, model.height
    intents: list[DimIntent] = []
    levels = [row["level"] for row in elevation_levels(model)]
    if len(levels) > 2:
        line = L + first
        for lo, hi in zip(levels, levels[1:], strict=False):
            intents.append(
                DimIntent("linear", (L, lo), (L, hi), (line, (lo + hi) / 2), feature="right:levels")
            )
    line = L + first + step
    intents.append(DimIntent("linear", (L, 0.0), (L, H), (line, H / 2), feature="right:overall"))
    xs = _positions(model)
    if len(xs) > 2:
        line = -first
        for lo, hi in zip(xs, xs[1:], strict=False):
            intents.append(
                DimIntent(
                    "linear",
                    (lo, 0.0),
                    (hi, 0.0),
                    ((lo + hi) / 2, line),
                    feature="bottom:positions",
                )
            )
    line = -(first + step)
    intents.append(
        DimIntent("linear", (0.0, 0.0), (L, 0.0), (L / 2, line), feature="bottom:overall")
    )
    return tuple(intents)


# -- primitives ----------------------------------------------------------------------


def _rect(x0, y0, x1, y1, role) -> Poly:
    return Poly(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), True, role)


def _label_in(box, text: str, h: float) -> Text:
    """In the top-left corner, clear of the service crosses a cabinet usually holds."""
    return Text((box[0] + 0.5 * h, box[3] - 1.5 * h), text, h, 0.0, "elev_text")


def _skirting_pieces(f: Fitting, openings) -> list[tuple[float, float]]:
    pieces = [(f.x, f.x + f.width)]
    for op in openings:
        if op.sill >= f.top - _EPS:
            continue
        cut = []
        for lo, hi in pieces:
            if op.x >= hi or op.x + op.width <= lo:
                cut.append((lo, hi))
                continue
            if op.x > lo + _EPS:
                cut.append((lo, op.x))
            if op.x + op.width < hi - _EPS:
                cut.append((op.x + op.width, hi))
        pieces = cut
    return pieces


def elevation_prims(model: Elevation, *, lang: str = "en", scale: float = 20) -> tuple[Prim, ...]:
    """The outline, level marks, openings, fittings, service axes and title - elevation-local."""
    k = _number(scale, "scale", minimum=0.0, strict=True)
    L, H = model.length, model.height
    h = TEXT_PAPER_MM * k
    prims: list[Prim] = [_rect(0.0, 0.0, L, H, "elev_outline")]
    # the level marks, left of the outline: apex, triangle, line, value
    apex = -(LEVEL_TRIANGLE + LEVEL_LINE + LEVEL_GAP_PAPER_MM) * k
    for y, value in ((0.0, model.floor_level), (H, model.floor_level + H / 1000.0)):
        prims += symbol_prims("level", (apex, y), lang=lang, scale=k, value=value)
    frame = FRAME_PAPER_MM * k
    for op in model.openings:
        x0, x1 = op.x, op.x + op.width
        prims.append(_rect(x0, op.sill, x1, op.head, "elev_fitting"))
        if op.kind == "window" and op.width > 4 * frame and op.head - op.sill > 4 * frame:
            prims.append(
                _rect(x0 + frame, op.sill + frame, x1 - frame, op.head - frame, "elev_fitting")
            )
        if op.hinge:
            hx, fx = (x0, x1) if op.hinge == "left" else (x1, x0)
            mid = (op.sill + op.head) / 2.0
            prims.append(Line((fx, op.sill), (hx, mid), "elev_fitting"))
            prims.append(Line((hx, mid), (fx, op.head), "elev_fitting"))
    for f in model.fittings:
        box = _box(f)
        if f.kind == "skirting":
            for lo, hi in _skirting_pieces(f, model.openings):
                prims.append(_rect(lo, 0.0, hi, f.top, "elev_fitting"))
            continue
        prims.append(_rect(box[0], box[2], box[1], box[3], "elev_fitting"))
        if f.kind in ("socket", "switch"):
            size = f.width / f.gangs
            for g in range(f.gangs):
                cx = f.x + size * (g + 0.5)
                if f.kind == "socket":
                    prims.append(Circle((cx, f.centre), 0.3 * size, "elev_fitting"))
                else:
                    q = 0.2 * size
                    prims.append(
                        _rect(
                            cx - q, f.centre - 1.5 * q, cx + q, f.centre + 1.5 * q, "elev_fitting"
                        )
                    )
        elif f.kind == "mirror":
            w, t = box[1] - box[0], box[3] - box[2]
            d = min(w, t) / 4.0
            for i in (1, 2):
                ox = box[0] + w * (0.15 + 0.1 * i)
                prims.append(
                    Line((ox, box[3] - t * 0.15 - d), (ox + d, box[3] - t * 0.15), "elev_fitting")
                )
        if f.label and box[1] - box[0] > (len(f.label) * 0.6 + 1.0) * h and box[3] - box[2] > 2 * h:
            prims.append(_label_in(box, f.label, h))
    arm = SERVICE_ARM_PAPER_MM * k
    r = SERVICE_RADIUS_PAPER_MM * k
    for s in model.services:
        prims.append(Line((s.x - arm, s.y), (s.x + arm, s.y), "elev_service"))
        prims.append(Line((s.x, s.y - arm), (s.x, s.y + arm), "elev_service"))
        prims.append(Circle((s.x, s.y), r, "elev_fitting"))
        prims.append(Text((s.x + r + 0.5 * k, s.y + r + 0.5 * k), s.label, h, 0.0, "elev_text"))
    if model.title:
        th = TITLE_PAPER_MM * k
        y = -(FIRST_OFFSET_PAPER_MM + 2 * STEP_PAPER_MM + TITLE_PAPER_MM) * k
        text = f"{model.title}  1:{k:g}"
        prims.append(centred_text(L / 2.0, y, text, th)._replace(role="elev_text"))
    return tuple(prims)


def shift_intents(intents, dx: float, dy: float) -> tuple[DimIntent, ...]:
    def move(p: Pt) -> Pt:
        return (p[0] + dx, p[1] + dy)

    return tuple(i._replace(p1=move(i.p1), p2=move(i.p2), text_at=move(i.text_at)) for i in intents)
