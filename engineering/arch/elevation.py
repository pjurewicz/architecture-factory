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
  by every opening that comes down into it), ``worktop``, ``shelf``,
  ``base_cabinet``, ``wall_cabinet``, ``mirror``, ``panel``, ``niche``,
  ``socket`` and ``switch`` (a box of ``size`` x ``size`` per gang around its
  centre). They may overlap: list order is the z-order, and a later fitting
  hides the lines, circles and labels an earlier one draws inside its box;
* the **finish zones** (``zones``): rectangles behind every fitting, filled
  with ``arch_floor_zone``'s engine (:func:`engineering.arch.floor.fill_hatches`,
  :func:`engineering.arch.tiles.layout`), each the rectangle minus the
  openings, its ``cut_around`` fittings and every later zone - measured, never
  typed; every part carries the zone's label as its ``arch_legend`` row;
* the **service axes**: a centre-line cross at each connection point (water,
  waste, gas, power, data, heating), with its short label;
* the **dimensions**, as real DIMENSIONs: on the right a chain of every level
  the elements define (skirting top, worktop top, cabinet bottoms and tops,
  mirror, sills and heads, socket and service centres) and the floor-to-ceiling
  overall; below, a chain of the jambs, the socket and switch centres and the
  service axes, and the overall length. A chain that would only restate the
  overall one is not drawn (ISO 129-1: each measurement once). Zone edges
  join both chains. ``dimensions``: ``none``, ``cm`` (the drawing's
  ``arch-cm`` style) or a style name; default the current style.

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
    "shelf",
    "niche",
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
class Zone:
    """A finish zone: a rectangle of the wall and how it is filled.

    ``fill`` holds ``pattern`` (a predefined name or None), ``pattern_scale``,
    ``angle``, ``joints`` (``[along, across]`` or None) and ``lay`` (the
    ``tiles.layout`` arguments, or None). ``cut_around`` are the fitting ids
    the zone stops at; openings and every later zone always cut it.
    """

    id: str
    x: float
    width: float
    bottom: float
    top: float
    label: str | None
    fill: dict
    cut_around: tuple[str, ...]
    edge: bool = False
    layer: str | None = None
    color: int | None = None


@dataclass(frozen=True)
class Elevation:
    length: float
    height: float
    floor_level: float
    openings: tuple[ElevOpening, ...]
    fittings: tuple[Fitting, ...]
    services: tuple[Service, ...]
    title: str | None = None
    zones: tuple[Zone, ...] = ()
    #: None: the current dimension style; "none": no dimensions; else a style name
    dimensions: str | None = None


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
    "zones",
    "dimensions",
)
#: ``spec.zones[]`` fields: the rectangle, its legend label, what it stops at, and
#: ``arch_floor_zone``'s fill fields (``start`` elevation-local).
ZONE_FIELDS = (
    "id",
    "x",
    "width",
    "bottom",
    "top",
    "label",
    "cut_around",
    "edge",
    "layer",
    "color",
    "pattern",
    "pattern_scale",
    "angle",
    "joints",
    "start",
    "lay",
    "plank_length",
    "plank_width",
    "gap",
    "bond",
    "cut_angle",
    "waste_pct",
    "kerf",
)
_LAY_ARGS = ("plank_length", "plank_width", "gap", "bond", "cut_angle", "waste_pct", "kerf")
#: Fittings a zone stops at unless ``cut_around`` names others (a mirror, a
#: niche, a socket and a switch sit on the finish).
CUT_AROUND_KINDS: tuple[str, ...] = ("worktop", "base_cabinet", "wall_cabinet", "panel", "shelf")


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
        "shelf": {"x", "width", "top", "thickness"},
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
    elif kind in ("worktop", "shelf"):
        x = _number(_required(d, "x", where), f"{where}.x", minimum=0.0)
        width = _number(_required(d, "width", where), f"{where}.width", minimum=0.0, strict=True)
        top = _number(_required(d, "top", where), f"{where}.top", minimum=0.0, strict=True)
        thick = d.get("thickness", WORKTOP_THICKNESS if kind == "worktop" else None)
        if thick is None:
            raise ValueError(f"{where}.thickness: required for a shelf")
        thickness = _number(thick, f"{where}.thickness", minimum=0.0, strict=True)
        bottom = top - thickness
        if bottom < -_EPS:
            raise ValueError(
                f"{where}: a {thickness:g} thick {kind} cannot have its top at {top:g}"
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
    for opening in openings:
        box = _box(opening)
        for fitting in fittings:
            if fitting.kind != "skirting" and _overlap(box, _box(fitting)):
                raise ValueError(f"fittings: {fitting.id} lies inside opening {opening.id}")
        for service in services:
            if box[0] < service.x < box[1] and box[2] < service.y < box[3]:
                raise ValueError(f"services: {service.id} lies inside opening {opening.id}")
    title = _label(spec.get("title"), "spec.title")
    zones_in = _list(spec.get("zones"), "spec.zones")
    zones = tuple(
        _zone(d, f"zones[{i}]", length, height, fittings, i) for i, d in enumerate(zones_in)
    )
    for i, zone in enumerate(zones):
        if zone.id in seen:
            raise ValueError(f"zones[{i}].id: {zone.id!r} is already used by {seen[zone.id]}")
        seen[zone.id] = f"zones[{i}]"
    dims = spec.get("dimensions")
    if dims is not None and (not isinstance(dims, str) or not dims.strip()):
        raise ValueError(f"spec.dimensions: none, cm or a dimension style name, got {dims!r}")
    model = Elevation(
        length, height, floor_level, openings, fittings, services, title, zones,
        dims.strip() if dims else None,
    )  # fmt: skip
    for i in range(len(zones)):
        zone_fill(model, i, scale=1.0)  # refuses an empty zone or a bad lay before drawing
    return model


def _zone(d, where: str, length: float, height: float, fittings, index: int) -> Zone:
    """One ``spec.zones[]`` row; the fill is checked by :func:`zone_fill`."""
    from backends.ezdxf_backend import _hatch_pattern
    from engineering.arch.floor import check_joints
    from engineering.arch.tiles import PATTERNS

    _known(d, where, ZONE_FIELDS)
    x = _number(d.get("x", 0.0), f"{where}.x", minimum=0.0)
    width = _number(d.get("width", length - x), f"{where}.width", minimum=0.0, strict=True)
    bottom = _number(d.get("bottom", 0.0), f"{where}.bottom", minimum=0.0)
    top = _number(d.get("top", height), f"{where}.top", minimum=0.0)
    if top <= bottom + _EPS:
        raise ValueError(f"{where}: top {top:g} must be above bottom {bottom:g}")
    if x + width > length + _EPS or top > height + _EPS:
        raise ValueError(f"{where}: runs past the {length:g} x {height:g} wall")
    ids = {f.id for f in fittings}
    if d.get("cut_around") is None:
        around = tuple(f.id for f in fittings if f.kind in CUT_AROUND_KINDS)
    else:
        around = tuple(_list(d["cut_around"], f"{where}.cut_around"))
        for i, name in enumerate(around):
            if name not in ids:
                raise ValueError(
                    f"{where}.cut_around[{i}]: {name!r} is not a fitting; the fittings are "
                    f"{', '.join(sorted(ids)) or 'none'} (openings are always cut around)"
                )
    pattern = d.get("pattern")
    if isinstance(pattern, str) and pattern.strip().lower() == "none":
        pattern = None
    elif pattern is not None:
        try:
            pattern = _hatch_pattern(pattern)
        except Exception as exc:  # the backend's invalid_args, named by path here
            raise ValueError(f"{where}.pattern: {exc}") from None
    joints = check_joints(d.get("joints"), f"{where}.joints")
    start = d.get("start")
    if start is not None:
        if not isinstance(start, (list, tuple)) or len(start) != 2:
            raise ValueError(f"{where}.start: give [x, y], elevation-local mm")
        start = tuple(_number(v, f"{where}.start") for v in start)
    lay = d.get("lay")
    fill = {
        "pattern": pattern,
        "pattern_scale": _number(d.get("pattern_scale", 1.0), f"{where}.pattern_scale",
                                 minimum=0.0, strict=True),
        "angle": _number(d.get("angle", 0.0), f"{where}.angle"),
        "joints": joints,
        "start": start,
        "lay": None,
    }  # fmt: skip
    if lay is not None:
        lay = _choice(lay, f"{where}.lay", PATTERNS)
        if joints is not None:
            raise ValueError(f"{where}.joints: lay draws the joints of its pieces; give one")
        for key in ("plank_length", "plank_width"):
            _required(d, key, where)
        fill["lay"] = {"pattern": lay, **{k: d[k] for k in _LAY_ARGS if k in d}}
    else:
        extra = sorted(k for k in _LAY_ARGS if k in d)
        if extra:
            raise ValueError(f"{where}.{extra[0]}: only with lay")
    edge = bool(d.get("edge", False))
    if pattern is None and joints is None and lay is None and not edge:
        raise ValueError(f"{where}: give pattern, joints, lay or edge; it draws nothing")
    color = d.get("color")
    if color is not None and (isinstance(color, bool) or not isinstance(color, int)):
        raise ValueError(f"{where}.color: an ACI number, got {color!r}")
    return Zone(
        _label(d.get("id"), f"{where}.id") or f"Z{index + 1}",
        x, width, bottom, top,
        _label(d.get("label"), f"{where}.label"),
        fill, around, edge,
        _label(d.get("layer"), f"{where}.layer"),
        color,
    )  # fmt: skip


def _rect_pts(box) -> tuple[Pt, ...]:
    x0, x1, y0, y1 = box
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def zone_fill(model: Elevation, index: int, *, scale: float) -> dict:
    """What zone ``index`` fills: its rectangle minus the openings, the
    ``cut_around`` fittings and every later zone, measured by the tiles engine.

    ``{zone, loops, area, cutouts, pattern, joints, lay, warnings}``: ``pattern``
    ``{name, hatch_scale, period_mm, repeats}``, ``joints`` ``{spacing_mm,
    period_mm, repeats}``, ``lay`` the ``tiles.layout`` result plus its ``args``.
    """
    from engineering.arch.floor import _LAY_NAMES, _fill_report, fill_period, hatch_scale
    from engineering.arch.tiles import Region, clip, layout

    zone = model.zones[index]
    where = f"zones[{index}]"
    box = (zone.x, zone.x + zone.width, zone.bottom, zone.top)
    fittings = {f.id: f for f in model.fittings}
    covers = [
        *((o.id, _box(o)) for o in model.openings),
        *((name, _box(fittings[name])) for name in zone.cut_around),
        *((z.id, (z.x, z.x + z.width, z.bottom, z.top)) for z in model.zones[index + 1 :]),
    ]
    cutouts, labels = [], []
    for name, b in covers:
        cut = (max(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]), min(b[3], box[3]))
        if cut[1] - cut[0] > _EPS and cut[3] - cut[2] > _EPS:
            cutouts.append(_rect_pts(cut))
            labels.append(name)
    outline = _rect_pts(box)
    pad = 1.0 + max(zone.width, zone.top - zone.bottom)
    cover = _rect_pts((box[0] - pad, box[1] + pad, box[2] - pad, box[3] + pad))
    area, loops = clip(cover, Region.of(outline, cutouts))
    if area <= _EPS or not loops:
        raise ValueError(f"{where}: nothing of it is left once {', '.join(labels)} cut it")
    side = min(zone.width, zone.top - zone.bottom)
    fill = zone.fill
    out = {"zone": zone, "loops": loops, "area": area, "cutouts": labels, "warnings": []}
    if fill["pattern"] not in (None, "SOLID"):
        from ezdxf.tools.pattern import ISO_PATTERN

        name = fill["pattern"]
        pscale = hatch_scale(name, scale, fill["pattern_scale"])
        row, warn = _fill_report(name, fill_period(ISO_PATTERN[name], pscale), side)
        out["pattern"] = {"name": name, "hatch_scale": pscale, **row}
        out["warnings"] += [warn] if warn else []
    elif fill["pattern"] == "SOLID":
        out["pattern"] = {"name": "SOLID", "hatch_scale": 1.0}
    if fill["joints"] is not None:
        row, warn = _fill_report("joints", max(fill["joints"]), side)
        out["joints"] = {"spacing_mm": fill["joints"], **row}
        out["warnings"] += [warn] if warn else []
    if fill["lay"] is not None:
        lay = fill["lay"]
        args = {
            "length": lay["plank_length"], "width": lay["plank_width"], "pattern": lay["pattern"],
            "joint": lay.get("gap", 3.0), "start": fill["start"], "angle_deg": fill["angle"],
            "bond": lay.get("bond", 0.5), "cut_deg": lay.get("cut_angle", 45.0),
            "waste_pct": lay.get("waste_pct", 10.0), "kerf": lay.get("kerf", 3.0),
            "cutouts": [{"points": c, "label": n} for c, n in zip(cutouts, labels, strict=True)],
        }  # fmt: skip
        try:
            laid = layout(outline, **args)
        except ValueError as exc:
            text = str(exc)
            for old, new in _LAY_NAMES:
                text = text.replace(old, new)
            raise ValueError(f"{where}.{text}") from None
        out["lay"] = {**laid, "args": args}
    if not zone.label and (fill["pattern"] or fill["joints"] or fill["lay"]):
        out["warnings"].append(
            {"code": "no_label", "message": f"{where}: no label: arch_legend shows the pattern"}
        )
    return out


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
        if f.bottom > _LEVEL_TOL and f.kind not in ("worktop", "shelf"):
            found.append((f.bottom, f"{f.id}.bottom"))
        found.append((f.top, f"{f.id}.top"))
    for s in model.services:
        found.append((s.y, f"{s.id}.axis"))
    for z in model.zones:
        found += [(z.bottom, f"{z.id}.bottom"), (z.top, f"{z.id}.top")]
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
    for z in model.zones:
        xs += [z.x, z.x + z.width]
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
    if model.dimensions is not None and model.dimensions.lower() == "none":
        return ()
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


def _inside_box(p: Pt, b) -> bool:
    return b[0] + _EPS < p[0] < b[1] - _EPS and b[2] + _EPS < p[1] < b[3] - _EPS


def _visible(p: Pt, q: Pt, boxes) -> list[tuple[Pt, Pt]]:
    """The parts of segment p-q outside the interior of every box (Liang-Barsky)."""
    spans = [(0.0, 1.0)]
    dx, dy = q[0] - p[0], q[1] - p[1]
    for b in boxes:
        t0, t1 = 0.0, 1.0
        for d, lo, hi, o in ((dx, b[0], b[1], p[0]), (dy, b[2], b[3], p[1])):
            if abs(d) < 1e-12:  # along the box: hidden only strictly inside it
                if not lo + _EPS < o < hi - _EPS:
                    t0, t1 = 1.0, 0.0
                continue
            a, c = (lo - o) / d, (hi - o) / d
            t0, t1 = max(t0, min(a, c)), min(t1, max(a, c))
        if t1 - t0 <= 1e-9:
            continue
        spans = [
            piece
            for s0, s1 in spans
            for piece in ((s0, min(s1, t0)), (max(s0, t1), s1))
            if piece[1] - piece[0] > 1e-9
        ]
    if spans == [(0.0, 1.0)]:
        return [(p, q)]
    return [
        ((p[0] + dx * s0, p[1] + dy * s0), (p[0] + dx * s1, p[1] + dy * s1)) for s0, s1 in spans
    ]


def _hidden(prims, boxes) -> list[Prim]:
    """``prims`` with what lies inside a later fitting's box taken out (z-order)."""
    if not boxes:
        return list(prims)
    out: list[Prim] = []
    for prim in prims:
        if isinstance(prim, (Line, Poly)):
            pts = (prim.p1, prim.p2) if isinstance(prim, Line) else prim.points
            n = len(pts) if isinstance(prim, Poly) and prim.closed else len(pts) - 1
            edges = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(n)]
            kept = [(e, _visible(*e, boxes)) for e in edges]
            if isinstance(prim, Poly) and all(v == [e] for e, v in kept):
                out.append(prim)
                continue
            out += [Line(a, b, prim.role) for _e, v in kept for a, b in v]
        elif isinstance(prim, (Circle, Text)):
            at = prim.center if isinstance(prim, Circle) else prim.at
            if not any(_inside_box(at, b) for b in boxes):
                out.append(prim)
        else:
            out.append(prim)
    return out


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
    groups: list[tuple[tuple, list[Prim]]] = []  # per fitting, in z-order
    for f in model.fittings:
        box = _box(f)
        own: list[Prim] = []
        groups.append((box, own))
        if f.kind == "skirting":
            for lo, hi in _skirting_pieces(f, model.openings):
                own.append(_rect(lo, 0.0, hi, f.top, "elev_fitting"))
            continue
        own.append(_rect(box[0], box[2], box[1], box[3], "elev_fitting"))
        if f.kind in ("socket", "switch"):
            size = f.width / f.gangs
            for g in range(f.gangs):
                cx = f.x + size * (g + 0.5)
                if f.kind == "socket":
                    own.append(Circle((cx, f.centre), 0.3 * size, "elev_fitting"))
                else:
                    q = 0.2 * size
                    own.append(
                        _rect(
                            cx - q, f.centre - 1.5 * q, cx + q, f.centre + 1.5 * q, "elev_fitting"
                        )
                    )
        elif f.kind == "mirror":
            w, t = box[1] - box[0], box[3] - box[2]
            d = min(w, t) / 4.0
            for i in (1, 2):
                ox = box[0] + w * (0.15 + 0.1 * i)
                own.append(
                    Line((ox, box[3] - t * 0.15 - d), (ox + d, box[3] - t * 0.15), "elev_fitting")
                )
        if f.label and box[1] - box[0] > (len(f.label) * 0.6 + 1.0) * h and box[3] - box[2] > 2 * h:
            own.append(_label_in(box, f.label, h))
    for i, (_box_i, own) in enumerate(groups):
        prims += _hidden(own, [b for b, _ in groups[i + 1 :]])
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
