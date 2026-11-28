"""The furniture and sanitary catalogue: authored outlines in, real blocks out.

The standard-parts pattern of track B (``engineering/mech/stdparts.py``)
applied to an architectural plan: :func:`catalogue` searches the items by
name, family and their English or Turkish label; :func:`block_spec` turns one
item into the typed primitives and the ATTDEF ``block_define`` takes;
:func:`insert_item` defines the block ``ARCH_<NAME>`` once, inserts it on the
furniture or sanitary layer and reports what it did.

**Sizes are nominal catalogue dimensions, not standards values.** A 900 x 2000
single bed or a 600 x 600 washing machine is what a furniture or appliance
catalogue lists as a common nominal size; no standard fixes it, and every row
says so in ``SIZE_BASIS``. The outlines are plan symbols drawn inside that
footprint, not manufacturer drawings.

Block-local frame, the same for every item: the origin is the **back-left
corner**, the back (the side that stands against a wall) runs along +X from the
origin, and the item extends towards +Y. So ``rotation=0`` puts an item's back
on a wall that runs along +X below it, and every primitive lies inside
``[0, w] x [0, d]`` - a test holds each item to that.

Block content is on layer ``0`` (colour and linetype ByBlock, which is how
``block_define`` writes them), so an INSERT takes the look of the layer it is
inserted on: ``ARCH_ROLE_LAYER["furniture"]`` or ``ARCH_ROLE_LAYER["sanitary"]``.

Every block starts with WIPEOUTs over the item's own outline (``mask``; the
footprint rectangle unless the item says otherwise - the WC bowl, the L of the
corner sofa, the table and chairs of a dining set), so a floor pattern drawn
before the INSERT does not show through the item. A curved edge is masked by
its inscribed polygon.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

from engineering.arch.lang import LANGS
from engineering.arch.layers import ARCH_LAYERS, ARCH_ROLE_LAYER

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

FAMILIES: tuple[str, ...] = ("furniture", "sanitary")

#: What every size in this module is - stated in each row and in the tool docstring.
SIZE_BASIS = "nominal catalogue dimensions (w x d, mm) - not a standards value"

BLOCK_PREFIX = "ARCH_"

#: The one ATTDEF every catalogue block carries: the item name, invisible, so a
#: count or a data extract can read what the INSERT is without parsing its name.
ITEM_TAG = "ITEM"
ITEM_TAG_HEIGHT = 100.0
#: The second ATTDEF: the footprint and height the block was drawn for, ``WxDxH`` mm.
SIZE_TAG = "SIZE"

#: The block-local frame every item shares, reported in each row.
ORIGIN = "back_left"
#: Where an item's front faces at ``rotation=0`` (block-local, unit vector).
FRONT_DIR = (0.0, 1.0)
#: Named points of the footprint as fractions of (w, d); ``anchor=`` on insert.
ANCHORS: dict[str, tuple[float, float]] = {
    "back_left": (0.0, 0.0),
    "back_center": (0.5, 0.0),
    "back_right": (1.0, 0.0),
    "center": (0.5, 0.5),
    "front_center": (0.5, 1.0),
}


# -- block_define spec helpers (block-local mm, layer 0) ---------------------


def _line(x1: float, y1: float, x2: float, y2: float) -> dict:
    return {
        "type": "line",
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "layer": "0",
    }


def _circle(cx: float, cy: float, r: float) -> dict:
    return {"type": "circle", "cx": float(cx), "cy": float(cy), "radius": float(r), "layer": "0"}


def _arc(cx: float, cy: float, r: float, start_deg: float, end_deg: float) -> dict:
    return {
        "type": "arc",
        "cx": float(cx),
        "cy": float(cy),
        "radius": float(r),
        "start_angle": float(start_deg),
        "end_angle": float(end_deg),
        "layer": "0",
    }


def _poly(points, closed: bool) -> dict:
    return {
        "type": "polyline",
        "points": [[float(x), float(y)] for x, y in points],
        "closed": bool(closed),
        "layer": "0",
    }


def _rect(x0: float, y0: float, x1: float, y1: float) -> dict:
    return _poly(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), True)


def _box_pts(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


#: Chords per half circle of a mask outline. A wipeout is a polygon, so a curved
#: edge is masked by its inscribed polygon: at a 400 mm radius the sliver left
#: between chord and arc is under 2 mm.
_MASK_STEPS = 24


def _arc_pts(cx: float, cy: float, r: float, start_deg: float, end_deg: float):
    """Points on a CCW arc from ``start_deg`` to ``end_deg``, both ends included."""
    sweep = (end_deg - start_deg) % 360.0 or 360.0
    n = max(2, math.ceil(_MASK_STEPS * sweep / 180.0))
    return [
        (
            cx + r * math.cos(math.radians(start_deg + sweep * k / n)),
            cy + r * math.sin(math.radians(start_deg + sweep * k / n)),
        )
        for k in range(n + 1)
    ]


def _wipeout(points) -> dict:
    return {"type": "wipeout", "points": [[float(x), float(y)] for x, y in points], "layer": "0"}


# -- the items ---------------------------------------------------------------


class CatalogueItem(NamedTuple):
    """One item: its nominal size and the generator that draws it at any size in range.

    ``draw(w, d)`` returns the block-local primitives for a ``w x d`` footprint;
    at the nominal ``(w, d)`` it draws the item's nominal symbol. ``w_range`` /
    ``d_range`` are the sizes the generator keeps a legible symbol for - a
    repository sanity bound, not a standards value. ``h`` is the nominal
    height: it changes nothing in plan and is written into the SIZE attribute.
    ``mask(w, d)`` returns the polygons the block's wipeouts cover - the
    item's own outline, so a floor pattern stops at it; ``None`` is the whole
    ``w x d`` footprint, :data:`NO_MASK` none at all.

    ``tier`` fixes the item's draw order among catalogue items (``TIERS``).
    ``options`` names the parameters beyond the size, ``{name: (low, high,
    default)}`` (integers; ``default`` may be ``callable(w, d)``); an item with
    options is drawn by ``draw(w, d, opts)`` / ``mask(w, d, opts)``. A
    table's chairs and a wardrobe's door swings lie beyond the ``w x d``
    footprint; ``block_spec`` reports the whole drawing as ``extent``.
    """

    name: str
    family: str
    label_en: str
    label_tr: str
    w: float
    d: float
    h: float
    w_range: tuple[float, float]
    d_range: tuple[float, float]
    draw: Callable[..., tuple[dict, ...]]
    mask: Callable[..., list[list[tuple[float, float]]]] | None = None
    tier: str = "furniture"
    options: dict | None = None

    def defaults(self, w: float, d: float) -> dict:
        """Every option at its default for a ``w x d`` item."""
        return {
            key: int(default(w, d) if callable(default) else default)
            for key, (_lo, _hi, default) in (self.options or {}).items()
        }

    def call(self, fn, w: float, d: float, opts: dict | None = None):
        """``fn(w, d)``, or ``fn(w, d, opts)`` for an item with options."""
        if not self.options:
            return fn(w, d)
        return fn(w, d, self.defaults(w, d) if opts is None else opts)

    @property
    def entities(self) -> tuple[dict, ...]:
        """The primitives at the nominal size."""
        return self.call(self.draw, self.w, self.d)


#: Draw order among catalogue items, lowest first (round 5, A6): floor
#: coverings and pre-walls, then furniture and sanitary, then small items on
#: top; ``wall`` items (a niche) go over everything, so their mask cuts the
#: wall poche they stand in. Every catalogue item but a ``wall`` one stays
#: under the rest of its space (LED strips, symbols, text).
TIERS: tuple[str, ...] = ("floor", "furniture", "decor", "wall")


def NO_MASK(*_args) -> list:  # noqa: N802 - a sentinel used like a constant
    """An item drawn without wipeouts (a concealed frame shown dashed)."""
    return []


#: A chair seat: square, its back rail 60 mm in from its back edge.
CHAIR = 450.0


def _chair(x0: float, y0: float, back_at_top: bool) -> tuple[dict, ...]:
    """A 450 x 450 chair seat with its back rail 60 mm in from the back edge."""
    back_y = y0 + CHAIR - 60.0 if back_at_top else y0 + 60.0
    return (_rect(x0, y0, x0 + CHAIR, y0 + CHAIR), _line(x0, back_y, x0 + CHAIR, back_y))


def _dining(per_side: int):
    """A table between two rows of ``per_side`` chairs, the chairs evenly spaced."""

    def draw(w: float, d: float) -> tuple[dict, ...]:
        if w < per_side * (CHAIR + 50.0):
            raise ValueError(
                f"{per_side} chairs a side need a width of at least "
                f"{per_side * (CHAIR + 50.0):g} mm, got {w:g}"
            )
        chairs: list[dict] = []
        for i in range(per_side):
            x = w * (2 * i + 1) / (2 * per_side) - CHAIR / 2.0
            chairs += _chair(x, 0.0, back_at_top=False)
            chairs += _chair(x, d - CHAIR, back_at_top=True)
        return (_rect(0, CHAIR, w, d - CHAIR), *chairs)

    return draw


def _dining_mask(per_side: int):
    """The table and each chair seat: the floor shows between the chairs."""

    def mask(w: float, d: float) -> list[list[tuple[float, float]]]:
        polys = [_box_pts(0, CHAIR, w, d - CHAIR)]
        for i in range(per_side):
            x = w * (2 * i + 1) / (2 * per_side) - CHAIR / 2.0
            polys += [_box_pts(x, 0, x + CHAIR, CHAIR), _box_pts(x, d - CHAIR, x + CHAIR, d)]
        return polys

    return mask


def _single_bed(w, d):
    return (_rect(0, 0, w, d), _rect(100, 50, w - 100, 450), _line(0, 700, w, 700))


def _double_bed(w, d):
    return (
        _rect(0, 0, w, d),
        _rect(100, 50, w / 2 - 50, 450),
        _rect(w / 2 + 50, 50, w - 100, 450),
        _line(0, 700, w, 700),
    )


def _dash_count(length: float, dash: float, gap: float) -> tuple[int, float]:
    """Dashes and their length so a dashed run starts and ends on a dash."""
    n = max(1, round((length + gap) / (dash + gap)))
    return n, max((length - (n - 1) * gap) / n, length / (2 * n))


def _dashed_line(x1, y1, x2, y2, dash: float = 80.0, gap: float = 50.0) -> tuple[dict, ...]:
    """A dashed line as short LINEs: it reads dashed whatever the linetype or LTSCALE."""
    length = math.hypot(x2 - x1, y2 - y1)
    n, piece = _dash_count(length, dash, gap)
    if n == 1:
        return (_line(x1, y1, x2, y2),)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    out = []
    for k in range(n):
        s = k * (piece + gap)
        out.append(_line(x1 + ux * s, y1 + uy * s, x1 + ux * (s + piece), y1 + uy * (s + piece)))
    return tuple(out)


def _dashed_arc(cx, cy, r, start_deg, end_deg, dash: float = 100.0, gap: float = 60.0):
    """A dashed CCW arc as short ARCs."""
    sweep = (end_deg - start_deg) % 360.0 or 360.0
    n, piece = _dash_count(math.radians(sweep) * r, dash, gap)
    to_deg = 180.0 / (math.pi * r)
    return tuple(
        _arc(
            cx,
            cy,
            r,
            start_deg + k * (piece + gap) * to_deg,
            start_deg + (k * (piece + gap) + piece) * to_deg,
        )
        for k in range(n)
    )


def _wardrobe(w, d, opts):
    """Front panel 20 mm, a hanging rail with hangers, a dashed swing per door.

    Doors are hinged left, right, left, ... and swing 90 degrees into the room,
    so the swings lie beyond the front (``extent``); they are not masked.
    """
    n = opts["doors"]
    leaf = w / n
    if not 300.0 <= leaf <= 1000.0:
        raise ValueError(f"{n} door(s) make {_mm(leaf)} mm leaves; a leaf takes 300-1000 mm")
    front = d - 20.0
    rail = front / 2.0
    hanger = min(440.0, front - 60.0) / 2.0
    count = max(1, int((w - 100.0) // 150.0))
    hangers = tuple(
        _line(x, rail - hanger, x, rail + hanger)
        for x in (100.0 + (w - 200.0) * (k + 0.5) / count for k in range(count))
    )
    swings: list[dict] = []
    for k in range(n):
        if k % 2 == 0:
            swings += _dashed_arc(k * leaf, d, leaf, 0.0, 90.0)
        else:
            swings += _dashed_arc((k + 1) * leaf, d, leaf, 90.0, 180.0)
    return (
        _rect(0, 0, w, d),
        _line(0, front, w, front),
        _line(50, rail, w - 50, rail),
        *hangers,
        *(_line(k * leaf, front, k * leaf, d) for k in range(1, n)),
        *swings,
    )


def _sofa(w, d):
    seat = (w - 400.0) / 3.0
    return (
        _rect(0, 0, w, d),
        _line(200, 0, 200, d),
        _line(w - 200, 0, w - 200, d),
        _line(200, 200, w - 200, 200),
        _line(200 + seat, 200, 200 + seat, d),
        _line(200 + 2 * seat, 200, 200 + 2 * seat, d),
    )


#: The seat depth of the corner sofa's two legs (back included).
CORNER_SEAT = 900.0


def _corner_sofa(w, d):
    """An L: the long leg along the back, the return leg on the right towards +Y."""
    s = CORNER_SEAT
    if d < s + 300.0:
        raise ValueError(f"the return leg needs a depth of at least {s + 300.0:g} mm, got {d:g}")
    seats = max(1, round((w - 200.0 - s) / 700.0))
    step = (w - 200.0 - s) / seats
    return (
        _poly(((0, 0), (w, 0), (w, d), (w - s, d), (w - s, s), (0, s)), True),
        _line(200, 0, 200, s),
        _line(200, 200, w - 200, 200),
        _line(w - 200, 200, w - 200, d),
        *(_line(200 + k * step, 200, 200 + k * step, s) for k in range(1, seats + 1)),
        _line(w - s, s, w - 200, s),
    )


def _corner_sofa_mask(w, d):
    s = CORNER_SEAT
    return [[(0, 0), (w, 0), (w, d), (w - s, d), (w - s, s), (0, s)]]


def _armchair(w, d):
    return (
        _rect(0, 0, w, d),
        _line(150, 0, 150, d),
        _line(w - 150, 0, w - 150, d),
        _line(150, 200, w - 150, 200),
    )


def _chair_item(w, d):
    return (_rect(0, 0, w, d), _line(0, 60, w, 60))


def _desk(w, d):
    return (_rect(0, 0, w, d), _line(w - 400, 0, w - 400, d))


def _counter(w, d):
    lines = tuple(_line(x, 0, x, d) for x in range(600, int(math.ceil(w)), 600) if x < w)
    return (_rect(0, 0, w, d), *lines)


def _fridge(w, d):
    return (_rect(0, 0, w, d), _line(0, d - 50, w, d - 50))


def _bookshelf(w, d):
    return (_rect(0, 0, w, d), _line(0, 0, w, d))


def _box(w, d):
    return (_rect(0, 0, w, d),)


def _hob(w, d):
    """Cooking zones: two side by side below 450 mm, four in a square above."""
    r = min(w / 4.0, d / 4.0) - 20.0
    if w < 450.0:
        zones = ((w / 4.0, d / 2.0), (3 * w / 4.0, d / 2.0))
        r = min(w / 4.0, d / 2.0) - 30.0
    else:
        zones = tuple((w * fx, d * fy) for fx in (0.25, 0.75) for fy in (0.27, 0.73))
    return (_rect(0, 0, w, d), _rect(20, 20, w - 20, d - 20), *(_circle(x, y, r) for x, y in zones))


def _tall_unit(w, d):
    """A tall unit in plan: its outline crossed corner to corner, its front 20 mm thick."""
    return (
        _rect(0, 0, w, d),
        _line(0, d - 20, w, d - 20),
        _line(0, 0, w, d - 20),
        _line(w, 0, 0, d - 20),
    )


def _dishwasher(w, d):
    return (_rect(0, 0, w, d), _line(0, d - 20, w, d - 20), _rect(50, 50, w - 50, d - 70))


def _wc(w, d):
    r = w / 2.0 - 40.0
    cy = d - 20.0 - r
    if cy < 180.0:
        raise ValueError(f"a {w:g} mm wide bowl needs a depth of at least {200.0 + r:g} mm")
    return (
        _rect(0, 0, w, 180),
        _line(40, 180, 40, cy),
        _line(w - 40, 180, w - 40, cy),
        _arc(w / 2, cy, r, 0, 180),
    )


def _wc_mask(w, d):
    r = w / 2.0 - 40.0
    cy = d - 20.0 - r
    return [
        [
            (0, 0),
            (w, 0),
            (w, 180),
            (w - 40, 180),
            *_arc_pts(w / 2, cy, r, 0, 180),
            (40, 180),
            (0, 180),
        ]
    ]


def _bidet(w, d):
    r = w / 2.0
    cy = d - r
    if cy < 120.0:
        raise ValueError(f"a {w:g} mm wide bidet needs a depth of at least {120.0 + r:g} mm")
    return (
        _line(0, 0, w, 0),
        _line(0, 0, 0, cy),
        _line(w, 0, w, cy),
        _arc(w / 2, cy, r, 0, 180),
        _circle(w / 2, 80, 20),
        _circle(w / 2, cy, 20),
    )


def _bidet_mask(w, d):
    r = w / 2.0
    return [[(0, 0), (w, 0), *_arc_pts(w / 2, d - r, r, 0, 180)]]


def _wall_basin(w, d):
    side = d - w / 2.0
    if side < 50.0:
        raise ValueError(f"a {w:g} mm wide basin needs a depth of at least {w / 2.0 + 50.0:g} mm")
    return (
        _poly(((0, side), (0, 0), (w, 0), (w, side)), False),
        _arc(w / 2, side, w / 2, 0, 180),
        _circle(w / 2, side + 50, min(w / 2 - 125, side - 25)),
        _circle(w / 2, side + 50, 20),
        _circle(w / 2, 50, 15),
    )


def _wall_basin_mask(w, d):
    side = d - w / 2.0
    return [[(0, 0), (w, 0), *_arc_pts(w / 2, side, w / 2, 0, 180)]]


def _countertop_basin(w, d):
    """A vanity top with a round vessel bowl on it and the tap behind the bowl."""
    r = min(w / 2.0 - 50.0, (d - 150.0) / 2.0, 225.0)
    cx, cy = w / 2.0, 120.0 + r
    return (
        _rect(0, 0, w, d),
        _circle(cx, cy, r),
        _circle(cx, cy, r - 15.0),
        _circle(cx, cy, 20),
        _circle(cx, 60, 15),
    )


def _shower_tray(w, d):
    return (_rect(0, 0, w, d), _rect(50, 50, w - 50, d - 50), _circle(w / 2, d / 2, 50))


def _walk_in(w, d):
    """Floor-level shower: a linear drain along the back, a fixed glass panel 10 mm
    thick on the front edge from the left, leaving an open entry on the right."""
    glass = max(w - 700.0, w / 2.0)
    return (
        _rect(0, 0, w, d),
        _rect(100, 60, w - 100, 120),
        _rect(0, d - 10, glass, d),
    )


def _bathtub(w, d):
    return (_rect(0, 0, w, d), _rect(60, 60, w - 60, d - 60), _circle(200, d / 2, 25))


def _stadium(x0, y0, x1, y1) -> tuple[dict, ...]:
    """A rounded oblong: two half circles joined by two straight lines."""
    r = (y1 - y0) / 2.0
    cy = (y0 + y1) / 2.0
    return (
        _line(x0 + r, y0, x1 - r, y0),
        _arc(x1 - r, cy, r, 270, 90),
        _line(x1 - r, y1, x0 + r, y1),
        _arc(x0 + r, cy, r, 90, 270),
    )


def _freestanding_bathtub(w, d):
    if w < d + 200.0:
        raise ValueError(f"a freestanding tub must be at least 200 mm longer than deep, got {w:g}")
    return (*_stadium(0, 0, w, d), *_stadium(70, 70, w - 70, d - 70), _circle(w / 2, d / 2, 25))


def _freestanding_bathtub_mask(w, d):
    r = d / 2.0
    return [[*_arc_pts(w - r, r, r, 270, 90), *_arc_pts(r, r, r, 90, 270)]]


def _sink(w, d):
    return (
        _rect(0, 0, w, d),
        _rect(50, 50, w / 2 - 50, d - 50),
        _circle(w / 4, d / 2, 30),
        _line(w / 2 + 50, 0.3 * d, w - 50, 0.3 * d),
        _line(w / 2 + 50, 0.5 * d, w - 50, 0.5 * d),
        _line(w / 2 + 50, 0.7 * d, w - 50, 0.7 * d),
    )


def _washing_machine(w, d):
    return (_rect(0, 0, w, d), _circle(w / 2, d / 2, min(w, d) / 3))


def _mirror(specs, w):
    """The specs mirrored about x = w / 2 (a left-hand twin of a right-hand item)."""
    out = []
    for spec in specs:
        spec = dict(spec)
        if spec["type"] == "line":
            spec["x1"], spec["x2"] = w - spec["x1"], w - spec["x2"]
        elif spec["type"] in ("polyline", "wipeout"):
            spec["points"] = [[w - x, y] for x, y in spec["points"]]
        elif spec["type"] in ("circle", "arc"):
            spec["cx"] = w - spec["cx"]
            if spec["type"] == "arc":
                spec["start_angle"], spec["end_angle"] = (
                    (180.0 - spec["end_angle"]) % 360.0,
                    (180.0 - spec["start_angle"]) % 360.0,
                )
        out.append(spec)
    return tuple(out)


def _corner_sofa_left(w, d):
    return _mirror(_corner_sofa(w, d), w)


def _corner_sofa_left_mask(w, d):
    return [[(w - x, y) for x, y in poly] for poly in _corner_sofa_mask(w, d)]


def _modular_sofa(w, d):
    """Armless seat modules of about 900 mm on one back cushion."""
    modules = max(2, round(w / 900.0))
    step = w / modules
    return (
        _rect(0, 0, w, d),
        _line(0, 200, w, 200),
        *(_line(k * step, 200, k * step, d) for k in range(1, modules)),
    )


def _oval_pts(w, d):
    """A circle (w = d) or a stadium along the longer side, as mask points."""
    if abs(w - d) < 1e-9:
        return _arc_pts(w / 2, d / 2, w / 2, 0, 360)[:-1]
    if w > d:
        r = d / 2.0
        return [*_arc_pts(w - r, r, r, 270, 90), *_arc_pts(r, r, r, 90, 270)]
    r = w / 2.0
    return [*_arc_pts(r, r, r, 180, 360), *_arc_pts(r, d - r, r, 0, 180)]


def _oval(x0, y0, x1, y1) -> tuple[dict, ...]:
    """A circle, or a stadium along the longer side of the box."""
    w, d = x1 - x0, y1 - y0
    if abs(w - d) < 1e-9:
        return (_circle(x0 + w / 2, y0 + d / 2, w / 2),)
    if w > d:
        return _stadium(x0, y0, x1, y1)
    r = w / 2.0
    cx = x0 + r
    return (
        _line(x1, y0 + r, x1, y1 - r),
        _arc(cx, y1 - r, r, 0, 180),
        _line(x0, y1 - r, x0, y0 + r),
        _arc(cx, y0 + r, r, 180, 360),
    )


#: How far a chair seat around a round table stands under the table's edge.
CHAIR_TUCK = 100.0


def _ring(w: float, d: float, off: float, n: int) -> list[tuple[float, float, float, float]]:
    """``n`` points evenly by arc length on the oval ``_oval(0, 0, w, d)`` offset
    outwards by ``off``, from the top centre: ``(x, y, ux, uy)`` with the outward
    normal. A taller-than-wide oval is the wide one with x and y swapped."""
    if d > w:
        return [(y, x, uy, ux) for x, y, ux, uy in _ring(d, w, off, n)]
    r = d / 2.0
    big = r + off
    flat = w - d
    segs = (flat / 2.0, math.pi * big, flat, math.pi * big, flat / 2.0)
    total = sum(segs)
    out = []
    for k in range(n):
        s = total * k / n
        if s < segs[0]:
            out.append((w / 2.0 - s, r + big, 0.0, 1.0))
            continue
        s -= segs[0]
        if s < segs[1]:
            a = math.pi / 2.0 + s / big
            out.append((r + big * math.cos(a), r + big * math.sin(a), math.cos(a), math.sin(a)))
            continue
        s -= segs[1]
        if s < segs[2]:
            out.append((r + s, r - big, 0.0, -1.0))
            continue
        s -= segs[2]
        if s < segs[3]:
            a = -math.pi / 2.0 + s / big
            out.append((w - r + big * math.cos(a), r + big * math.sin(a), math.cos(a), math.sin(a)))
            continue
        out.append((w - r - (s - segs[3]), r + big, 0.0, 1.0))
    return out


def _seat_pts(x, y, ux, uy) -> list[tuple[float, float]]:
    """A chair seat's corners, its centre at (x, y), its back towards (ux, uy)."""
    h = CHAIR / 2.0
    tx, ty = -uy, ux
    return [
        (x - tx * h - ux * h, y - ty * h - uy * h),
        (x + tx * h - ux * h, y + ty * h - uy * h),
        (x + tx * h + ux * h, y + ty * h + uy * h),
        (x - tx * h + ux * h, y - ty * h + uy * h),
    ]


def _round_seats(w, d, n):
    if not n:
        return []
    off = CHAIR / 2.0 - CHAIR_TUCK
    ring = _ring(w, d, off, n)
    r = min(w, d) / 2.0
    perimeter = math.pi * 2 * (r + off) + 2 * abs(w - d)
    need = n * (CHAIR + 50.0)
    if perimeter < need - 1e-6:
        least = 2 * (need / (2 * math.pi) - off)
        raise ValueError(
            f"{n} chairs need {_mm(need)} mm round the table at the seats, it gives "
            f"{perimeter:.0f}; a round table seating {n} is at least {least:.0f} mm across"
        )
    return ring


def _round_table(w, d, opts):
    """The table, and ``chairs`` seats evenly round it, tucked 100 mm under its edge."""
    chairs: list[dict] = []
    h = CHAIR / 2.0 - 60.0
    for x, y, ux, uy in _round_seats(w, d, opts["chairs"]):
        chairs.append(_poly(_seat_pts(x, y, ux, uy), True))
        tx, ty = -uy, ux
        bx, by = x + ux * h, y + uy * h
        chairs.append(
            _line(
                bx - tx * CHAIR / 2, by - ty * CHAIR / 2, bx + tx * CHAIR / 2, by + ty * CHAIR / 2
            )
        )
    return (*_oval(0, 0, w, d), *chairs)


def _round_table_mask(w, d, opts):
    return [_oval_pts(w, d), *(_seat_pts(*p) for p in _round_seats(w, d, opts["chairs"]))]


def _round_rug(w, d):
    """A round or oval rug with a border 80 mm in."""
    inner = _oval(80, 80, w - 80, d - 80) if min(w, d) > 400 else ()
    return (*_oval(0, 0, w, d), *inner)


def _radiator(w, d):
    """A panel radiator in plan: the panel off the wall on two brackets, its fins."""
    gap = min(50.0, d / 3.0)
    fins = tuple(_line(x, gap, x, d) for x in range(100, int(math.ceil(w)), 100) if x < w)
    br = min(150.0, w / 4.0)
    return (_rect(0, gap, w, d), *fins, _line(br, 0, br, gap), _line(w - br, 0, w - br, gap))


def _radiator_mask(w, d):
    return [_box_pts(0, min(50.0, d / 3.0), w, d)]


def _wc_frame(w, d):
    """A concealed WC frame, dashed (inside the pre-wall), with the Ø110 soil outlet."""
    return (
        *_dashed_line(0, 0, w, 0),
        *_dashed_line(w, 0, w, d),
        *_dashed_line(w, d, 0, d),
        *_dashed_line(0, d, 0, 0),
        *_dashed_line(0, 0, w, d),
        *_dashed_line(w, 0, 0, d),
        _circle(w / 2.0, d / 2.0, min(55.0, d / 2.0 - 5.0)),
    )


def _prewall(w, d):
    """A plasterboard pre-wall: the board 25 mm on the room side, insulation between."""
    out = [_rect(0, 0, w, d)]
    if d >= 60.0:
        out.append(_line(0, d - 25.0, w, d - 25.0))
    if d >= 100.0:
        low, high = 15.0, d - 40.0
        step = high - low
        n = max(1, round(w / step))
        out.append(_poly([(w * k / n, high if k % 2 else low) for k in range(n + 1)], False))
    return tuple(out)


#: How far a niche's mask reaches into the room past the wall face, so the
#: face line across the recess is hidden too.
NICHE_LIP = 10.0


def _niche(w, d):
    """A wall niche: the recess's back and sides; its front is the wall face."""
    return (_poly(((0, d), (0, 0), (w, 0), (w, d)), False),)


def _niche_mask(w, d):
    return [_box_pts(0, 0, w, d + NICHE_LIP)]


def _stool(w, d):
    return (*_oval(0, 0, w, d), *_oval(60, 60, w - 60, d - 60))


def _plant(w, d):
    """Foliage (the outline), a pot in the middle, six leaves between them."""
    r = 0.2 * min(w, d)
    cx, cy = w / 2.0, d / 2.0
    leaves = []
    for k in range(6):
        a = math.radians(30 + 60 * k)
        c, s_ = math.cos(a), math.sin(a)
        leaves.append(_line(cx + r * c, cy + r * s_, cx + 0.45 * w * c, cy + 0.45 * d * s_))
    return (*_oval(0, 0, w, d), _circle(cx, cy, r), *leaves)


def _curtain(w, d):
    """A rod along the back and the folded curtain as a zigzag in front of it."""
    folds = max(2, round(w / 150.0))
    step = w / folds
    pts = [(k * step, d if k % 2 else 40.0) for k in range(folds + 1)]
    return (_line(0, 20, w, 20), _poly(pts, False))


def _rug(w, d):
    """The rug and a fringe along both short ends."""
    fringe = min(60.0, w / 10.0)
    count = max(2, int(d // 60.0))
    ys = [d * (k + 0.5) / count for k in range(count)]
    return (
        _rect(fringe, 0, w - fringe, d),
        *(_line(0, y, fringe, y) for y in ys),
        *(_line(w - fringe, y, w, y) for y in ys),
    )


def _built_in(w, d):
    """Built-in joinery as plans letter it: the outline crossed corner to corner."""
    return (_rect(0, 0, w, d), _line(0, 0, w, d), _line(w, 0, 0, d))


def _sink_bowl(w, d):
    return (_rect(0, 0, w, d), _rect(40, 40, w - 40, d - 40), _circle(w / 2, d / 2, 30))


def _item(
    name,
    family,
    en,
    tr,
    w,
    d,
    h,
    w_range,
    d_range,
    draw,
    mask=None,
    *,
    tier="furniture",
    options=None,
) -> CatalogueItem:
    return CatalogueItem(
        name,
        family,
        en,
        tr,
        float(w),
        float(d),
        float(h),
        w_range,
        d_range,
        draw,
        mask,
        tier,
        options,
    )


ITEMS: tuple[CatalogueItem, ...] = (
    # -- furniture -----------------------------------------------------------
    _item(
        "single_bed",
        "furniture",
        "single bed",
        "tek kişilik yatak",
        900,
        2000,
        450,
        (800, 1200),
        (1900, 2200),
        _single_bed,
    ),
    _item(
        "double_bed",
        "furniture",
        "double bed",
        "çift kişilik yatak",
        1600,
        2000,
        450,
        (1200, 2200),
        (1900, 2200),
        _double_bed,
    ),
    _item(
        "wardrobe",
        "furniture",
        "wardrobe",
        "gardırop",
        1200,
        600,
        2200,
        (400, 4000),
        (350, 700),
        _wardrobe,
        options={"doors": (1, 12, lambda w, d: max(1, round(w / 600.0)))},
    ),
    _item(
        "sofa_3_seat",
        "furniture",
        "sofa, three seats",
        "üçlü kanepe",
        2200,
        900,
        850,
        (1800, 2800),
        (800, 1100),
        _sofa,
    ),
    _item(
        "armchair",
        "furniture",
        "armchair",
        "berjer koltuk",
        900,
        900,
        850,
        (700, 1100),
        (700, 1000),
        _armchair,
    ),
    _item(
        "chair",
        "furniture",
        "chair",
        "sandalye",
        450,
        450,
        900,
        (380, 600),
        (380, 600),
        _chair_item,
    ),
    _item(
        "dining_table_4",
        "furniture",
        "dining table with four chairs",
        "dört kişilik yemek masası",
        1400,
        1700,
        750,
        (1200, 1800),
        (1600, 1900),
        _dining(2),
        _dining_mask(2),
    ),
    _item(
        "desk",
        "furniture",
        "desk",
        "çalışma masası",
        1400,
        700,
        750,
        (800, 2000),
        (500, 900),
        _desk,
    ),
    _item(
        "kitchen_counter",
        "furniture",
        "kitchen counter run",
        "mutfak tezgâhı",
        2400,
        600,
        900,
        (600, 6000),
        (500, 700),
        _counter,
    ),
    _item(
        "fridge",
        "furniture",
        "fridge",
        "buzdolabı",
        600,
        650,
        1850,
        (500, 900),
        (550, 750),
        _fridge,
    ),
    _item(
        "bookshelf",
        "furniture",
        "bookshelf",
        "kitaplık",
        800,
        300,
        1800,
        (400, 3000),
        (200, 450),
        _bookshelf,
    ),
    _item(
        "nightstand",
        "furniture",
        "nightstand",
        "komodin",
        500,
        400,
        550,
        (300, 900),
        (250, 800),
        _box,
    ),
    _item(
        "coffee_table",
        "furniture",
        "coffee table",
        "orta sehpa",
        1100,
        600,
        450,
        (500, 1600),
        (400, 1000),
        _box,
    ),
    _item(
        "corner_sofa",
        "furniture",
        "corner sofa, L-shaped",
        "köşe koltuk",
        2600,
        1600,
        850,
        (2000, 3600),
        (1400, 2400),
        _corner_sofa,
        _corner_sofa_mask,
    ),
    _item(
        "dining_table_6",
        "furniture",
        "dining table with six chairs",
        "altı kişilik yemek masası",
        1800,
        1800,
        750,
        (1600, 2400),
        (1700, 2000),
        _dining(3),
        _dining_mask(3),
    ),
    _item(
        "dining_table_8",
        "furniture",
        "dining table with eight chairs",
        "sekiz kişilik yemek masası",
        2400,
        1900,
        750,
        (2200, 3000),
        (1800, 2100),
        _dining(4),
        _dining_mask(4),
    ),
    _item(
        "induction_hob",
        "furniture",
        "induction hob",
        "indüksiyonlu ocak",
        600,
        520,
        50,
        (300, 900),
        (450, 550),
        _hob,
    ),
    _item(
        "oven_tall_unit",
        "furniture",
        "tall unit with built-in oven",
        "ankastre fırın dolabı",
        600,
        600,
        2100,
        (450, 900),
        (550, 650),
        _tall_unit,
    ),
    _item(
        "dishwasher",
        "furniture",
        "dishwasher",
        "bulaşık makinesi",
        600,
        600,
        820,
        (450, 600),
        (550, 650),
        _dishwasher,
    ),
    # -- sanitary ------------------------------------------------------------
    _item(
        "wc",
        "sanitary",
        "wc with cistern",
        "klozet",
        380,
        700,
        800,
        (340, 420),
        (480, 750),
        _wc,
        _wc_mask,
    ),
    _item(
        "bidet",
        "sanitary",
        "bidet",
        "bide",
        360,
        560,
        400,
        (340, 400),
        (480, 600),
        _bidet,
        _bidet_mask,
    ),
    _item(
        "wall_basin",
        "sanitary",
        "wall basin",
        "lavabo",
        550,
        450,
        850,
        (400, 700),
        (350, 600),
        _wall_basin,
        _wall_basin_mask,
    ),
    _item(
        "shower_tray",
        "sanitary",
        "shower tray",
        "duş teknesi",
        900,
        900,
        50,
        (700, 1800),
        (700, 1200),
        _shower_tray,
    ),
    _item(
        "bathtub",
        "sanitary",
        "bathtub",
        "küvet",
        1700,
        700,
        550,
        (1400, 2000),
        (650, 900),
        _bathtub,
    ),
    _item(
        "kitchen_sink",
        "sanitary",
        "kitchen sink with drainer",
        "eviye",
        1000,
        500,
        900,
        (600, 1600),
        (400, 650),
        _sink,
    ),
    _item(
        "washing_machine",
        "sanitary",
        "washing machine",
        "çamaşır makinesi",
        600,
        600,
        850,
        (400, 700),
        (400, 700),
        _washing_machine,
    ),
    _item(
        "walk_in_shower",
        "sanitary",
        "walk-in shower with a glass panel",
        "walk-in duş",
        1200,
        900,
        2000,
        (700, 2400),
        (700, 1400),
        _walk_in,
    ),
    _item(
        "freestanding_bathtub",
        "sanitary",
        "freestanding bathtub",
        "ayaklı küvet",
        1700,
        800,
        600,
        (1400, 1900),
        (700, 900),
        _freestanding_bathtub,
        _freestanding_bathtub_mask,
    ),
    _item(
        "sink_bowl",
        "sanitary",
        "kitchen sink, single bowl",
        "tek gözlü eviye",
        450,
        450,
        900,
        (300, 1000),
        (350, 650),
        _sink_bowl,
    ),
    # -- furniture added for interiors (2026-11) -----------------------------
    _item(
        "modular_sofa",
        "furniture",
        "modular sofa",
        "modüler kanepe",
        2700,
        1000,
        800,
        (1200, 6000),
        (700, 1300),
        _modular_sofa,
    ),
    _item(
        "corner_sofa_left",
        "furniture",
        "corner sofa, L-shaped, return on the left",
        "köşe koltuk, sol",
        2600,
        1600,
        850,
        (2000, 3600),
        (1400, 2400),
        _corner_sofa_left,
        _corner_sofa_left_mask,
    ),
    _item(
        "round_table",
        "furniture",
        "round or oval table, chairs round it",
        "yuvarlak masa",
        1000,
        1000,
        750,
        (500, 2400),
        (500, 2400),
        _round_table,
        _round_table_mask,
        options={"chairs": (0, 12, 0)},
    ),
    _item(
        "bar_stool",
        "furniture",
        "stool or bar stool",
        "tabure",
        380,
        380,
        750,
        (280, 600),
        (280, 600),
        _stool,
        lambda w, d: [_oval_pts(w, d)],
    ),
    _item(
        "plant",
        "furniture",
        "plant in a pot",
        "saksı bitkisi",
        600,
        600,
        1200,
        (200, 2000),
        (200, 2000),
        _plant,
        lambda w, d: [_oval_pts(w, d)],
        tier="decor",
    ),
    _item(
        "curtain",
        "furniture",
        "curtain on a rod",
        "perde",
        2000,
        150,
        2600,
        (300, 10000),
        (80, 400),
        _curtain,
        tier="decor",
    ),
    _item(
        "rug",
        "furniture",
        "rug",
        "halı",
        2000,
        1400,
        10,
        (600, 6000),
        (400, 5000),
        _rug,
        tier="floor",
    ),
    _item(
        "built_in_wardrobe",
        "furniture",
        "built-in wardrobe or joinery (crossed)",
        "gömme dolap",
        1800,
        600,
        2600,
        (300, 8000),
        (250, 1000),
        _built_in,
    ),
    _item(
        "countertop_basin",
        "sanitary",
        "countertop basin on a vanity top",
        "tezgah üstü lavabo",
        600,
        500,
        850,
        (400, 2400),
        (300, 700),
        _countertop_basin,
    ),
    # -- round 5 (B3) --------------------------------------------------------
    _item(
        "round_rug",
        "furniture",
        "round or oval rug",
        "yuvarlak halı",
        2000,
        2000,
        10,
        (600, 6000),
        (600, 6000),
        _round_rug,
        lambda w, d: [_oval_pts(w, d)],
        tier="floor",
    ),
    _item(
        "radiator",
        "sanitary",
        "panel radiator",
        "panel radyatör",
        1000,
        150,
        600,
        (300, 3000),
        (60, 300),
        _radiator,
        _radiator_mask,
    ),
    _item(
        "concealed_wc_frame",
        "sanitary",
        "concealed WC frame (behind the WC, in the pre-wall)",
        "gömme rezervuar",
        500,
        150,
        1120,
        (400, 600),
        (80, 300),
        _wc_frame,
        NO_MASK,
    ),
    _item(
        "prewall",
        "sanitary",
        "plasterboard pre-wall",
        "alçıpan ön duvar",
        1200,
        200,
        1200,
        (300, 6000),
        (80, 600),
        _prewall,
        tier="floor",
    ),
    _item(
        "wall_niche",
        "furniture",
        "wall niche (recess at a wall face)",
        "duvar nişi",
        600,
        150,
        400,
        (200, 3000),
        (50, 600),
        _niche,
        _niche_mask,
        tier="wall",
    ),
)

#: The height a SIZE attribute may carry: it does not change the plan symbol.
HEIGHT_RANGE: tuple[float, float] = (10.0, 4000.0)

CATALOGUE_NAMES: tuple[str, ...] = tuple(item.name for item in ITEMS)
#: Fitted items - joinery, appliances and sanitary ware fixed to the building,
#: which a floor zone (underfloor heating, a floor finish) stops at by default;
#: the rest is movable furniture a floor runs under. A test holds it to ITEMS.
FITTED: frozenset[str] = frozenset(
    {
        "kitchen_counter",
        "fridge",
        "induction_hob",
        "oven_tall_unit",
        "dishwasher",
        "wc",
        "bidet",
        "wall_basin",
        "shower_tray",
        "bathtub",
        "kitchen_sink",
        "washing_machine",
        "walk_in_shower",
        "freestanding_bathtub",
        "sink_bowl",
        "built_in_wardrobe",
        "countertop_basin",
        "radiator",
        "concealed_wc_frame",
        "prewall",
        "wall_niche",
    }
)
_BY_NAME: dict[str, CatalogueItem] = {item.name: item for item in ITEMS}

#: Turkish letters folded to ASCII so "buzdolabi" finds "buzdolabı" and a
#: query typed on an English keyboard still reaches the Turkish label.
#: Polish labels by item name; a test holds the keys equal to ``ITEMS``.
LABEL_PL: dict[str, str] = {
    "single_bed": "łóżko pojedyncze",
    "double_bed": "łóżko podwójne",
    "wardrobe": "szafa",
    "sofa_3_seat": "sofa trzyosobowa",
    "armchair": "fotel",
    "chair": "krzesło",
    "dining_table_4": "stół jadalny z czterema krzesłami",
    "desk": "biurko",
    "kitchen_counter": "blat kuchenny",
    "fridge": "lodówka",
    "bookshelf": "regał na książki",
    "nightstand": "szafka nocna",
    "coffee_table": "stolik kawowy",
    "wc": "miska ustępowa ze spłuczką",
    "bidet": "bidet",
    "wall_basin": "umywalka",
    "shower_tray": "brodzik",
    "bathtub": "wanna",
    "kitchen_sink": "zlewozmywak z ociekaczem",
    "washing_machine": "pralka",
    "corner_sofa": "sofa narożna L",
    "dining_table_6": "stół jadalny z sześcioma krzesłami",
    "dining_table_8": "stół jadalny z ośmioma krzesłami",
    "induction_hob": "płyta indukcyjna",
    "oven_tall_unit": "słupek z piekarnikiem do zabudowy",
    "dishwasher": "zmywarka",
    "walk_in_shower": "kabina walk-in ze ścianką szklaną",
    "freestanding_bathtub": "wanna wolnostojąca",
    "countertop_basin": "umywalka nablatowa z blatem",
    "sink_bowl": "zlewozmywak jednokomorowy",
    "modular_sofa": "sofa modułowa",
    "corner_sofa_left": "sofa narożna L, lewa",
    "round_table": "stół okrągły lub owalny z krzesłami",
    "bar_stool": "stołek / hoker",
    "plant": "roślina w donicy",
    "curtain": "zasłona na karniszu",
    "rug": "dywan",
    "built_in_wardrobe": "zabudowa meblowa / szafa wnękowa (z krzyżem)",
    "round_rug": "dywan okrągły lub owalny",
    "radiator": "grzejnik płytowy",
    "concealed_wc_frame": "stelaż podtynkowy WC",
    "prewall": "przedścianka GK",
    "wall_niche": "wnęka w ścianie",
}

_FOLD = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
    }
)


def _fold(text: str) -> str:
    """Lower-case and fold Turkish and Polish letters. ``İ`` is mapped before ``lower()``,
    which would otherwise turn it into ``i`` plus a combining dot."""
    return " ".join(str(text).replace("İ", "i").lower().translate(_FOLD).split())


def _check_lang(lang: str) -> str:
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {', '.join(LANGS)}, got {lang!r}.")
    return lang


def _row(item: CatalogueItem, lang: str) -> dict:
    return {
        "name": item.name,
        "family": item.family,
        "label": {"tr": item.label_tr, "pl": LABEL_PL[item.name]}.get(lang, item.label_en),
        "label_en": item.label_en,
        "label_tr": item.label_tr,
        "label_pl": LABEL_PL[item.name],
        "size": [item.w, item.d],
        "width": item.w,
        "depth": item.d,
        "origin": ORIGIN,
        "front_dir": list(FRONT_DIR),
        "height": item.h,
        "size_range": {"width": list(item.w_range), "depth": list(item.d_range)},
        "size_basis": SIZE_BASIS,
        "block": f"{BLOCK_PREFIX}{item.name.upper()}",
        "layer": ARCH_ROLE_LAYER[item.family],
        "tier": item.tier,
        "fitted": item.name in FITTED,
        **(
            {
                "options": {
                    k: {"range": [lo, hi], "default": item.defaults(item.w, item.d)[k]}
                    for k, (lo, hi, _default) in item.options.items()
                }
            }
            if item.options
            else {}
        ),
    }


def catalogue(
    query: str | None = None, family: str | None = None, *, lang: str = "en"
) -> tuple[dict, ...]:
    """Every catalogue item, optionally filtered.

    ``family`` is ``furniture`` or ``sanitary`` (anything else is refused with
    the list). ``query`` is split into words and every word must occur in the
    item's name, family, English, Turkish or Polish label; matching ignores case
    and Turkish and Polish diacritics, so ``'bed'``, ``'yatak'``, ``'KLOZET'``,
    ``'buzdolabi'`` and ``'lozko'`` all work. ``lang`` picks the ``label`` column.
    """
    _check_lang(lang)
    if family is not None and family not in FAMILIES:
        raise ValueError(f"family must be one of {', '.join(FAMILIES)}, got {family!r}.")
    words = _fold(query).split() if query else []
    rows: list[dict] = []
    for item in ITEMS:
        if family is not None and item.family != family:
            continue
        haystack = _fold(
            f"{item.name.replace('_', ' ')} {item.family} {item.label_en} {item.label_tr} "
            f"{LABEL_PL[item.name]}"
        )
        if all(word in haystack for word in words):
            rows.append(_row(item, lang))
    return tuple(rows)


def _resolve(name: str) -> CatalogueItem:
    key = "_".join(str(name).replace("-", " ").lower().split())
    item = _BY_NAME.get(key)
    if item is not None:
        return item
    close = difflib.get_close_matches(key, CATALOGUE_NAMES, n=3, cutoff=0.4)
    hint = (
        f"nearest: {', '.join(close)}"
        if close
        else f"the catalogue holds {', '.join(CATALOGUE_NAMES)}"
    )
    raise ValueError(f"arch catalogue: no item {name!r}; {hint}.")


def _size(value, default: float, bounds: tuple[float, float], what: str, item: str) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"arch catalogue: {item} {what} must be a number in mm, got {value!r}.")
    number = float(value)
    low, high = bounds
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(
            f"arch catalogue: {item} {what} {value!r} mm is outside {low:g}-{high:g} mm."
        )
    return number


def _holds(item: CatalogueItem, width, depth) -> str:
    """`` Items of the family that take this size: ...``: those that share a word of
    the item's English label (empty when none does)."""
    w = item.w if width is None else width
    d = item.d if depth is None else depth
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (w, d)):
        return ""
    words = {w for w in _fold(item.label_en).replace(",", " ").split() if len(w) > 3} - {"with"}
    fits = [
        other.name
        for other in ITEMS
        if other.family == item.family
        and other.name != item.name
        and words & set(_fold(other.label_en).replace(",", " ").split())
        and other.w_range[0] <= w <= other.w_range[1]
        and other.d_range[0] <= d <= other.d_range[1]
    ]
    return f" Items of the family that take {_mm(w)} x {_mm(d)}: {', '.join(fits)}." if fits else ""


def _mm(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _options(item: CatalogueItem, given, w: float, d: float) -> dict:
    """The item's options with ``given`` validated by name (integers in range)."""
    opts = item.defaults(w, d)
    if given is None:
        return opts
    if not isinstance(given, dict):
        raise ValueError(f"options must be an object, got {given!r}")
    allowed = item.options or {}
    for key, value in given.items():
        if key not in allowed:
            takes = f"takes {', '.join(allowed)}" if allowed else "takes no options"
            raise ValueError(f"options.{key}: unknown; {item.name} {takes}")
        low, high, _default = allowed[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
            raise ValueError(f"options.{key} must be a whole number, got {value!r}")
        if not low <= value <= high:
            raise ValueError(f"options.{key} {value!r} is outside {low}-{high}")
        opts[key] = int(value)
    return opts


def _bounds(entities) -> tuple[float, float, float, float]:
    """The block-local box of primitive specs, an arc by its real sweep."""
    pts: list[tuple[float, float]] = []
    for e in entities:
        kind = e["type"]
        if kind == "line":
            pts += [(e["x1"], e["y1"]), (e["x2"], e["y2"])]
        elif kind == "circle":
            cx, cy, r = e["cx"], e["cy"], e["radius"]
            pts += [(cx - r, cy - r), (cx + r, cy + r)]
        elif kind == "arc":
            cx, cy, r = e["cx"], e["cy"], e["radius"]
            a0, a1 = e["start_angle"], e["end_angle"]
            sweep = (a1 - a0) % 360.0 or 360.0
            angles = [a0, a0 + sweep] + [q for q in range(0, 720, 90) if a0 < q < a0 + sweep]
            pts += [
                (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
                for a in angles
            ]
        else:
            pts += [tuple(p) for p in e["points"]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def block_spec(
    name: str,
    width: float | None = None,
    depth: float | None = None,
    height: float | None = None,
    options: dict | None = None,
) -> dict:
    """The typed primitives and the ITEM / SIZE ATTDEFs ``block_define`` takes for one item.

    ``width`` / ``depth`` / ``height`` (mm) override the nominal size; the
    symbol is redrawn for the footprint, not scaled, so rails, rims and chairs
    keep their real sizes. ``options`` sets an item's own parameters by name
    (``CatalogueItem.options``: a round table's ``chairs``, a wardrobe's
    ``doors``). The nominal size with the default options keeps the block
    ``ARCH_<NAME>``; any other is ``ARCH_<NAME>_<W>x<D>``, then ``_<KEY><n>``
    per option off its default (``ARCH_ROUND_TABLE_1800x1800_C8``). The height
    changes nothing in plan and is written into the SIZE attribute (``WxDxH``).

    ``entities`` begins with the item's wipeouts (see the module docstring).
    ``extent`` is the block-local box of everything drawn - wider than the
    ``w x d`` footprint where chairs or door swings reach past it.

    Refusals (``ValueError``, before anything is built): an unknown name (with
    the nearest names), a size outside the item's range, an unknown or
    out-of-range option, a size the symbol cannot hold.
    """
    item = _resolve(name)
    try:
        w = _size(width, item.w, item.w_range, "width", item.name)
        d = _size(depth, item.d, item.d_range, "depth", item.name)
    except ValueError as exc:
        raise ValueError(f"{exc}{_holds(item, width, depth)}") from None
    h = _size(height, item.h, HEIGHT_RANGE, "height", item.name)
    try:
        opts = _options(item, options, w, d)
    except ValueError as exc:
        raise ValueError(f"arch catalogue: {item.name} {exc}.") from None
    try:
        entities = item.call(item.draw, w, d, opts)
    except ValueError as exc:
        raise ValueError(f"arch catalogue: {item.name} {_mm(w)} x {_mm(d)}: {exc}.") from None
    # The wipeouts come first: inside a block a WIPEOUT hides what is drawn
    # before it, so the item's own lines, listed after, stay on top of it.
    masks = item.call(item.mask, w, d, opts) if item.mask else [_box_pts(0.0, 0.0, w, d)]
    entities = (*(_wipeout(poly) for poly in masks), *entities)
    defaults = item.defaults(w, d)
    changed = {k: v for k, v in opts.items() if v != defaults[k]}
    nominal = (w, d) == (item.w, item.d) and not changed
    block = f"{BLOCK_PREFIX}{item.name.upper()}"
    if not nominal:
        block = f"{block}_{_mm(w)}x{_mm(d)}".replace(".", "_")
        block += "".join(f"_{k[0].upper()}{v}" for k, v in sorted(changed.items()))
    common = {"x": w / 2.0, "height": ITEM_TAG_HEIGHT, "align": "middle_center", "invisible": True}
    return {
        "name": block,
        "entities": [dict(spec) for spec in entities],
        "attdefs": [
            {
                "tag": ITEM_TAG,
                "prompt": "Catalogue item",
                "default": item.name,
                "y": d / 2.0,
                **common,
            },
            {
                "tag": SIZE_TAG,
                "prompt": "Size W x D x H (mm)",
                "default": f"{_mm(w)}x{_mm(d)}x{_mm(h)}",
                "y": max(0.0, d / 2.0 - 1.5 * ITEM_TAG_HEIGHT),
                **common,
            },
        ],
        "base_x": 0.0,
        "base_y": 0.0,
        "item": item.name,
        "family": item.family,
        "tier": item.tier,
        "size": [w, d],
        "extent": [round(v, 6) + 0.0 for v in _bounds(entities)],
        "height": h,
        "options": opts,
        "nominal": nominal,
        "size_basis": SIZE_BASIS,
    }


# -- many items in one call (round 5, A1) -------------------------------------

#: The keys an ``items[]`` entry takes.
ITEM_KEYS: tuple[str, ...] = (
    "name",
    "x",
    "y",
    "rotation",
    "anchor",
    "width",
    "depth",
    "height",
    "options",
)


def _finite(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return number


def parse_item(entry) -> dict:
    """One ``items[]`` entry checked and resolved, without touching a drawing.

    Returns ``{"at", "rotation", "anchor", "spec"}`` (``spec`` is
    :func:`block_spec`'s answer). Raises ``ValueError`` naming the key - an
    unknown key, a missing name or coordinate, a non-finite number, an unknown
    anchor, anything :func:`block_spec` refuses.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"must be an object {{name, x, y, ...}}, got {entry!r}")
    unknown = sorted(str(k) for k in entry if k not in ITEM_KEYS)
    if unknown:
        raise ValueError(
            f"unknown key(s) {', '.join(unknown)}; an item takes {', '.join(ITEM_KEYS)}"
        )
    for key in ("name", "x", "y"):
        if key not in entry:
            raise ValueError(f"{key} is required")
    if not isinstance(entry["name"], str):
        raise ValueError(f"name must be a string, got {entry['name']!r}")
    at = (_finite(entry["x"], "x"), _finite(entry["y"], "y"))
    rotation = _finite(entry.get("rotation", 0.0), "rotation")
    anchor = entry.get("anchor", ORIGIN)
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {', '.join(ANCHORS)}, got {anchor!r}")
    spec = block_spec(
        entry["name"],
        entry.get("width"),
        entry.get("depth"),
        entry.get("height"),
        entry.get("options"),
    )
    return {"at": at, "rotation": rotation, "anchor": anchor, "spec": spec}


def _strip(message: str) -> str:
    return message.removeprefix("arch catalogue: ").rstrip(".")


def check_items(items) -> list[dict]:
    """Every problem of an ``arch_catalogue_insert`` ``items`` list, nothing written.

    ``[{i, name, message}]``, empty when every item would be inserted. A list
    that is not a list (or is empty) raises ``ValueError``: the call itself is
    malformed. Pure - the batch planner calls it at plan time.
    """
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError(f"items must be a non-empty list of {{name, x, y, ...}}, got {items!r}")
    problems = []
    for i, entry in enumerate(items):
        try:
            parse_item(entry)
        except ValueError as exc:
            name = entry.get("name") if isinstance(entry, dict) else None
            problems.append({"i": i, "name": name, "message": f"items[{i}]: {_strip(str(exc))}"})
    return problems


def check_insert_args(args: dict) -> None:
    """``cad_batch``'s plan-time check of an ``arch_catalogue_insert`` step.

    Raises ``ValueError`` only when the call would refuse as a whole: ``items``
    malformed or every item refused. A step with some refused items runs - each
    is reported in its ``refused`` and the others are inserted.
    """
    items = args.get("items") if isinstance(args, dict) else None
    problems = check_items(items)
    if len(problems) == len(items):
        raise ValueError(_all_refused(problems))


def _all_refused(problems: list[dict]) -> str:
    shown = "; ".join(p["message"] for p in problems[:5])
    more = f"; and {len(problems) - 5} more" if len(problems) > 5 else ""
    return f"arch_catalogue_insert: every item was refused, nothing inserted: {shown}{more}"


# -- draw order by tier (round 5, A6) -----------------------------------------

_TIER_RANK = {tier: k for k, tier in enumerate(TIERS)}


def item_of_block(block: str) -> CatalogueItem | None:
    """``ARCH_DOUBLE_BED`` / ``ARCH_DOUBLE_BED_1800x2000`` -> its item (not ``ARCH_E_*``)."""
    upper = block.upper()
    if not upper.startswith(BLOCK_PREFIX) or upper.startswith(f"{BLOCK_PREFIX}E_"):
        return None
    rest = block[len(BLOCK_PREFIX) :].lower()
    for name in sorted(CATALOGUE_NAMES, key=len, reverse=True):
        if rest == name or rest.startswith(f"{name}_"):
            return _BY_NAME[name]
    return None


def intended_cover(ref, entity) -> bool:
    """Whether catalogue INSERT ``ref``'s mask covers ``entity`` by design.

    A catalogue item masks the floor finish (``A-FLOR-E-N``) and any catalogue
    item of a lower tier (a table over its rug); a ``wall`` item (a niche) also
    masks the wall it is cut into (``A-WALL*``).
    """
    mine = item_of_block(ref.dxf.name)
    if mine is None:
        return False
    layer = str(entity.dxf.get("layer", "")).upper()
    if layer == ARCH_ROLE_LAYER["floor"].upper():
        return True
    if mine.tier == "wall" and layer.startswith("A-WALL"):
        return True
    if entity.dxftype() == "INSERT":
        other = item_of_block(entity.dxf.name)
        return other is not None and _TIER_RANK[other.tier] < _TIER_RANK[mine.tier]
    return False


def _layer_new(order: list, new: dict[str, str]) -> list:
    """``order`` (the space's entities, drawn first first) with each new catalogue
    INSERT (``new``: handle -> tier) moved to its tier's place.

    A ``wall`` item goes on top of everything. Any other goes just under the
    first catalogue item of a higher tier already placed; failing one, just over
    the last catalogue item of its tier or lower; failing one, under everything.
    Nothing else moves, so what was drawn over the catalogue items stays over.
    """
    placed = [e for e in order if e.dxf.handle not in new]
    by_handle = {e.dxf.handle: e for e in order}

    def rank(e) -> int | None:
        if e.dxftype() != "INSERT":
            return None
        if e.dxf.handle in new:
            return _TIER_RANK[new[e.dxf.handle]]
        item = item_of_block(e.dxf.name)
        return None if item is None or item.tier == "wall" else _TIER_RANK[item.tier]

    for handle in (e.dxf.handle for e in order if e.dxf.handle in new):
        entity, mine = by_handle[handle], _TIER_RANK[new[handle]]
        if new[handle] == "wall":
            placed.append(entity)
            continue
        ranks = [rank(e) for e in placed]
        higher = next((k for k, r in enumerate(ranks) if r is not None and r > mine), None)
        if higher is None:
            lower = [k for k, r in enumerate(ranks) if r is not None and r <= mine]
            higher = lower[-1] + 1 if lower else 0
        placed.insert(higher, entity)
    return placed


def _apply_order(backend, new: dict[str, str]):
    """Write the tier order of the new INSERTs into model space's SORTENTSTABLE,
    then read what masks touch them. Returns ``(changed, hits)``."""
    from engineering.wipeout_mask import mask_hits

    doc = backend._require_doc()
    msp = doc.modelspace()
    current = list(msp.entities_in_redraw_order())
    target = _layer_new(current, new)
    changed = target != current
    if changed:
        # the backend's draw_order rule: the space's own handles, ascending,
        # reassigned to the new order (unique, never "0")
        keys = sorted((e.dxf.handle for e in current), key=lambda h: int(h, 16))
        msp.set_redraw_order(
            {e.dxf.handle: k for e, k in zip(target, keys, strict=True) if e.dxf.handle != k}
        )
    return changed, mask_hits(doc, msp, focus=set(new))[1]


# -- insert ---------------------------------------------------------------------


async def _ensure_layer(backend: AutoCADBackend, name: str) -> bool:
    """Create the INSERT's layer from its ``ARCH_LAYERS`` row when the drawing lacks it.

    The headless engine would otherwise create a missing layer silently with
    no colour or lineweight. Returns whether it was created.
    """
    if name.lower() in {layer.name.lower() for layer in await backend.layer_list()}:
        return False
    row = next((row for row in ARCH_LAYERS if row[0] == name), None)
    color, linetype, lineweight = (row[1], row[2], row[3]) if row else (7, "Continuous", 0.25)
    await backend.layer_create(name=name, color=color, linetype=linetype, lineweight=lineweight)
    return True


def _placement(w: float, d: float, at, angle: float, anchor: str) -> dict:
    """The WCS geometry of an item whose ``anchor`` lands on ``at`` turned by ``angle``."""
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    fx, fy = ANCHORS[anchor]
    ox, oy = at[0] - (c * fx * w - s * fy * d), at[1] - (s * fx * w + c * fy * d)

    def wcs(lx: float, ly: float) -> list[float]:
        return [round(ox + c * lx - s * ly, 6) + 0.0, round(oy + s * lx + c * ly, 6) + 0.0]

    return {
        "insert_point": wcs(0.0, 0.0),
        "back_center": wcs(w / 2.0, 0.0),
        "center": wcs(w / 2.0, d / 2.0),
        "footprint": [wcs(0.0, 0.0), wcs(w, 0.0), wcs(w, d), wcs(0.0, d)],
        "front_dir": [round(-s, 9) + 0.0, round(c, 9) + 0.0],
    }


def _warnings(hits, index: dict[str, int]) -> list[dict]:
    """Mask hits that concern the new items, as ``[{i, handle, masks | hidden_by}]``."""
    out: list[dict] = []
    for insert, _block, hit in hits:
        if insert in index:
            out.append(
                {
                    "i": index[insert],
                    "handle": insert,
                    "masks": [h for h, _t in hit[:20]],
                    "message": f"its mask hides {len(hit)} entit"
                    f"{'y' if len(hit) == 1 else 'ies'} drawn before it",
                }
            )
            continue
        for handle, _t in hit:
            if handle in index:
                out.append(
                    {
                        "i": index[handle],
                        "handle": handle,
                        "hidden_by": insert,
                        "message": f"INSERT {insert}'s mask hides it",
                    }
                )
    return sorted(out, key=lambda row: row["i"])


async def _insert_items(backend: AutoCADBackend, items) -> dict:
    """Insert many catalogue items in one call; an item refused does not stop the others.

    Every item is checked first (:func:`parse_item`); the valid ones are
    inserted in model space (each block defined once per footprint and options,
    on the furniture or sanitary layer) and layered by tier (``TIERS``: rugs and
    pre-walls lowest, then furniture and sanitary, then plants and decor; a
    niche on top). The answer lists ``inserted [{i, handle, block, footprint}]``
    (``extent`` too when the drawing reaches past the footprint),
    ``refused [{i, name, message}]`` and ``warnings`` - a new item whose mask
    hides something, or one another item's mask hides (the ``wipeout_mask``
    focus, scoped to the new items; intended layering is not a warning).

    Raises ``InvalidArgsError`` naming the items when every one is refused
    (nothing is written then) or ``items`` is not a non-empty list.
    """
    from engineering.errors import InvalidArgsError

    try:
        problems = check_items(items)
    except ValueError as exc:
        raise InvalidArgsError(f"invalid_args: {exc}") from None
    if len(problems) == len(items):
        raise InvalidArgsError(f"invalid_args: {_all_refused(problems)}")
    bad = {p["i"] for p in problems}
    parsed = [(i, parse_item(entry)) for i, entry in enumerate(items) if i not in bad]

    layers_created = []
    for layer in dict.fromkeys(ARCH_ROLE_LAYER[p["spec"]["family"]] for _i, p in parsed):
        if await _ensure_layer(backend, layer):
            layers_created.append(layer)
    existing = {block.name.upper() for block in await backend.block_list()}
    defined: list[str] = []
    inserted: list[dict] = []
    new: dict[str, str] = {}
    for i, p in parsed:
        spec = p["spec"]
        if spec["name"].upper() not in existing:
            await backend.block_define(
                spec["name"],
                spec["entities"],
                spec["attdefs"],
                spec["base_x"],
                spec["base_y"],
                False,
                False,
            )
            existing.add(spec["name"].upper())
            defined.append(spec["name"])
        w, d = spec["size"]
        placed = _placement(w, d, p["at"], p["rotation"], p["anchor"])
        ref = await backend.block_insert(
            spec["name"],
            placed["insert_point"][0],
            placed["insert_point"][1],
            1.0,
            1.0,
            p["rotation"],
            {ITEM_TAG: spec["item"], SIZE_TAG: spec["attdefs"][1]["default"]},
            ARCH_ROLE_LAYER[spec["family"]],
        )
        new[ref.handle] = spec["tier"]
        row = {"i": i, "handle": ref.handle, "block": spec["name"], **placed}
        x0, y0, x1, y1 = spec["extent"]
        if x0 < -1e-6 or y0 < -1e-6 or x1 > w + 1e-6 or y1 > d + 1e-6:
            ox, oy = placed["insert_point"]
            c, s = math.cos(math.radians(p["rotation"])), math.sin(math.radians(p["rotation"]))
            row["extent"] = [
                [round(ox + c * x - s * y, 6) + 0.0, round(oy + s * x + c * y, 6) + 0.0]
                for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            ]
        row["_spec"] = spec
        inserted.append(row)

    changed, hits = await backend._async(_apply_order, backend, new)
    if changed:
        backend._mark_dirty()
    index = {row["handle"]: row["i"] for row in inserted}
    return {
        "ok": True,
        "count": len(inserted),
        "inserted": inserted,
        "refused": problems,
        "warnings": _warnings(hits, index),
        "defined": defined,
        "layers_created": layers_created,
        "size_basis": SIZE_BASIS,
        "backend": backend.name,
    }


async def insert_items(backend: AutoCADBackend, items) -> dict:
    """:func:`_insert_items`, each inserted row cut to ``i``, ``handle``, ``block``,
    ``footprint`` (and ``extent`` when the drawing is wider) - the tool's answer."""
    result = await _insert_items(backend, items)
    keep = ("i", "handle", "block", "footprint", "extent")
    return {
        **result,
        "inserted": [{k: row[k] for k in keep if k in row} for row in result["inserted"]],
    }


async def insert_item(
    backend: AutoCADBackend,
    name: str,
    *,
    at,
    rotation: float = 0.0,
    anchor: str = ORIGIN,
    width: float | None = None,
    depth: float | None = None,
    height: float | None = None,
    options: dict | None = None,
) -> dict:
    """One item through :func:`insert_items`, its refusal a ``ValueError``.

    The library route (scripts, tests): the result carries the block's
    ``insert_point``, the WCS ``footprint`` (back-left, back-right,
    front-right, front-left), ``back_center``, ``center``, ``front_dir``,
    ``block_name``, ``defined``, ``layer``, ``tier``, ``size`` and ``warnings``.
    """
    entry = {"name": name, "rotation": rotation, "anchor": anchor}
    try:
        entry["x"], entry["y"] = at[0], at[1]
    except (TypeError, IndexError) as exc:
        raise ValueError(f"arch catalogue: 'at' must be two numbers, got {at!r}.") from exc
    for key, value in (("width", width), ("depth", depth), ("height", height)):
        if value is not None:
            entry[key] = value
    if options is not None:
        entry["options"] = options
    problems = check_items([entry])
    if problems:
        raise ValueError(f"arch catalogue: {problems[0]['message'].removeprefix('items[0]: ')}.")
    result = await _insert_items(backend, [entry])
    row = result["inserted"][0]
    spec = row.pop("_spec")
    layer = ARCH_ROLE_LAYER[spec["family"]]
    return {
        "ok": True,
        "handle": row["handle"],
        "block_name": spec["name"],
        "defined": bool(result["defined"]),
        "item": spec["item"],
        "family": spec["family"],
        "tier": spec["tier"],
        "layer": layer,
        "layer_created": layer in result["layers_created"],
        "at": [float(at[0]), float(at[1])],
        "anchor": anchor,
        "rotation": float(rotation),
        **{k: row[k] for k in ("insert_point", "back_center", "center", "footprint", "front_dir")},
        "size": spec["size"],
        "height": spec["height"],
        "options": spec["options"],
        "nominal": spec["nominal"],
        "warnings": result["warnings"],
        "size_basis": SIZE_BASIS,
        "backend": backend.name,
    }
