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
    return {"type": "circle", "cx": float(cx), "cy": float(cy), "r": float(r), "layer": "0"}


def _arc(cx: float, cy: float, r: float, start_deg: float, end_deg: float) -> dict:
    return {
        "type": "arc",
        "cx": float(cx),
        "cy": float(cy),
        "r": float(r),
        "start_deg": float(start_deg),
        "end_deg": float(end_deg),
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


# -- the items ---------------------------------------------------------------


class CatalogueItem(NamedTuple):
    """One item: its nominal size and the generator that draws it at any size in range.

    ``draw(w, d)`` returns the block-local primitives for a ``w x d`` footprint;
    at the nominal ``(w, d)`` it draws the item's nominal symbol. ``w_range`` /
    ``d_range`` are the sizes the generator keeps a legible symbol for - a
    repository sanity bound, not a standards value. ``h`` is the nominal
    height: it changes nothing in plan and is written into the SIZE attribute.
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
    draw: Callable[[float, float], tuple[dict, ...]]

    @property
    def entities(self) -> tuple[dict, ...]:
        """The primitives at the nominal size."""
        return self.draw(self.w, self.d)


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


def _single_bed(w, d):
    return (_rect(0, 0, w, d), _rect(100, 50, w - 100, 450), _line(0, 700, w, 700))


def _double_bed(w, d):
    return (
        _rect(0, 0, w, d),
        _rect(100, 50, w / 2 - 50, 450),
        _rect(w / 2 + 50, 50, w - 100, 450),
        _line(0, 700, w, 700),
    )


def _wardrobe(w, d):
    return (_rect(0, 0, w, d), _line(50, d / 2, w - 50, d / 2))


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


def _item(name, family, en, tr, w, d, h, w_range, d_range, draw) -> CatalogueItem:
    return CatalogueItem(name, family, en, tr, float(w), float(d), float(h), w_range, d_range, draw)


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
        (300, 700),
        (300, 600),
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
        "wc", "sanitary", "wc with cistern", "klozet", 380, 700, 800, (340, 420), (480, 750), _wc
    ),
    _item("bidet", "sanitary", "bidet", "bide", 360, 560, 400, (340, 400), (480, 600), _bidet),
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
        (800, 1200),
        (450, 600),
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
        (1000, 2400),
        (800, 1400),
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
    ),
    _item(
        "countertop_basin",
        "sanitary",
        "countertop basin on a vanity top",
        "tezgah üstü lavabo",
        600,
        500,
        850,
        (400, 1600),
        (400, 600),
        _countertop_basin,
    ),
)

#: The height a SIZE attribute may carry: it does not change the plan symbol.
HEIGHT_RANGE: tuple[float, float] = (10.0, 4000.0)

CATALOGUE_NAMES: tuple[str, ...] = tuple(item.name for item in ITEMS)
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


def _mm(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def block_spec(
    name: str,
    width: float | None = None,
    depth: float | None = None,
    height: float | None = None,
) -> dict:
    """The typed primitives and the ITEM / SIZE ATTDEFs ``block_define`` takes for one item.

    ``width`` / ``depth`` / ``height`` (mm) override the nominal size; the
    symbol is redrawn for the footprint, not scaled, so rails, rims and chairs
    keep their real sizes. The nominal size keeps the block ``ARCH_<NAME>``;
    any other footprint is ``ARCH_<NAME>_<W>x<D>``. The height changes nothing
    in plan and is written into the SIZE attribute (``WxDxH``).

    Returns ``{"name", "entities", "attdefs", "base_x", "base_y", "item",
    "family", "size", "height", "size_basis"}``. An unknown name (with the
    nearest names) and a size outside the item's range are refused before
    anything is built.
    """
    item = _resolve(name)
    w = _size(width, item.w, item.w_range, "width", item.name)
    d = _size(depth, item.d, item.d_range, "depth", item.name)
    h = _size(height, item.h, HEIGHT_RANGE, "height", item.name)
    try:
        entities = item.draw(w, d)
    except ValueError as exc:
        raise ValueError(f"arch catalogue: {item.name} {_mm(w)} x {_mm(d)}: {exc}.") from None
    nominal = (w, d) == (item.w, item.d)
    block = f"{BLOCK_PREFIX}{item.name.upper()}"
    if not nominal:
        block = f"{block}_{_mm(w)}x{_mm(d)}".replace(".", "_")
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
                "y": d / 2.0 - 1.5 * ITEM_TAG_HEIGHT,
                **common,
            },
        ],
        "base_x": 0.0,
        "base_y": 0.0,
        "item": item.name,
        "family": item.family,
        "size": [w, d],
        "height": h,
        "nominal": nominal,
        "size_basis": SIZE_BASIS,
    }


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
) -> dict:
    """Define the item's block once per footprint, insert it so its ``anchor`` lands on ``at``.

    ``anchor`` names a point of the footprint (``ANCHORS``; the block's own
    base is ``back_left``); ``rotation`` is degrees CCW about ``at``;
    ``width`` / ``depth`` / ``height`` as in :func:`block_spec`. The result
    carries the block's real ``insert_point``, the WCS ``footprint`` (back-left,
    back-right, front-right, front-left), ``back_center``, ``center`` and the WCS
    ``front_dir``. Refusals - an unknown item (with the nearest names) or
    anchor, a size out of range, a non-finite coordinate or rotation - fire
    before any write.
    """
    spec = block_spec(name, width, depth, height)
    if anchor not in ANCHORS:
        raise ValueError(
            f"arch catalogue: anchor must be one of {', '.join(ANCHORS)}, got {anchor!r}."
        )
    try:
        x, y = float(at[0]), float(at[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError(f"arch catalogue: 'at' must be two numbers, got {at!r}.") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"arch catalogue: 'at' must be two finite numbers, got {at!r}.")
    angle = float(rotation)
    if not math.isfinite(angle):
        raise ValueError(f"arch catalogue: rotation must be finite, got {rotation!r}.")

    w, d = spec["size"]
    placed = _placement(w, d, (x, y), angle, anchor)

    layer = ARCH_ROLE_LAYER[spec["family"]]
    layer_created = await _ensure_layer(backend, layer)
    existing = {block.name.upper() for block in await backend.block_list()}
    defined = spec["name"].upper() not in existing
    if defined:
        await backend.block_define(
            spec["name"],
            spec["entities"],
            spec["attdefs"],
            spec["base_x"],
            spec["base_y"],
            False,
            False,
        )
    ref = await backend.block_insert(
        spec["name"],
        placed["insert_point"][0],
        placed["insert_point"][1],
        1.0,
        1.0,
        angle,
        {ITEM_TAG: spec["item"], SIZE_TAG: spec["attdefs"][1]["default"]},
        layer,
    )
    return {
        "ok": True,
        "handle": ref.handle,
        "block_name": spec["name"],
        "defined": defined,
        "item": spec["item"],
        "family": spec["family"],
        "layer": layer,
        "layer_created": layer_created,
        "at": [x, y],
        "anchor": anchor,
        "rotation": angle,
        **placed,
        "size": spec["size"],
        "height": spec["height"],
        "nominal": spec["nominal"],
        "size_basis": SIZE_BASIS,
        "backend": backend.name,
    }
