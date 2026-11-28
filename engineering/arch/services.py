"""Installation points on an interior plan - every trade, one engine, real blocks.

:func:`insert_points` places a batch of symbols in one call: each item names
a :data:`KINDS` row and a point, or a :data:`RUN_KINDS` row and a ``path``.
The kinds come in per-trade vocabularies (:data:`TRADES`): electrical and
lighting, plumbing (water and drainage), heating and ventilation; a kind name
is unique across trades, so the trade is the kind's. A wall point (socket,
switch, water point, radiator, wall intake ...) snaps to the nearest wall face
within ``snap``: the block's origin lands on the face and its +Y turns into
the room, so the rotation comes from the wall, never from the caller. A free
point (ceiling light, detector, stack, floor gully, ceiling diffuser) stays
where it is put.

**The graphics are this repository's nominal convention, not a standard**:
shapes in the spirit of IEC 60617 / ISO 4067 plan symbols, sized in paper
millimetres times the plot-scale denominator (``scale``), so a 4 mm socket
reads 4 mm on the sheet; the ventilation codes are EN 16798-3's air names
(SUP, ETA, ODA, EHA). Two things are real size: the hot / cold pair's two
outlets stand :data:`PAIR_SPACING` mm apart on the wall face, and a radiator
is drawn at its ``length``. Every result and the legend say so
(:data:`SYMBOL_BASIS`).

Block-local frame, the catalogue's: the origin is the anchor (on the wall
face for a wall point, the centre for a free one), the wall runs along +X,
+Y faces into the room. One block per kind and scale,
``ARCH_<T>_<KIND>_<scale>`` with ``<T>`` the trade's letter (E, P, H, V); a
radiator's block also carries its length (``..._L600``).

Every block carries ATTDEFs: ``ITEM`` (the kind, invisible), ``CODE`` (the
letters inside a coded symbol; ``CODE2`` for the pair's second outlet), ``L``
(the item's label), ``N`` (``2x`` for a double outlet) and ``H`` (the mounting
height, ``H=30CM``). After the insert every visible ATTRIB is turned back to 0
degrees and set beside the symbol on its room side, so the lettering reads
horizontally whatever the wall's direction; a label never replaces the height.

Runs (LED strip, track, circuit link, linear drain) are one LWPOLYLINE each
along a ``path``, in their own linetype and lineweight; ``switch_frame`` is a
switch with a framed list of the circuits it serves. ``symbols={kind: block}``
draws a kind with the project's own block, anchored the same way, its
attributes filled where the block has them and lettered as TEXT where it does
not. ``dim_from`` dimensions a point's position from a named ``arch_wall``'s
face or axis, measured square to that wall.

Every INSERT, run, frame and dimension carries an ``ACADMCP_SERVICE`` record
(:data:`APP_ID`: ``{kind: device|run|frame|dim, trade, item, circuit?,
circuits?, id}``), which is how ``arch_legend`` names a project block and a
run, and what a circuit / loop list reads.
"""

from __future__ import annotations

import difflib
import math
from typing import TYPE_CHECKING, NamedTuple

from engineering.arch.layers import ARCH_ROLE_LAYER
from engineering.errors import InvalidArgsError

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

APP_ID = "ACADMCP_SERVICE"
RECORD_KINDS = ("device", "run", "frame", "dim")
SYMBOL_BASIS = (
    "nominal repository symbol convention (paper mm x scale) - not a standard; "
    "the hot/cold pair spacing and a radiator's length are real size"
)
#: trade -> its block letter; the order is the vocabulary's.
TRADES: dict[str, str] = {"electrical": "E", "plumbing": "P", "heating": "H", "ventilation": "V"}
BLOCK_PREFIXES = tuple(f"ARCH_{letter}_" for letter in TRADES.values())

#: Paper millimetres (times ``scale``).
TEXT_HEIGHT = 1.8  # the smallest ISO 3098 size
LABEL_GAP = 0.8
DIM_OFFSET = 8.0  # a dim_from dimension line past the symbol
MAX_ITEMS = 500
MAX_COUNT = 9
#: Real millimetres: the two outlets of a hot / cold pair, centre to centre.
PAIR_SPACING = 150.0
RADIATOR_LENGTH = 600.0
RADIATOR_RANGE = (200.0, 3000.0)


def _refuse(message: str) -> InvalidArgsError:
    return InvalidArgsError(f"invalid_args: {message}")


class Kind(NamedTuple):
    name: str
    trade: str
    role: str  # ARCH_ROLE_LAYER key of its default layer
    wall: bool  # snaps to a wall face; else free
    code: str  # letters inside the symbol ("" none; "A|B" one per outlet)
    shape: str
    en: str
    pl: str
    tr: str


def _k(name, trade, role, wall, code, shape, en, pl, tr) -> Kind:
    return Kind(name, trade, role, wall, code, shape, en, pl, tr)


_E, _P, _H, _V = "electrical", "plumbing", "heating", "ventilation"
_EL = ("electrical", "electrical")

KINDS: dict[str, Kind] = {
    k.name: k
    for k in (
        # electrical and lighting
        _k("socket", *_EL, True, "", "socket", "socket outlet", "gniazdo wtyczkowe", "priz"),
        _k(
            "floor_box",
            *_EL,
            False,
            "",
            "floor_box",
            "floor box with sockets",
            "puszka podłogowa z gniazdami",
            "döşeme prizi",
        ),
        _k(
            "light_outlet",
            *_EL,
            False,
            "",
            "light",
            "lighting outlet (ceiling)",
            "wypust oświetleniowy (sufit)",
            "tavan aydınlatma çıkışı",
        ),
        _k(
            "wall_light",
            *_EL,
            True,
            "",
            "wall_light",
            "lighting outlet (wall)",
            "wypust oświetleniowy ścienny",
            "duvar aydınlatma çıkışı",
        ),
        _k("switch", *_EL, True, "", "switch", "light switch", "łącznik oświetlenia", "anahtar"),
        _k(
            "data", *_EL, True, "RJ45", "box", "data outlet RJ45", "gniazdo RJ45", "data prizi RJ45"
        ),
        _k("tv", *_EL, True, "TV", "box", "TV outlet", "gniazdo RTV", "TV prizi"),
        _k("sat", *_EL, True, "SAT", "box", "satellite outlet", "gniazdo SAT", "uydu prizi"),
        _k("usb", *_EL, True, "USB", "box", "USB charging outlet", "gniazdo USB", "USB prizi"),
        _k(
            "smoke_detector",
            *_EL,
            False,
            "SD",
            "detector",
            "smoke detector",
            "czujka dymu",
            "duman dedektörü",
        ),
        _k(
            "co_detector",
            *_EL,
            False,
            "CO",
            "detector",
            "carbon monoxide detector",
            "czujka czadu (CO)",
            "karbonmonoksit dedektörü",
        ),
        _k(
            "flood_detector",
            *_EL,
            False,
            "FD",
            "detector",
            "flood detector",
            "czujnik zalania",
            "su baskını dedektörü",
        ),
        _k(
            "intercom",
            *_EL,
            True,
            "",
            "intercom",
            "video door phone",
            "wideodomofon",
            "görüntülü diyafon",
        ),
        _k(
            "panel",
            *_EL,
            True,
            "",
            "panel",
            "distribution board",
            "rozdzielnica",
            "elektrik panosu",
        ),
        _k(
            "ceiling_light",
            *_EL,
            False,
            "",
            "ceiling_light",
            "ceiling luminaire",
            "oprawa sufitowa",
            "tavan armatürü",
        ),
        _k(
            "pendant_light",
            *_EL,
            False,
            "",
            "pendant",
            "pendant luminaire",
            "oprawa wisząca",
            "sarkıt armatür",
        ),
        _k("spot", *_EL, False, "", "spot", "spotlight", "reflektor", "spot"),
        _k(
            "spot_pair",
            *_EL,
            False,
            "",
            "spot_pair",
            "double spotlight",
            "reflektor podwójny",
            "çift spot",
        ),
        _k(
            "track_head",
            *_EL,
            False,
            "",
            "track_head",
            "track spotlight",
            "głowica szynoprzewodu",
            "ray spot",
        ),
        _k(
            "motion_sensor",
            *_EL,
            False,
            "PIR",
            "detector",
            "motion sensor",
            "czujka ruchu",
            "hareket sensörü",
        ),
        _k(
            "led_driver",
            *_EL,
            False,
            "ZL",
            "driver",
            "LED driver access panel",
            "rewizja zasilacza LED",
            "LED sürücü kapağı",
        ),
        _k(
            "switch_frame",
            *_EL,
            True,
            "",
            "switch",
            "switch with circuit list",
            "łącznik z listą obwodów",
            "devre listeli anahtar",
        ),
        # plumbing: water on P-DOMW, drainage on P-SANR
        _k(
            "cold_water",
            _P,
            "water",
            True,
            "CW",
            "water",
            "cold water point",
            "punkt wody zimnej",
            "soğuk su çıkışı",
        ),
        _k(
            "hot_water",
            _P,
            "water",
            True,
            "HW",
            "water",
            "hot water point",
            "punkt ciepłej wody",
            "sıcak su çıkışı",
        ),
        _k(
            "water_pair",
            _P,
            "water",
            True,
            "HW|CW",
            "pair",
            "hot + cold water pair (150 mm)",
            "punkt wody ciepłej + zimnej (150 mm)",
            "sıcak + soğuk su çifti (150 mm)",
        ),
        _k(
            "drain",
            _P,
            "drainage",
            True,
            "",
            "drain",
            "drain point",
            "podejście kanalizacyjne",
            "pis su çıkışı",
        ),
        _k(
            "stack",
            _P,
            "drainage",
            False,
            "S",
            "stack",
            "soil stack / riser",
            "pion kanalizacyjny",
            "pis su kolonu",
        ),
        _k(
            "floor_gully",
            _P,
            "drainage",
            False,
            "",
            "gully",
            "floor gully",
            "wpust podłogowy",
            "yer süzgeci",
        ),
        _k(
            "outdoor_tap",
            _P,
            "water",
            True,
            "",
            "tap",
            "outdoor tap",
            "zawór czerpalny zewnętrzny",
            "bahçe musluğu",
        ),
        _k(
            "rain_shower",
            _P,
            "water",
            False,
            "",
            "shower",
            "rain shower head",
            "deszczownica",
            "yağmur duş başlığı",
        ),
        # heating
        _k(
            "radiator",
            _H,
            "heating",
            True,
            "",
            "radiator",
            "radiator",
            "grzejnik",
            "radyatör",
        ),
        _k(
            "manifold",
            _H,
            "heating",
            True,
            "MF",
            "manifold",
            "heating manifold",
            "rozdzielacz ogrzewania",
            "ısıtma kollektörü",
        ),
        _k(
            "thermostat",
            _H,
            "heating",
            True,
            "T",
            "round",
            "room thermostat",
            "termostat pokojowy",
            "termostat",
        ),
        # ventilation (EN 16798-3 air names)
        _k(
            "supply",
            _V,
            "ventilation",
            False,
            "SUP",
            "diffuser",
            "supply air terminal",
            "anemostat nawiewny",
            "üfleme menfezi",
        ),
        _k(
            "extract",
            _V,
            "ventilation",
            False,
            "ETA",
            "diffuser",
            "extract air terminal",
            "anemostat wywiewny",
            "emiş menfezi",
        ),
        _k(
            "intake",
            _V,
            "ventilation",
            True,
            "ODA",
            "box",
            "outdoor air intake",
            "czerpnia powietrza",
            "taze hava girişi",
        ),
        _k(
            "exhaust",
            _V,
            "ventilation",
            True,
            "EHA",
            "box",
            "exhaust air outlet",
            "wyrzutnia powietrza",
            "egzoz havası çıkışı",
        ),
    )
}


class Run(NamedTuple):
    name: str
    trade: str
    role: str
    linetype: str  # loaded on demand
    lineweight: int  # 1/100 mm
    dash: float  # paper mm the linetype's pattern should repeat over (0: continuous)
    en: str
    pl: str
    tr: str


#: Kinds drawn along a ``path`` (one LWPOLYLINE each).
RUN_KINDS: dict[str, Run] = {
    r.name: r
    for r in (
        Run("led_strip", *_EL, "Continuous", 50, 0.0, "LED strip", "taśma LED", "LED şerit"),
        Run(
            "track",
            *_EL,
            "Continuous",
            70,
            0.0,
            "surface track",
            "szynoprzewód natynkowy",
            "sıva üstü ray",
        ),
        Run(
            "track_pendant",
            *_EL,
            "DASHED",
            70,
            4.0,
            "pendant track",
            "szynoprzewód wiszący",
            "sarkıt ray",
        ),
        Run(
            "circuit_link",
            *_EL,
            "CENTER",
            18,
            8.0,
            "circuit link (switch to luminaires)",
            "powiązanie obwodu (łącznik - oprawy)",
            "devre bağlantısı",
        ),
        Run(
            "linear_drain",
            _P,
            "drainage",
            "Continuous",
            100,
            0.0,
            "linear drain",
            "odwodnienie liniowe",
            "lineer gider",
        ),
    )
}

ALL_KINDS = (*KINDS, *RUN_KINDS)
#: Default layer of every service role - kept out of ``arch_underlay``.
SERVICE_ROLES = ("electrical", "water", "drainage", "heating", "ventilation")


def trade_of(kind: str) -> str:
    return (KINDS.get(kind) or RUN_KINDS[kind]).trade


def vocabulary(trade: str | None = None) -> dict[str, dict[str, list[str]]]:
    """``{trade: {points: [...], runs: [...]}}``, for a refusal or a result."""
    out: dict[str, dict[str, list[str]]] = {}
    for name in TRADES:
        if trade not in (None, name):
            continue
        out[name] = {
            "points": [k for k, row in KINDS.items() if row.trade == name],
            "runs": [k for k, row in RUN_KINDS.items() if row.trade == name],
        }
    return out


def _vocabulary_text(trade: str | None) -> str:
    return "; ".join(
        f"{name}: {', '.join(v['points'] + v['runs'])}" for name, v in vocabulary(trade).items()
    )


def unknown_kind(kind, where: str) -> InvalidArgsError:
    """The refusal of an unknown kind: the nearest names and their trade's list."""
    near = difflib.get_close_matches(str(kind), ALL_KINDS, n=3, cutoff=0.6)
    trade = trade_of(near[0]) if near else None
    hint = f" Nearest: {', '.join(near)}." if near else ""
    return _refuse(f"{where}: {kind!r} is not a kind.{hint} Kinds - {_vocabulary_text(trade)}.")


def label(kind: str, lang: str = "en") -> str:
    """The kind's description in ``lang`` (en / tr / pl)."""
    row = KINDS.get(kind) or RUN_KINDS[kind]
    return {"pl": row.pl, "tr": row.tr}.get(lang, row.en)


def layer_of(kind: str) -> str:
    return ARCH_ROLE_LAYER[(KINDS.get(kind) or RUN_KINDS[kind]).role]


# -- block geometry (paper mm; multiplied by the scale when the block is defined) --


def _line(x1, y1, x2, y2):
    return {"type": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _circle(cx, cy, r):
    return {"type": "circle", "cx": cx, "cy": cy, "radius": r}


def _arc(cx, cy, r, a0, a1):
    return {"type": "arc", "cx": cx, "cy": cy, "radius": r, "start_angle": a0, "end_angle": a1}


def _disc(cx, cy, r):
    return {"type": "hatch", "circle": [cx, cy, r]}


def _rect(x0, y0, x1, y1):
    return {"type": "polyline", "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "closed": True}


def _shape(shape: str) -> tuple[list[dict], tuple[float, float], float]:
    """``(primitives, code centre, depth)`` in paper mm; depth is the symbol's
    extent along +Y from the anchor (a free symbol is centred on it)."""
    if shape == "socket":  # a half disc on the wall face
        return [_line(-2.0, 0.0, 2.0, 0.0), _arc(0.0, 0.0, 2.0, 0.0, 180.0)], (0.0, 1.0), 2.0
    if shape == "floor_box":
        prims = [_rect(-2.5, -2.5, 2.5, 2.5), _line(-1.5, -0.75, 1.5, -0.75)]
        return [*prims, _arc(0.0, -0.75, 1.5, 0.0, 180.0)], (0.0, 0.0), 2.5
    if shape == "light":  # a circle crossed
        d = 2.0 * math.sqrt(0.5)
        return (
            [_circle(0.0, 0.0, 2.0), _line(-d, -d, d, d), _line(-d, d, d, -d)],
            (0.0, 0.0),
            2.0,
        )
    if shape == "wall_light":
        d = 2.0 * math.sqrt(0.5)
        return (
            [_line(0, 0, 0, 1.0), _circle(0, 3.0, 2.0), _line(-d, 3 - d, d, 3 + d)]
            + [_line(-d, 3 + d, d, 3 - d)],
            (0.0, 3.0),
            5.0,
        )
    if shape == "switch":  # a ring on a stem, its lever at 45 degrees with a tick
        c = 2.0
        u = math.sqrt(0.5)
        lever = (u * 1.0, c + u * 1.0), (u * 3.5, c + u * 3.5)
        tick = (lever[1][0] + u * 0.8, lever[1][1] - u * 0.8)
        return (
            [
                _line(0, 0, 0, c - 1.0),
                _circle(0, c, 1.0),
                _line(*lever[0], *lever[1]),
                _line(*lever[1], *tick),
            ],
            (0.0, c),
            c + 1.0,
        )
    if shape == "box":  # a square on the wall face, its code inside whatever the wall
        return [_rect(-2.75, 0.0, 2.75, 5.5)], (0.0, 2.75), 5.5
    if shape == "detector":
        return [_circle(0.0, 0.0, 2.5)], (0.0, 0.0), 2.5
    if shape in ("round", "water"):  # a coded circle on a stem
        return [_line(0, 0, 0, 0.5), _circle(0, 2.5, 2.0)], (0.0, 2.5), 4.5
    if shape == "intercom":
        return (
            [_rect(-2.0, 0.0, 2.0, 5.0), _rect(-1.3, 0.7, 1.3, 3.0), _circle(0.0, 4.0, 0.4)],
            (0.0, 2.5),
            5.0,
        )
    if shape == "ceiling_light":  # a ring
        return [_circle(0.0, 0.0, 2.5), _circle(0.0, 0.0, 1.25)], (0.0, 0.0), 2.5
    if shape == "pendant":  # a circle with a filled centre
        return [_circle(0.0, 0.0, 2.5), _disc(0.0, 0.0, 1.0)], (0.0, 0.0), 2.5
    if shape == "spot":
        return [_circle(0.0, 0.0, 1.5), _disc(0.0, 0.0, 0.5)], (0.0, 0.0), 1.5
    if shape == "spot_pair":
        prims = [_circle(-2.0, 0.0, 1.5), _circle(2.0, 0.0, 1.5), _line(-0.5, 0.0, 0.5, 0.0)]
        return [*prims, _disc(-2.0, 0.0, 0.5), _disc(2.0, 0.0, 0.5)], (0.0, 0.0), 3.5
    if shape == "track_head":  # a spot on a stem
        return [_line(0.0, 0.0, 0.0, -1.5), _circle(0.0, 0.0, 1.2)], (0.0, 0.0), 1.2
    if shape == "driver":  # a hatch frame with its code
        return [_rect(-3.0, -2.0, 3.0, 2.0), _rect(-2.5, -1.5, 2.5, 1.5)], (0.0, 0.0), 3.0
    if shape == "panel":  # a box with its wall-side half filled
        solid = {"type": "solid", "points": [[-5.0, 0.0], [5.0, 0.0], [-5.0, 4.0]]}
        return [_rect(-5.0, 0.0, 5.0, 4.0), _line(-5.0, 4.0, 5.0, 0.0), solid], (0.0, 2.0), 4.0
    if shape == "drain":  # a circle on a stem with a filled centre
        return (
            [_line(0, 0, 0, 0.5), _circle(0, 2.5, 2.0), _disc(0.0, 2.5, 0.9)],
            (0.0, 2.5),
            4.5,
        )
    if shape == "stack":  # a double circle with its code
        return [_circle(0.0, 0.0, 3.0), _circle(0.0, 0.0, 2.3)], (0.0, 0.0), 3.0
    if shape == "gully":  # a square grate
        grate = [_line(-1.2, y, 1.2, y) for y in (-1.0, 0.0, 1.0)]
        return [_rect(-2.0, -2.0, 2.0, 2.0), _rect(-1.5, -1.5, 1.5, 1.5), *grate], (0.0, 0.0), 2.0
    if shape == "tap":  # a circle on a stem with its spout
        return (
            [_line(0, 0, 0, 0.5), _circle(0, 2.5, 2.0), _line(2.0, 2.5, 3.5, 2.5)]
            + [_line(3.5, 2.5, 3.5, 1.5), _disc(0.0, 2.5, 0.5)],
            (0.0, 2.5),
            4.5,
        )
    if shape == "shower":  # a rose: a circle and its jets
        jets = [
            _disc(1.5 * math.cos(a), 1.5 * math.sin(a), 0.35) for a in (0.0, 1.571, 3.142, 4.712)
        ]
        return [_circle(0.0, 0.0, 3.0), *jets, _disc(0.0, 0.0, 0.35)], (0.0, 0.0), 3.0
    if shape == "manifold":  # a bar on the wall face with its outlets
        ticks = [_line(x, 3.0, x, 4.0) for x in (-3.0, -1.0, 1.0, 3.0)]
        return [_rect(-4.0, 0.0, 4.0, 3.0), *ticks], (0.0, 1.5), 4.0
    if shape == "diffuser":  # a square terminal with its code
        d = 2.5
        corners = [_line(sx * d, sy * d, sx * 1.9, sy * 1.9) for sx in (-1, 1) for sy in (-1, 1)]
        return [_rect(-d, -d, d, d), *corners], (0.0, 0.0), d
    raise AssertionError(shape)


def _geometry(kind: str, scale: float, length: float | None = None):
    """``(primitives, code centres, depth, half width)`` of one kind in paper mm;
    the pair and the radiator carry real millimetres divided by ``scale``."""
    row = KINDS[kind]
    if row.shape == "pair":
        # each outlet's stem starts at its real position on the face; the circles
        # spread apart only when the pair's real spacing is too tight to read
        s0 = PAIR_SPACING / 2.0 / scale
        s = max(s0, 2.3)
        prims = []
        for sx in (-1.0, 1.0):
            prims += [_line(sx * s0, 0.0, sx * s, 0.5), _circle(sx * s, 2.5, 2.0)]
        return prims, [(-s, 2.5), (s, 2.5)], 4.5, s + 2.0
    if row.shape == "radiator":
        half = (length or RADIATOR_LENGTH) / 2.0 / scale
        prims = [
            _rect(-half, 0.6, half, 2.2),
            _line(-half, 1.4, half, 1.4),
            _line(-half, 0.0, -half, 0.6),
        ]
        return prims, [], 2.2, half
    prims, centre, depth = _shape(row.shape)
    return prims, [centre], depth, depth


def _scaled(spec: dict, k: float) -> dict:
    out = {"layer": "0", **spec}
    for key in ("x1", "y1", "x2", "y2", "cx", "cy", "radius"):
        if key in out:
            out[key] = float(out[key]) * k
    if "circle" in out:
        out["circle"] = [float(v) * k for v in out["circle"]]
    if "points" in out:
        out["points"] = [[float(x) * k, float(y) * k] for x, y in out["points"]]
    return out


def block_name(kind: str, scale: float, length: float | None = None) -> str:
    letter = TRADES[KINDS[kind].trade]
    name = f"ARCH_{letter}_{kind.upper()}_{scale:g}"
    if KINDS[kind].shape == "radiator":
        name += f"_L{(length or RADIATOR_LENGTH):g}"
    return name.replace(".", "_")


def kind_of_block(block: str) -> str | None:
    """``ARCH_E_SOCKET_50`` -> ``socket``: our own symbol block's kind, or None."""
    upper = block.upper()
    for kind in sorted(KINDS, key=len, reverse=True):
        prefix = f"ARCH_{TRADES[KINDS[kind].trade]}_{kind.upper()}_"
        if upper.startswith(prefix):
            return kind
    return None


def block_spec(kind: str, scale: float, length: float | None = None) -> dict:
    """The primitives and ATTDEFs ``block_define`` takes for one kind at one scale."""
    row = KINDS[kind]
    prims, centres, _depth, _half = _geometry(kind, scale, length)
    h = TEXT_HEIGHT * scale
    attdefs = [
        {"tag": "ITEM", "default": kind, "x": 0.0, "y": 0.0, "height": h, "invisible": True},
        {"tag": "L", "default": "", "x": 0.0, "y": 0.0, "height": h, "align": "middle_left"},
        {"tag": "N", "default": "", "x": 0.0, "y": 0.0, "height": h, "align": "middle_left"},
        {"tag": "H", "default": "", "x": 0.0, "y": 0.0, "height": h, "align": "middle_left"},
    ]
    codes = row.code.split("|") if row.code else []
    for i, (code, (cx, cy)) in enumerate(zip(codes, centres, strict=False)):
        attdefs.append(
            {
                "tag": "CODE" if i == 0 else f"CODE{i + 1}",
                "default": code,
                "x": cx * scale,
                "y": cy * scale,
                "height": h * (0.8 if len(code) > 2 else 1.0),
                "align": "middle_center",
            }
        )
    return {
        "name": block_name(kind, scale, length),
        "entities": [_scaled(p, scale) for p in prims],
        "attdefs": [{"prompt": a["tag"], **a} for a in attdefs],
    }


# -- wall faces ------------------------------------------------------------------

Pt = tuple[float, float]


def _inside(pt: Pt, ring) -> bool:
    x, y = pt
    inside = False
    n = len(ring)
    for i in range(n):
        (x1, y1), (x2, y2) = ring[i], ring[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _room_normal(p: Pt, q: Pt, bodies) -> Pt | None:
    """The unit normal of face p->q pointing away from the wall body, or None
    when no body touches it (a bare line: the side comes from the point)."""
    length = math.hypot(q[0] - p[0], q[1] - p[1])
    ux, uy = (q[0] - p[0]) / length, (q[1] - p[1]) / length
    mx, my = (p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0
    eps = min(1.0, 0.01 * length)
    left = (-uy, ux)
    in_left = any(_inside((mx + left[0] * eps, my + left[1] * eps), b) for b in bodies)
    in_right = any(_inside((mx - left[0] * eps, my - left[1] * eps), b) for b in bodies)
    if in_left == in_right:
        return None
    return (uy, -ux) if in_left else left


async def wall_faces(backend: AutoCADBackend) -> list[tuple[Pt, Pt, Pt | None]]:
    """``(p, q, room normal or None)`` for every wall face on the drawing: the
    junction engine's faces of the ``arch_wall`` records, the edges of the wall
    outlines ``read_wall_outlines`` finds, and bare LINEs / open polylines on
    an ``A-WALL*`` layer (a traced plan). Reads only."""
    from engineering.arch.draw import WALL_LAYER_PREFIX, read_plan, read_wall_outlines
    from engineering.arch.walls import wall_geometry
    from engineering.sheet.bom import all_entities

    plan = await read_plan(backend)
    outlines = await read_wall_outlines(backend, plan)
    segments: list[tuple[Pt, Pt]] = []
    bodies: list = [list(o) for o in outlines]
    if plan["walls"]:
        geometry = wall_geometry(plan["walls"], plan["openings"], poche=False)
        segments += [(line.p1, line.p2) for line in geometry.faces]
        bodies += [list(r) for r in geometry.regions]
    for outline in outlines:
        segments += [(outline[i], outline[(i + 1) % len(outline)]) for i in range(len(outline))]
    hatch = ARCH_ROLE_LAYER["wall_hatch"].casefold()
    for layer in await backend.layer_list():
        name = layer.name.casefold()
        if not name.startswith(WALL_LAYER_PREFIX.casefold()) or name == hatch:
            continue
        for entity in await all_entities(backend, layer=layer.name):
            props = entity.properties or {}
            kind = str(entity.type).upper()
            if kind == "LINE":
                segments.append(
                    (tuple(props["start"][:2]), tuple(props["end"][:2]))  # type: ignore[arg-type]
                )
            elif kind == "LWPOLYLINE" and not props.get("closed"):
                pts = [tuple(float(v) for v in p[:2]) for p in props.get("points") or ()]
                segments += list(zip(pts, pts[1:], strict=False))
    out = []
    for p, q in segments:
        p, q = (float(p[0]), float(p[1])), (float(q[0]), float(q[1]))
        if math.hypot(q[0] - p[0], q[1] - p[1]) > 1e-6:
            out.append((p, q, _room_normal(p, q, bodies)))
    return out


def snap_to_face(at: Pt, faces, snap: float) -> dict | None:
    """The nearest face within ``snap``: ``{anchor, normal, distance}``, or None."""
    best = None
    for p, q, normal in faces:
        dx, dy = q[0] - p[0], q[1] - p[1]
        t = max(0.0, min(1.0, ((at[0] - p[0]) * dx + (at[1] - p[1]) * dy) / (dx * dx + dy * dy)))
        foot = (p[0] + t * dx, p[1] + t * dy)
        dist = math.hypot(at[0] - foot[0], at[1] - foot[1])
        if dist <= snap and (best is None or dist < best["distance"] - 1e-9):
            if normal is None:  # a bare line: the room is the side the point is on
                if dist < 1e-6:
                    continue
                normal = ((at[0] - foot[0]) / dist, (at[1] - foot[1]) / dist)
            best = {"anchor": foot, "normal": normal, "distance": dist}
    return best


# -- the batch -----------------------------------------------------------------


def _finite(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _refuse(f"{where}: expected a finite number, got {value!r}.")
    return float(value)


def _point(value, where: str) -> Pt:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise _refuse(f"{where}: expected [x, y], got {value!r}.")
    return (_finite(value[0], f"{where}[0]"), _finite(value[1], f"{where}[1]"))


def _name(value, where: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise _refuse(f"{where}: expected text, got {value!r}.")
    return " ".join(str(value).split())


_ITEM_KEYS = {
    "kind",
    "at",
    "path",
    "count",
    "height",
    "rotation",
    "label",
    "circuit",
    "circuits",
    "length",
    "dim_from",
}
_DIM_REFS = ("face", "axis")


def _dim_from(value, where: str) -> dict | None:
    """``"W1"`` or ``{wall, ref: face|axis}`` -> ``{wall, ref}``."""
    if value is None:
        return None
    if isinstance(value, str):
        value = {"wall": value}
    if not isinstance(value, dict) or not isinstance(value.get("wall"), str):
        raise _refuse(f"{where}: expected a wall id or {{wall, ref: face|axis}}, got {value!r}.")
    extra = sorted(set(value) - {"wall", "ref"})
    if extra:
        raise _refuse(f"{where}: unknown key(s) {extra}; keys are wall, ref.")
    ref = value.get("ref", "face")
    if ref not in _DIM_REFS:
        raise _refuse(f"{where}.ref: {ref!r}; refs are face, axis.")
    return {"wall": value["wall"], "ref": ref}


def parse_items(items, default_kind: str | None) -> list[dict]:
    """Every item checked by path before anything is drawn."""
    if not isinstance(items, (list, tuple)) or not items:
        raise _refuse("items: give at least one {kind, at} or {kind, path} item.")
    if len(items) > MAX_ITEMS:
        raise _refuse(f"items: at most {MAX_ITEMS} per call, got {len(items)}.")
    out = []
    for i, item in enumerate(items):
        where = f"items[{i}]"
        if not isinstance(item, dict):
            raise _refuse(f"{where}: expected an object {{kind, at, ...}}, got {item!r}.")
        extra = sorted(set(item) - _ITEM_KEYS)
        if extra:
            raise _refuse(f"{where}: unknown key(s) {extra}; keys are {sorted(_ITEM_KEYS)}.")
        kind = item.get("kind", default_kind)
        if kind not in ALL_KINDS:
            raise unknown_kind(kind, f"{where}.kind")
        circuit = (
            None if item.get("circuit") is None else _name(item["circuit"], f"{where}.circuit")
        )
        if kind in RUN_KINDS:
            bad = sorted(
                {"at", "count", "height", "rotation", "label", "circuits", "length", "dim_from"}
                & set(item)
            )
            if bad:
                raise _refuse(f"{where}: a {kind} is a run along path; drop {bad}.")
            path = item.get("path")
            if not isinstance(path, (list, tuple)) or len(path) < 2:
                raise _refuse(f"{where}.path: a {kind} needs [[x, y], [x, y], ...] (2+ points).")
            pts = [_point(p, f"{where}.path[{j}]") for j, p in enumerate(path)]
            out.append({"kind": kind, "path": pts, "circuit": circuit})
            continue
        if "path" in item:
            raise _refuse(f"{where}.path: only {', '.join(RUN_KINDS)} take a path; use at.")
        count = item.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
            raise _refuse(f"{where}.count: a whole number 1-{MAX_COUNT}, got {count!r}.")
        height = item.get("height")
        if height is not None and _finite(height, f"{where}.height") <= 0:
            raise _refuse(f"{where}.height: the mounting height (mm) must be > 0.")
        length = item.get("length")
        if length is not None:
            if KINDS[kind].shape != "radiator":
                raise _refuse(f"{where}.length: only a radiator takes a length.")
            lo, hi = RADIATOR_RANGE
            if not lo <= _finite(length, f"{where}.length") <= hi:
                raise _refuse(f"{where}.length: a radiator is {lo:g}-{hi:g} mm long.")
            length = float(length)
        rotation = item.get("rotation")
        circuits = item.get("circuits")
        if kind == "switch_frame":
            if not isinstance(circuits, (list, tuple)) or not circuits:
                raise _refuse(
                    f"{where}.circuits: a switch_frame lists its circuits, e.g. ['SALON 01']."
                )
            circuits = [_name(c, f"{where}.circuits[{j}]") for j, c in enumerate(circuits)]
        elif circuits is not None:
            raise _refuse(f"{where}.circuits: only a switch_frame lists circuits; use circuit.")
        out.append(
            {
                "kind": kind,
                "at": _point(item.get("at"), f"{where}.at"),
                "count": count,
                "height": None if height is None else float(height),
                "rotation": None if rotation is None else _finite(rotation, f"{where}.rotation"),
                "label": None if item.get("label") is None else str(item["label"]).strip(),
                "circuit": circuit,
                "circuits": circuits,
                "length": length,
                "dim_from": _dim_from(item.get("dim_from"), f"{where}.dim_from"),
            }
        )
    return out


def height_text(height_mm: float | None) -> str:
    """``300`` -> ``H=30CM``; a mounting height is written in centimetres on a plan."""
    if height_mm is None:
        return ""
    cm = round(height_mm / 10.0, 1)
    return f"H={cm:g}CM"


def _label_spots(tip: Pt, front: Pt, h: float, gap: float, lines: int) -> tuple[str, list[Pt]]:
    """``(align, one point per line, top line first)``: the label lines beside
    the symbol's tip on its room side, horizontal whatever the wall."""
    fx, fy = front
    x, y = tip[0] + fx * gap, tip[1] + fy * gap
    pitch = 1.3 * h
    if abs(fx) >= abs(fy):
        top = y + pitch * (lines - 1) / 2.0
        align = "middle_left" if fx > 0 else "middle_right"
    elif fy > 0:
        top, align = y + pitch * (lines - 1), "bottom_center"
    else:
        top, align = y, "top_center"
    return align, [(x, top - i * pitch) for i in range(lines)]


def _frame_box(tip: Pt, front: Pt, width: float, height: float, gap: float):
    """``(x0, y0)`` lower-left corner of a ``width`` x ``height`` box beside the
    tip on the room side (the label rule, for a box)."""
    fx, fy = front
    x, y = tip[0] + fx * gap, tip[1] + fy * gap
    if abs(fx) >= abs(fy):
        return (x if fx > 0 else x - width), y - height / 2.0
    return x - width / 2.0, (y if fy > 0 else y - height)


def _symbols(symbols, blocks: dict[str, str]) -> dict[str, str]:
    """``{kind: existing block name}``, refused by key."""
    if symbols is None:
        return {}
    if not isinstance(symbols, dict):
        raise _refuse("symbols: expected {kind: block name}.")
    out = {}
    for kind, name in symbols.items():
        if kind not in KINDS:
            if kind in RUN_KINDS:
                raise _refuse(f"symbols.{kind}: a run is a line, not a block.")
            raise unknown_kind(kind, f"symbols.{kind}")
        found = blocks.get(str(name).upper())
        if found is None:
            raise _refuse(
                f"symbols.{kind}: no block {name!r} in the drawing; define it first (block_define)."
            )
        out[kind] = found
    return out


def _wall_reference(wall, anchor: Pt, ref: str) -> tuple[Pt, Pt]:
    """``(foot on the wall's face / axis, unit normal)``: the point square to the
    wall segment nearest ``anchor`` that a ``dim_from`` dimension measures from."""
    from engineering.arch.walls import face_offsets, wall_segments

    best = None
    for p, q in wall_segments(wall):
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        t = ((anchor[0] - p[0]) * dx + (anchor[1] - p[1]) * dy) / (length * length)
        tc = max(0.0, min(1.0, t))
        gap = math.dist(anchor, (p[0] + tc * dx, p[1] + tc * dy))
        if best is None or gap < best[0] - 1e-9:
            best = (gap, p, (-dy / length, dx / length))
    _gap, p, (nx, ny) = best
    d = (anchor[0] - p[0]) * nx + (anchor[1] - p[1]) * ny  # signed, along the left normal
    offset = 0.0
    if ref == "face":
        left, right = face_offsets(wall)
        offset = left if d >= 0 else right
    back = d - offset
    return (anchor[0] - nx * back, anchor[1] - ny * back), (nx, ny)


async def insert_points(
    backend: AutoCADBackend,
    items,
    *,
    kind: str | None = None,
    scale: float = 50,
    snap: float = 300.0,
    text_height: float | None = None,
    layer: str | None = None,
    color: int | None = None,
    symbols: dict | None = None,
) -> dict:
    """Place every item; see the module docstring. Refusals - an unknown kind
    (with its trade's list), a bad count / height / length / point / path, a
    wall point with no wall face within ``snap`` and no ``rotation``, a
    ``symbols`` block the drawing lacks, a ``dim_from`` wall the plan lacks -
    fire before anything is drawn."""
    from engineering.arch.draw import ensure_arch_layers, read_plan

    k = _finite(scale, "scale")
    if k <= 0:
        raise _refuse("scale: the plot-scale denominator must be > 0.")
    reach = _finite(snap, "snap")
    th = TEXT_HEIGHT if text_height is None else _finite(text_height, "text_height")
    if th <= 0:
        raise _refuse("text_height: paper mm, must be > 0.")
    if color is not None and (
        isinstance(color, bool) or not isinstance(color, int) or not 0 <= color <= 256
    ):
        raise _refuse(f"color: an ACI 0-256 (256 = ByLayer), got {color!r}.")
    override = None if layer is None else _name(layer, "layer")
    if override == "":
        raise _refuse("layer: must not be empty.")
    rows = parse_items(items, kind)
    blocks = {b.name.upper(): b.name for b in await backend.block_list()}
    own = _symbols(symbols, blocks)
    points = [r for r in rows if r["kind"] in KINDS]
    walls = {}
    if any(r.get("dim_from") for r in points):
        walls = {w.id: w for w in (await read_plan(backend))["walls"]}
        for i, r in enumerate(rows):
            wanted = (r.get("dim_from") or {}).get("wall")
            if wanted is not None and wanted not in walls:
                raise _refuse(
                    f"items[{i}].dim_from.wall: no arch_wall {wanted!r} on the drawing; "
                    f"walls are {sorted(walls) or 'none'}."
                )
    faces = (
        await wall_faces(backend)
        if any(KINDS[r["kind"]].wall and r["rotation"] is None for r in points)
        else []
    )
    placed = []
    for i, row in enumerate(rows):
        row["layer"] = override or layer_of(row["kind"])
        if row["kind"] in RUN_KINDS:
            placed.append((row, None, 0.0, None))
        elif KINDS[row["kind"]].wall and row["rotation"] is None:
            hit = snap_to_face(row["at"], faces, reach)
            if hit is None:
                raise _refuse(
                    f"items[{i}]: no wall face within snap={reach:g} of {list(row['at'])}; "
                    "move the point onto a wall face (arch_wall, an A-WALL* outline or line) "
                    "or give its rotation."
                )
            nx, ny = hit["normal"]
            rotation = round(math.degrees(math.atan2(ny, nx)) - 90.0, 9) % 360.0
            placed.append((row, hit["anchor"], rotation, hit["distance"]))
        else:
            placed.append((row, row["at"], row["rotation"] or 0.0, None))

    await ensure_arch_layers(backend, [r["layer"] for r in rows])
    for run in {RUN_KINDS[r["kind"]].linetype for r in rows if r["kind"] in RUN_KINDS}:
        if run != "Continuous":
            await backend.linetype_load(run)
    defined = []
    shapes = {}
    for row in points:
        if row["kind"] in own:
            continue
        base_kind = "switch" if row["kind"] == "switch_frame" else row["kind"]
        shapes.setdefault(block_name(base_kind, k, row["length"]), (base_kind, row["length"]))
    for name, (base_kind, length) in shapes.items():
        spec = block_spec(base_kind, k, length)
        have = blocks.get(name.upper())
        if have is None or not await backend._async(_has_tag, backend, have, "L", readonly=True):
            await backend.block_define(
                spec["name"], spec["entities"], spec["attdefs"], 0.0, 0.0, have is not None, False
            )
            defined.append(spec["name"])
    work = []
    for row, anchor, rotation, distance in placed:
        if row["kind"] in RUN_KINDS:
            work.append({"row": row})
            continue
        base_kind = "switch" if row["kind"] == "switch_frame" else row["kind"]
        block = own.get(row["kind"]) or block_name(base_kind, k, row["length"])
        values = {
            "ITEM": row["kind"],
            "L": row["label"] or "",
            "N": f"{row['count']}x" if row["count"] > 1 else "",
            "H": height_text(row["height"]),
        }
        work.append(
            {
                "row": row,
                "block": block,
                "anchor": anchor,
                "rotation": rotation,
                "values": values,
                "distance": distance,
                "own": row["kind"] in own,
            }
        )
    out = await backend._async(_draw, backend, work, k, th, color)
    dims = await _dimensions(backend, work, out, walls, k, color)
    counts: dict[str, int] = {}
    circuits: dict[str, int] = {}
    trades: dict[str, int] = {}
    for row, *_ in placed:
        counts[row["kind"]] = counts.get(row["kind"], 0) + row.get("count", 1)
        trade = trade_of(row["kind"])
        trades[trade] = trades.get(trade, 0) + 1
        if row.get("circuit"):
            circuits[row["circuit"]] = circuits.get(row["circuit"], 0) + 1
    result = {
        "ok": True,
        "items": out,
        "counts": counts,
        "trades": trades,
        "blocks_defined": defined,
        "layers": sorted({r["layer"] for r in rows}),
        "symbol_basis": SYMBOL_BASIS,
        "backend": backend.name,
    }
    if dims:
        result["dimensions"] = dims
    if circuits:
        result["circuits"] = circuits
    if own:
        result["symbols"] = own
    return result


async def _dimensions(backend, work, out, walls, k: float, color) -> list[dict]:
    """One linear DIMENSION per ``dim_from`` item: from the named wall's face or
    axis to the point's anchor, square to that wall, its line past the symbol
    on the room side; on the item's layer with its record (``kind: dim``)."""
    from engineering.mech.xdata import to_values

    jobs = [
        (job, entry) for job, entry in zip(work, out, strict=True) if job["row"].get("dim_from")
    ]
    if not jobs:
        return []
    previous = await backend.system_get_variable("DIMSCALE")
    previous = 1.0 if previous is None else float(previous)
    if abs(previous - k) > 1e-9:
        await backend.system_set_variable("DIMSCALE", k)
    rows = []
    try:
        for job, entry in jobs:
            row, anchor = job["row"], job["anchor"]
            ref = row["dim_from"]
            foot, (nx, ny) = _wall_reference(walls[ref["wall"]], anchor, ref["ref"])
            a = math.radians(job["rotation"])
            front = (-math.sin(a), math.cos(a))
            _p, _c, depth, _h = _geometry(row["kind"], k, row["length"])
            reach = (depth + DIM_OFFSET) * k
            at = (
                (foot[0] + anchor[0]) / 2.0 + front[0] * reach,
                (foot[1] + anchor[1]) / 2.0 + front[1] * reach,
            )
            payload = {
                "v": 1,
                "kind": "dim",
                "trade": trade_of(row["kind"]),
                "item": row["kind"],
                "id": entry["handle"],
            }
            info = await backend.dimension_linear(
                foot[0],
                foot[1],
                anchor[0],
                anchor[1],
                at[0],
                at[1],
                math.degrees(math.atan2(ny, nx)),
                row["layer"],
                xdata={APP_ID: to_values(payload, app_id=APP_ID, kinds=RECORD_KINDS)},
            )
            value = round(math.dist(foot, anchor), 3)
            entry["dimension"] = info.handle
            rows.append(
                {
                    "handle": info.handle,
                    "item": entry["handle"],
                    "from": {**ref, "point": [round(c, 3) for c in foot]},
                    "value": value,
                }
            )
    finally:
        if abs(previous - k) > 1e-9:
            await backend.system_set_variable("DIMSCALE", previous)
    return rows


def _has_tag(backend, block: str, tag: str) -> bool:
    doc = backend._require_doc()
    return any(a.dxf.tag.upper() == tag for a in doc.blocks.get(block).query("ATTDEF"))


def _record(entity, payload: dict) -> None:
    from engineering.mech.xdata import to_values

    doc = entity.doc
    if APP_ID not in doc.appids:
        doc.appids.add(APP_ID)
    values = to_values({"v": 1, **payload}, app_id=APP_ID, kinds=RECORD_KINDS)
    entity.set_xdata(APP_ID, [(1000, v) for v in values])


def _block_reach(block) -> tuple[float, float]:
    """``(depth along +Y, half width along X)`` of a block's graphics, its units."""
    from ezdxf import bbox

    box = bbox.extents(e for e in block if e.dxftype() != "ATTDEF")
    if not box.has_data:
        return 0.0, 0.0
    return max(box.extmax.y, 0.0), max(abs(box.extmin.x), abs(box.extmax.x))


def _draw(backend, work, k: float, th: float, color) -> list[dict]:
    """Everything in one pass over the document: INSERTs with their ATTRIBs at
    0 degrees beside the symbol (TEXT where a project block lacks the tag),
    runs, frames, and the ``ACADMCP_SERVICE`` record on each."""
    from ezdxf.enums import TextEntityAlignment

    from engineering.annotation import text_width

    doc = backend._require_doc()
    msp = backend._msp()
    h, gap = th * k, LABEL_GAP * k
    ltscale = float(doc.header.get("$LTSCALE", 1.0) or 1.0)
    out = []
    for job in work:
        row = job["row"]
        base = {"layer": row["layer"], **({} if color is None else {"color": color})}
        trade = trade_of(row["kind"])
        if "block" not in job:
            run = RUN_KINDS[row["kind"]]
            attribs = {**base, "lineweight": run.lineweight}
            if run.linetype != "Continuous":
                pattern = (
                    sum(abs(v) for v in doc.linetypes.get(run.linetype).pattern_tags.compile())
                    or 1.0
                )
                attribs.update(linetype=run.linetype, ltscale=run.dash * k / pattern / ltscale)
            poly = msp.add_lwpolyline(row["path"], dxfattribs=attribs)
            payload = {"kind": "run", "trade": trade, "item": row["kind"], "id": poly.dxf.handle}
            if row["circuit"]:
                payload["circuit"] = row["circuit"]
            _record(poly, payload)
            length = sum(
                math.dist(a, b) for a, b in zip(row["path"], row["path"][1:], strict=False)
            )
            entry = {"handle": poly.dxf.handle, "kind": row["kind"], "length": round(length, 1)}
            if row["circuit"]:
                entry["circuit"] = row["circuit"]
            out.append(entry)
            continue
        anchor, rotation, values = job["anchor"], job["rotation"], job["values"]
        ref = msp.add_blockref(job["block"], anchor, dxfattribs={**base, "rotation": rotation})
        block = doc.blocks.get(job["block"])
        tags = {a.dxf.tag.upper(): a for a in block.query("ATTDEF")}
        ref.add_auto_attribs({t: v for t, v in values.items() if t in tags})
        a = math.radians(rotation)
        front = (-math.sin(a), math.cos(a))
        meta = KINDS[row["kind"]]
        if job["own"]:
            depth, half = _block_reach(block)
            centres = []
        else:
            base_kind = "switch" if row["kind"] == "switch_frame" else row["kind"]
            _prims, centres, depth, half = _geometry(base_kind, k, row["length"])
            depth, half = depth * k, half * k
            centres = [(cx * k, cy * k) for cx, cy in centres]
        if meta.wall:
            tip = (anchor[0] + front[0] * depth, anchor[1] + front[1] * depth)
        else:  # a free symbol: its labels to the right of the symbol
            front = (1.0, 0.0)
            tip = (anchor[0] + half, anchor[1])
        code_at = [
            (
                anchor[0] + math.cos(a) * cx - math.sin(a) * cy,
                anchor[1] + math.sin(a) * cx + math.cos(a) * cy,
            )
            for cx, cy in centres
        ]
        lines = [t for t in ("L", "N", "H") if values[t]]
        align, spots = _label_spots(tip, front, h, gap, max(1, len(lines)))
        spot = dict(zip(lines, spots, strict=False))
        record = {"kind": "device", "trade": trade, "item": row["kind"], "id": ref.dxf.handle}
        if row["circuit"]:
            record["circuit"] = row["circuit"]
        how = TextEntityAlignment[align.upper()]
        for attrib in ref.attribs:
            tag = attrib.dxf.tag.upper()
            if tag.startswith("CODE") and not job["own"]:
                n = int(tag[4:] or 1) - 1
                if n < len(code_at):
                    attrib.dxf.rotation = 0.0
                    attrib.set_placement(code_at[n], align=TextEntityAlignment.MIDDLE_CENTER)
            elif tag in spot:
                attrib.dxf.rotation = 0.0
                attrib.dxf.height = h
                attrib.set_placement(spot[tag], align=how)
        for tag in lines:
            if tag not in tags:  # a project block without the tag: lettered beside it
                text = msp.add_text(values[tag], dxfattribs={**base, "height": h})
                text.set_placement(spot[tag], align=how)
                _record(text, record)
        _record(ref, record)
        entry = {"handle": ref.dxf.handle, "kind": row["kind"], "at": [round(c, 3) for c in anchor]}
        if job["distance"] is not None:
            entry["rotation"] = round(rotation, 3)
            entry["snapped"] = round(job["distance"], 3)
        if row["circuit"]:
            entry["circuit"] = row["circuit"]
        if row["kind"] == "switch_frame":
            names = row["circuits"]
            pad, pitch = 0.6 * h, 1.4 * h
            width = max(text_width(n, h) for n in names) + 2 * pad
            height = len(names) * pitch + pad
            # past the label lines: their width across a side wall, their height
            # off a wall above or below
            if not lines:
                beyond = 0.0
            elif abs(front[0]) >= abs(front[1]):
                beyond = max(text_width(values[t], h) for t in lines) + gap
            else:
                beyond = len(lines) * 1.3 * h + gap
            far = (tip[0] + front[0] * beyond, tip[1] + front[1] * beyond)
            x0, y0 = _frame_box(far, front, width, height, gap)
            frame = msp.add_lwpolyline(
                [(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)],
                close=True,
                dxfattribs=base,
            )
            frame_record = {**record, "kind": "frame", "circuits": names}
            _record(frame, frame_record)
            for j, name in enumerate(names):
                text = msp.add_text(name, dxfattribs={**base, "height": h})
                text.set_placement(
                    (x0 + pad, y0 + height - pad / 2.0 - (j + 0.5) * pitch),
                    align=TextEntityAlignment.MIDDLE_LEFT,
                )
                _record(text, frame_record)
            entry["circuits"] = names
        out.append(entry)
    backend._mark_dirty()
    return out


def read_records(doc) -> list[tuple[object, dict]]:
    """``(entity, payload)`` of every ``ACADMCP_SERVICE`` record in model space."""
    from engineering.mech.xdata import decode

    found = []
    for e in doc.modelspace():
        if not e.has_xdata(APP_ID):
            continue
        try:
            payload = decode(list(e.get_xdata(APP_ID)), app_id=APP_ID, kinds=RECORD_KINDS)
        except ValueError:
            continue
        found.append((e, payload))
    return found
