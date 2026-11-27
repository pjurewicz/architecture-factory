"""Electrical outlets, switches and devices on an interior plan, as real blocks.

:func:`insert_devices` places a batch of symbols in one call: each item names
a :data:`KINDS` row and a point. A wall device (socket, switch, RJ45 ...)
snaps to the nearest wall face within ``snap``: the block's origin lands on
the face and its +Y turns into the room, so the rotation comes from the wall,
never from the caller. A free device (ceiling light, floor box, smoke / CO /
flood detector) stays where it is put.

**The graphics are this repository's nominal convention, not a standard**:
shapes in the spirit of IEC 60617 plan symbols, sized in paper millimetres
times the plot-scale denominator (``scale``), so a 4 mm socket reads 4 mm on
the sheet. Every result and the legend say so (:data:`SYMBOL_BASIS`).

Block-local frame, the catalogue's: the origin is the anchor (on the wall
face for a wall device, the centre for a free one), the wall runs along +X,
+Y faces into the room. One block per kind and scale, ``ARCH_E_<KIND>_<scale>``.

Every block carries ATTDEFs: ``ITEM`` (the kind, invisible, for a count or a
legend), ``CODE`` (the letters inside a coded symbol), ``N`` (``2x`` for a
double outlet; empty for one) and ``H`` (the mounting height, ``H=30CM``;
empty when not given). After the insert every visible ATTRIB is turned back
to 0 degrees and set beside the symbol on its room side, so the lettering
reads horizontally whatever the wall's direction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

from engineering.arch.layers import ARCH_ROLE_LAYER

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

SYMBOL_BASIS = "nominal repository symbol convention (paper mm x scale) - not a standard"
BLOCK_PREFIX = "ARCH_E_"
LAYER = ARCH_ROLE_LAYER["electrical"]

#: Paper millimetres (times ``scale``).
TEXT_HEIGHT = 1.8  # the smallest ISO 3098 size
LABEL_GAP = 0.8
MAX_ITEMS = 500
MAX_COUNT = 9


class Kind(NamedTuple):
    name: str
    wall: bool  # snaps to a wall face; else free
    code: str  # letters inside the symbol ("" for none)
    shape: str
    en: str
    pl: str
    tr: str


KINDS: dict[str, Kind] = {
    k.name: k
    for k in (
        Kind("socket", True, "", "socket", "socket outlet", "gniazdo wtyczkowe", "priz"),
        Kind(
            "floor_box",
            False,
            "",
            "floor_box",
            "floor box with sockets",
            "puszka podłogowa z gniazdami",
            "döşeme prizi",
        ),
        Kind(
            "light_outlet",
            False,
            "",
            "light",
            "lighting outlet (ceiling)",
            "wypust oświetleniowy (sufit)",
            "tavan aydınlatma çıkışı",
        ),
        Kind(
            "wall_light",
            True,
            "",
            "wall_light",
            "lighting outlet (wall)",
            "wypust oświetleniowy ścienny",
            "duvar aydınlatma çıkışı",
        ),
        Kind("switch", True, "", "switch", "light switch", "łącznik oświetlenia", "anahtar"),
        Kind("data", True, "RJ45", "box", "data outlet RJ45", "gniazdo RJ45", "data prizi RJ45"),
        Kind("tv", True, "TV", "box", "TV outlet", "gniazdo RTV", "TV prizi"),
        Kind("sat", True, "SAT", "box", "satellite outlet", "gniazdo SAT", "uydu prizi"),
        Kind("usb", True, "USB", "box", "USB charging outlet", "gniazdo USB", "USB prizi"),
        Kind(
            "smoke_detector",
            False,
            "SD",
            "detector",
            "smoke detector",
            "czujka dymu",
            "duman dedektörü",
        ),
        Kind(
            "co_detector",
            False,
            "CO",
            "detector",
            "carbon monoxide detector",
            "czujka czadu (CO)",
            "karbonmonoksit dedektörü",
        ),
        Kind(
            "flood_detector",
            False,
            "FD",
            "detector",
            "flood detector",
            "czujnik zalania",
            "su baskını dedektörü",
        ),
        Kind(
            "thermostat", True, "T", "round", "room thermostat", "termostat pokojowy", "termostat"
        ),
        Kind(
            "intercom",
            True,
            "",
            "intercom",
            "video door phone",
            "wideodomofon",
            "görüntülü diyafon",
        ),
        Kind(
            "panel",
            True,
            "",
            "panel",
            "distribution board",
            "rozdzielnica",
            "elektrik panosu",
        ),
    )
}


def label(kind: str, lang: str = "en") -> str:
    """The kind's description in ``lang`` (en / tr / pl)."""
    row = KINDS[kind]
    return {"pl": row.pl, "tr": row.tr}.get(lang, row.en)


# -- block geometry (paper mm; multiplied by the scale when the block is defined) --


def _line(x1, y1, x2, y2):
    return {"type": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _circle(cx, cy, r):
    return {"type": "circle", "cx": cx, "cy": cy, "radius": r}


def _arc(cx, cy, r, a0, a1):
    return {"type": "arc", "cx": cx, "cy": cy, "radius": r, "start_angle": a0, "end_angle": a1}


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
    if shape == "round":
        return [_line(0, 0, 0, 0.5), _circle(0, 2.5, 2.0)], (0.0, 2.5), 4.5
    if shape == "intercom":
        return (
            [_rect(-2.0, 0.0, 2.0, 5.0), _rect(-1.3, 0.7, 1.3, 3.0), _circle(0.0, 4.0, 0.4)],
            (0.0, 2.5),
            5.0,
        )
    if shape == "panel":  # a box with its wall-side half filled
        solid = {"type": "solid", "points": [[-5.0, 0.0], [5.0, 0.0], [-5.0, 4.0]]}
        return [_rect(-5.0, 0.0, 5.0, 4.0), _line(-5.0, 4.0, 5.0, 0.0), solid], (0.0, 2.0), 4.0
    raise AssertionError(shape)


def _scaled(spec: dict, k: float) -> dict:
    out = {"layer": "0", **spec}
    for key in ("x1", "y1", "x2", "y2", "cx", "cy", "radius"):
        if key in out:
            out[key] = float(out[key]) * k
    if "points" in out:
        out["points"] = [[float(x) * k, float(y) * k] for x, y in out["points"]]
    return out


def block_name(kind: str, scale: float) -> str:
    return f"{BLOCK_PREFIX}{kind.upper()}_{scale:g}".replace(".", "_")


def block_spec(kind: str, scale: float) -> dict:
    """The primitives and ATTDEFs ``block_define`` takes for one kind at one scale."""
    row = KINDS[kind]
    prims, (cx, cy), _depth = _shape(row.shape)
    h = TEXT_HEIGHT * scale
    attdefs = [
        {"tag": "ITEM", "default": kind, "x": 0.0, "y": 0.0, "height": h, "invisible": True},
        {"tag": "N", "default": "", "x": 0.0, "y": 0.0, "height": h, "align": "middle_left"},
        {"tag": "H", "default": "", "x": 0.0, "y": 0.0, "height": h, "align": "middle_left"},
    ]
    if row.code:
        attdefs.append(
            {
                "tag": "CODE",
                "default": row.code,
                "x": cx * scale,
                "y": cy * scale,
                "height": h * (0.8 if len(row.code) > 2 else 1.0),
                "align": "middle_center",
            }
        )
    return {
        "name": block_name(kind, scale),
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
        raise ValueError(f"{where}: expected a finite number, got {value!r}.")
    return float(value)


def _point(value, where: str) -> Pt:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{where}: expected [x, y], got {value!r}.")
    return (_finite(value[0], f"{where}[0]"), _finite(value[1], f"{where}[1]"))


_ITEM_KEYS = {"kind", "at", "count", "height", "rotation", "label"}


def parse_items(items, default_kind: str | None) -> list[dict]:
    """Every item checked by path before anything is drawn."""
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("items: give at least one {kind, at} item.")
    if len(items) > MAX_ITEMS:
        raise ValueError(f"items: at most {MAX_ITEMS} per call, got {len(items)}.")
    out = []
    for i, item in enumerate(items):
        where = f"items[{i}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where}: expected an object {{kind, at, ...}}, got {item!r}.")
        extra = sorted(set(item) - _ITEM_KEYS)
        if extra:
            raise ValueError(f"{where}: unknown key(s) {extra}; keys are {sorted(_ITEM_KEYS)}.")
        kind = item.get("kind", default_kind)
        if kind not in KINDS:
            raise ValueError(f"{where}.kind: {kind!r} is not one of {', '.join(KINDS)}.")
        count = item.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
            raise ValueError(f"{where}.count: a whole number 1-{MAX_COUNT}, got {count!r}.")
        height = item.get("height")
        if height is not None and _finite(height, f"{where}.height") <= 0:
            raise ValueError(f"{where}.height: the mounting height (mm) must be > 0.")
        rotation = item.get("rotation")
        out.append(
            {
                "kind": kind,
                "at": _point(item.get("at"), f"{where}.at"),
                "count": count,
                "height": None if height is None else float(height),
                "rotation": None if rotation is None else _finite(rotation, f"{where}.rotation"),
                "label": None if item.get("label") is None else str(item["label"]).strip(),
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


async def insert_devices(
    backend: AutoCADBackend,
    items,
    *,
    kind: str | None = None,
    scale: float = 50,
    snap: float = 300.0,
) -> dict:
    """Place every item; see the module docstring. Refusals - an unknown kind
    (with the list), a bad count / height / point, a wall device with no wall
    face within ``snap`` and no ``rotation`` - fire before anything is drawn."""
    from engineering.arch.draw import ensure_arch_layers

    k = _finite(scale, "scale")
    if k <= 0:
        raise ValueError("scale: the plot-scale denominator must be > 0.")
    reach = _finite(snap, "snap")
    rows = parse_items(items, kind)
    faces = (
        await wall_faces(backend)
        if any(KINDS[r["kind"]].wall and r["rotation"] is None for r in rows)
        else []
    )
    placed = []
    for i, row in enumerate(rows):
        if KINDS[row["kind"]].wall and row["rotation"] is None:
            hit = snap_to_face(row["at"], faces, reach)
            if hit is None:
                raise ValueError(
                    f"items[{i}]: no wall face within snap={reach:g} of {list(row['at'])}; "
                    "move the point onto a wall face (arch_wall, an A-WALL* outline or line) "
                    "or give its rotation."
                )
            nx, ny = hit["normal"]
            rotation = round(math.degrees(math.atan2(ny, nx)) - 90.0, 9) % 360.0
            placed.append((row, hit["anchor"], rotation, hit["distance"]))
        else:
            placed.append((row, row["at"], row["rotation"] or 0.0, None))

    await ensure_arch_layers(backend, (LAYER,))
    existing = {b.name.upper() for b in await backend.block_list()}
    defined = []
    for name in dict.fromkeys(r["kind"] for r, *_ in placed):
        spec = block_spec(name, k)
        if spec["name"].upper() not in existing:
            await backend.block_define(
                spec["name"], spec["entities"], spec["attdefs"], 0.0, 0.0, False, False
            )
            defined.append(spec["name"])
    h, gap = TEXT_HEIGHT * k, LABEL_GAP * k
    out, fixes = [], []
    for row, anchor, rotation, distance in placed:
        values = {
            "N": f"{row['count']}x" if row["count"] > 1 else "",
            "H": row["label"] if row["label"] is not None else height_text(row["height"]),
        }
        ref = await backend.block_insert(
            block_name(row["kind"], k), anchor[0], anchor[1], 1.0, 1.0, rotation, values, LAYER
        )
        a = math.radians(rotation)
        front = (-math.sin(a), math.cos(a))
        _prims, (cx, cy), depth = _shape(KINDS[row["kind"]].shape)
        free = not KINDS[row["kind"]].wall
        reach_tip = depth * k
        tip = (anchor[0] + front[0] * reach_tip, anchor[1] + front[1] * reach_tip)
        if free:  # a free symbol: its labels to the right of the symbol
            front = (1.0, 0.0)
            tip = (anchor[0] + depth * k, anchor[1])
        code_at = (
            anchor[0] + math.cos(a) * cx * k - math.sin(a) * cy * k,
            anchor[1] + math.sin(a) * cx * k + math.cos(a) * cy * k,
        )
        tags = [tag for tag in ("N", "H") if values[tag]]
        align, spots = _label_spots(tip, front, h, gap, max(1, len(tags)))
        fixes.append((ref.handle, align, dict(zip(tags, spots, strict=False)), code_at))
        entry = {"handle": ref.handle, "kind": row["kind"], "at": [round(c, 3) for c in anchor]}
        if distance is not None:
            entry["rotation"] = round(rotation, 3)
            entry["snapped"] = round(distance, 3)
        out.append(entry)
    await backend._async(_level_attribs, backend, fixes)
    counts: dict[str, int] = {}
    for row, *_ in placed:
        counts[row["kind"]] = counts.get(row["kind"], 0) + row["count"]
    return {
        "ok": True,
        "items": out,
        "counts": counts,
        "blocks_defined": defined,
        "layer": LAYER,
        "symbol_basis": SYMBOL_BASIS,
        "backend": backend.name,
    }


def _level_attribs(backend, fixes) -> None:
    """Every visible ATTRIB of the new INSERTs at 0 degrees: N and H beside the
    symbol on its room side, CODE centred in the symbol."""
    from ezdxf.enums import TextEntityAlignment

    doc = backend._require_doc()
    for handle, align, lines, code_at in fixes:
        ref = doc.entitydb.get(handle)
        spot = {tag: (at, align) for tag, at in lines.items()} | {
            "CODE": (code_at, "middle_center")
        }
        for attrib in ref.attribs:
            tag = attrib.dxf.tag.upper()
            if tag not in spot:
                continue
            at, how = spot[tag]
            attrib.dxf.rotation = 0.0
            attrib.set_placement(at, align=TextEntityAlignment[how.upper()])
    backend._mark_dirty()
