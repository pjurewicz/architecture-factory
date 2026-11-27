"""The ``text_overlap`` critique focus: lettering printed over other lettering or a symbol.

``dim_overlap`` compares dimensions with dimensions; nothing compared a room
label with the dimension text under it, two notes with each other, or a label
with the elevation mark it was dropped on - the sheet printed them on top of
each other and every focus stayed silent. This focus tests, within one space
(model space, or one paper layout), three kinds of pair:

1. ``text_overlaps_text``: TEXT / MTEXT / ATTRIB x TEXT / MTEXT / ATTRIB;
2. ``text_overlaps_dimension``: TEXT / MTEXT x a DIMENSION's text;
3. ``text_overlaps_symbol``: TEXT / MTEXT x an architectural symbol - a
   CIRCLE, a closed polyline or an INSERT on the arch symbol layer
   (``A-SYMB-T-N``, where ``arch_symbol`` and ``arch_grid`` draw their
   bubbles and arrows as plain entities, not blocks). Furniture and equipment
   blocks are never tested: their label lies inside the outline on purpose.

Measurement is :func:`engineering.viewport_clip.text_box` (``mtext_size`` for
MTEXT, ``ezdxf.disassemble`` for TEXT / ATTRIB), so both focuses see one box;
a DIMENSION's text is the TEXT / MTEXT its rendering draws. A circle is a
48-gon; a polyline's bulges are ignored. A pair overlaps when the area of the
intersection is more than :data:`OVERLAP_RATIO` of the smaller outline:
two labels that only touch, or whose descenders graze, are not reported.

Not tested: text on a frozen or off layer or an invisible entity; a text in a
plane tilted out of WCS XY; the cells of a table (an MTEXT with a column
width whose top-left corner sits in a ruled cell: a horizontal and a vertical
LINE of its layer within one text height above and left of its insert, as
``entity_create_table`` draws it); two items of one INSERT, one DXF GROUP
(a callout), one DIMENSION, or carrying the same ``ACADMCP_*`` record; the
symbol layer's own lettering against the symbol layer's outlines; the text
inside a dimension's ``*D`` block as a text of its own (the snapshot never
reads blocks). Nothing is repairable: which label moves is a person's call.

Coverage: one ``checked`` row per space with the number of items tested.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .plan_spec import Issue
from .qa_coverage import DOCUMENT, ledger_of

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

FOCUS = "text_overlap"
CODES = {
    "text": "text_overlaps_text",
    "dimension": "text_overlaps_dimension",
    "symbol": "text_overlaps_symbol",
}

#: A pair overlaps above this share of the smaller outline's area.
OVERLAP_RATIO = 0.15

#: The layer ``arch_symbol`` / ``arch_grid`` draw bubbles and arrows on.
SYMBOL_LAYER = "a-symb-t-n"

#: A circle's outline is this regular polygon.
CIRCLE_SIDES = 48

_TEXT = frozenset({"TEXT", "MTEXT"})

Pt = tuple[float, float]


def _area(poly: list[Pt]) -> float:
    return 0.5 * sum(
        x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1], strict=True)
    )


def _ccw(poly: list[Pt]) -> list[Pt]:
    return poly if _area(poly) >= 0 else poly[::-1]


def _clip(subject: list[Pt], convex: list[Pt]) -> list[Pt]:
    """``subject`` clipped by the convex CCW polygon ``convex`` (Sutherland-Hodgman;
    a concave subject's degenerate edges carry no area)."""
    out = subject
    for (ax, ay), (bx, by) in zip(convex, convex[1:] + convex[:1], strict=True):
        if not out:
            break
        inp, out = out, []

        def side(p, ax=ax, ay=ay, bx=bx, by=by):
            return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)

        for p, q in zip(inp, inp[1:] + inp[:1], strict=True):
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def overlap_ratio(a: list[Pt], b: list[Pt]) -> float:
    """Area of ``a`` ∩ ``b`` over the smaller area; ``b`` must be convex."""
    a, b = _ccw(a), _ccw(b)
    small = min(abs(_area(a)), abs(_area(b)))
    if small <= 0:
        return 0.0
    cut = _clip(a, b)
    return abs(_area(cut)) / small if len(cut) >= 3 else 0.0


def _symbol_outline(e) -> list[Pt]:
    kind = e.dxftype()
    if kind == "CIRCLE":
        c, r = e.dxf.center, float(e.dxf.radius)
        return [
            (
                c.x + r * math.cos(2 * math.pi * i / CIRCLE_SIDES),
                c.y + r * math.sin(2 * math.pi * i / CIRCLE_SIDES),
            )
            for i in range(CIRCLE_SIDES)
        ]
    if kind == "LWPOLYLINE" and e.closed:
        return [(x, y) for x, y, *_ in e.get_points("xy")]
    if kind == "POLYLINE" and e.is_2d_polyline and e.is_closed:
        return [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
    if kind == "INSERT":
        from ezdxf import bbox

        box = bbox.extents([e], fast=True)
        if box.has_data:
            (x0, y0), (x1, y1) = box.extmin.vec2, box.extmax.vec2
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return []


def _ruled(rules, key: tuple, x: float, y: float, h: float) -> bool:
    """A horizontal rule of ``key`` (space, layer) within ``h`` above (x, y) and a vertical
    one within ``h`` left of it: the corner of a table cell."""
    horizontal, vertical = rules.get(key, ([], []))
    above = any(y < ry <= y + h and lo <= x <= hi for ry, lo, hi in horizontal)
    return above and any(x - h <= rx < x and lo <= y <= hi for rx, lo, hi in vertical)


def _rules(records) -> dict:
    """Per (space, layer): horizontal ``(y, x0, x1)`` and vertical ``(x, y0, y1)`` LINEs."""
    out: dict = {}
    for r in records:
        if r.type != "LINE" or len(r.points) != 2:
            continue
        (x0, y0), (x1, y1) = r.points
        h, v = out.setdefault((r.space, str(r.layer).lower()), ([], []))
        if abs(y0 - y1) < 1e-9:
            h.append((y0, min(x0, x1), max(x0, x1)))
        elif abs(x0 - x1) < 1e-9:
            v.append((x0, min(y0, y1), max(y0, y1)))
    return out


def _items(doc, records, hidden: set[str]) -> dict[str, list[dict]]:
    """Every item this focus tests, per space: ``{handle, type, role, text,
    poly, keys, layer}`` (role ``text`` / ``attrib`` / ``dimension`` / ``symbol``)."""
    from .viewport_clip import text_box

    groups: dict[str, str] = {}
    for _name, group in doc.groups:
        for member in group:
            groups[member.dxf.handle] = group.dxf.handle
    rules = _rules(records)
    out: dict[str, list[dict]] = {}

    def add(space, handle, kind, role, text, poly, keys, layer):
        if len(poly) >= 3:
            out.setdefault(space, []).append(
                {
                    "handle": handle,
                    "type": kind,
                    "role": role,
                    "text": text,
                    "poly": poly,
                    "keys": keys,
                    "layer": layer,
                }
            )

    for r in records:
        layer = str(r.layer).lower()
        if not r.visible or r.tilted or layer in hidden:
            continue
        on_symbol_layer = layer == SYMBOL_LAYER
        if r.type not in _TEXT and r.type not in ("INSERT", "DIMENSION") and not on_symbol_layer:
            continue
        e = doc.entitydb.get(r.handle)
        if e is None or not e.is_alive:
            continue
        keys = {f"xd:{app}:{payload}" for app, payload in r.xdata if app.startswith("ACADMCP")}
        if r.handle in groups:
            keys.add(f"group:{groups[r.handle]}")
        if r.type in _TEXT:
            if r.type == "MTEXT" and float(e.dxf.get("width", 0) or 0) > 0:
                ins = e.dxf.insert
                h = float(e.dxf.get("char_height", 0) or 0)
                if _ruled(rules, (r.space, layer), ins.x, ins.y, h):
                    continue  # a table cell
            text = e.plain_text() if r.type == "MTEXT" else str(e.dxf.text)
            add(r.space, r.handle, r.type, "text", text, text_box(e), keys, layer)
        elif r.type == "DIMENSION":
            for piece in e.virtual_entities():
                if piece.dxftype() in _TEXT:
                    text = piece.plain_text() if piece.dxftype() == "MTEXT" else piece.dxf.text
                    keys_d = keys | {f"dim:{r.handle}"}
                    add(
                        r.space,
                        r.handle,
                        "DIMENSION",
                        "dimension",
                        str(text),
                        text_box(piece),
                        keys_d,
                        layer,
                    )
        if r.type == "INSERT":
            for a in e.attribs:
                if a.is_invisible or str(a.dxf.layer).lower() in hidden:
                    continue
                keys_a = keys | {f"insert:{r.handle}"}
                add(
                    r.space,
                    a.dxf.handle,
                    "ATTRIB",
                    "attrib",
                    str(a.dxf.text),
                    text_box(a),
                    keys_a,
                    layer,
                )
        if on_symbol_layer and r.type not in _TEXT and r.type != "DIMENSION":
            add(r.space, r.handle, r.type, "symbol", "", _symbol_outline(e), keys, layer)
    return out


def _pair_kind(a: dict, b: dict) -> str | None:
    roles = {a["role"], b["role"]}
    if roles <= {"text", "attrib"}:
        return "text"
    if "text" not in roles:
        return None  # ATTRIB x dimension / symbol, dimension x dimension, ...
    other = b if a["role"] == "text" else a
    text = a if other is b else b
    if other["role"] == "dimension":
        return "dimension"
    if other["role"] == "symbol" and text["layer"] != SYMBOL_LAYER:
        return "symbol"
    return None


def _box(poly):
    xs, ys = [x for x, _ in poly], [y for _, y in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _overlaps(items: list[dict]) -> list[tuple[str, float, dict, dict]]:
    """``(kind, ratio, a, b)`` of every overlapping pair (sweep on x)."""
    for it in items:
        it["box"] = _box(it["poly"])
    order = sorted(items, key=lambda it: it["box"][0])
    found = []
    for i, a in enumerate(order):
        ax0, ay0, ax1, ay1 = a["box"]
        for b in order[i + 1 :]:
            bx0, by0, bx1, by1 = b["box"]
            if bx0 > ax1:
                break
            if by0 > ay1 or by1 < ay0 or a["keys"] & b["keys"] or a["handle"] == b["handle"]:
                continue
            kind = _pair_kind(a, b)
            if kind is None:
                continue
            # the TEXT / MTEXT first; a symbol (maybe concave) is the clipped subject
            text, other = (a, b) if a["role"] == "text" else (b, a)
            ratio = overlap_ratio(other["poly"], text["poly"])
            if ratio > OVERLAP_RATIO:
                found.append((kind, ratio, text, other))
    return found


def _scan(doc, snap) -> dict[str, tuple[int, list]]:
    hidden = {
        name.lower() for name, row in snap.layers.items() if row.get("frozen") or not row.get("on")
    }
    items = _items(doc, snap.records, hidden)
    return {
        space: (len(items.get(space, [])), _overlaps(items.get(space, [])))
        for space in snap.layouts
    }


def _issue(space: str, kind: str, ratio: float, a: dict, b: dict) -> Issue:
    what = {"text": "", "dimension": "the text of ", "symbol": "the symbol "}[kind]
    return Issue(
        severity="warning",
        focus=FOCUS,  # type: ignore[arg-type]
        message=(
            f"{a['type']} {a['handle']} '{a['text'][:30]}' overlaps {what}{b['type']} "
            f"{b['handle']}{' ' + repr(b['text'][:30]) if b['text'] else ''} on '{space}' "
            f"({ratio:.0%} of the smaller outline). Move one of them."
        ),
        handles=[a["handle"], b["handle"]],
        detail={
            "code": CODES[kind],
            "space": space,
            "ratio": round(ratio, 3),
            "items": [
                {"handle": it["handle"], "type": it["type"], "text": it["text"][:60]}
                for it in (a, b)
            ],
            "repairable": False,
        },
    )


async def check_text_overlap(backend: AutoCADBackend, shared: dict) -> list[Issue]:
    """The focus: every space of the run's QA snapshot."""
    from .critique import scan_incomplete_issue
    from .qa_scan import _async_readonly, qa_snapshot

    snap, coverage = await qa_snapshot(backend, shared)
    ledger = ledger_of(shared)
    if snap is None or not coverage.complete:
        ledger.not_checked(FOCUS, DOCUMENT, "scan_incomplete")
        return [scan_incomplete_issue(FOCUS, coverage)]
    found = await _async_readonly(backend, lambda: _scan(backend._require_doc(), snap))
    issues: list[Issue] = []
    for space in snap.layouts:
        n, pairs = found[space]
        ledger.checked(FOCUS, space, n)
        issues.extend(_issue(space, *pair) for pair in pairs)
    return issues


check_text_overlap.needs_shared = True  # type: ignore[attr-defined]
