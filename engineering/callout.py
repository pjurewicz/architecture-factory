"""Geometry of one technical callout: a terminator on each target, a leader
(straight, or through the caller's points) to a knee, a horizontal shelf, and
a note or a numbered circle on the shelf, optionally over a background mask.

Pure: `plan_callout` validates every argument and computes every vertex
before the backend draws anything, so a refusal leaves the drawing untouched.
Widths come from the ``measure`` the caller passes: the backend passes
`engineering.annotation.style_text_width` of the note's text style (font and
width factor), the box `text_overlap` and the renderer see.

Layout, in multiples of the text height ``h`` (AutoCAD's MLEADER defaults at
4 mm text: an 8 mm dogleg and a 2 mm landing gap):

* the leader is ``target -> landing``, ``leader`` points (first the target,
  last the knee), or one straight branch per ``targets`` point to one knee;
* the shelf runs ``SHELF`` from the knee, away from the target's side
  (``side="auto"``), to the ``side`` the caller names, or to ``shelf_end``;
* ``shelf="under"`` runs it under the note, ``"over"`` over it (the note
  below); without ``shelf_end`` it is as long as the note, with it the note
  stands at the shelf's end (``align="auto"``) and must fit on it;
* the terminator is none, a 45-degree tick, a filled dot or a filled arrow;
* a note starts ``GAP`` past the shelf end, its middle on the shelf;
* a number sits in a circle tangent to the shelf end, the numeral ``h`` high;
* the mask is a WIPEOUT: a rectangle ``GAP`` round the note, or a polygon
  circumscribing the circle. It is drawn before the leader and the lettering,
  so it hides what lies under the callout and nothing of the callout itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engineering.annotation import LINE_PITCH, text_width

STYLES = ("none", "tick", "dot", "arrow")
SIDES = ("auto", "left", "right")
#: ``end``: the note past the shelf's end; ``under``: the shelf under the note
#: (round 7, the architect's underlined label); ``over``: the note under it.
SHELVES = ("end", "under", "over")
ALIGNS = ("auto", "left", "right")
#: Architectural tick: a stroke this many text heights long at 45 degrees.
TICK_LENGTH = 1.0
#: Shelf (dogleg) length, in text heights.
SHELF = 2.0
#: Shelf end to note, and the mask's margin round the note, in text heights.
GAP = 0.5
#: Filled arrow: length 1 h, half-width 1/6 of it (AutoCAD's closed filled).
ARROW_LENGTH = 1.0
ARROW_HALF_WIDTH = 1.0 / 6.0
#: Dot radius in text heights: 0.5 at 2.5 mm text, the balloon dot.
DOT_RADIUS = 0.2
#: Description of the anonymous GROUP that holds one callout's parts, so
#: ``entity_delete`` of any part erases the whole callout.
GROUP_DESCRIPTION = "ACADMCP callout"
#: Sides of the polygon that masks a numbered circle.
MASK_SIDES = 32
#: MTEXT line spacing factor range (AutoCAD's).
LINE_SPACING = (0.25, 4.0)

Pt = tuple[float, float]


@dataclass(frozen=True)
class CalloutPlan:
    kind: str  # "text" | "number"
    style: str  # STYLES
    side: str  # "right" | "left": where the shelf runs from the knee
    height: float
    leader: tuple[Pt, ...]  # first target ... knee, shelf end
    branches: tuple[tuple[Pt, Pt], ...]  # every further target -> knee
    terminators: tuple  # per target: arrow triangle (tip first) | (centre, r) | tick (p, q)
    content: str  # MTEXT content (lines joined by \\P) or the numeral
    content_at: Pt
    content_align: str  # MTEXT attachment / TEXT alignment name
    circle: tuple[Pt, float] | None  # the numbered circle
    mask: tuple[Pt, ...] | None
    box: tuple[Pt, ...]  # what the content covers (the mask's outline), mask or not
    line_spacing: float = 1.0
    overflow: tuple[str, ...] = ()  # words wider than max_width


def _point(value, name: str) -> Pt:
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        raise ValueError(f"{name} must be an [x, y] pair; got {value!r}") from None
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return x, y


def _points(value, name: str, least: int) -> list[Pt]:
    if not isinstance(value, (list, tuple)) or len(value) < least:
        raise ValueError(f"{name} must list at least {least} [x, y] points; got {value!r}")
    return [_point(p, f"{name}[{i}]") for i, p in enumerate(value)]


def _paths(target, landing, targets, leader) -> list[list[Pt]]:
    """One leader path per target, each ending at the shared knee."""
    given = [
        k
        for k, v in (("target", target), ("targets", targets), ("leader", leader))
        if v is not None
    ]
    if len(given) != 1:
        raise ValueError(
            f"pass exactly one of target, targets or leader; got {', '.join(given) or 'none'}"
        )
    if leader is not None:
        if landing is not None:
            raise ValueError("leader ends at the knee: its last point is the landing; drop landing")
        return [_points(leader, "leader", 2)]
    if landing is None:
        raise ValueError("landing (the knee) is required with target / targets")
    knee = _point(landing, "landing")
    if target is not None:
        return [[_point(target, "target"), knee]]
    return [[p, knee] for p in _points(targets, "targets", 1)]


def _content(text, number) -> tuple[str, list[str] | str]:
    if (text is None) == (number is None):
        raise ValueError(
            "pass exactly one of text (a descriptive callout) or number (a numbered one)"
        )
    if text is not None:
        lines = [line.strip() for line in str(text).replace("\r\n", "\n").split("\n")]
        if not any(lines):
            raise ValueError("text must not be empty")
        return "text", lines
    if isinstance(number, bool):
        raise ValueError(f"number must be a positive integer or a short label; got {number!r}")
    if isinstance(number, int):
        if number < 1:
            raise ValueError(f"number must be a positive integer; got {number}")
        return "number", str(number)
    label = str(number).strip()
    if not label or "\n" in label:
        raise ValueError(f"number must be a single-line label; got {number!r}")
    return "number", label


def _wrap(lines: list[str], limit: float, h: float, measure) -> tuple[list[str], tuple[str, ...]]:
    """Each line broken at ASCII spaces into lines no wider than ``limit``
    (a no-break space U+00A0 keeps "18 mm" together), and the words wider
    than ``limit`` on their own, which stand alone on a line, never split."""
    out: list[str] = []
    overflow: list[str] = []
    for line in lines:
        current = ""
        for word in line.split(" "):
            if not word:
                continue
            if measure(word, h) > limit:
                overflow.append(word)
            trial = f"{current} {word}" if current else word
            if current and measure(trial, h) > limit:
                out.append(current)
                current = word
            else:
                current = trial
        out.append(current)
    return out, tuple(overflow)


def _terminator(style: str, path: list[Pt], h: float):
    """The terminator on ``path[0]``, pointing along the path's first segment,
    or None; refuses a first segment not longer than the terminator."""
    (tx, ty), (nx, ny) = path[0], path[1]
    leg = math.hypot(nx - tx, ny - ty)
    reach = {"arrow": ARROW_LENGTH * h, "dot": DOT_RADIUS * h}.get(style, 0.0)
    if leg <= reach:
        raise ValueError(
            f"the leader from target ({tx:g}, {ty:g}) runs {leg:g} to ({nx:g}, {ny:g}); it "
            f"must be longer than its {style} ({reach:g}). Move the next point away from the target."
        )
    if style == "none":
        return None
    ux, uy = (tx - nx) / leg, (ty - ny) / leg  # next point -> target
    if style == "arrow":
        bx, by = tx - ux * ARROW_LENGTH * h, ty - uy * ARROW_LENGTH * h
        px, py = -uy * ARROW_HALF_WIDTH * h, ux * ARROW_HALF_WIDTH * h
        return ((tx, ty), (bx + px, by + py), (bx - px, by - py))
    if style == "dot":
        return ((tx, ty), DOT_RADIUS * h)
    c = s45 = math.sqrt(0.5)
    vx, vy = ux * c - uy * s45, ux * s45 + uy * c
    half = TICK_LENGTH * h / 2.0
    return ((tx - vx * half, ty - vy * half), (tx + vx * half, ty + vy * half))


def plan_callout(
    target=None,
    landing=None,
    *,
    targets=None,
    leader=None,
    text: str | None = None,
    number: int | str | None = None,
    style: str = "dot",
    mask: bool = True,
    text_height: float = 2.5,
    side: str = "auto",
    max_width: float | None = None,
    shelf: str = "end",
    shelf_end: float | None = None,
    align: str = "auto",
    line_spacing: float = 1.0,
    measure=text_width,
) -> CalloutPlan:
    """Every vertex of one callout, or ``ValueError`` naming what is wrong.

    ``measure(text, height)`` is a line's width (default: the Standard face
    at width factor 1). ``max_width`` (drawing units, like ``text_height``)
    wraps a note at spaces into explicit MTEXT lines; the MTEXT keeps width 0,
    so AutoCAD's own font metrics never re-break it differently from the mask."""
    kind, body = _content(text, number)
    if style not in STYLES:
        raise ValueError(f"style must be one of {', '.join(STYLES)}; got {style!r}")
    if side not in SIDES:
        raise ValueError(f"side must be one of {', '.join(SIDES)}; got {side!r}")
    if shelf not in SHELVES:
        raise ValueError(f"shelf must be one of {', '.join(SHELVES)}; got {shelf!r}")
    if align not in ALIGNS:
        raise ValueError(f"align must be one of {', '.join(ALIGNS)}; got {align!r}")
    paths = _paths(target, landing, targets, leader)
    lx, ly = paths[0][-1]
    if kind == "number" and shelf != "end":
        raise ValueError(f"shelf {shelf!r} lines a text note; a numbered callout sits at the end")
    h = float(text_height)
    if not (math.isfinite(h) and h > 0):
        raise ValueError(f"text_height must be positive; got {text_height!r}")
    try:
        spacing = float(line_spacing)
    except (TypeError, ValueError):
        spacing = math.nan
    lo, hi = LINE_SPACING
    if not lo <= spacing <= hi:
        raise ValueError(f"line_spacing must be {lo:g} to {hi:g}; got {line_spacing!r}")
    if kind == "number" and spacing != 1.0:
        raise ValueError("line_spacing spaces a text note's lines; a numbered callout has one")
    if max_width is not None:
        if kind != "text":
            raise ValueError("max_width wraps a text note; a numbered callout has none")
        try:
            limit = float(max_width)
        except (TypeError, ValueError):
            limit = math.nan
        if not (math.isfinite(limit) and limit > 0):
            raise ValueError(f"max_width must be positive; got {max_width!r}")
        body, overflow = _wrap(body, limit, h, measure)
    else:
        overflow = ()
    terminators = tuple(_terminator(style, path, h) for path in paths)

    if shelf_end is not None:
        if side != "auto":
            raise ValueError("pass side or shelf_end, not both: shelf_end sets the side")
        try:
            end = float(shelf_end)
        except (TypeError, ValueError):
            end = math.nan
        if not math.isfinite(end) or abs(end - lx) < 1e-9:
            raise ValueError(
                f"shelf_end must be an x other than the knee's {lx:g}; got {shelf_end!r}"
            )
        d = 1.0 if end > lx else -1.0
    elif side == "auto":
        before = sum(path[-2][0] for path in paths) / len(paths)
        d = 1.0 if lx >= before else -1.0
    else:
        d = 1.0 if side == "right" else -1.0
    margin = GAP * h
    lined = kind == "text" and shelf != "end"
    if kind == "text":
        lines = body
        width = max(measure(line, h) for line in lines)
        block = h * (1.0 + (len(lines) - 1) * LINE_PITCH * spacing)
    if shelf_end is not None:
        run = abs(end - lx)
        if lined and run < width + 2 * margin - 1e-9:
            raise ValueError(
                f"shelf_end {end:g} leaves a {run:g} shelf; the note needs {width + 2 * margin:g} "
                f"(its width and a {margin:g} gap each side). Move shelf_end out or wrap it (max_width)."
            )
    else:
        run = width + 2 * margin if lined else SHELF * h
    sx, sy = lx + d * run, ly

    circle = None
    if kind == "text":
        content = "\\P".join(lines)
        if lined:
            at_end = shelf_end is not None
            hand = align if align != "auto" else ("right" if (d > 0) == at_end else "left")
            x0, x1 = sorted((lx, sx))
            anchor_x = x0 + margin if hand == "left" else x1 - margin
            nx0 = anchor_x if hand == "left" else anchor_x - width
            bx0, bx1 = nx0 - margin, nx0 + width + margin
            if shelf == "under":
                y0, y1, content_at, row = (
                    sy,
                    sy + block + 2 * margin,
                    (anchor_x, sy + margin),
                    "bottom",
                )
            else:
                y0, y1, content_at, row = (
                    sy - block - 2 * margin,
                    sy,
                    (anchor_x, sy - margin),
                    "top",
                )
        else:
            hand = align if align != "auto" else ("left" if d > 0 else "right")
            tx0 = sx + margin if d > 0 else sx - margin - width
            bx0, bx1 = (sx, tx0 + width + margin) if d > 0 else (tx0 - margin, sx)
            y0, y1 = sy - block / 2.0 - margin, sy + block / 2.0 + margin
            content_at = (tx0 if hand == "left" else tx0 + width, sy)
            row = "middle"
        content_align = f"{row}_{hand}"
        outline = ((bx0, y0), (bx1, y0), (bx1, y1), (bx0, y1))
    else:
        content = body
        radius = max(h, 0.5 * measure(content, h) + 0.4 * h)
        centre = (sx + d * radius, sy)
        circle = (centre, radius)
        content_at = centre
        content_align = "middle_center"
        # circumscribed, so the circle's own line lies inside the mask
        reach_out = radius / math.cos(math.pi / MASK_SIDES)
        outline = tuple(
            (
                centre[0] + reach_out * math.cos(2.0 * math.pi * i / MASK_SIDES),
                centre[1] + reach_out * math.sin(2.0 * math.pi * i / MASK_SIDES),
            )
            for i in range(MASK_SIDES)
        )

    return CalloutPlan(
        kind=kind,
        style=style,
        side="right" if d > 0 else "left",
        height=h,
        leader=(*paths[0], (sx, sy)),
        branches=tuple((path[0], path[-1]) for path in paths[1:]),
        terminators=terminators,
        content=content,
        content_at=content_at,
        content_align=content_align,
        circle=circle,
        mask=outline if mask else None,
        box=outline,
        line_spacing=spacing,
        overflow=overflow,
    )
