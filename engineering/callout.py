"""Geometry of one technical callout: a terminator on the target, a leader with
a knee and a horizontal shelf, and a note or a numbered circle on the shelf,
optionally over a background mask.

Pure: `plan_callout` validates every argument and computes every vertex
before the backend draws anything, so a refusal leaves the drawing untouched.

Layout, in multiples of the text height ``h`` (AutoCAD's MLEADER defaults at
4 mm text: an 8 mm dogleg and a 2 mm landing gap):

* the shelf runs ``SHELF`` from the landing, away from the target's side
  (``side="auto"``), or to the ``side`` the caller names; ``shelf="under"``
  runs it under the note instead, as long as the note, the note on top;
* the terminator is none, a 45-degree tick, a filled dot or a filled arrow;
* a note starts ``GAP`` past the shelf end, its middle on the shelf;
* a number sits in a circle tangent to the shelf end, the numeral ``h`` high;
* the mask is a WIPEOUT: a rectangle from the shelf end to ``GAP`` past the
  note, or a polygon circumscribing the circle. It is drawn before the leader
  and the lettering, so it hides what lies under the callout and nothing of
  the callout itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engineering.annotation import LINE_PITCH, text_width

STYLES = ("none", "tick", "dot", "arrow")
SIDES = ("auto", "left", "right")
#: ``end``: the note past the shelf's end; ``under``: the shelf under the note,
#: as long as it (round 7, the architect's underlined label).
SHELVES = ("end", "under")
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

Pt = tuple[float, float]


@dataclass(frozen=True)
class CalloutPlan:
    kind: str  # "text" | "number"
    style: str  # STYLES
    side: str  # "right" | "left": where the content sits from the landing
    height: float
    leader: tuple[Pt, ...]  # target, landing, shelf end
    arrow: tuple[Pt, ...] | None  # filled triangle, tip first
    dot: tuple[Pt, float] | None  # centre, radius
    tick: tuple[Pt, Pt] | None
    content: str  # MTEXT content (lines joined by \P) or the numeral
    content_at: Pt
    content_align: str  # MTEXT attachment / TEXT alignment name
    circle: tuple[Pt, float] | None  # the numbered circle
    mask: tuple[Pt, ...] | None
    box: tuple[Pt, ...]  # what the content covers (the mask's outline), mask or not
    overflow: tuple[str, ...] = ()  # words wider than max_width


def _point(value, name: str) -> Pt:
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"{name} must be an [x, y] pair; got {value!r}") from None
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return x, y


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


def plan_callout(
    target,
    landing,
    *,
    text: str | None = None,
    number: int | str | None = None,
    style: str = "dot",
    mask: bool = True,
    text_height: float = 2.5,
    side: str = "auto",
    max_width: float | None = None,
    shelf: str = "end",
    align: str = "auto",
    measure=text_width,
) -> CalloutPlan:
    """Every vertex of one callout, or ``ValueError`` naming what is wrong.

    ``measure(text, height)`` is a line's width (default: the Standard face at
    width factor 1); the backend passes the note's text style's
    (`engineering.annotation.style_text_width`), the box `text_overlap` reads.

    ``max_width`` (drawing units, like ``text_height``) wraps a note at spaces
    into explicit MTEXT lines; the MTEXT keeps width 0, so AutoCAD's own font
    metrics never re-break it differently from the mask."""
    tx, ty = _point(target, "target")
    lx, ly = _point(landing, "landing")
    kind, body = _content(text, number)
    if style not in STYLES:
        raise ValueError(f"style must be one of {', '.join(STYLES)}; got {style!r}")
    if side not in SIDES:
        raise ValueError(f"side must be one of {', '.join(SIDES)}; got {side!r}")
    if shelf not in SHELVES:
        raise ValueError(f"shelf must be one of {', '.join(SHELVES)}; got {shelf!r}")
    if align not in ALIGNS:
        raise ValueError(f"align must be one of {', '.join(ALIGNS)}; got {align!r}")
    if kind == "number" and shelf == "under":
        raise ValueError("shelf 'under' underlines a text note; a numbered callout sits at the end")
    h = float(text_height)
    if not (math.isfinite(h) and h > 0):
        raise ValueError(f"text_height must be positive; got {text_height!r}")
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

    leg = math.hypot(lx - tx, ly - ty)
    reach = {"arrow": ARROW_LENGTH * h, "dot": DOT_RADIUS * h}.get(style, 0.0)
    if leg <= reach:
        raise ValueError(
            f"landing ({lx:g}, {ly:g}) is {leg:g} from target ({tx:g}, {ty:g}); the leader "
            f"must be longer than its {style} ({reach:g}). Move the landing away from the target."
        )

    if side == "auto":
        d = 1.0 if lx >= tx else -1.0
    else:
        d = 1.0 if side == "right" else -1.0
    margin = GAP * h
    if kind == "text":
        lines = body
        width = max(measure(line, h) for line in lines)
        block = h * (1.0 + (len(lines) - 1) * LINE_PITCH)
    run = width + 2 * margin if kind == "text" and shelf == "under" else SHELF * h
    sx, sy = lx + d * run, ly
    ux, uy = (tx - lx) / leg, (ty - ly) / leg  # landing -> target
    arrow = dot = tick = None
    if style == "arrow":
        bx, by = tx - ux * ARROW_LENGTH * h, ty - uy * ARROW_LENGTH * h
        px, py = -uy * ARROW_HALF_WIDTH * h, ux * ARROW_HALF_WIDTH * h
        arrow = ((tx, ty), (bx + px, by + py), (bx - px, by - py))
    elif style == "dot":
        dot = ((tx, ty), DOT_RADIUS * h)
    elif style == "tick":
        c, s45 = math.cos(math.pi / 4), math.sin(math.pi / 4)
        vx, vy = ux * c - uy * s45, ux * s45 + uy * c
        half = TICK_LENGTH * h / 2.0
        tick = ((tx - vx * half, ty - vy * half), (tx + vx * half, ty + vy * half))

    circle = None
    if kind == "text":
        if shelf == "under":
            x0, x1 = sorted((lx, sx))
            tx0, y0, y1 = x0 + margin, sy, sy + block + 2 * margin
        else:
            tx0 = sx + margin if d > 0 else sx - margin - width
            x0, x1 = (sx, tx0 + width + margin) if d > 0 else (tx0 - margin, sx)
            y0, y1 = sy - block / 2.0 - margin, sy + block / 2.0 + margin
        hand = align if align != "auto" else ("left" if d > 0 else "right")
        content = "\\P".join(lines)
        anchor_x = tx0 if hand == "left" else tx0 + width
        content_at = (anchor_x, sy + margin if shelf == "under" else sy)
        align = f"{'bottom' if shelf == 'under' else 'middle'}_{hand}"
        outline = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    else:
        content = body
        radius = max(h, 0.5 * measure(content, h) + 0.4 * h)
        centre = (sx + d * radius, sy)
        circle = (centre, radius)
        content_at = centre
        align = "middle_center"
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
        leader=((tx, ty), (lx, ly), (sx, sy)),
        arrow=arrow,
        dot=dot,
        tick=tick,
        content=content,
        content_at=content_at,
        content_align=align,
        circle=circle,
        mask=outline if mask else None,
        box=outline,
        overflow=overflow,
    )
