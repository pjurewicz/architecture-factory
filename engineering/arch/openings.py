"""Opening symbols in plan, hosted in their wall.

The wall engine (`walls.py`) cuts the gap; this module draws what stands in it.
When the engine's own cut row is given (``WallGeometry.cuts``), every symbol is
built from its four corners - the very vertices the jambs are drawn from - so a
door hinges exactly on its jamb even where the engine moved the axis (a T-stem
end snapped onto its host's axis), and a corner window's glazing runs to the
mitre line its neighbour's glazing meets. Without a cut row the corners are
computed from the wall's axis, its face offsets and ``model.segment_of``.

Conventions (this repository's, stated because plan symbols vary by office):

* The wall's normal rule: the **left** of a segment is the side on your left
  as you walk along it from the axis point before it to the one after.
  ``swing="in"`` opens to the left, ``"out"`` to the right; a perimeter drawn
  counter-clockwise has its inside on the left, so "in" means into the building.
  ``swing_towards=[x, y]`` names the side by a WCS point instead and does not
  depend on the axis order: the side of the hosting segment's line the point
  is on. A point on or between the wall's faces says no side and is refused,
  and so is ``swing`` beside it.
* ``hinge`` names the jamb that carries the hinges by the axis: ``"start"`` is
  the jamb at ``offset`` (nearer the axis's first point), ``"end"`` the jamb at
  ``offset + width``. Reversing an axis re-measures ``offset`` from the other
  end, so the same physical door then has the other ``hinge`` value.
* ``hand`` names the same jamb as seen by a person standing on the side the
  door swings towards and facing the wall: ``"left"`` hinges on their left-hand
  jamb. ``hinge`` and ``hand`` are one choice (give one); ``resolve_door``
  derives each from the other, and the record stores both.
* ``hinge`` is for ``door`` only (a double door hinges on both jambs, a sliding
  or garage door on neither); ``swing_towards`` for ``door``, ``double_door``
  and ``garage_door`` (where its panels are stored). Other kinds refuse them.
* ``width`` is the **structural opening** (otwór w murze): the gap the wall
  engine cuts, the length the dimension chains measure and the width the
  schedule lists. A frame stands inside it, so the leaf and the clear passage
  are narrower: with the defaults below a 900 opening has a 830 leaf and a 800
  clear passage (a Polish "80" door).
* **Frame** (ościeżnica), ``door`` / ``double_door`` / ``sliding_door``, on by
  default (``frame=False`` drops it): one closed polyline in each jamb, across
  the **full wall thickness** face to face (a wrap-around frame), so it sits
  the same in a left-, centre- or right-justified wall - it is built from the
  cut's corners, not the axis. ``FRAME_FACE`` wide along the wall; a hinged
  kind's frame has a rebate (przylga) ``FRAME_REBATE`` wide and
  ``FRAME_REBATE_DEPTH`` deep on the swing-side face, which the leaf closes
  into: the hinge is the rebate's vertex on that face. A sliding door's frame
  is plain (nothing closes into it). Drawn at every plot scale: the leaf and
  the passage must not change with the plot.
* ``door`` is drawn **open at 90°**: the leaf is a line from the hinge, square
  to the wall on the swing-side face, as long as the gap between the two
  frames' rebate vertices (the whole width without a frame); the swing is a
  quarter arc about the hinge from the closed position (the other frame's
  vertex) to the open one.
* ``double_door``: two such leaves, each half that length, hinged on both
  jambs and meeting in the middle; ``hand`` does not apply.
* ``sliding_door``: two panels inside the wall thickness, at 35 % and 65 % of
  it, each half the width between the frames plus ``SLIDING_OVERLAP`` so they
  overlap in the middle; ``swing`` and ``hand`` do not apply.
* ``garage_door``: no frame - a sectional door hangs on its own rails behind
  the opening. The door panel on the swing-side face across the opening;
  with a ``height`` the open sectional door is shown above the cut plane as a
  dashed outline that deep into the swing side (an overhead door stores its
  panels about its own height behind the opening). No height, no outline.
* ``cased_opening``: no leaf and no frame - the head above the cut plane,
  dashed on both faces across the gap.
* ``window``: a frame line on each wall face across the opening and two glazing
  lines between them at the thirds of the wall thickness.
* ``curtain_wall``: two glazing lines at 40 % and 60 % of the thickness and a
  mullion across the full thickness between panels no wider than
  ``CURTAIN_PANEL_MAX``. Its ends get no mullion: the wall engine closes them
  with a jamb (or, at a corner, a jamb along the mitre line or nothing), and a
  mullion there would be the same line drawn twice.

The overlap, the panel width, the fractions and the frame's sizes are drafting
conventions of this module, not standards. The first primitive is the one `draw.py` writes the
opening record on.
"""

from __future__ import annotations

import math
from dataclasses import replace

from engineering.arch.model import (
    EPS,
    FRAMED_KINDS,
    OPENING_KINDS,
    SIDED_KINDS,
    Opening,
    OpeningArgsError,
    Wall,
    segment_of,
)
from engineering.arch.walls import face_offsets, wall_segments
from engineering.mech.primitives import Arc, Line, Poly, Prim, Pt

#: How far the two panels of a sliding door overlap in the middle (mm).
SLIDING_OVERLAP = 50.0
#: The widest panel of a curtain wall between two mullions (mm).
CURTAIN_PANEL_MAX = 1500.0
#: A door frame's width along the wall, jamb to clear passage (mm).
FRAME_FACE = 50.0
#: How far the leaf laps onto the frame's stop, i.e. the rebate's width (mm).
FRAME_REBATE = 15.0
#: The rebate's depth from the swing-side face - the leaf's thickness (mm).
FRAME_REBATE_DEPTH = 40.0


def _deg(v: Pt) -> float:
    return math.degrees(math.atan2(v[1], v[0])) % 360.0


def _unit(p: Pt, q: Pt) -> Pt:
    length = math.hypot(q[0] - p[0], q[1] - p[1])
    return ((q[0] - p[0]) / length, (q[1] - p[1]) / length)


def _lerp(p: Pt, q: Pt, k: float) -> Pt:
    """The point at fraction ``k`` from ``p`` to ``q``: ``p`` and ``q`` themselves,
    bit for bit, at 0 and 1, so a symbol's end is the jamb's own vertex."""
    if k == 0.0:
        return p
    if k == 1.0:
        return q
    return (p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k)


def _leaf(hinge: Pt, free: Pt, n: Pt, side: float, length: float) -> tuple[Prim, Prim]:
    """A leaf open at 90° from ``hinge`` and its quarter-arc swing from ``free``."""
    opened = (hinge[0] + n[0] * side * length, hinge[1] + n[1] * side * length)
    a_closed = _deg((free[0] - hinge[0], free[1] - hinge[1]))
    a_open = _deg((opened[0] - hinge[0], opened[1] - hinge[1]))
    if abs(((a_open - a_closed) % 360.0) - 90.0) < 1e-6:
        swing = Arc(hinge, length, a_closed, a_open, "door")
    else:
        swing = Arc(hinge, length, a_open, a_closed, "door")
    return (Line(hinge, opened, "door"), swing)


def resolve_door(wall: Wall, opening: Opening, path: str | None = None) -> Opening:
    """``opening`` as its record stores it: ``swing`` resolved from
    ``swing_towards`` (then cleared), and ``hinge`` and ``hand`` from each other.

    The side is the sign of the point's offset from the hosting segment's axis
    line; a point on or between the faces is refused by name.
    """
    where = path or f"opening {opening.id}"
    if opening.kind not in SIDED_KINDS:
        if opening.swing_towards is not None or opening.hinge is not None:
            raise OpeningArgsError(f"{where}: a {opening.kind} takes no hinge or swing_towards")
        return opening
    swing = opening.swing
    if opening.swing_towards is not None:
        index, _along = segment_of(wall, float(opening.offset))
        (ax, ay), (bx, by) = wall_segments(wall)[index]
        ux, uy = _unit((ax, ay), (bx, by))
        px, py = (float(v) for v in opening.swing_towards)
        s = ux * (py - ay) - uy * (px - ax)
        left, right = face_offsets(wall)
        if s > left + EPS:
            swing = "in"
        elif s < right - EPS:
            swing = "out"
        else:
            raise OpeningArgsError(
                f"{where}.swing_towards: ({px:g}, {py:g}) is {s:g} mm off wall {wall.id}'s "
                f"axis, not clear of its faces ({right:g} to {left:g}); name a point beyond "
                "the face the door opens to"
            )
    if opening.kind != "door":
        return replace(opening, swing=swing, swing_towards=None)
    side = 1 if swing == "in" else -1
    # Facing the wall from the swing side, the viewer's left hand points along
    # +u when they stand on the axis's left, and along -u when on its right.
    if opening.hinge is not None:
        hinge = opening.hinge
        toward = 1 if hinge == "end" else -1
        hand = "left" if toward == side else "right"
    else:
        hand = opening.hand
        toward = side if hand == "left" else -side
        hinge = "end" if toward > 0 else "start"
    return replace(opening, swing=swing, hand=hand, hinge=hinge, swing_towards=None)


def opening_prims(wall: Wall, opening: Opening, cut: dict | None = None) -> tuple[Prim, ...]:
    """The symbol of ``opening`` in WCS - see the module docstring for each kind.

    ``cut`` is the opening's row of ``WallGeometry.cuts``; without it the ends
    are the jambs square to the axis, which is where the engine cuts every
    opening that does not run into a corner.
    """
    if opening.wall != wall.id:
        raise ValueError(
            f"opening {opening.id!r} is hosted in wall {opening.wall!r}, not in {wall.id!r}"
        )
    if opening.kind not in OPENING_KINDS:
        raise ValueError(
            f"opening {opening.id!r}: kind {opening.kind!r}; kinds are {', '.join(OPENING_KINDS)}"
        )
    if opening.swing not in ("in", "out"):
        raise ValueError(f"opening {opening.id!r}: swing {opening.swing!r}; swings are in, out")
    if opening.hand not in ("left", "right"):
        raise ValueError(f"opening {opening.id!r}: hand {opening.hand!r}; hands are left, right")
    opening = resolve_door(wall, opening)
    width = float(opening.width)
    if cut is not None:
        (la, lb), (ra, rb) = cut["left"], cut["right"]
    else:
        index, along = segment_of(wall, float(opening.offset))
        (ax, ay), (bx, by) = wall_segments(wall)[index]
        length = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / length, (by - ay) / length
        left, right = face_offsets(wall)
        la, lb, ra, rb = (
            (ax + ux * s - uy * offset, ay + uy * s + ux * offset)
            for s, offset in (
                (along, left),
                (along + width, left),
                (along, right),
                (along + width, right),
            )
        )
    # From here on every point is read off the gap's four corners - the vertices
    # the engine closes the jambs with - so no symbol can miss its jamb, whatever
    # the wall engine did to the axis at a junction (a T-stem snapped onto its host).
    u = _unit(ra, rb)
    n = (-u[1], u[0])

    def inside(t: float, k: float) -> Pt:
        """The point at fraction ``t`` of the width (0 = the near jamb) and ``k`` of
        the thickness (0 = the right face)."""
        return _lerp(_lerp(ra, rb, t), _lerp(la, lb, t), k)

    def across(k: float) -> Line:
        """The line across the gap at fraction ``k`` of the thickness, right face = 0."""
        return Line(_lerp(ra, la, k), _lerp(rb, lb, k), "window")

    kind = opening.kind
    if kind == "window":
        return (Line(la, lb, "window"), Line(ra, rb, "window"), across(1 / 3), across(2 / 3))
    if kind == "curtain_wall":
        panels = max(1, math.ceil(width / CURTAIN_PANEL_MAX - 1e-9))
        mullions = tuple(
            Line(_lerp(ra, rb, i / panels), _lerp(la, lb, i / panels), "window")
            for i in range(1, panels)
        )
        return (across(0.6), across(0.4), *mullions)
    if kind == "cased_opening":
        return (Line(la, lb, "overhead"), Line(ra, rb, "overhead"))
    side = 1.0 if opening.swing == "in" else -1.0
    # the swing-side face across the gap, from the near jamb to the far one
    near, far = (la, lb) if side > 0 else (ra, rb)
    back_near, back_far = (ra, rb) if side > 0 else (la, lb)
    framed = kind in FRAMED_KINDS and opening.frame is not False
    face = FRAME_FACE if framed else 0.0
    lap = FRAME_REBATE if framed and kind != "sliding_door" else 0.0
    if framed and width <= 2.0 * face:
        raise OpeningArgsError(
            f"opening {opening.id}: width {width:g} mm leaves no passage inside a "
            f"{face:g} mm frame on each jamb; widen it or give frame false"
        )

    def at(a: float, k: float, *, mirrored: bool = False) -> Pt:
        """The point ``a`` mm along the gap from the near jamb (from the far one
        when ``mirrored``), at fraction ``k`` of the thickness from the swing-side face."""
        t = (width - a if mirrored else a) / width
        return _lerp(_lerp(near, far, t), _lerp(back_near, back_far, t), k)

    def frame(mirrored: bool) -> Poly:
        """One jamb's frame across the full thickness: the stop ``face`` wide, the
        rebate cut out of it on the swing-side face; plain when ``lap`` is 0."""
        thick = math.dist(far, back_far) if mirrored else math.dist(near, back_near)
        k = min(FRAME_REBATE_DEPTH, thick / 2.0) / thick
        rebate = ((face - lap, 0.0), (face - lap, k), (face, k)) if lap else ((face, 0.0),)
        outline = ((0.0, 0.0), *rebate, (face, 1.0), (0.0, 1.0))
        return Poly(tuple(at(a, k, mirrored=mirrored) for a, k in outline), True, "door")

    frames: tuple[Prim, ...] = (frame(False), frame(True)) if framed else ()
    if kind == "sliding_door":
        clear = width - 2.0 * face
        overlap = min(SLIDING_OVERLAP, clear / 4.0)
        a0, a1 = face, width - face
        mid = width / 2.0
        return (
            Line(inside(a0 / width, 0.65), inside((mid + overlap) / width, 0.65), "door"),
            Line(inside((mid - overlap) / width, 0.35), inside(a1 / width, 0.35), "door"),
            *frames,
        )
    if kind == "garage_door":
        prims: list[Prim] = [Line(near, far, "door")]
        if opening.height:
            depth = side * float(opening.height)
            far0 = (near[0] + n[0] * depth, near[1] + n[1] * depth)
            far1 = (far[0] + n[0] * depth, far[1] + n[1] * depth)
            prims += [
                Line(near, far0, "overhead"),
                Line(far0, far1, "overhead"),
                Line(far1, far, "overhead"),
            ]
        return tuple(prims)
    # the leaf hangs on the rebate's vertex on the swing-side face: the very
    # vertex its frame polyline is drawn through (the jamb's own without a frame)
    hinge_near, hinge_far = at(face - lap, 0.0), at(face - lap, 0.0, mirrored=True)
    leaf = width - 2.0 * (face - lap)
    if kind == "double_door":
        middle = at(width / 2.0, 0.0)
        return (
            *_leaf(hinge_near, middle, n, side, leaf / 2.0),
            *_leaf(hinge_far, middle, n, side, leaf / 2.0),
            *frames,
        )

    hinge, free = (hinge_far, hinge_near) if opening.hinge == "end" else (hinge_near, hinge_far)
    return (*_leaf(hinge, free, n, side, leaf), *frames)
