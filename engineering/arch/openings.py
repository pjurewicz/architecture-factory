"""Opening symbols in plan, hosted in their wall.

The wall engine (`walls.py`) cuts the gap; this module draws what stands in it.
Both read the opening's place from the same numbers - the wall's axis, its face
offsets and ``model.segment_of`` - so the symbol cannot drift off its gap. When
the engine's own cut row is given (``WallGeometry.cuts``), the glazed and
cased kinds take their ends from it, which is how a corner window's glazing
runs to the mitre line its neighbour's glazing meets.

Conventions (this repository's, stated because plan symbols vary by office):

* ``swing="in"`` opens to the **left** of the wall's axis, looking along it
  from its first point; ``"out"`` to the right. A perimeter drawn
  counter-clockwise has its inside on the left, so "in" means into the building.
* ``hand`` names the jamb that carries the hinges, seen by a person standing on
  the side the door swings towards and facing the wall: ``"left"`` hinges on
  their left-hand jamb.
* ``door`` is drawn **open at 90°**: the leaf is a line as long as the opening
  is wide, square to the wall from the hinge, on the swing-side face; the swing
  is a quarter arc about the hinge from the closed position to the open one.
* ``double_door``: two such leaves, each half the width, hinged on both jambs
  and meeting in the middle; ``hand`` does not apply.
* ``sliding_door``: two panels inside the wall thickness, at 35 % and 65 % of
  it, each half the width plus ``SLIDING_OVERLAP`` so they overlap in the
  middle; ``swing`` and ``hand`` do not apply.
* ``garage_door``: the door panel on the swing-side face across the opening;
  with a ``height`` the open sectional door is shown above the cut plane as a
  dashed outline that deep into the swing side (an overhead door stores its
  panels about its own height behind the opening). No height, no outline.
* ``cased_opening``: no leaf - the head above the cut plane, dashed on both
  faces across the gap.
* ``window``: a frame line on each wall face across the opening and two glazing
  lines between them at the thirds of the wall thickness.
* ``curtain_wall``: two glazing lines at 40 % and 60 % of the thickness and a
  mullion across the full thickness at both ends and between panels no wider
  than ``CURTAIN_PANEL_MAX``.

The overlap, the panel width and the fractions are drafting conventions of this
module, not standards. The first primitive is the one `draw.py` writes the
opening record on.
"""

from __future__ import annotations

import math

from engineering.arch.model import OPENING_KINDS, Opening, Wall, segment_of
from engineering.arch.walls import face_offsets, wall_segments
from engineering.mech.primitives import Arc, Line, Prim, Pt

#: How far the two panels of a sliding door overlap in the middle (mm).
SLIDING_OVERLAP = 50.0
#: The widest panel of a curtain wall between two mullions (mm).
CURTAIN_PANEL_MAX = 1500.0


def _deg(v: Pt) -> float:
    return math.degrees(math.atan2(v[1], v[0])) % 360.0


def _lerp(p: Pt, q: Pt, k: float) -> Pt:
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
    width = float(opening.width)
    index, along = segment_of(wall, float(opening.offset))
    (ax, ay), (bx, by) = wall_segments(wall)[index]
    length = math.hypot(bx - ax, by - ay)
    u = ((bx - ax) / length, (by - ay) / length)
    n = (-u[1], u[0])
    left, right = face_offsets(wall)

    def at(s: float, offset: float) -> Pt:
        return (ax + u[0] * s + n[0] * offset, ay + u[1] * s + n[1] * offset)

    s0, s1 = along, along + width
    if cut is not None:
        (la, lb), (ra, rb) = cut["left"], cut["right"]
    else:
        la, lb, ra, rb = at(s0, left), at(s1, left), at(s0, right), at(s1, right)

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
            for i in range(panels + 1)
        )
        return (across(0.6), across(0.4), *mullions)
    if kind == "cased_opening":
        return (Line(la, lb, "overhead"), Line(ra, rb, "overhead"))

    side = 1.0 if opening.swing == "in" else -1.0
    face = left if side > 0 else right
    if kind == "sliding_door":
        middle = (s0 + s1) / 2.0
        overlap = min(SLIDING_OVERLAP, width / 4.0)
        k_out, k_in = right + 0.65 * (left - right), right + 0.35 * (left - right)
        return (
            Line(at(s0, k_out), at(middle + overlap, k_out), "door"),
            Line(at(middle - overlap, k_in), at(s1, k_in), "door"),
        )
    if kind == "garage_door":
        prims: list[Prim] = [Line(at(s0, face), at(s1, face), "door")]
        if opening.height:
            depth = side * float(opening.height)
            far0, far1 = at(s0, face + depth), at(s1, face + depth)
            prims += [
                Line(at(s0, face), far0, "overhead"),
                Line(far0, far1, "overhead"),
                Line(far1, at(s1, face), "overhead"),
            ]
        return tuple(prims)
    if kind == "double_door":
        middle = (s0 + s1) / 2.0
        return (
            *_leaf(at(s0, face), at(middle, face), n, side, width / 2.0),
            *_leaf(at(s1, face), at(middle, face), n, side, width / 2.0),
        )

    # Facing the wall from the swing side, the viewer's left hand points along
    # +u when they stand on the axis's left, and along -u when on its right.
    toward = side if opening.hand == "left" else -side
    hinge_s, free_s = (s1, s0) if toward > 0 else (s0, s1)
    return _leaf(at(hinge_s, face), at(free_s, face), n, side, width)
