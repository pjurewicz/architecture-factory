"""The wall engine: axis polylines in, clean wall outlines out.

A wall is an axis polyline with a thickness and a justification - the side of
the axis the wall lies on, looking along it from its first point. This module
offsets every axis segment into its two faces and resolves every place where
segments meet:

* **L and straight joins** - two segment ends at one point. Each face is met by
  the face of its neighbour and both are trimmed or extended to their
  intersection, so the corner is a clean mitre. A near-straight join whose
  faces are offset from each other (a thickness or justification change, or a
  rounding kink between two such walls) would mitre kilometres away; past
  ``_MITRE_REACH`` thicknesses it steps instead: both faces stop on the line
  bisecting the join and a short face closes the step. Two free ends that
  meet end to end, parallel, with their axes up to half the thicker wall apart
  (a thickness change with one face flush) are joined at one node and step
  the same way (`_resolve_steps`).
* **T** - a segment ends *inside* another wall. Its axis is carried to the
  crossing axis, the crossing segment is split there, and the same rule stops
  the stem's faces on the crossing wall's near face. That face is cut where
  the stem meets it; the far face stays one line.
* **X** - two axes cross in their interiors. Both are split and all four
  faces are cut.

All three are one algorithm: at every node the segment ends are taken as
outgoing directions and sorted by angle, and the left face of each end meets
the right face of the next one counter-clockwise. A node of three or more ends
leaves a *hub* - the square an X crosses in, the triangle a centred T leaves -
that no single segment covers; it is a region of its own, so the poche has no
hole in it. The hub belongs to the wall that runs straight through the node
(the crossing wall of a T; the first of the two at an X), and a wall of another
material ending on it gets a *seam* - a line across its end, on the crossing
wall's near face - so two poches never meet without a line between them.

An opening removes its width from both faces and each edge is closed by a jamb;
a free wall end is closed by a jamb too. A *corner opening* - a window, a
curtain wall or a cased opening whose span starts at offset 0 or ends at the
axis end, where the wall turns at an L - runs into the junction: its face lines
stop on the L's mitre line instead of a jamb, so a window on each of the two
walls meets its neighbour at the corner with no post between them. When only
one of the two walls is open there, the other one is closed by a jamb along
that mitre line. A corner at a T or an X (three or more ends at the node) is
not one wall corner and stays in ``omitted``. Every solid piece - one axis segment's
stretch between junctions, openings and free ends, plus each hub - is one CCW
region and, with ``poche=True``, one ``HatchArea`` in the wall's material.

Refused by name before anything is computed: two walls with one id, an unknown
material or justification, a zero-length segment, two segments that meet at
less than ``MIN_JOIN_ANGLE_DEG`` (the mitre would run away), an end two walls
would both continue as a step, a wall reaching into a step's cut, a wall corner that
lies inside another wall off its axis, an end inside a wall whose axis would
have to run past that wall's end to meet it, a segment that lies wholly
inside the wall it joins, and one too short for its junctions (its outline
would cross itself - `engineering.measure.is_self_intersecting`). An opening that cannot be cut because it runs into a
junction is returned in ``omitted``, never dropped.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from engineering.arch.materials import normalise_material, wall_hatch
from engineering.arch.model import CORNER_KINDS, Opening, Wall, segment_of, validate_openings
from engineering.measure import is_self_intersecting, polygon_area_perimeter
from engineering.mech.primitives import HatchArea, Line, Pt

#: The shallowest angle two wall axes may meet at. Below it the mitre point of
#: the two faces runs more than eleven wall thicknesses away from the node, so
#: the junction is refused rather than drawn as a spike. A drafting limit of
#: this engine, not a building rule.
MIN_JOIN_ANGLE_DEG = 5.0

#: How far from its node, in thicknesses of the thicker wall, the face mitre of
#: a near-straight join may lie. Two faces at equal offsets meet within 1.42
#: thicknesses of the node at any kink up to 90 degrees; faces at different
#: offsets (a thickness or justification change) meet at offset / sin(kink),
#: kilometres away for a rounding kink, so past this reach the join steps
#: instead of flaring.
_MITRE_REACH = 2.0

_EPS = 1e-9
#: Decimals a point is rounded to when face lines are matched end to end.
_KEY_DECIMALS = 6


@dataclass(frozen=True)
class WallGeometry:
    faces: tuple[Line, ...]
    jambs: tuple[Line, ...]
    regions: tuple[tuple[Pt, ...], ...]
    hatches: tuple[HatchArea, ...]
    omitted: tuple[dict, ...]
    #: Each opening's two face lines across its gap. Never drawn: they are what
    #: `room_segments` adds so a doorway does not merge two rooms into one face.
    thresholds: tuple[Line, ...] = ()
    #: One row per cut opening: ``{"element", "left": (p, q), "right": (p, q),
    #: "corner": ["start"|"end", ...]}`` - its gap on the wall's left and right
    #: face, from the near jamb to the far one (a corner end on the mitre line).
    #: The symbols and the dimension chains read the opening's ends from here.
    cuts: tuple[dict, ...] = ()
    #: Where a wall of one material ends on the hub of a T or an X owned by a
    #: wall of another: the line across its end that parts the two poches. It
    #: belongs to the wall whose end it closes. Drawn, but not a room boundary:
    #: it lies inside the wall solid.
    seams: tuple[Line, ...] = ()


# -- vectors ------------------------------------------------------------------


def _sub(p: Pt, q: Pt) -> Pt:
    return (p[0] - q[0], p[1] - q[1])


def _add(p: Pt, q: Pt) -> Pt:
    return (p[0] + q[0], p[1] + q[1])


def _mul(v: Pt, k: float) -> Pt:
    return (v[0] * k, v[1] * k)


def _dot(u: Pt, v: Pt) -> float:
    return u[0] * v[0] + u[1] * v[1]


def _cross(u: Pt, v: Pt) -> float:
    return u[0] * v[1] - u[1] * v[0]


def _left(u: Pt) -> Pt:
    return (-u[1], u[0])


def _unit(p: Pt, q: Pt) -> Pt:
    length = math.dist(p, q)
    return ((q[0] - p[0]) / length, (q[1] - p[1]) / length)


def _meet(p: Pt, u: Pt, q: Pt, v: Pt) -> Pt:
    """Where the line through ``p`` along ``u`` meets the line through ``q`` along ``v``."""
    t = _cross(_sub(q, p), v) / _cross(u, v)
    return _add(p, _mul(u, t))


def _heading(u: Pt) -> float:
    return math.degrees(math.atan2(u[1], u[0])) % 360.0


def _line_angle(u: Pt, v: Pt) -> float:
    """The angle between two lines (not directions), 0-90 degrees."""
    return math.degrees(math.asin(min(1.0, abs(_cross(u, v)))))


def _polygon(points: Sequence[Pt]) -> tuple[Pt, ...] | None:
    """Consecutive duplicates removed; None if nothing is enclosed.

    Every caller builds its ring counter-clockwise - a piece as right face out,
    left face back; a hub in the angular order of its corners - so the ring is
    not turned here; a ring that comes out clockwise is inside out, and the
    callers refuse it (`_counter_clockwise`). The area is
    `engineering.measure`'s, not a second shoelace.
    """
    kept: list[Pt] = []
    for p in points:
        if not kept or math.dist(kept[-1], p) > _EPS:
            kept.append(p)
    if len(kept) > 1 and math.dist(kept[0], kept[-1]) <= _EPS:
        kept.pop()
    if len(kept) < 3:
        return None
    area, _perimeter = polygon_area_perimeter([(x, y, 0.0) for x, y in kept], True)
    return tuple(kept) if area > _EPS else None


def _counter_clockwise(ring: Sequence[Pt]) -> bool:
    """The ring's turning sense - the sign `polygon_area_perimeter` folds away."""
    twice = sum(_cross(p, q) for p, q in zip(ring, (*ring[1:], ring[0]), strict=True))
    return twice > 0.0


def _key(p: Pt) -> tuple[float, float]:
    return (round(p[0], _KEY_DECIMALS) + 0.0, round(p[1], _KEY_DECIMALS) + 0.0)


# -- the wall's own numbers -----------------------------------------------------


def face_offsets(wall: Wall) -> tuple[float, float]:
    """``(left, right)``: the signed offsets of a wall's two faces along the axis's left normal."""
    thickness = float(wall.thickness)
    if wall.justification == "center":
        return (thickness / 2.0, -thickness / 2.0)
    if wall.justification == "left":
        return (thickness, 0.0)
    if wall.justification == "right":
        return (0.0, -thickness)
    raise ValueError(
        f"wall {wall.id!r}: justification {wall.justification!r}; "
        "justifications are center, left, right"
    )


def wall_segments(wall: Wall) -> tuple[tuple[Pt, Pt], ...]:
    """The wall's axis segments in order, the closing one last when ``closed``."""
    points = [(float(x), float(y)) for x, y in wall.axis]
    pairs = list(zip(points, points[1:], strict=False))
    if wall.closed:
        pairs.append((points[-1], points[0]))
    return tuple(pairs)


# -- the network ----------------------------------------------------------------


@dataclass
class _Seg:
    wall: int
    index: int
    va: tuple[int, int]
    vb: tuple[int, int]
    left: float
    right: float
    #: Where other axes stop on or cross this one, as points, never as distances:
    #: a distance is measured from the segment's start, and that start may still
    #: be carried onto a third wall later in the same pass.
    cuts: list[Pt] = field(default_factory=list)


@dataclass
class _Piece:
    seg: _Seg
    a: Pt
    b: Pt
    s0: float
    corners: dict = field(default_factory=dict)  # (end 0|1, "left"|"right") -> Pt
    nodes: dict = field(default_factory=dict)  # end 0|1 -> index of its node


def _validate(walls: Sequence[Wall], openings: Sequence[Opening], join_tol: float) -> None:
    if not isinstance(join_tol, (int, float)) or not math.isfinite(join_tol) or join_tol <= 0:
        raise ValueError(f"join_tol: must be a finite number above zero, got {join_tol!r}")
    seen: dict[str, int] = {}
    for i, wall in enumerate(walls):
        if not isinstance(wall, Wall):
            raise ValueError(f"walls[{i}]: expected a Wall, got {type(wall).__name__}")
        if wall.id in seen:
            raise ValueError(
                f"walls[{i}].id: {wall.id!r} is already used by walls[{seen[wall.id]}]"
            )
        seen[wall.id] = i
        try:
            wall_hatch(wall.material)
        except ValueError as exc:
            # wall_hatch's message already names its field; the path replaces it
            reason = str(exc).removeprefix("material: ")
            raise ValueError(f"walls[{i}].material: {reason}") from None
        try:
            face_offsets(wall)
        except ValueError as exc:
            raise ValueError(f"walls[{i}].justification: {exc}") from None
        segments = wall_segments(wall)
        if not segments:
            raise ValueError(f"walls[{i}].axis: needs at least two points")
        for k, (a, b) in enumerate(segments):
            if math.dist(a, b) <= _EPS:
                raise ValueError(f"walls[{i}].axis[{k}]: zero-length segment at {a}")
    validate_openings(list(walls), list(openings))


def _segments(walls: Sequence[Wall]) -> tuple[dict, list[_Seg]]:
    pos: dict[tuple[int, int], Pt] = {}
    segs: list[_Seg] = []
    for wi, wall in enumerate(walls):
        left, right = face_offsets(wall)
        count = len(wall.axis)
        for k, (x, y) in enumerate(wall.axis):
            pos[(wi, k)] = (float(x), float(y))
        for i in range(count if wall.closed else count - 1):
            segs.append(_Seg(wi, i, (wi, i), (wi, (i + 1) % count), left, right))
    return pos, segs


def _name(walls: Sequence[Wall], index: int) -> str:
    return f"walls[{index}] ({walls[index].id!r})"


def _refuse_shallow(walls, one: _Seg, other: _Seg, u: Pt, v: Pt) -> None:
    angle = _line_angle(u, v)
    if angle < MIN_JOIN_ANGLE_DEG:
        raise ValueError(
            f"{_name(walls, one.wall)} meets {_name(walls, other.wall)} at {angle:.1f}°; "
            f"a junction below {MIN_JOIN_ANGLE_DEG:g}° cannot be resolved into a clean "
            "outline - end the wall at a steeper angle."
        )


def _touching(segs) -> dict[tuple[int, int], list[_Seg]]:
    touching: dict[tuple[int, int], list[_Seg]] = {}
    for seg in segs:
        touching.setdefault(seg.va, []).append(seg)
        touching.setdefault(seg.vb, []).append(seg)
    return touching


def _ends_faces(pos, seg: _Seg, vertex) -> tuple[Pt, Pt]:
    """Where a segment's two faces cross its axis end at ``vertex``."""
    n = _left(_unit(pos[seg.va], pos[seg.vb]))
    return _add(pos[vertex], _mul(n, seg.left)), _add(pos[vertex], _mul(n, seg.right))


def _crosses_body(pos, seg: _Seg, p: Pt, q: Pt, tol: float) -> bool:
    """Whether the segment ``pq`` runs more than ``tol`` inside ``seg``'s wall body."""
    a, b = pos[seg.va], pos[seg.vb]
    u = _unit(a, b)
    n = _left(u)
    lo = [tol, seg.right + tol]
    hi = [math.dist(a, b) - tol, seg.left - tol]
    start = (_dot(_sub(p, a), u), _dot(_sub(p, a), n))
    delta = (_dot(_sub(q, p), u), _dot(_sub(q, p), n))
    t0, t1 = 0.0, 1.0
    for k in (0, 1):  # Liang-Barsky against the body shrunk by tol
        if abs(delta[k]) <= _EPS:
            if not lo[k] < start[k] < hi[k]:
                return False
            continue
        ta, tb = (lo[k] - start[k]) / delta[k], (hi[k] - start[k]) / delta[k]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 >= t1:
            return False
    return True


def _resolve_steps(walls, pos, segs, tol: float) -> dict[tuple[int, int], tuple[int, int]]:
    """Pair the free ends where one wall continues another with its axis offset sideways.

    Two free axis ends that meet end to end (within ``tol`` along the wall),
    their segments parallel (below ``MIN_JOIN_ANGLE_DEG``) and running apart,
    with their axes up to half the thicker wall apart: a thickness change
    (24 cm to 16 cm, one face flush) or a shifted axis. They join at one node
    and step there like a straight join whose faces do not line up. Ends
    further apart, along or across, stay two walls. Refused by name: an end
    two walls would both continue (a fork), and a third wall reaching into
    the step's cut. ``{vertex: partner vertex}``, both ways.
    """
    touching = _touching(segs)
    free = {vertex: mine[0] for vertex, mine in touching.items() if len(mine) == 1}
    found: list[tuple] = []
    for (v1, s1), (v2, s2) in itertools.combinations(free.items(), 2):
        if s1 is s2:
            continue
        p, q = pos[v1], pos[v2]
        if math.dist(p, q) <= tol:
            continue  # one node already
        u1 = _unit(pos[s1.vb] if s1.va == v1 else pos[s1.va], p)  # arriving at p
        u2 = _unit(q, pos[s2.vb] if s2.va == v2 else pos[s2.va])  # leaving q
        if _dot(u1, u2) <= 0.0 or _line_angle(u1, u2) >= MIN_JOIN_ANGLE_DEG:
            continue
        if abs(_dot(_sub(q, p), u1)) > tol:
            continue  # not end to end
        offset = abs(_cross(u1, _sub(q, p)))
        if offset > max(s1.left - s1.right, s2.left - s2.right) / 2.0 + tol:
            continue
        # and the two walls share some of their width there: faces that only
        # touch (a justified wall beside the other) are two walls, not a step
        n = _left(u1)
        (a1, b1), (a2, b2) = (
            sorted(_dot(_sub(face, p), n) for face in _ends_faces(pos, seg, vertex))
            for seg, vertex in ((s1, v1), (s2, v2))
        )
        if min(b1, b2) - max(a1, a2) > tol:
            found.append((v1, s1, v2, s2))
    steps: dict[tuple[int, int], tuple[int, int]] = {}
    for v1, s1, v2, s2 in found:
        for vertex in (v1, v2):
            rivals = [row for row in found if vertex in (row[0], row[2])]
            if len(rivals) > 1:
                p = pos[vertex]
                names = " and ".join(
                    _name(walls, (r[3] if r[0] == vertex else r[1]).wall) for r in rivals
                )
                raise ValueError(
                    f"{_name(walls, vertex[0])} ends at ({p[0]:g}, {p[1]:g}) where {names} "
                    "both continue it with their axes to the side; a step joins one wall to "
                    "one - end the others clear of it or join them at a corner."
                )
        # the cut across the step, face to outermost face: no third wall may reach into it
        p = pos[v1]
        n = _left(_unit(pos[s1.vb] if s1.va == v1 else pos[s1.va], p))
        across = [
            _dot(_sub(face, p), n)
            for seg, vertex in ((s1, v1), (s2, v2))
            for face in _ends_faces(pos, seg, vertex)
        ]
        lo, hi = _add(p, _mul(n, min(across))), _add(p, _mul(n, max(across)))
        for other in segs:
            if other is not s1 and other is not s2 and _crosses_body(pos, other, lo, hi, tol):
                raise ValueError(
                    f"{_name(walls, other.wall)} reaches into the step where "
                    f"{_name(walls, s2.wall)} continues {_name(walls, s1.wall)} at "
                    f"({p[0]:g}, {p[1]:g}); a wall cannot meet a step - end it clear of the "
                    "step, or align the two walls' axes."
                )
        steps[v1], steps[v2] = v2, v1
    return steps


def _resolve_tees(walls, pos, segs, tol: float, steps: dict) -> None:
    """Carry every axis end that stops inside another wall onto that wall's axis, and split it."""
    touching = _touching(segs)
    for vertex, mine in touching.items():
        for other in segs:
            if any(other is seg for seg in mine):
                continue
            if steps.get(vertex) in (other.va, other.vb):
                continue  # a step: joined end to end, not a T
            p = pos[vertex]
            a, b = pos[other.va], pos[other.vb]
            if math.dist(p, a) <= tol or math.dist(p, b) <= tol:
                continue
            length = math.dist(a, b)
            u = _unit(a, b)
            s = _dot(_sub(p, a), u)
            d = _dot(_sub(p, a), _left(u))
            if s < -tol or s > length + tol or d < other.right - tol or d > other.left + tol:
                continue
            for seg in mine:
                _refuse_shallow(walls, seg, other, _unit(pos[seg.va], pos[seg.vb]), u)
            if abs(d) <= tol:
                x = _add(a, _mul(u, s))
            elif len(mine) == 1:
                seg = mine[0]
                far = pos[seg.vb] if seg.va == vertex else pos[seg.va]
                x = _meet(far, _unit(far, p), a, u)
            else:
                raise ValueError(
                    f"the corner of {_name(walls, vertex[0])} at ({p[0]:g}, {p[1]:g}) lies inside "
                    f"{_name(walls, other.wall)} off its axis; end the wall there or move the corner "
                    "onto the other wall's axis."
                )
            sx = _dot(_sub(x, a), u)
            if sx < -tol or sx > length + tol:
                raise ValueError(
                    f"{_name(walls, vertex[0])} ends inside {_name(walls, other.wall)} at "
                    f"({p[0]:g}, {p[1]:g}), but its axis meets that wall's axis beyond its end; "
                    "join the two axes at a common point."
                )
            if sx <= tol:
                pos[vertex] = a
            elif sx >= length - tol:
                pos[vertex] = b
            else:
                pos[vertex] = x
                other.cuts.append(x)
            break


def _resolve_crossings(walls, pos, segs, tol: float) -> None:
    """Split both segments wherever two axes cross in their interiors (an X)."""
    for i, one in enumerate(segs):
        for other in segs[i + 1 :]:
            if {one.va, one.vb} & {other.va, other.vb}:
                continue
            a1, b1 = pos[one.va], pos[one.vb]
            a2, b2 = pos[other.va], pos[other.vb]
            r = _sub(b1, a1)
            q = _sub(b2, a2)
            denom = _cross(r, q)
            if abs(denom) <= _EPS:
                continue
            t = _cross(_sub(a2, a1), q) / denom
            w = _cross(_sub(a2, a1), r) / denom
            l1 = math.hypot(*r)
            l2 = math.hypot(*q)
            if tol < t * l1 < l1 - tol and tol < w * l2 < l2 - tol:
                _refuse_shallow(walls, one, other, _unit(a1, b1), _unit(a2, b2))
                crossing = _add(a1, _mul(r, t))
                one.cuts.append(crossing)
                other.cuts.append(crossing)


def _pieces(walls, pos, segs, tol: float) -> list[_Piece]:
    pieces: list[_Piece] = []
    for seg in segs:
        a, b = pos[seg.va], pos[seg.vb]
        length = math.dist(a, b)
        if length <= tol:
            raise ValueError(
                f"{_name(walls, seg.wall)}: segment {seg.index} lies inside the wall it joins - "
                "nothing of it stands clear of that wall; lengthen it or leave it out."
            )
        u = _unit(a, b)
        stops: list[float] = []
        for cut in sorted(_dot(_sub(point, a), u) for point in seg.cuts):
            if tol < cut < length - tol and (not stops or cut - stops[-1] > tol):
                stops.append(cut)
        bounds = [0.0, *stops, length]
        for s0, s1 in zip(bounds, bounds[1:], strict=False):
            start = a if s0 == 0.0 else _add(a, _mul(u, s0))
            end = b if s1 == length else _add(a, _mul(u, s1))
            pieces.append(_Piece(seg, start, end, s0))
    return pieces


def _outgoing(piece: _Piece, end: int):
    """(origin, direction, left normal, left offset, right offset) leaving the node at ``end``."""
    u = _unit(piece.a, piece.b)
    if end == 0:
        return piece.a, u, _left(u), piece.seg.left, piece.seg.right
    back = (-u[0], -u[1])
    return piece.b, back, _left(back), -piece.seg.right, -piece.seg.left


def _canonical(end: int, side: str) -> str:
    """Leaving through its end, a piece's left face is its canonical right one."""
    if end == 0:
        return side
    return "right" if side == "left" else "left"


def _thickness(piece: _Piece) -> float:
    return piece.seg.left - piece.seg.right


def _build(walls: Sequence[Wall], openings: Sequence[Opening], join_tol: float) -> dict:
    walls = list(walls)
    openings = list(openings)
    _validate(walls, openings, join_tol)
    pos, segs = _segments(walls)
    steps = _resolve_steps(walls, pos, segs, join_tol)
    _resolve_tees(walls, pos, segs, join_tol, steps)
    _resolve_crossings(walls, pos, segs, join_tol)
    pieces = _pieces(walls, pos, segs, join_tol)

    faces: list[tuple[Line, str]] = []
    jambs: list[tuple[Line, str]] = []
    thresholds: list[tuple[Line, str]] = []
    cuts: list[tuple[dict, str]] = []
    regions: list[tuple[tuple[Pt, ...], str]] = []
    seams: list[tuple[Line, str]] = []
    omitted: list[tuple[dict, str]] = []

    # -- the nodes ------------------------------------------------------------
    nodes: list[Pt] = []
    ends: list[list[tuple[_Piece, int]]] = []
    for piece in pieces:
        for end, point in ((0, piece.a), (1, piece.b)):
            for k, node in enumerate(nodes):
                if math.dist(point, node) <= join_tol:
                    ends[k].append((piece, end))
                    piece.nodes[end] = k
                    break
            else:
                piece.nodes[end] = len(nodes)
                nodes.append(point)
                ends.append([(piece, end)])
    # the two free ends of a step are one node; the step below draws it. The node
    # stays at the first end: both lie on the square cut through it.
    node_of = {_key(node): k for k, node in enumerate(nodes)}
    step_nodes: set[int] = set()
    for vertex, partner in steps.items():
        k1, k2 = node_of.get(_key(pos[vertex])), node_of.get(_key(pos[partner]))
        if vertex < partner and k1 is not None and k2 is not None:
            if len(ends[k1]) == len(ends[k2]) == 1:
                piece, end = ends[k2].pop()
                piece.nodes[end] = k1
                ends[k1].append((piece, end))
                step_nodes.add(k1)

    for node, members in zip(nodes, ends, strict=True):
        if not members:
            continue
        if len(members) == 1:
            piece, end = members[0]
            origin, _u, n, left, right = _outgoing(piece, end)
            pl = _add(origin, _mul(n, left))
            pr = _add(origin, _mul(n, right))
            piece.corners[(end, _canonical(end, "left"))] = pl
            piece.corners[(end, _canonical(end, "right"))] = pr
            jambs.append((Line(pl, pr, "wall"), walls[piece.seg.wall].id))
            continue
        ordered = sorted(members, key=lambda m: _heading(_outgoing(*m)[1]))
        pairs: list[list] = []
        for idx, (pi, ei) in enumerate(ordered):
            pj, ej = ordered[(idx + 1) % len(ordered)]
            oi, ui, ni, li, _ri = _outgoing(pi, ei)
            oj, uj, nj, _lj, rj = _outgoing(pj, ej)
            gap = (_heading(uj) - _heading(ui)) % 360.0
            if gap < MIN_JOIN_ANGLE_DEG:
                raise ValueError(
                    f"{_name(walls, pi.seg.wall)} and {_name(walls, pj.seg.wall)} meet at "
                    f"({node[0]:g}, {node[1]:g}) {gap:.1f}° apart; a junction below "
                    f"{MIN_JOIN_ANGLE_DEG:g}° cannot be resolved into a clean outline."
                )
            face_i = _add(oi, _mul(ni, li))
            face_j = _add(oj, _mul(nj, rj))
            mitre = None
            if abs(_cross(ui, uj)) > 1e-12:
                mitre = _meet(face_i, ui, face_j, uj)
                reach = _MITRE_REACH * max(_thickness(pi), _thickness(pj))
                if _dot(ui, uj) < 0.0 and math.dist(mitre, node) > reach:
                    mitre = None  # a near-straight join whose faces are offset: a step
            pairs.append([pi, ei, pj, ej, face_i, ui, face_j, uj, mitre])
        if len(pairs) == 2 and any(pair[8] is None for pair in pairs):
            # both sides of one straight-through join step on the same cut line,
            # or the two pieces would leave a sliver between their ends
            for pair in pairs:
                pair[8] = None
        hub: list[Pt] = []
        for pi, ei, pj, ej, face_i, ui, face_j, uj, mitre in pairs:
            if mitre is not None:
                ci = cj = mitre
            else:
                # the step: both faces stop on the line through the node that
                # bisects the join - square across a straight join, and the line
                # every equal-offset mitre of this node lies on anyway
                cut = _left(_sub(ui, uj))
                ci = _meet(face_i, ui, node, cut)
                cj = _meet(face_j, uj, node, cut)
                if math.dist(ci, cj) <= _EPS:
                    cj = ci
                else:
                    faces.append((Line(ci, cj, "wall"), walls[pi.seg.wall].id))
            pi.corners[(ei, _canonical(ei, "left"))] = ci
            pj.corners[(ej, _canonical(ej, "right"))] = cj
            hub.append(ci)
            if cj is not ci:
                hub.append(cj)
        if len(ordered) >= 3:
            polygon = _polygon(hub)
            if polygon and not _counter_clockwise(polygon):
                raise ValueError(
                    f"the junction at ({node[0]:g}, {node[1]:g}) cannot be resolved into a "
                    "clean outline - its middle turns inside out; move the walls' ends apart."
                )
            if polygon:
                # the wall that runs straight through the node owns its hub (the
                # crossing wall of a T); with none, or two at an X, the first one
                through = [
                    pi.seg.wall
                    for (pi, ei), (pj, ej) in itertools.combinations(ordered, 2)
                    if pi.seg.wall == pj.seg.wall
                    and _dot(_outgoing(pi, ei)[1], _outgoing(pj, ej)[1]) < -1.0 + 1e-9
                ]
                owner = min(through or [piece.seg.wall for piece, _end in ordered])
                regions.append((polygon, walls[owner].id))
                # a piece of another material ends on the hub: its end is a seam
                hub_material = normalise_material(walls[owner].material)
                for piece, end in ordered:
                    wall = walls[piece.seg.wall]
                    if normalise_material(wall.material) == hub_material:
                        continue
                    left, right = piece.corners[(end, "left")], piece.corners[(end, "right")]
                    if math.dist(left, right) > _EPS:
                        seams.append((Line(left, right, "wall"), wall.id))

    # -- the openings -----------------------------------------------------------
    by_id = {wall.id: i for i, wall in enumerate(walls)}
    by_seg = {(seg.wall, seg.index): seg for seg in segs}
    cut_in: dict[int, list[tuple[float, float, bool, bool, str]]] = {}
    open_ends: set[tuple[int, int]] = set()  # (id(piece), end) a corner opening runs into
    for opening in openings:
        wi = by_id[opening.wall]
        wall = walls[wi]
        si, along = segment_of(wall, float(opening.offset))
        a0, b0 = wall_segments(wall)[si]
        start = _add(a0, _mul(_unit(a0, b0), along))
        seg = by_seg[(wi, si)]
        u = _unit(pos[seg.va], pos[seg.vb])
        s0 = _dot(_sub(start, pos[seg.va]), u)
        s1 = s0 + float(opening.width)
        host = None
        for piece in pieces:
            length = math.dist(piece.a, piece.b)
            if piece.seg is seg and piece.s0 - _EPS <= s0 and s1 <= piece.s0 + length + _EPS:
                host = piece
                break
        if host is None:
            omitted.append(
                (
                    {
                        "element": opening.id,
                        "reason": f"it crosses a junction on wall {wall.id!r} segment {si}; "
                        "an opening must lie inside one stretch of wall",
                    },
                    wall.id,
                )
            )
            continue
        d0 = s0 - host.s0
        d1 = s1 - host.s0
        clear_from = max(
            _dot(_sub(host.corners[(0, side)], host.a), u) for side in ("left", "right")
        )
        clear_to = min(_dot(_sub(host.corners[(1, side)], host.a), u) for side in ("left", "right"))
        host_length = math.dist(host.a, host.b)
        corner0 = d0 <= _EPS and clear_from > _EPS
        corner1 = d1 >= host_length - _EPS and clear_to < host_length - _EPS
        refusal = None
        for flag, end in ((corner0, 0), (corner1, 1)):
            if not flag:
                continue
            size = len(ends[host.nodes[end]])
            if opening.kind not in CORNER_KINDS:
                refusal = (
                    f"a {opening.kind} cannot run into the junction of wall {wall.id!r} - "
                    f"only {', '.join(CORNER_KINDS)} may run to the corner; "
                    "end it where the solid wall starts"
                )
            elif size != 2:
                refusal = (
                    f"it runs to a node of wall {wall.id!r} where {size} wall ends meet; "
                    "a corner opening runs to an L (two ends), not into a T or an X"
                )
        if refusal is not None:
            omitted.append(({"element": opening.id, "reason": refusal}, wall.id))
            continue
        if (d0 < clear_from - _EPS and not corner0) or (d1 > clear_to + _EPS and not corner1):
            omitted.append(
                (
                    {
                        "element": opening.id,
                        "reason": f"it runs into a junction of wall {wall.id!r}: on segment {si} "
                        f"the solid wall runs from {host.s0 + clear_from:g} to "
                        f"{host.s0 + clear_to:g} mm along the axis, the opening from "
                        f"{s0:g} to {s1:g}",
                    },
                    wall.id,
                )
            )
            continue
        cut_in.setdefault(id(host), []).append((d0, d1, corner0, corner1, opening.id))
        if corner0:
            open_ends.add((id(host), 0))
        if corner1:
            open_ends.add((id(host), 1))

    # -- faces, jambs and regions, piece by piece ---------------------------------
    for piece in pieces:
        seg = piece.seg
        owner = walls[seg.wall].id
        u = _unit(piece.a, piece.b)
        n = _left(u)

        def at(s: float, offset: float, piece=piece, u=u, n=n) -> Pt:
            return _add(_add(piece.a, _mul(u, s)), _mul(n, offset))

        cur: tuple[Pt, Pt] | None = (piece.corners[(0, "left")], piece.corners[(0, "right")])
        runs: list[tuple[Pt, Pt, Pt, Pt]] = []
        for d0, d1, corner0, corner1, element in sorted(cut_in.get(id(piece), [])):
            if corner0:  # the gap starts on the mitre line: no solid run, no jamb before it
                la, ra = piece.corners[(0, "left")], piece.corners[(0, "right")]
            else:
                la, ra = at(d0, seg.left), at(d0, seg.right)
                runs.append((cur[0], cur[1], la, ra))
                jambs.append((Line(la, ra, "wall"), owner))
            if corner1:
                lb, rb = piece.corners[(1, "left")], piece.corners[(1, "right")]
            else:
                lb, rb = at(d1, seg.left), at(d1, seg.right)
                jambs.append((Line(lb, rb, "wall"), owner))
            thresholds.append((Line(la, lb, "wall"), owner))
            thresholds.append((Line(ra, rb, "wall"), owner))
            corner = [name for flag, name in ((corner0, "start"), (corner1, "end")) if flag]
            cuts.append(
                ({"element": element, "left": (la, lb), "right": (ra, rb), "corner": corner}, owner)
            )
            cur = None if corner1 else (lb, rb)
        if cur is not None:
            runs.append((cur[0], cur[1], piece.corners[(1, "left")], piece.corners[(1, "right")]))
        for sl, sr, el, er in runs:
            if math.dist(sl, el) > _EPS:
                faces.append((Line(sl, el, "wall"), owner))
            if math.dist(sr, er) > _EPS:
                faces.append((Line(sr, er, "wall"), owner))
            polygon = _polygon((sr, er, el, sl))
            if polygon and (is_self_intersecting(polygon) or not _counter_clockwise(polygon)):
                stepped = [nodes[piece.nodes[e]] for e in (0, 1) if piece.nodes[e] in step_nodes]
                if stepped:
                    x, y = stepped[0]
                    raise ValueError(
                        f"{_name(walls, seg.wall)}: segment {seg.index} steps to the next wall "
                        f"at ({x:g}, {y:g}), and a wall ending on it there leaves the stretch "
                        "between them with no room - end that wall clear of the step, or "
                        "align the two walls' axes."
                    )
                raise ValueError(
                    f"{_name(walls, seg.wall)}: segment {seg.index} is too short for the "
                    "junctions at its ends - its outline would cross itself; lengthen it "
                    "or join it elsewhere."
                )
            if polygon:
                regions.append((polygon, owner))

    # A wall left solid at an L whose other wall a corner opening runs into ends
    # on the mitre line, and is closed there by a jamb along it.
    for members in ends:
        if len(members) != 2:
            continue
        opened = [(piece, end) for piece, end in members if (id(piece), end) in open_ends]
        if len(opened) != 1:
            continue
        piece, end = next(m for m in members if (id(m[0]), m[1]) not in open_ends)
        left, right = piece.corners[(end, "left")], piece.corners[(end, "right")]
        if math.dist(left, right) > _EPS:
            jambs.append((Line(left, right, "wall"), walls[piece.seg.wall].id))

    return {
        "walls": walls,
        "faces": _merge_collinear(faces, jambs),
        "jambs": jambs,
        "thresholds": thresholds,
        "cuts": cuts,
        "regions": regions,
        "seams": seams,
        "omitted": omitted,
    }


def _merge_collinear(faces, jambs):
    """Join two faces of one wall that continue each other in a straight line.

    That is what keeps a crossing wall's far face one line past a T, and a
    straight axis with an intermediate vertex one face. A point touched by a
    third line (a jamb, another face) is a real corner and is left alone.
    """
    items = list(faces)
    while True:
        touching: dict[tuple[float, float], list[int]] = {}
        for idx, (line, _owner) in enumerate(items):
            for p in (line.p1, line.p2):
                touching.setdefault(_key(p), []).append(idx)
        for line, _owner in jambs:
            for p in (line.p1, line.p2):
                touching.setdefault(_key(p), []).append(-1)
        for key, idxs in touching.items():
            if len(idxs) != 2 or -1 in idxs or idxs[0] == idxs[1]:
                continue
            i, j = sorted(idxs)
            (li, wi), (lj, wj) = items[i], items[j]
            if wi != wj:
                continue
            near_i, far_i = (li.p2, li.p1) if _key(li.p2) == key else (li.p1, li.p2)
            far_j = lj.p2 if _key(lj.p1) == key else lj.p1
            d1 = _unit(far_i, near_i)
            d2 = _unit(near_i, far_j)
            if abs(_cross(d1, d2)) > 1e-9 or _dot(d1, d2) <= 0.0:
                continue
            items[i] = (Line(far_i, far_j, "wall"), wi)
            del items[j]
            break
        else:
            return items


def _assemble(parts: dict, poche: bool, owner: str | None) -> WallGeometry:
    def mine(rows):
        return [item for item, who in rows if owner is None or who == owner]

    material = {wall.id: wall.material for wall in parts["walls"]}
    regions = [(polygon, who) for polygon, who in parts["regions"] if owner in (None, who)]
    hatches = (
        tuple(HatchArea((polygon,), material[who]) for polygon, who in regions) if poche else ()
    )
    return WallGeometry(
        faces=tuple(mine(parts["faces"])),
        jambs=tuple(mine(parts["jambs"])),
        regions=tuple(polygon for polygon, _who in regions),
        hatches=hatches,
        omitted=tuple(mine(parts["omitted"])),
        thresholds=tuple(mine(parts["thresholds"])),
        cuts=tuple(mine(parts["cuts"])),
        seams=tuple(mine(parts["seams"])),
    )


def wall_geometry(walls, openings=(), *, poche=True, join_tol=1.0) -> WallGeometry:
    """The whole network's outlines, jambs, regions and poche - see the module docstring."""
    return _assemble(_build(walls, openings, join_tol), poche, None)


def wall_geometry_by_wall(
    walls, openings=(), *, poche=True, join_tol=1.0
) -> dict[str, WallGeometry]:
    """The same geometry, split by the wall each element belongs to.

    A hub (the middle of a T or an X) belongs to the wall running straight
    through it - the first of them in the order given, or the first wall at a
    node no wall runs through. `engineering.arch.draw` uses the split to redraw only the walls
    whose outline a new wall or opening changes.
    """
    parts = _build(walls, openings, join_tol)
    return {wall.id: _assemble(parts, poche, wall.id) for wall in parts["walls"]}


def room_segments(geometry: WallGeometry) -> tuple[tuple[Pt, Pt], ...]:
    """Faces, jambs and thresholds as bare segments - the input of the planar-face finder."""
    return tuple(
        (line.p1, line.p2) for line in (*geometry.faces, *geometry.jambs, *geometry.thresholds)
    )
