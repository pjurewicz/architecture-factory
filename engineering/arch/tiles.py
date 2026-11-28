"""Floor tile and panel layouts: the cut grid and its takeoff. Pure; no backend.

A layout is a pattern of tiles laid from a start point over a floor outline,
minus cutouts (fitted furniture, a shower tray, a bath). Every tile is
intersected with that region and each piece is classified ``full`` or ``cut``;
a tile that misses the region is dropped.

How a piece is computed, for any simple outline and any simple cutouts
(convex or not, touching each other or the walls): its boundary is the tile's
edges where the region lies just to their left, plus the region's edges
(outline counter-clockwise, cutouts clockwise) strictly inside the tile where
the region lies to their left and not to their right. Each kept edge is
oriented with the piece on its left, so the piece's area is Green's sum over
those edges - exact, holes included - and the edges chain into the closed
loops that are drawn. A tile edge lying on a wall is kept once, from the tile.

Patterns, in a frame turned by ``angle_deg`` about the start point - the
direction the rows (grid, running bond) or the pattern's axis (herringbone,
chevron) run:

* ``grid`` - tile corners on a lattice of pitch (length + joint, width + joint);
* ``running_bond`` - every row shifted by ``bond`` x pitch further than the one
  below it (0.5 = half bond, 1/3 = third bond);
* ``herringbone`` - pairs of one plank along +45° and one along -45° to the
  axis, the second standing at the first one's end; the pair repeats on the
  lattice (W', W') and (L', -L') in the planks' frame, primes meaning size +
  joint, so the pattern repeats every W' x sqrt 2 along the axis. Any length >
  width tiles the plane (lengths need not be a multiple of the width);
* ``chevron`` (French herringbone) - parallelogram planks in bands along the
  axis, their ends on the band edges and their long sides at ``cut_deg`` to the
  axis, leaning one way in even bands and the other in odd ones, so the joints
  zigzag. ``length`` is the long edge, so a band is length x sin(cut) wide and
  its planks repeat every (width + joint) / sin(cut) along the axis. The planks
  are handed (A in even bands, B in odd), counted apart.

The quantities: a full piece is one tile. Cut pieces are, by default, packed
into tiles the way a fitter reuses offcuts (``cut_tiles``): each piece needs
its bounding rectangle in the tile's own frame (grain and pattern direction
kept, no 90° turn), cut guillotine-style from a fresh tile or from the
best-fitting offcut left by an earlier cut, every cut losing ``kerf``. A
diagonal piece still needs its whole bounding rectangle, so the matching
triangle of a 45° cut is not reused - the count is conservative there; a
chevron plank is a parallelogram, so a chevron reuses no offcut at all.
``tiles_no_reuse`` (every cut piece a whole tile) is reported beside it as the
upper bound. The waste allowance is applied on top of the chosen count. Units
are drawing units, taken to be millimetres (areas in m²).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Pt = tuple[float, float]

PATTERNS = ("grid", "running_bond", "herringbone", "chevron")

#: A piece is full when it holds this share of the tile's area.
FULL_SHARE = 1.0 - 1e-9
#: Offset of the side probes, drawing units (1e-4 mm): far below any joint.
_PROBE = 1e-4
#: Endpoint snap when edges are chained into loops.
_SNAP = 1e-6
#: Coverage limit: a layout of more tiles is refused (the call would draw them all).
MAX_TILES = 20000


@dataclass(frozen=True)
class Piece:
    """One tile's share of the region: ``loops`` are closed, the piece on their left."""

    index: int
    corners: tuple[Pt, ...]
    loops: tuple[tuple[Pt, ...], ...]
    area: float
    full: bool
    hand: str | None = None


# -- validation -------------------------------------------------------------------


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where}: must be finite, got {value!r}")
    return number


def _positive(value, where: str) -> float:
    number = _number(value, where)
    if number <= 0.0:
        raise ValueError(f"{where}: must be positive, got {value!r}")
    return number


def signed_area(points: Sequence[Pt]) -> float:
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _segments_cross(p1: Pt, p2: Pt, q1: Pt, q2: Pt) -> bool:
    def orient(a: Pt, b: Pt, c: Pt) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and 0 not in (d1, d2, d3, d4)


def polygon(value, where: str) -> tuple[Pt, ...]:
    """A simple polygon of at least three points, counter-clockwise, without a repeated close."""
    if isinstance(value, (str, dict)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{where}: give a list of [x, y] points")
    points: list[Pt] = []
    for i, item in enumerate(value):
        if isinstance(item, (str, dict)) or not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(f"{where}[{i}]: expected [x, y], got {item!r}")
        point = (_number(item[0], f"{where}[{i}][0]"), _number(item[1], f"{where}[{i}][1]"))
        if points and math.dist(point, points[-1]) <= _SNAP:
            continue
        points.append(point)
    if len(points) > 1 and math.dist(points[0], points[-1]) <= _SNAP:
        points.pop()
    if len(points) < 3:
        raise ValueError(f"{where}: a polygon needs at least 3 distinct points, got {len(points)}")
    area = signed_area(points)
    if abs(area) <= 1e-9:
        raise ValueError(f"{where}: the points enclose no area")
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _segments_cross(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                raise ValueError(
                    f"{where}: edges {i} and {j} cross; a floor outline or cutout must be "
                    "a simple polygon"
                )
    return tuple(points if area > 0 else reversed(points))


# -- the region ---------------------------------------------------------------------


def _inside(point: Pt, loop: Sequence[Pt]) -> bool:
    x, y = point
    inside = False
    j = len(loop) - 1
    for i in range(len(loop)):
        xi, yi = loop[i]
        xj, yj = loop[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


@dataclass(frozen=True)
class Region:
    """The outline minus every cutout; ``edges`` oriented with the region on their left."""

    outline: tuple[Pt, ...]
    cutouts: tuple[tuple[Pt, ...], ...]
    edges: tuple[tuple[Pt, Pt], ...]

    @classmethod
    def of(cls, outline, cutouts=()) -> Region:
        edges: list[tuple[Pt, Pt]] = []
        for loop, ccw in ((outline, True), *((c, False) for c in cutouts)):
            ordered = loop if ccw else tuple(reversed(loop))
            edges += [(ordered[i], ordered[(i + 1) % len(ordered)]) for i in range(len(ordered))]
        return cls(tuple(outline), tuple(cutouts), tuple(edges))

    def contains(self, point: Pt) -> bool:
        return _inside(point, self.outline) and not any(_inside(point, c) for c in self.cutouts)


# -- one piece ------------------------------------------------------------------------


def _split_params(a: Pt, b: Pt, others) -> list[float]:
    """Where segment a-b meets each of ``others`` (crossings and collinear overlaps), as t."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    ts = [0.0, 1.0]
    for p, q in others:
        ex, ey = q[0] - p[0], q[1] - p[1]
        denom = dx * ey - dy * ex
        wx, wy = p[0] - a[0], p[1] - a[1]
        if abs(denom) > 1e-12 * math.sqrt(length2 * (ex * ex + ey * ey)):
            t = (wx * ey - wy * ex) / denom
            s = (wx * dy - wy * dx) / denom
            if -1e-12 <= s <= 1 + 1e-12 and 0.0 < t < 1.0:
                ts.append(t)
        else:  # parallel: on the same line, the other's ends split this one
            if abs(wx * dy - wy * dx) <= 1e-9 * math.sqrt(length2):
                for end in (p, q):
                    t = ((end[0] - a[0]) * dx + (end[1] - a[1]) * dy) / length2
                    if 0.0 < t < 1.0:
                        ts.append(t)
    return sorted(set(ts))


def _probes(a: Pt, b: Pt) -> tuple[Pt, Pt, Pt]:
    """The midpoint of a-b and a point just to its left and just to its right."""
    mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    length = math.dist(a, b)
    nx, ny = -(b[1] - a[1]) / length * _PROBE, (b[0] - a[0]) / length * _PROBE
    return (mx, my), (mx + nx, my + ny), (mx - nx, my - ny)


def _strictly_inside_convex(point: Pt, corners: Sequence[Pt]) -> bool:
    for i, (x1, y1) in enumerate(corners):
        x2, y2 = corners[(i + 1) % len(corners)]
        length = math.hypot(x2 - x1, y2 - y1)
        if ((x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)) / length <= _PROBE / 2:
            return False
    return True


def _chain(edges: list[tuple[Pt, Pt]]) -> tuple[tuple[Pt, ...], ...]:
    """Oriented edges -> closed loops (collinear runs merged)."""

    def key(p: Pt) -> tuple[int, int]:
        return (round(p[0] / _SNAP), round(p[1] / _SNAP))

    starts: dict[tuple[int, int], list[int]] = {}
    for i, (a, _b) in enumerate(edges):
        starts.setdefault(key(a), []).append(i)
    used = [False] * len(edges)
    loops: list[tuple[Pt, ...]] = []
    for first in range(len(edges)):
        if used[first]:
            continue
        loop: list[Pt] = []
        i = first
        while i is not None and not used[i]:
            used[i] = True
            a, b = edges[i]
            loop.append(a)
            nxt = [j for j in starts.get(key(b), ()) if not used[j]]
            i = nxt[0] if nxt else None
        loops.append(tuple(_merge_collinear(loop)))
    return tuple(loop for loop in loops if len(loop) >= 3)


def _merge_collinear(loop: list[Pt]) -> list[Pt]:
    out = list(loop)
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cross) <= 1e-9 * max(math.dist(a, b) * math.dist(b, c), 1e-12):
                out.pop(i)
                changed = True
                break
    return out


def clip(corners: Sequence[Pt], region: Region) -> tuple[float, tuple[tuple[Pt, ...], ...]]:
    """(area, loops) of a convex counter-clockwise tile intersected with ``region``."""
    tile_edges = [(corners[i], corners[(i + 1) % len(corners)]) for i in range(len(corners))]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    near = [
        (p, q)
        for p, q in region.edges
        if max(p[0], q[0]) >= lo_x - _PROBE
        and min(p[0], q[0]) <= hi_x + _PROBE
        and max(p[1], q[1]) >= lo_y - _PROBE
        and min(p[1], q[1]) <= hi_y + _PROBE
    ]
    if not near:  # the tile is wholly inside or wholly outside the region
        if region.contains(((lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0)):
            return signed_area(corners), (tuple(corners),)
        return 0.0, ()

    def inside_piece(point: Pt) -> bool:
        return _strictly_inside_convex(point, corners) and region.contains(point)

    kept: list[tuple[Pt, Pt]] = []
    for a, b in tile_edges:
        ts = _split_params(a, b, near)
        for t0, t1 in zip(ts, ts[1:], strict=False):
            p = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
            q = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
            if math.dist(p, q) <= _SNAP:
                continue
            _mid, left, _right = _probes(p, q)
            if region.contains(left):
                kept.append((p, q))
    others = [*tile_edges, *near]
    for a, b in near:
        ts = _split_params(a, b, others)
        for t0, t1 in zip(ts, ts[1:], strict=False):
            p = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
            q = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
            if math.dist(p, q) <= _SNAP:
                continue
            mid, left, right = _probes(p, q)
            if not _strictly_inside_convex(mid, corners):
                continue  # on the tile's own edge: the tile edge carries it
            if inside_piece(left) and not inside_piece(right):
                kept.append((p, q))
    area = sum(p[0] * q[1] - q[0] * p[1] for p, q in kept) / 2.0
    return area, _chain(kept)


# -- patterns -------------------------------------------------------------------------


def _rotate(point: Pt, origin: Pt, cos_a: float, sin_a: float) -> Pt:
    x, y = point
    return (origin[0] + x * cos_a - y * sin_a, origin[1] + x * sin_a + y * cos_a)


def _local_bbox(points, origin: Pt, cos_a: float, sin_a: float) -> tuple[float, ...]:
    local = [
        (
            (x - origin[0]) * cos_a + (y - origin[1]) * sin_a,
            -(x - origin[0]) * sin_a + (y - origin[1]) * cos_a,
        )
        for x, y in points
    ]
    xs = [p[0] for p in local]
    ys = [p[1] for p in local]
    return min(xs), min(ys), max(xs), max(ys)


def _rect(x0: float, y0: float, x1: float, y1: float) -> tuple[Pt, ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def local_tiles(
    pattern: str, length: float, width: float, joint: float, bond: float, box, cut=45.0
):
    """Local-frame ``(corners, hand)`` of the tiles covering ``box`` = (x0, y0, x1, y1);
    corners counter-clockwise, ``hand`` "A" / "B" for a chevron plank, else None."""
    lp, wp = length + joint, width + joint
    bx0, by0, bx1, by1 = box
    if pattern in ("grid", "running_bond"):
        shift = bond if pattern == "running_bond" else 0.0
        rows = range(math.floor(by0 / wp) - 1, math.ceil(by1 / wp) + 1)
        for k in rows:
            dx = ((k * shift) % 1.0) * lp
            for i in range(math.floor((bx0 - dx) / lp) - 1, math.ceil((bx1 - dx) / lp) + 1):
                x, y = i * lp + dx, k * wp
                yield _rect(x, y, x + length, y + width), None
        return
    if pattern == "chevron":
        sin_c, cot_c = math.sin(math.radians(cut)), 1.0 / math.tan(math.radians(cut))
        band, end = length * sin_c, width / sin_c  # band width; an end's length on the band edge
        run, step = band * cot_c, wp / sin_c  # a long side's run along the axis; plank pitch
        for k in range(math.floor(by0 / (band + joint)) - 1, math.ceil(by1 / (band + joint)) + 1):
            y = k * (band + joint)
            for i in range(math.floor((bx0 - run) / step) - 1, math.ceil(bx1 / step) + 1):
                x = i * step
                if k % 2 == 0:
                    yield (
                        ((x, y), (x + end, y), (x + end + run, y + band), (x + run, y + band)),
                        "A",
                    )
                else:
                    yield (
                        ((x + run, y), (x + run + end, y), (x + end, y + band), (x, y + band)),
                        "B",
                    )
        return
    # herringbone, in the planks' frame (the axis along (1, 1)): cell origin
    # o = m*(wp, wp) + n*(lp, -lp); solve the box corners for (m, n)
    det = wp * (-lp) - lp * wp
    ms, ns = [], []
    for cx, cy in ((bx0, by0), (bx1, by0), (bx0, by1), (bx1, by1)):
        ms.append((cx * (-lp) - lp * cy) / det)
        ns.append((wp * cy - wp * cx) / det)
    for m in range(math.floor(min(ms)) - 2, math.ceil(max(ms)) + 2):
        for n in range(math.floor(min(ns)) - 2, math.ceil(max(ns)) + 2):
            ox, oy = m * wp + n * lp, m * wp - n * lp
            yield _rect(ox, oy, ox + length, oy + width), None
            yield _rect(ox + lp, oy + wp - lp, ox + lp + width, oy + wp - lp + length), None


# -- the layout --------------------------------------------------------------------------


def _piece_size(piece: Piece) -> tuple[float, float]:
    """The piece's bounding rectangle in its tile's frame, long side of the tile first."""
    c0, c1, _c2, c3 = piece.corners
    du, dv = math.dist(c0, c1), math.dist(c0, c3)
    ex = ((c1[0] - c0[0]) / du, (c1[1] - c0[1]) / du)
    ey = ((c3[0] - c0[0]) / dv, (c3[1] - c0[1]) / dv)
    points = [p for loop in piece.loops for p in loop]
    us = [(p[0] - c0[0]) * ex[0] + (p[1] - c0[1]) * ex[1] for p in points]
    vs = [(p[0] - c0[0]) * ey[0] + (p[1] - c0[1]) * ey[1] for p in points]
    a, b = max(us) - min(us), max(vs) - min(vs)
    return (a, b) if du >= dv else (b, a)


def cut_tiles(pieces, length: float, width: float, kerf: float) -> int:
    """Tiles the cut pieces consume when offcuts are reused (guillotine, best fit)."""
    sizes = sorted(
        (_piece_size(p) for p in pieces if not p.full), key=lambda s: (-s[0] * s[1], -s[0], -s[1])
    )
    free: list[tuple[float, float]] = []
    opened = 0
    eps = 1e-6
    for a, b in sizes:
        fits = [i for i, (w, h) in enumerate(free) if a <= w + eps and b <= h + eps]
        if fits:
            i = min(fits, key=lambda k: (free[k][0] * free[k][1], k))
            w, h = free.pop(i)
        else:
            opened += 1
            w, h = length, width
        # split along the shorter leftover, keeping the larger offcut whole
        right, top = w - a - kerf, h - b - kerf
        if right * h >= w * top:
            rests = [(right, h), (a, top)]
        else:
            rests = [(right, b), (w, top)]
        free += [(rw, rh) for rw, rh in rests if rw > eps and rh > eps]
    return opened


def layout(
    outline,
    *,
    length,
    width,
    pattern: str = "grid",
    joint=3.0,
    start=None,
    angle_deg=0.0,
    bond=0.5,
    cutouts=(),
    waste_pct=10.0,
    offcut_reuse: bool = True,
    kerf=3.0,
    cut_deg=45.0,
) -> dict:
    """Every piece of the layout and its takeoff. Refuses by name before computing."""
    if pattern not in PATTERNS:
        raise ValueError(f"pattern: {pattern!r} is not one of {', '.join(PATTERNS)}")
    length = _positive(length, "tile_length")
    width = _positive(width, "tile_width")
    if pattern == "herringbone" and length <= width:
        raise ValueError(
            f"tile_length: herringbone needs a plank longer than it is wide, got "
            f"{length:g} x {width:g}"
        )
    if not isinstance(offcut_reuse, bool):
        raise ValueError(f"offcut_reuse: expected true or false, got {offcut_reuse!r}")
    kerf = _number(kerf, "kerf")
    if kerf < 0.0:
        raise ValueError(f"kerf: must not be negative, got {kerf:g}")
    joint = _number(joint, "joint")
    if joint < 0.0:
        raise ValueError(f"joint: must not be negative, got {joint:g}")
    bond = _number(bond, "bond")
    if not 0.0 < bond < 1.0:
        raise ValueError(
            f"bond: the row shift is a share of the tile length in (0, 1), got {bond:g}"
        )
    angle = _number(angle_deg, "angle_deg")
    cut = _number(cut_deg, "cut_deg")
    if pattern == "chevron" and not 10.0 <= cut <= 80.0:
        raise ValueError(f"cut_deg: a chevron's cut is 10-80 degrees to its axis, got {cut:g}")
    waste = _number(waste_pct, "waste_pct")
    if not 0.0 <= waste <= 100.0:
        raise ValueError(f"waste_pct: expected 0-100, got {waste:g}")
    floor = polygon(outline, "outline")
    if isinstance(cutouts, (str, dict)) or not isinstance(cutouts, (list, tuple)):
        raise ValueError("cutouts: give a list of {points, label}")
    holes: list[tuple[Pt, ...]] = []
    labels: list[str] = []
    for i, cutout in enumerate(cutouts):
        if not isinstance(cutout, dict) or "points" not in cutout:
            raise ValueError(f"cutouts[{i}]: expected {{points, label}}, got {cutout!r}")
        unknown = sorted(set(cutout) - {"points", "label"})
        if unknown:
            raise ValueError(
                f"cutouts[{i}]: unknown key(s) {', '.join(unknown)}; use points, label"
            )
        holes.append(polygon(cutout["points"], f"cutouts[{i}].points"))
        labels.append(str(cutout.get("label") or f"cutout {i + 1}"))
    if start is None:
        origin = (min(p[0] for p in floor), min(p[1] for p in floor))
    else:
        origin = (_number(start[0], "start_x"), _number(start[1], "start_y"))

    region = Region.of(floor, holes)
    frame = angle - 45.0 if pattern == "herringbone" else angle  # its planks' frame
    cos_a, sin_a = math.cos(math.radians(frame)), math.sin(math.radians(frame))
    box = _local_bbox(floor, origin, cos_a, sin_a)
    lp, wp = length + joint, width + joint
    estimate = (box[2] - box[0] + 2 * lp) * (box[3] - box[1] + 2 * lp) / (lp * wp)
    if estimate > MAX_TILES:
        raise ValueError(
            f"tile_length: about {estimate:.0f} tiles would cover this outline, over the "
            f"{MAX_TILES} one call draws; lay larger tiles or split the floor"
        )

    tile_area = length * width
    pieces: list[Piece] = []
    for local, hand in local_tiles(pattern, length, width, joint, bond, box, cut):
        corners = tuple(_rotate(p, origin, cos_a, sin_a) for p in local)
        area, loops = clip(corners, region)
        if area <= tile_area * 1e-9 or not loops:
            continue
        full = area >= tile_area * FULL_SHARE
        pieces.append(Piece(len(pieces), corners, (corners,) if full else loops, area, full, hand))

    # The region's own area: the same clip against a tile that covers all of it.
    fx = [p[0] for p in floor]
    fy = [p[1] for p in floor]
    pad = 1.0 + max(max(fx) - min(fx), max(fy) - min(fy))
    cover = (
        (min(fx) - pad, min(fy) - pad),
        (max(fx) + pad, min(fy) - pad),
        (max(fx) + pad, max(fy) + pad),
        (min(fx) - pad, max(fy) + pad),
    )
    floor_area, _ = clip(cover, region)

    full_count = sum(1 for p in pieces if p.full)
    cut_count = len(pieces) - full_count
    no_reuse = full_count + cut_count
    offcut_reuse = offcut_reuse and pattern != "chevron"  # a parallelogram's offcut fits no plank
    used_for_cuts = cut_tiles(pieces, length, width, kerf) if offcut_reuse else cut_count
    total = full_count + used_for_cuts
    factor = 1.0 + waste / 100.0
    order = math.ceil(round(total * factor, 9))
    return {
        "pieces": pieces,
        "origin": origin,
        "cutout_labels": labels,
        "takeoff": {
            "floor_area_m2": round(floor_area / 1e6, 4),
            "outline_area_m2": round(signed_area(floor) / 1e6, 4),
            "tiled_area_m2": round(sum(p.area for p in pieces) / 1e6, 4),
            "tile_area_m2": round(tile_area / 1e6, 6),
            "full": full_count,
            "cut": cut_count,
            "offcut_reuse": offcut_reuse,
            "kerf": kerf,
            "cut_tiles": used_for_cuts,
            "tiles": total,
            "tiles_no_reuse": no_reuse,
            "waste_pct": waste,
            "floor_area_with_waste_m2": round(floor_area * factor / 1e6, 4),
            "order_tiles": order,
            "order_area_m2": round(order * tile_area / 1e6, 4),
            **(
                {"by_hand": {h: sum(p.hand == h for p in pieces) for h in ("A", "B")}}
                if pattern == "chevron"
                else {}
            ),
        },
    }
