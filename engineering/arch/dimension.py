"""Exterior dimension chains: openings, walls, overall - three rows per side.

An architectural plan carries its dimensions outside the building, in rows
parallel to each facade, nearest first:

1. **openings** - the facade's extremes and both jambs of every opening in the
   exterior wall on that side;
2. **walls** - the extremes and both faces of every wall that meets that
   exterior wall, so the chain reads wall thickness, clear room width, wall
   thickness, ...;
3. **overall** - the building's extreme faces.

The "exterior wall on a side" is every axis segment parallel to that side
whose outer face no other parallel segment overlapping it lies beyond. Points
come from the wall engine's own faces, so a chain can never disagree with the
outline it measures.

Wall outlines that are not in the junction engine - a bay, a column, a wall
imported as a polygon (``polygons``, tuples of points) - widen the box the
overall row spans, and one whose outline reaches a facade (a vertex inside an
exterior wall's body or on the box edge) adds its two extremes along that
facade to the walls row. A corner opening's jamb is its gap's end on the outer
face, which at an L is the outer mitre point, not the axis.

ISO 129-1: each measurement appears once. A row whose points are exactly those
of a farther row (a facade with no opening, a side no wall meets) is not
emitted; the rows that are keep their places, so the overall row is always at
the same distance.

Offsets are paper millimetres times the plot-scale denominator: the first row
10 mm from the building and 8 mm between rows. Those two numbers are this
repository's drafting defaults, not standard values; pass ``first_offset`` /
``step`` in drawing units to override them.

A stepped facade (a garage set forward, a porch, a wing set back) projects every
step onto one row, so short spans - a wall's 120, a return's 240 - stand side
by side and their texts would print over each other. :func:`chain_text_positions`
places every text of a side as a box: centred on its span where it fits, else
on the other side of its dimension line, else slid along the line to the
nearest free place - no two boxes overlap and none crosses another row's line.
A text's width is estimated from its formatted value (``TEXT_WIDTH_FACTOR``
text heights per character), not measured from a font.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from engineering.arch.model import Wall, segment_of
from engineering.arch.walls import face_offsets, wall_geometry, wall_segments
from engineering.mech.primitives import DimIntent

SIDES: tuple[str, ...] = ("bottom", "top", "left", "right")
ROWS: tuple[str, ...] = ("openings", "walls", "overall")
#: How far out a row lies, for placing texts row by row: the plan's three rows
#: and the interior elevation's (``engineering/arch/elevation.py``) inner rows.
ROW_RANK: dict[str, int] = {"openings": 0, "levels": 0, "positions": 0, "walls": 1, "overall": 2}
FIRST_OFFSET_PAPER_MM = 10.0
STEP_PAPER_MM = 8.0
#: Width of one character of dimension text, in text heights: an estimate a
#: little wider than a digit of the fonts the engines draw with, so a box is
#: never narrower than its text.
TEXT_WIDTH_FACTOR = 0.75
_TOL = 1.0
#: A walls-row span shorter than this, ending on an oblique wall's projected
#: face, is merged away (`_merge_oblique`).
OBLIQUE_MERGE_MM = 35.0


def _positive(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{where}: must be a finite number above zero, got {value!r}")
    return number


def _sides(sides) -> tuple[str, ...]:
    if isinstance(sides, str) or not isinstance(sides, (list, tuple)) or not sides:
        raise ValueError(f"sides: name at least one of {', '.join(SIDES)}, got {sides!r}")
    chosen: list[str] = []
    for i, side in enumerate(sides):
        if side not in SIDES:
            raise ValueError(f"sides[{i}]: {side!r}; sides are {', '.join(SIDES)}")
        if side in chosen:
            raise ValueError(f"sides[{i}]: {side!r} is named twice")
        chosen.append(side)
    return tuple(chosen)


def _unit(a, b) -> tuple[float, float]:
    length = math.dist(a, b)
    return ((b[0] - a[0]) / length, (b[1] - a[1]) / length)


def _exterior(walls: Sequence[Wall], ai: int, sign: float) -> list[dict]:
    """The segments parallel to a side whose outer face nothing parallel lies beyond."""
    ci = 1 - ai
    parallel: list[dict] = []
    for wall in walls:
        left, right = face_offsets(wall)
        for index, (a, b) in enumerate(wall_segments(wall)):
            u = _unit(a, b)
            if abs(u[ci]) > 1e-9:
                continue
            n = (-u[1], u[0])
            faces = (a[ci] + n[ci] * left, a[ci] + n[ci] * right)
            parallel.append(
                {
                    "wall": wall,
                    "index": index,
                    "a": a,
                    "b": b,
                    "span": (min(a[ai], b[ai]), max(a[ai], b[ai])),
                    "outer": min(faces) if sign < 0 else max(faces),
                }
            )
    exterior = []
    for item in parallel:
        lo, hi = item["span"]
        covered = any(
            other is not item
            and min(hi, other["span"][1]) - max(lo, other["span"][0]) > _TOL
            and (
                other["outer"] < item["outer"] - _TOL
                if sign < 0
                else other["outer"] > item["outer"] + _TOL
            )
            for other in parallel
        )
        if not covered:
            exterior.append(item)
    return exterior


def _inside(point, item: dict) -> bool:
    a, b = item["a"], item["b"]
    left, right = face_offsets(item["wall"])
    u = _unit(a, b)
    s = (point[0] - a[0]) * u[0] + (point[1] - a[1]) * u[1]
    d = (point[0] - a[0]) * -u[1] + (point[1] - a[1]) * u[0]
    return -_TOL <= s <= math.dist(a, b) + _TOL and right - _TOL <= d <= left + _TOL


def _points(values: set[float]) -> list[float]:
    out: list[float] = []
    for value in sorted(values):
        if not out or value - out[-1] > 1e-6:
            out.append(value)
    return out


def _anchor(anchors: dict, value: float, default: float) -> float:
    """The across-facade coordinate recorded for ``value`` (merged within 1e-6 by `_points`)."""
    for key, across in anchors.items():
        if abs(key - value) <= 1e-6:
            return across
    return default


def _merge_oblique(
    points: list[float], oblique: dict[float, float], real: set[float]
) -> list[float]:
    """The walls row without the slivers an oblique wall's end projection leaves.

    An oblique wall's two faces at its end project onto the facade
    ``thickness * |sin|`` apart, at points that are no feature of the plan. A
    span under ``OBLIQUE_MERGE_MM`` ending on such a point loses it: next to a
    point another source gives (a face, a jamb, a step) the projected point
    goes, and the two points of one wall end become the one toward the wall it
    meets (``oblique`` maps each to it), which lies on that wall's faces. A
    point any other source gives is never dropped.
    """

    def near(value, values, within=1e-6) -> bool:
        return any(abs(value - v) <= within for v in values)

    short = OBLIQUE_MERGE_MM - 1e-9
    known = [p for p in points if near(p, real) or not near(p, oblique)]
    by_end: dict[float, list[float]] = {}
    for p in points:
        if p not in known and not near(p, known, short):
            kept = next(k for o, k in oblique.items() if abs(p - o) <= 1e-6)
            by_end.setdefault(kept, []).append(p)
    out = list(known)
    for kept, projected in by_end.items():
        if len(projected) > 1 and max(projected) - min(projected) <= short:
            out.append(kept)
        else:
            out.extend(projected)
    return _points(set(out))


def exterior_chains(
    walls,
    openings=(),
    *,
    sides=("bottom", "left"),
    first_offset=None,
    step=None,
    scale=50,
    polygons=(),
) -> tuple[DimIntent, ...]:
    """Linear ``DimIntent`` rows outside each named side, nearest first.

    ``feature`` names the row as ``"<side>:<row>"`` (``"bottom:openings"``);
    ``text_at`` is the point the dimension line passes through.
    """
    walls = list(walls)
    openings = list(openings)
    chosen = _sides(sides)
    factor = _positive(scale, "scale")
    first = (
        FIRST_OFFSET_PAPER_MM * factor
        if first_offset is None
        else _positive(first_offset, "first_offset")
    )
    gap = STEP_PAPER_MM * factor if step is None else _positive(step, "step")
    outlines = [tuple((float(x), float(y)) for x, y in polygon) for polygon in polygons]
    geometry = wall_geometry(walls, openings, poche=False) if walls else None
    faces_drawn = list(geometry.faces) if geometry else []
    if not faces_drawn and not outlines:
        raise ValueError("walls: there is no wall to dimension")
    xs = [p[0] for line in faces_drawn for p in (line.p1, line.p2)]
    ys = [p[1] for line in faces_drawn for p in (line.p1, line.p2)]
    xs += [p[0] for outline in outlines for p in outline]
    ys += [p[1] for outline in outlines for p in outline]
    box = {"bottom": min(ys), "top": max(ys), "left": min(xs), "right": max(xs)}
    # the facade of the walls alone: where a polygon's tied extremes are measured from
    wx = [p[0] for line in faces_drawn for p in (line.p1, line.p2)] or xs
    wy = [p[1] for line in faces_drawn for p in (line.p1, line.p2)] or ys
    facade = {"bottom": min(wy), "top": max(wy), "left": min(wx), "right": max(wx)}
    # every point a chain may measure from: the wall faces' ends and the outlines' vertices
    candidates = [p for line in faces_drawn for p in (line.p1, line.p2)]
    candidates += [p for outline in outlines for p in outline]
    skipped = {entry["element"] for entry in geometry.omitted} if geometry else set()
    cuts = {cut["element"]: cut for cut in geometry.cuts} if geometry else {}
    by_id = {wall.id: wall for wall in walls}

    intents: list[DimIntent] = []
    for side in chosen:
        ai = 0 if side in ("bottom", "top") else 1
        sign = -1.0 if side in ("bottom", "left") else 1.0
        edge = box[side]
        lo, hi = (box["left"], box["right"]) if ai == 0 else (box["bottom"], box["top"])
        ci = 1 - ai
        exterior = _exterior(walls, ai, sign)
        hosts = {(item["wall"].id, item["index"]): item for item in exterior}

        def extreme(value: float, ai=ai, ci=ci, edge=edge) -> float:
            """Where across the facade the overall extreme ``value`` is a real point."""
            near = [p for p in candidates if abs(p[ai] - value) <= 1e-6]
            return min(near, key=lambda p: abs(p[ci] - edge))[ci]

        # value along the facade -> the coordinate across it of the feature it measures:
        # a chain's points stay on the walls they dimension, whatever lies farther out
        jambs = {lo: extreme(lo), hi: extreme(hi)}
        for opening in openings:
            if opening.id in skipped:
                continue
            wall = by_id[opening.wall]
            index, along = segment_of(wall, float(opening.offset))
            host = hosts.get((wall.id, index))
            if host is None:
                continue
            cut = cuts.get(opening.id)
            if cut is not None:
                outer = min(
                    (cut["left"], cut["right"]), key=lambda pq: abs(pq[0][ci] - host["outer"])
                )
                for point in outer:
                    jambs.setdefault(point[ai], point[ci])
                continue
            a, b = wall_segments(wall)[index]
            u = _unit(a, b)
            jambs.setdefault(a[ai] + u[ai] * along, host["outer"])
            jambs.setdefault(a[ai] + u[ai] * (along + float(opening.width)), host["outer"])

        faces = {lo: jambs[lo], hi: jambs[hi]}
        # an oblique wall's two faces projected at its end: value -> the one of the
        # two that lies toward the wall it meets, which is on that wall's faces
        oblique: dict[float, float] = {}
        real = {lo, hi}  # every value another source gives
        for wall in walls:
            left, right = face_offsets(wall)
            for a, b in wall_segments(wall):
                u = _unit(a, b)
                if abs(u[1 - ai]) <= 1e-9:
                    continue
                n = (-u[1], u[0])
                for end, away in ((a, u[ai]), (b, -u[ai])):
                    met = next((item for item in exterior if _inside(end, item)), None)
                    if met is None:
                        continue
                    pair = [end[ai] + n[ai] * left, end[ai] + n[ai] * right]
                    for value in pair:
                        faces.setdefault(value, met["outer"])
                    if abs(u[ai]) > 1e-9:
                        kept = min(pair, key=lambda v, away=away: v * away)
                        oblique.update(dict.fromkeys(pair, kept))
                    else:
                        real.update(pair)
        for outline in outlines:
            on_facade = any(
                abs(p[ci] - edge) <= _TOL or any(_inside(p, item) for item in exterior)
                for p in outline
            )
            if on_facade:
                for value in (min(p[ai] for p in outline), max(p[ai] for p in outline)):
                    tied = [p for p in outline if abs(p[ai] - value) <= 1e-6]
                    nearest = min(tied, key=lambda p: abs(p[ci] - facade[side]))
                    faces.setdefault(value, nearest[ci])
                    real.add(value)

        anchors = {**faces, **jambs}
        walls_row = _merge_oblique(_points(set(faces)), oblique, real)
        rows = [_points(set(jambs)), walls_row, _points({lo, hi})]
        for r, points in enumerate(rows):
            if any(points == later for later in rows[r + 1 :]):
                continue
            offset = edge + sign * (first + r * gap)
            for p, q in zip(points, points[1:], strict=False):
                cp, cq = _anchor(anchors, p, edge), _anchor(anchors, q, edge)
                if ai == 0:
                    p1, p2, at = (p, cp), (q, cq), ((p + q) / 2.0, offset)
                else:
                    p1, p2, at = (cp, p), (cq, q), (offset, (p + q) / 2.0)
                intents.append(DimIntent("linear", p1, p2, at, feature=f"{side}:{ROWS[r]}"))
    return tuple(intents)


def chain_text_positions(
    intents: Sequence[DimIntent], *, text_height, gap, tad=1, style_vars=None
) -> list[tuple[float, float]]:
    """The text centre of each of ``exterior_chains``' dimensions, in its order.

    ``text_height`` and ``gap`` are drawing units (DIMTXT and DIMGAP times
    DIMSCALE); ``tad`` is DIMTAD (0: the text sits in the dimension line, else
    above it); ``style_vars`` formats the values the way the drawing does, to
    size each box (``engineering/dimtext.py``).

    Per side, texts whose box fits inside their span go first, centred where
    the style puts them. Every other text then takes, row by row from the
    building outwards and along the row, the free place nearest its span's
    middle: the style's place, the other side of its own line, or one of those
    slid along the line clear of the boxes already placed and of the extension
    lines. A place is free when its box keeps a text height from its neighbours
    along the line and ``gap`` from every other row's line; a moved text also
    keeps ``gap`` from every extension line where the row leaves room.
    """
    from engineering.dimtext import format_value

    h = _positive(text_height, "text_height")
    g = abs(float(gap))
    near = g + h / 2.0
    # neighbours along a line keep a text height apart: closer, "60" and
    # "240" read as "60240"
    sep = max(g, h)
    out: list[tuple[float, float] | None] = [None] * len(intents)
    by_side: dict[str, list[int]] = {}
    for i, intent in enumerate(intents):
        by_side.setdefault(intent.feature.split(":")[0], []).append(i)

    def place_side(side: str, members: list[int]) -> None:
        ai = 0 if side in ("bottom", "top") else 1
        ci = 1 - ai
        # "above" a horizontal text is +y; a vertical one reads upwards, so its
        # "above" is -x
        up = 1.0 if ai == 0 else -1.0
        slots = (up * near, -up * near) if tad else (0.0, up * near, -up * near)
        lines = sorted({intents[i].text_at[ci] for i in members})
        placed: list[tuple[float, float, float, float]] = []
        # every extension line: its position along, and the stretch across it
        # covers - from the building to its own row's line, plus the overshoot
        exts: set[tuple[float, float, float]] = set()
        for j in members:
            edge, line = intents[j].p1[ci], intents[j].text_at[ci]
            tip = line + math.copysign(h / 2.0, line - edge)
            for end in (intents[j].p1[ai], intents[j].p2[ai]):
                exts.add((end, min(edge, tip), max(edge, tip)))

        def box(i, along, slot):
            intent = intents[i]
            label = format_value(abs(intent.p2[ai] - intent.p1[ai]), 0, style_vars)
            half = len(label) * TEXT_WIDTH_FACTOR * h / 2.0
            across = intent.text_at[ci] + slot
            return (along - half, along + half, across - h / 2.0, across + h / 2.0)

        def free(i, rect, clear_ext=False) -> bool:
            own = intents[i].text_at[ci]
            if any(line != own and rect[2] - g < line < rect[3] + g for line in lines):
                return False
            if clear_ext and any(
                rect[0] - g < at < rect[1] + g and lo < rect[3] and rect[2] < hi
                for at, lo, hi in exts
            ):
                return False
            return not any(
                rect[0] < o[1] + sep and o[0] < rect[1] + sep and rect[2] < o[3] and o[2] < rect[3]
                for o in placed
            )

        def put(i, rect) -> None:
            placed.append(rect)
            along = (rect[0] + rect[1]) / 2.0
            across = (rect[2] + rect[3]) / 2.0
            out[i] = (along, across) if ai == 0 else (across, along)

        def middle(i) -> float:
            return (intents[i].p1[ai] + intents[i].p2[ai]) / 2.0

        later: list[int] = []
        for i in members:
            rect = box(i, middle(i), slots[0])
            span = abs(intents[i].p2[ai] - intents[i].p1[ai])
            if rect[1] - rect[0] + 2.0 * g <= span and free(i, rect):
                put(i, rect)
            else:
                later.append(i)
        later.sort(key=lambda i: (ROW_RANK[intents[i].feature.split(":")[1]], middle(i)))
        for i in later:
            mid = middle(i)
            first = box(i, mid, slots[0])
            half = (first[1] - first[0]) / 2.0
            alongs = {mid}
            for o in placed:
                alongs.add(o[1] + sep + half)
                alongs.add(o[0] - sep - half)
            for at, _, _ in exts:
                alongs.add(at + g + half)
                alongs.add(at - g - half)
            candidates = sorted(
                ((abs(a - mid), s, a) for a in alongs for s in range(len(slots))),
                key=lambda c: (c[0], c[1], c[2]),
            )
            # clear of every extension line if there is room, else only of
            # the other texts and rows
            for clear_ext in (True, False):
                spot = next(
                    (
                        rect
                        for _, s, along in candidates
                        if free(i, rect := box(i, along, slots[s]), clear_ext)
                    ),
                    None,
                )
                if spot is not None:
                    put(i, spot)
                    break
            else:
                # rows closer than a text height leave no free place: the
                # style's place, overlapping, rather than none
                put(i, box(i, mid, slots[0]))

    for side, members in by_side.items():
        place_side(side, members)
    return [p for p in out if p is not None]
