"""A continuous chain of linear dimensions through any points, with tight texts moved apart.

``dimension_chain`` dimensions a run of points - a section's steps, a detail's
holes, a polyline's vertices - as one row of linear DIMENSIONs sharing one
dimension line, plus an overall dimension one row farther out. Every
dimension measures along the chain's direction:

- ``horizontal`` (0°) and ``vertical`` (90°), or ``auto``: horizontal when the
  points spread at least as far in X as in Y;
- ``aligned``: along the first point -> last point direction, the line
  parallel to it (a linear DIMENSION rotated to that angle, so every span is
  the projection a chain needs, not a point-to-point distance).

The points are projected onto that direction and sorted; two points with the
same projection are one station, and the one nearer the dimension line keeps
its extension line (reported in ``merged``). The dimension line lies
``offset`` beyond the outermost point: positive is above a horizontal chain,
right of a vertical one, left of an aligned one's first -> last direction.
``offset`` 0 puts it through the outermost point - on the points themselves
when they are collinear, for internal dimensions - and every row then goes to
the side the style puts the text on: above a horizontal or aligned chain, left
of a vertical one.

A text that fits its span is centred where the style puts it (DIMTAD). A
tight one - wider than its span plus a gap each side - is staggered: centred
on its span on the first free of two tiers beyond the dimension line, past the
extension lines' overshoot (DIMEXE), so neighbouring tight spans alternate
tiers. With neither tier free it slides along them to the nearest free place;
a text that leaves its own span gets a leader to its dimension line
(``leader=True``), so it cannot be read as its neighbour's. The overall row
moves out until it clears every chain text. Text widths are estimated from the
formatted value (``TEXT_WIDTH_FACTOR``), not measured from a font.

ISO 129-1: each measurement once - the overall row is drawn only when the
chain has two spans or more.
"""

from __future__ import annotations

import math

from engineering.arch.dimension import TEXT_WIDTH_FACTOR
from engineering.dimtext import format_value

ORIENTATIONS: tuple[str, ...] = ("auto", "horizontal", "vertical", "aligned")
#: Default distance between the chain row and the overall row: 8 mm on paper.
STEP_PAPER_MM = 8.0
_STYLE_VARS = (
    "dimtxt",
    "dimgap",
    "dimexe",
    "dimtad",
    "dimdec",
    "dimzin",
    "dimdsep",
    "dimrnd",
    "dimpost",
)


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{where}: expected a finite number, got {value!r}")
    return value


def _points(points) -> list[tuple[float, float]]:
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        raise ValueError("points: give at least two [x, y] points")
    out = []
    for i, p in enumerate(points):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"points[{i}]: expected [x, y], got {p!r}")
        out.append((_number(p[0], f"points[{i}][0]"), _number(p[1], f"points[{i}][1]")))
    return out


def plan_chain(
    points,
    offset,
    *,
    orientation: str = "auto",
    overall: bool = True,
    step=None,
    text_height: float = 2.5,
    gap: float = 0.625,
    tad: int = 1,
    style_vars=None,
    ext_beyond=None,
    leader: bool = True,
) -> dict:
    """The chain's dimensions and their text places, in drawing units; draws nothing.

    ``text_height`` and ``gap`` are DIMTXT and DIMGAP times DIMSCALE, ``step``
    the distance from the chain row to the overall row (default
    ``STEP_PAPER_MM`` times ``text_height / 2.5``). Returns ``{orientation,
    angle, rows: [{row, from, to, value, dim_at, text_at, placement, leader}],
    merged}``; ``placement`` is ``inline`` (where the style puts it),
    ``staggered`` (the other side of the line, inside its span) or ``moved``
    (slid out of its span).
    """
    pts = _points(points)
    offset = _number(offset, "offset")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation: {orientation!r} is not one of {', '.join(ORIENTATIONS)}")
    h = _number(text_height, "text_height")
    if h <= 0:
        raise ValueError(f"text_height: must be positive, got {h:g}")
    g = abs(_number(gap, "gap"))
    exe = 1.25 * h / 2.5 if ext_beyond is None else abs(_number(ext_beyond, "ext_beyond"))
    step = STEP_PAPER_MM * h / 2.5 if step is None else _number(step, "step")
    if step <= 0:
        raise ValueError(f"step: must be positive, got {step:g}")

    if orientation == "auto":
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        orientation = "horizontal" if max(xs) - min(xs) >= max(ys) - min(ys) else "vertical"
    if orientation == "horizontal":
        angle, user_normal = 0.0, (0.0, 1.0)
    elif orientation == "vertical":
        angle, user_normal = 90.0, (1.0, 0.0)
    else:
        dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
        if math.hypot(dx, dy) <= 1e-9:
            raise ValueError(
                "points: the first and last point coincide; an aligned chain has no direction"
            )
        angle = math.degrees(math.atan2(dy, dx))
        user_normal = (-dy / math.hypot(dx, dy), dx / math.hypot(dx, dy))
        # read the way the text reads: ezdxf turns a text in (90, 270] round
        angle = (angle + 90.0) % 180.0 - 90.0
        if angle == -90.0:
            angle = 90.0
    rad = math.radians(angle)
    u = (math.cos(rad), math.sin(rad))
    # n: the side a linear dimension's text stands on ("above" its line)
    n = (-u[1], u[0])
    if offset == 0.0:
        side = 1.0  # on the points: the text's own side (above / left), no sign to read
    else:
        side = math.copysign(1.0, offset) * math.copysign(
            1.0, user_normal[0] * n[0] + user_normal[1] * n[1]
        )

    def local(p):
        return (p[0] * u[0] + p[1] * u[1], p[0] * n[0] + p[1] * n[1])

    def world(s, t):
        return (s * u[0] + t * n[0], s * u[1] + t * n[1])

    # stations: one point per projection, the one nearest the dimension line
    extent = max(1.0, *(abs(c) for p in pts for c in p))
    eps = 1e-9 * extent
    stations: list[tuple[float, float, tuple[float, float]]] = []
    merged: list[dict] = []
    for p in sorted(pts, key=lambda q: local(q)[0]):
        s, t = local(p)
        if stations and abs(s - stations[-1][0]) <= eps:
            keep = stations[-1]
            nearer = t * side > keep[1] * side
            drop = keep[2] if nearer else p
            if nearer:
                stations[-1] = (keep[0], t, p)
            merged.append({"point": list(drop), "kept": list(stations[-1][2])})
            continue
        stations.append((s, t, p))
    if len(stations) < 2:
        raise ValueError(
            f"points: all project onto one station along the {orientation} direction; "
            "nothing to dimension"
        )

    outer = max(t * side for _, t, _ in stations) * side
    chain_line = outer + side * abs(offset)
    spans = list(zip(stations, stations[1:], strict=False))
    with_overall = bool(overall) and len(spans) >= 2

    # across positions as distances from a row's line, + away from the geometry;
    # a linear dimension's text stands on local +t ("above") when DIMTAD is set
    above = side  # local +t, measured in "away" distances
    t0 = (g + h / 2.0) * above if tad else 0.0
    # the stagger tiers lie beyond the line, clear of the extension lines' overshoot
    # and of the style's own place when that is beyond the line too
    tier1 = max(exe, t0 + h / 2.0) + g + h / 2.0
    tiers = (tier1, tier1 + h + g)
    sep = max(g, h)  # closer, "60" and "105" read as "60105"
    boxes: list[tuple[float, float, float, float]] = []  # along lo/hi, away lo/hi

    def width(value: float) -> float:
        return len(format_value(value, 0, style_vars)) * TEXT_WIDTH_FACTOR * h

    def free(rect, lines_away=()) -> bool:
        if any(rect[2] - g < at < rect[3] + g for at in lines_away):
            return False
        return not any(
            rect[0] < o[1] + sep and o[0] < rect[1] + sep and rect[2] < o[3] and o[2] < rect[3]
            for o in boxes
        )

    def rect(along: float, w: float, away: float, base: float = 0.0):
        return (along - w / 2.0, along + w / 2.0, base + away - h / 2.0, base + away + h / 2.0)

    texts: list[tuple[float, float, str]] = []  # along, away from the chain line, placement
    tight: list[int] = []
    for i, (a, b) in enumerate(spans):
        mid, w = (a[0] + b[0]) / 2.0, width(b[0] - a[0])
        r = rect(mid, w, t0)
        texts.append((mid, t0, "inline"))
        if w + 2.0 * g <= b[0] - a[0] and free(r):
            boxes.append(r)
        else:
            tight.append(i)
    # the overall row's extension lines run through the tiers at the chain's ends
    ends = (stations[0][0], stations[-1][0]) if with_overall else ()
    for i in tight:
        a, b = spans[i]
        mid, w = (a[0] + b[0]) / 2.0, width(b[0] - a[0])
        spot = next(
            ((mid, away, "staggered") for away in tiers if free(rect(mid, w, away))),
            None,
        )
        if spot is None:
            alongs = {mid}
            for o in boxes:
                alongs |= {o[1] + sep + w / 2.0, o[0] - sep - w / 2.0}
            for e in (s for s, _, _ in stations):
                alongs |= {e + g + w / 2.0, e - g - w / 2.0}
            spot = next(
                (
                    (along, away, "staggered" if a[0] < along < b[0] else "moved")
                    for _, _, along, away in sorted(
                        (abs(x - mid), k, x, tiers[k]) for x in alongs for k in range(len(tiers))
                    )
                    if free(r := rect(along, w, away))
                    and not any(r[0] - g < e < r[1] + g for e in ends)
                ),
                (mid, tiers[0], "staggered"),
            )
        boxes.append(rect(spot[0], w, spot[1]))
        texts[i] = spot

    rows = []
    for (a, b), (along, away, placement) in zip(spans, texts, strict=True):
        rows.append(_row("chain", a, b, chain_line, along, chain_line + side * away, placement))
    overall_away = None
    if with_overall:
        # beyond every chain text by a gap, at least ``step`` from the chain line
        reach = max((o[3] for o in boxes), default=0.0)
        # the line clears them, and so does the overall text's near edge (t0 - h/2)
        overall_away = max(step, reach + g + max(0.0, h / 2.0 - t0))
        line = chain_line + side * overall_away
        a, b = stations[0], stations[-1]
        rows.append(_row("overall", a, b, line, (a[0] + b[0]) / 2.0, line + side * t0, "inline"))
    for row in rows:
        row["dim_at"] = [round(c, 9) for c in world(*row.pop("_dim"))]
        row["text_at"] = [round(c, 9) for c in world(*row.pop("_text"))]
        row["leader"] = bool(leader) and row["placement"] == "moved"
    return {
        "orientation": orientation,
        "angle": angle,
        "rows": rows,
        "merged": merged,
        "overall_offset": overall_away,
    }


def _row(row, a, b, line, along, across, placement) -> dict:
    return {
        "row": row,
        "from": list(a[2]),
        "to": list(b[2]),
        "value": round(b[0] - a[0], 9),
        "placement": placement,
        "_dim": ((a[0] + b[0]) / 2.0, line),
        "_text": (along, across),
    }


async def draw_chain(
    backend,
    points,
    offset,
    *,
    orientation: str = "auto",
    overall: bool = True,
    step=None,
    scale=None,
    layer: str | None = None,
    leader: bool = True,
    dry_run: bool = False,
) -> dict:
    """``plan_chain`` drawn as real linear DIMENSIONs inside one transaction.

    ``scale`` (a plot-scale denominator) is DIMSCALE while the chain is drawn,
    the drawing's own put back afterwards; None draws at the drawing's DIMSCALE.
    A failure half way rolls the whole chain back.
    """
    from engineering.mech.draw import guarded_transaction

    if scale is not None:
        scale = _number(scale, "scale")
        if scale <= 0:
            raise ValueError(f"scale: must be positive, got {scale:g}")
    previous = await backend.system_get_variable("DIMSCALE")
    previous = 1.0 if previous in (None, 0, 0.0) else float(previous)
    factor = previous if scale is None else scale
    style = {name: await backend.system_get_variable(name.upper()) for name in _STYLE_VARS}
    plan = plan_chain(
        points,
        offset,
        orientation=orientation,
        overall=overall,
        step=STEP_PAPER_MM * factor if step is None else step,
        text_height=float(style["dimtxt"] or 2.5) * factor,
        gap=float(style["dimgap"] or 0.0) * factor,
        ext_beyond=float(1.25 if style["dimexe"] is None else style["dimexe"]) * factor,
        tad=int(style["dimtad"] or 0),
        style_vars={k: v for k, v in style.items() if v is not None},
        leader=leader,
    )
    layer = layer or backend._role_layer("dim")
    result = {
        "ok": True,
        "orientation": plan["orientation"],
        "angle": plan["angle"],
        "layer": layer,
        "dimscale": factor,
        "count": len(plan["rows"]),
        "staggered": sum(r["placement"] == "staggered" for r in plan["rows"]),
        "moved": sum(r["placement"] == "moved" for r in plan["rows"]),
        "dimensions": plan["rows"],
        "merged": plan["merged"],
        "overall": any(r["row"] == "overall" for r in plan["rows"]),
        "dry_run": bool(dry_run),
        "backend": backend.name,
    }
    if dry_run:
        return result

    async def job() -> list[str]:
        await backend._ensure_layer(layer)
        moved_scale = abs(previous - factor) > 1e-9
        if moved_scale:
            await backend.system_set_variable("DIMSCALE", factor)
        handles = []
        try:
            for row in plan["rows"]:
                info = await backend.dimension_linear(
                    row["from"][0],
                    row["from"][1],
                    row["to"][0],
                    row["to"][1],
                    row["dim_at"][0],
                    row["dim_at"][1],
                    plan["angle"],
                    layer,
                    text_at=tuple(row["text_at"]),
                    text_leader=row["leader"],
                )
                row["handle"] = info.handle
                handles.append(info.handle)
        finally:
            if moved_scale:
                await backend.system_set_variable("DIMSCALE", previous)
        return handles

    result["handles"] = await guarded_transaction(
        backend, job, what="dimension_chain", done="drawn"
    )
    return result
