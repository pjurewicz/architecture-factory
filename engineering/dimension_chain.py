"""A continuous chain of linear dimensions through any points, with tight texts moved apart.

``dimension_chain`` dimensions a run of points - a section's steps, a detail's
holes, a polyline's vertices - as one row of linear DIMENSIONs sharing one
dimension line, plus (``overall=True``, off by default) an overall dimension
one row farther out. Every dimension measures along the chain's direction:

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
(``leader=True``), so it cannot be read as its neighbour's. A span narrower
than half a text height (``NARROW_SPAN``: a 1 cm step at 1:50 is 0.2 mm on
paper, its two extension lines read as one) never takes a text under it: the
text slides out beside the neighbours and a leader ties it to its span. The
leader is AutoCAD's own: the text carries a user location with DIMTMOVE 1, the
renderer draws the broken leader with its shelf under the text. The overall
row moves out until it clears every chain text. Text widths are estimated from
the formatted value (``TEXT_WIDTH_FACTOR``), not measured from a font.

``texts`` overrides a row's text (``<>`` is the measured value) and, as
``{"text"?, "side"?, "shift"?}``, places it: ``side`` ``above`` / ``below`` the
dimension line in the dimension's reading frame (above a horizontal chain,
left of a vertical one: the DIMTAD side), ``shift`` ``[along, up]`` drawing
units from there (along the chain, toward its above side). A text so placed is
never moved by the planner, the others keep clear of it, and one that leaves
its span gets the leader.

Repeated tight spans - two or more equal neighbours whose texts do not fit,
a stair's or a joist spacing's 4 x 205 - are reported in ``repeats`` with a
``tight_repeats`` warning. ``merge_repeats=True`` draws each such run as one
dimension over the whole run with the override ``4 × 205 = <>``: the number it
claims is its own measurement (820), so the sheet stays honest under
``dimension_truth`` (a count prefix is never the claimed size); the stations
inside the run lose their extension lines.

``axis`` names the points (indices into ``points``) that lie on a wall's or a
partition's axis rather than on a face: every extension line at such a
station is drawn with the CENTER linetype (DIMLTEX1 / DIMLTEX2), the axis
convention; the dimension's linetype scale is fitted so that line holds two
CENTER sequences on paper (a shorter line plots solid, in AutoCAD too). An extension line of zero length - a point on its own dimension
line, as ``offset`` 0 gives - is suppressed (DIMSE1 / DIMSE2) instead of
leaving a DIMEXE tick, unless ``ext_length`` fixes every extension line's
length (DIMFXL): then each point gets a tick through it; an axis station on
the line is warned ``axis_on_line``.

ISO 129-1: each measurement once - the overall row is drawn only when the
chain has two rows or more.
"""

from __future__ import annotations

import math

from engineering.arch.dimension import TEXT_WIDTH_FACTOR
from engineering.dimtext import format_value

ORIENTATIONS: tuple[str, ...] = ("auto", "horizontal", "vertical", "aligned")
#: Default distance between the chain row and the overall row: 8 mm on paper.
STEP_PAPER_MM = 8.0
#: A span narrower than this many text heights takes no text beneath it.
NARROW_SPAN = 0.5
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


def _text_spec(key, value) -> tuple[str | None, str | None, tuple[float, float] | None]:
    """``texts[key]``: a string, or ``{text?, side?, shift?}`` -> (text, side, shift)."""
    if isinstance(value, str):
        return value, None, None
    if not isinstance(value, dict) or not value or set(value) - {"text", "side", "shift"}:
        raise ValueError(
            f"texts[{key!r}]: expected a string or {{text, side, shift}}, got {value!r}"
        )
    text, where, shift = value.get("text"), value.get("side"), value.get("shift")
    if text is not None and not isinstance(text, str):
        raise ValueError(f"texts[{key!r}].text: expected a string, got {type(text).__name__}")
    if where not in (None, "above", "below"):
        raise ValueError(f"texts[{key!r}].side: expected above or below, got {where!r}")
    if shift is not None:
        if not isinstance(shift, (list, tuple)) or len(shift) != 2:
            raise ValueError(f"texts[{key!r}].shift: expected [along, up], got {shift!r}")
        shift = (
            _number(shift[0], f"texts[{key!r}].shift[0]"),
            _number(shift[1], f"texts[{key!r}].shift[1]"),
        )
    return text, where, shift


def _axis(axis, count: int) -> set[int]:
    if axis is None:
        return set()
    if not isinstance(axis, (list, tuple)):
        raise ValueError(f"axis: expected a list of point indices, got {axis!r}")
    out = set()
    for i, k in enumerate(axis):
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < count:
            raise ValueError(
                f"axis[{i}]: expected an index into points (0..{count - 1}), got {k!r}"
            )
        out.add(k)
    return out


def plan_chain(
    points,
    offset,
    *,
    orientation: str = "auto",
    overall: bool = False,
    step=None,
    text_height: float = 2.5,
    gap: float = 0.625,
    tad: int = 1,
    style_vars=None,
    ext_beyond=None,
    leader: bool = True,
    merge_repeats: bool = False,
    axis=None,
    texts=None,
) -> dict:
    """The chain's dimensions and their text places, in drawing units; draws nothing.

    ``texts`` ``{index: text}`` overrides a chain row's text (0 the first span,
    after ``merge_repeats``; ``"overall"`` the overall row); ``<>`` in it is
    the measured value; a value may be ``{text?, side?, shift?}`` to place it
    (module docstring). A text is placed by its own width.

    ``text_height`` and ``gap`` are DIMTXT and DIMGAP times DIMSCALE, ``step``
    the distance from the chain row to the overall row (default
    ``STEP_PAPER_MM`` times ``text_height / 2.5``). Returns ``{orientation,
    angle, rows: [{row, from, to, value, dim_at, text_at, placement, leader,
    extension}], merged, repeats, warnings}``; a merged repeat row also carries
    ``repeat {count, value}`` and its ``text`` override; ``placement`` is ``inline`` (where the style puts it),
    ``staggered`` (the other side of the line, inside its span) or ``moved``
    (slid out of its span).
    """
    pts = _points(points)
    on_axis = _axis(axis, len(pts))
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
    # (along, across, point, on an axis)
    stations: list[tuple[float, float, tuple[float, float], bool]] = []
    merged: list[dict] = []
    for i, p in sorted(enumerate(pts), key=lambda q: local(q[1])[0]):
        s, t = local(p)
        if stations and abs(s - stations[-1][0]) <= eps:
            keep = stations[-1]
            nearer = t * side > keep[1] * side
            drop = keep[2] if nearer else p
            kept = (keep[0], t, p) if nearer else keep[:3]
            stations[-1] = (*kept, keep[3] or i in on_axis)
            merged.append({"point": list(drop), "kept": list(stations[-1][2])})
            continue
        stations.append((s, t, p, i in on_axis))
    if len(stations) < 2:
        raise ValueError(
            f"points: all project onto one station along the {orientation} direction; "
            "nothing to dimension"
        )

    outer = max(st[1] * side for st in stations) * side
    chain_line = outer + side * abs(offset)
    spans = list(zip(stations, stations[1:], strict=False))

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

    def fmt(value: float) -> str:
        return format_value(value, 0, style_vars)

    def width(text: str) -> float:
        return len(text) * TEXT_WIDTH_FACTOR * h

    def tight(span) -> bool:
        length = span[1][0] - span[0][0]
        return width(fmt(length)) + 2.0 * g > length

    # runs of two or more equal neighbouring spans whose texts do not fit
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(spans):
        j = i + 1
        if tight(spans[i]):
            length = spans[i][1][0] - spans[i][0][0]
            while (
                j < len(spans)
                and abs(spans[j][1][0] - spans[j][0][0] - length) <= eps
                and tight(spans[j])
            ):
                j += 1
        if j - i >= 2:
            runs.append((i, j))
        i = j
    repeats, warnings = [], []
    # units: (start station, end station, repeat (count, value) or None, text or None)
    units = [(a, b, None, None) for a, b in spans]
    for i, j in reversed(runs):
        a, b = spans[i][0], spans[j - 1][1]
        count, value = j - i, round(spans[i][1][0] - spans[i][0][0], 9)
        total = b[0] - a[0]
        repeats.append({"count": count, "value": value, "from": list(a[2]), "to": list(b[2])})
        if merge_repeats:
            units[i:j] = [(a, b, (count, value), f"{count} × {fmt(value)} = <>")]
        else:
            warnings.append(
                {
                    "code": "tight_repeats",
                    "count": count,
                    "value": value,
                    "from": list(a[2]),
                    "to": list(b[2]),
                    "message": f"{count} equal spans of {fmt(value)} do not fit their texts; "
                    f"merge_repeats=True draws one {count} × {fmt(value)} = {fmt(total)}",
                }
            )
    repeats.reverse()
    warnings.reverse()
    with_overall = bool(overall) and len(units) >= 2
    overall_text = None
    want: dict[int, tuple[str | None, tuple[float, float] | None]] = {}
    for key, value in (texts or {}).items():
        text, where, shift = _text_spec(key, value)
        if str(key) == "overall" and with_overall:
            if where or shift:
                raise ValueError(f"texts[{key!r}]: side and shift place a chain row's text")
            overall_text = text
            continue
        try:
            index = int(key)
        except (TypeError, ValueError):
            index = -1
        if not 0 <= index < len(units):
            raise ValueError(
                f"texts: key {key!r} names no row; the chain has rows 0..{len(units) - 1}"
                + (" and 'overall'" if with_overall else "")
            )
        a, b, repeat, old = units[index]
        units[index] = (a, b, repeat, old if text is None else text)
        if where or shift:
            want[index] = (where, shift)

    def label(unit) -> str:
        a, b, _, text = unit
        total = fmt(b[0] - a[0])
        return text.replace("<>", total) if text else total

    def free(rect, lines_away=()) -> bool:
        if any(rect[2] - g < at < rect[3] + g for at in lines_away):
            return False
        return not any(
            rect[0] < o[1] + sep and o[0] < rect[1] + sep and rect[2] < o[3] and o[2] < rect[3]
            for o in boxes
        )

    def rect(along: float, w: float, away: float, base: float = 0.0):
        return (along - w / 2.0, along + w / 2.0, base + away - h / 2.0, base + away + h / 2.0)

    # explicit places first: the others keep clear of them
    placed: dict[int, tuple[float, float, str]] = {}
    for i, (where, shift) in want.items():
        a, b = units[i][0], units[i][1]
        along, w = (a[0] + b[0]) / 2.0, width(label(units[i]))
        away = t0 if where is None else side * (g + h / 2.0) * (1 if where == "above" else -1)
        if shift:
            along, away = along + shift[0], away + side * shift[1]
        boxes.append(rect(along, w, away))
        kind = (
            "moved"
            if not a[0] < along < b[0]
            else "staggered"
            if abs(away - t0) > eps
            else "inline"
        )
        placed[i] = (along, away, kind)
    texts: list[tuple[float, float, str]] = []  # along, away from the chain line, placement
    tight: list[int] = []
    for i, unit in enumerate(units):
        if i in placed:
            texts.append(placed[i])
            continue
        a, b = unit[0], unit[1]
        mid, w = (a[0] + b[0]) / 2.0, width(label(unit))
        r = rect(mid, w, t0)
        texts.append((mid, t0, "inline"))
        if w + 2.0 * g <= b[0] - a[0] and free(r):
            boxes.append(r)
        else:
            tight.append(i)
    # the overall row's extension lines run through the tiers at the chain's ends
    ends = (stations[0][0], stations[-1][0]) if with_overall else ()
    for i in tight:
        a, b = units[i][0], units[i][1]
        mid, w = (a[0] + b[0]) / 2.0, width(label(units[i]))
        # a narrow span takes no text under it: out beside the neighbours, with a leader
        narrow = b[0] - a[0] < NARROW_SPAN * h
        spot = (
            None
            if narrow
            else next(
                ((mid, away, "staggered") for away in tiers if free(rect(mid, w, away))),
                None,
            )
        )
        if spot is None:
            alongs = {mid}
            for o in boxes:
                alongs |= {o[1] + sep + w / 2.0, o[0] - sep - w / 2.0}
            for e in (st[0] for st in stations):
                alongs |= {e + g + w / 2.0, e - g - w / 2.0}
            spot = next(
                (
                    (along, away, "staggered" if a[0] < along < b[0] else "moved")
                    for _, _, along, away in sorted(
                        # a narrow span's text takes the first tier beside it, not the nearest
                        (k, abs(x - mid), x, tiers[k]) if narrow else (abs(x - mid), k, x, tiers[k])
                        for x in alongs
                        for k in range(len(tiers))
                    )
                    if not (narrow and a[0] < along < b[0])
                    and free(r := rect(along, w, away))
                    and not any(r[0] - g < e < r[1] + g for e in ends)
                ),
                (mid, tiers[0], "staggered"),
            )
        boxes.append(rect(spot[0], w, spot[1]))
        texts[i] = spot

    rows = []
    for (a, b, repeat, text), (along, away, placement) in zip(units, texts, strict=True):
        row = _row("chain", a, b, chain_line, along, chain_line + side * away, placement)
        if repeat:
            row["repeat"] = {"count": repeat[0], "value": repeat[1]}
        if text:
            row["text"] = text
        rows.append(row)
    overall_away = None
    if with_overall:
        # beyond every chain text by a gap, at least ``step`` from the chain line
        reach = max((o[3] for o in boxes), default=0.0)
        # the line clears them, and so does the overall text's near edge (t0 - h/2)
        overall_away = max(step, reach + g + max(0.0, h / 2.0 - t0))
        line = chain_line + side * overall_away
        a, b = stations[0], stations[-1]
        rows.append(_row("overall", a, b, line, (a[0] + b[0]) / 2.0, line + side * t0, "inline"))
        if overall_text:
            rows[-1]["text"] = overall_text
    for row in rows:
        line = row.pop("_line")
        row["extension"] = [
            "axis"
            if st[3] and abs(st[1] - line) > eps
            else "line"
            if abs(st[1] - line) > eps
            else "suppressed"
            for st in row.pop("_ends")
        ]
        row["dim_at"] = [round(c, 9) for c in world(*row.pop("_dim"))]
        row["text_at"] = [round(c, 9) for c in world(*row.pop("_text"))]
        row["leader"] = bool(leader) and row["placement"] == "moved"
    for st in stations:
        if st[3] and abs(st[1] - chain_line) <= eps:
            warnings.append(
                {
                    "code": "axis_on_line",
                    "point": list(st[2]),
                    "message": "an axis point on the chain's own dimension line has no "
                    "extension line to draw dash-dot; give an offset to show it",
                }
            )
    return {
        "orientation": orientation,
        "angle": angle,
        "rows": rows,
        "merged": merged,
        "repeats": repeats,
        "warnings": warnings,
        "overall_offset": overall_away,
    }


def _row(row, a, b, line, along, across, placement) -> dict:
    return {
        "row": row,
        "from": list(a[2]),
        "to": list(b[2]),
        "value": round(b[0] - a[0], 9),
        "placement": placement,
        "_line": line,
        "_ends": (a, b),
        "_dim": ((a[0] + b[0]) / 2.0, line),
        "_text": (along, across),
    }


def _extension_vars(extension, fixed: bool = False) -> dict:
    """DIMSE / DIMLTEX overrides of a row; a fixed length keeps every line."""
    out = {}
    for end, kind in enumerate(extension, start=1):
        if kind == "suppressed" and not fixed:
            out[f"dimse{end}"] = 1
        elif kind == "axis":
            out[f"dimltex{end}"] = "CENTER"
    return out


async def draw_chain(
    backend,
    points,
    offset,
    *,
    orientation: str = "auto",
    overall: bool = False,
    step=None,
    scale=None,
    layer: str | None = None,
    leader: bool = True,
    merge_repeats: bool = False,
    axis=None,
    dry_run: bool = False,
    ext_length=None,
    style: str | None = None,
    texts=None,
) -> dict:
    """``plan_chain`` drawn as real linear DIMENSIONs inside one transaction.

    ``style`` draws (and plans) the chain in that dimension style, the
    current one left current; ``texts`` as :func:`plan_chain`.

    ``scale`` (a plot-scale denominator) is DIMSCALE while the chain is drawn,
    the drawing's own put back afterwards; None draws at the drawing's DIMSCALE.
    A failure half way rolls the whole chain back. ``ext_length`` (drawing
    units) gives every extension line that fixed length from the dimension
    line toward its point (DIMFXLON / DIMFXL), so a chain at ``offset`` 0
    still shows a tick through each point.
    """
    if style is None:
        return await _draw_chain(
            backend, points, offset, orientation, overall, step, scale, layer, leader,
            merge_repeats, axis, dry_run, ext_length, texts, None,
        )  # fmt: skip
    async with backend.dimstyle_scope(style) as name:
        return await _draw_chain(
            backend, points, offset, orientation, overall, step, scale, layer, leader,
            merge_repeats, axis, dry_run, ext_length, texts, name,
        )  # fmt: skip


async def _draw_chain(
    backend, points, offset, orientation, overall, step, scale, layer, leader,
    merge_repeats, axis, dry_run, ext_length, texts, style_name,
) -> dict:  # fmt: skip
    from engineering.mech.draw import guarded_transaction

    if scale is not None:
        scale = _number(scale, "scale")
        if scale <= 0:
            raise ValueError(f"scale: must be positive, got {scale:g}")
    previous = await backend.system_get_variable("DIMSCALE")
    previous = 1.0 if previous in (None, 0, 0.0) else float(previous)
    factor = previous if scale is None else scale
    fixed = {}
    if ext_length is not None:
        ext_length = _number(ext_length, "ext_length")
        if ext_length <= 0:
            raise ValueError(f"ext_length: must be positive, got {ext_length:g}")
        fixed = {"dimfxlon": 1, "dimfxl": ext_length / factor}  # DIMFXL is scaled by DIMSCALE
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
        merge_repeats=merge_repeats,
        axis=axis,
        texts=texts,
    )
    layer = layer or backend._role_layer("dim")
    result = {
        "ok": True,
        "orientation": plan["orientation"],
        "angle": plan["angle"],
        "layer": layer,
        "dimscale": factor,
        **({"style": style_name} if style_name else {}),
        "count": len(plan["rows"]),
        "staggered": sum(r["placement"] == "staggered" for r in plan["rows"]),
        "moved": sum(r["placement"] == "moved" for r in plan["rows"]),
        "dimensions": plan["rows"],
        "merged": plan["merged"],
        "repeats": plan["repeats"],
        "warnings": plan["warnings"],
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
                    text_override=row.get("text"),
                    text_at=tuple(row["text_at"]),
                    text_leader=row["leader"],
                    dimvars={**_extension_vars(row["extension"], bool(fixed)), **fixed},
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
