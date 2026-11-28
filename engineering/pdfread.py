"""Vector content of one PDF page - texts, lines, rectangles, fills - read only.

The reader is pypdf (BSD, pure Python), the ``pdfread`` extra; without it
:func:`read_pdf` refuses with the ``pdf_read`` capability naming the extra.

**Coordinates.** Every object is reported in *page millimetres*: ``u`` from
the left edge, ``v`` from the top edge, v down - the page as displayed (the
CropBox, ``/Rotate`` applied), which is the convention of an image pixel in
``view_screenshot``'s ``overlay_align``, with millimetres instead of pixels.
Pass ``align`` (pairs ``[u, v, x, y]``: page mm -> WCS point, the same list
shape ``overlay_align`` / ``image_map`` take; two pairs exact, three or more a
least-squares similarity with residuals) and every object also carries WCS
``x, y`` / ``points``; ``region`` is then a WCS window, otherwise a page-mm
window ``[u0, v0, u1, v1]``.

**Kinds.** ``text`` (a shown string at its insertion point, ``height`` its
font size and ``rotation`` its baseline angle, degrees CCW), ``rect`` (a
closed four-corner path with right angles - ``re`` or m / l / l / l / h, as
plotters write it: its ``corners``, ``stroke`` / ``fill`` colours), ``line`` (each
stroked subpath that is not a rect: ``points``, ``closed``; Bezier curves
flattened, ``curved: true``) and ``fill`` (each filled subpath that is not a
rect). A clip path (``W n``) paints nothing and is not reported. Images and
shadings are not read.

**Limits**, each named in the result or the refusal: :data:`MAX_BYTES` and
:data:`MAX_PAGES` refuse before the page is parsed; :data:`MAX_SECONDS` (wall
clock) and ``limit`` (at most :data:`MAX_OBJECTS`) truncate and say so in
``truncated``; Form XObjects nest at most :data:`MAX_FORM_DEPTH` deep.
pypdf's own decompression ceilings (``pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH``
and siblings) bound a stream bomb.
"""

from __future__ import annotations

import math
import time
from importlib.util import find_spec
from pathlib import Path

from backends.capability import UnsupportedCapabilityError
from backends.region_render import overlay_fit, overlay_transform

MAX_BYTES = 64 * 1024 * 1024
MAX_PAGES = 2000
MAX_SECONDS = 20.0
MAX_OBJECTS = 20000
DEFAULT_LIMIT = 2000
MAX_FORM_DEPTH = 8
KINDS = ("text", "line", "rect", "fill")
PT_MM = 25.4 / 72.0
_CURVE_STEPS = 8


class _Stop(Exception):
    """The object limit or the wall clock ran out; the reason is its argument."""


def available() -> bool:
    return find_spec("pypdf") is not None


def _refuse_missing() -> None:
    raise UnsupportedCapabilityError(
        "pdf_read",
        "pdf_read needs pypdf, which is not installed. Install it with: "
        "pip install 'autocad-mcp-pro[pdfread]' (or pip install pypdf).",
    )


def _mul(m, n):
    """m then n (row vectors, PDF order): the matrix of applying m, then n."""
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = n
    return (
        a * a2 + b * c2,
        a * b2 + b * d2,
        c * a2 + d * c2,
        c * b2 + d * d2,
        e * a2 + f * c2 + e2,
        e * b2 + f * d2 + f2,
    )


def _apply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def _color(operator: bytes, operands) -> str | None:
    try:
        values = [float(v) for v in operands if isinstance(v, int | float)]
    except (TypeError, ValueError):
        return None
    if operator in (b"g", b"G") and len(values) == 1:
        rgb = values * 3
    elif operator in (b"rg", b"RG") and len(values) == 3:
        rgb = values
    elif operator in (b"k", b"K") and len(values) == 4:
        c, m, y, k = values
        rgb = [(1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)]
    elif operator in (b"sc", b"SC", b"scn", b"SCN") and len(values) in (1, 3, 4):
        return _color({1: b"g", 3: b"rg", 4: b"k"}[len(values)], values)
    else:
        return None
    return "#" + "".join(f"{round(max(0.0, min(1.0, v)) * 255):02x}" for v in rgb)


class _Page:
    """Page user space (points) -> displayed page mm (origin top-left, v down)."""

    def __init__(self, page):
        box = page.cropbox
        self.x0, self.y0 = float(box.left), float(box.bottom)
        self.x1, self.y1 = float(box.right), float(box.top)
        self.rotate = int(page.get("/Rotate", 0) or 0) % 360
        unit = page.get("/UserUnit", 1.0)
        self.k = PT_MM * float(unit or 1.0)
        w, h = self.x1 - self.x0, self.y1 - self.y0
        if self.rotate in (90, 270):
            w, h = h, w
        self.size = (round(w * self.k, 3), round(h * self.k, 3))

    def uv(self, x: float, y: float) -> tuple[float, float]:
        r = self.rotate
        if r == 90:
            u, v = y - self.y0, x - self.x0
        elif r == 180:
            u, v = self.x1 - x, y - self.y0
        elif r == 270:
            u, v = self.y1 - y, self.x1 - x
        else:
            u, v = x - self.x0, self.y1 - y
        return (u * self.k, v * self.k)


def _bezier(p0, p1, p2, p3):
    out = []
    for i in range(1, _CURVE_STEPS + 1):
        t = i / _CURVE_STEPS
        s = 1 - t
        out.append(
            (
                s**3 * p0[0] + 3 * s * s * t * p1[0] + 3 * s * t * t * p2[0] + t**3 * p3[0],
                s**3 * p0[1] + 3 * s * s * t * p1[1] + 3 * s * t * t * p2[1] + t**3 * p3[1],
            )
        )
    return out


class _Collector:
    def __init__(self, page: _Page, kinds, limit: int, deadline: float, transform, window):
        self.page = page
        self.kinds = set(kinds)
        self.limit = limit
        self.deadline = deadline
        self.transform = transform
        self.window = window
        self.items: list[dict] = []
        self.ops = 0

    def tick(self) -> None:
        self.ops += 1
        if self.ops % 256 == 0 and time.monotonic() > self.deadline:
            raise _Stop("max_seconds")

    def add(self, item: dict) -> None:
        """Map ``item`` (page mm) to the output units, keep it if it meets the window."""
        if item["kind"] not in self.kinds:
            return
        row = _finish(item, self.transform)
        if self.window is not None:
            anchor = (
                row.get("corners")
                or row.get("points")
                or [[row.get("x", row["u"]), row.get("y", row["v"])]]
            )
            x0, y0, x1, y1 = _bbox(anchor)
            w = self.window
            if x1 < w[0] or x0 > w[2] or y1 < w[1] or y0 > w[3]:
                return
        if len(self.items) >= self.limit:
            raise _Stop("limit")
        self.items.append(row)

    def pts(self, matrix, points):
        return [self.page.uv(*_apply(matrix, x, y)) for x, y in points]

    def scale(self, matrix) -> float:
        return math.sqrt(abs(matrix[0] * matrix[3] - matrix[1] * matrix[2])) * self.page.k


def _finish(item: dict, transform) -> dict:
    """``item`` rounded, with WCS coordinates when ``transform`` is set."""
    row = dict(item)
    scale = abs(transform[0]) if transform else 1.0

    def to_out(u, v):
        if not transform:
            return [_r(u), _r(v)]
        z = transform[0] * complex(u, -v) + transform[1]
        return [_r(z.real), _r(z.imag)]

    if row["kind"] == "text":
        if transform:
            row["x"], row["y"] = to_out(row["u"], row["v"])
            row["rotation"] += math.degrees(math.atan2(transform[0].imag, transform[0].real))
        row["u"], row["v"] = _r(row["u"]), _r(row["v"])
        row["height"] = _r(row["height"] * scale)
        row["rotation"] = _r(row["rotation"] % 360.0)
        return row
    key = "corners" if row["kind"] == "rect" else "points"
    row[key] = [to_out(u, v) for u, v in row[key]]
    if row.get("width") is not None:
        row["width"] = _r(row["width"] * scale)
    return {k: v for k, v in row.items() if v is not None and v is not False}


def _corners(uv):
    """The four corners when the closed path ``uv`` is a rectangle, else None.

    A plotter writes a rectangle as m / l / l / l / h as often as ``re``, so
    the shape decides, not the operator."""
    pts = []
    for p in uv:
        if not pts or math.dist(p, pts[-1]) > 1e-6:
            pts.append(p)
    if len(pts) > 1 and math.dist(pts[0], pts[-1]) <= 1e-6:
        pts.pop()
    if len(pts) != 4:
        return None
    for i in range(4):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % 4]
        e1 = (a[0] - b[0], a[1] - b[1])
        e2 = (c[0] - b[0], c[1] - b[1])
        n1, n2 = math.hypot(*e1), math.hypot(*e2)
        if n1 <= 1e-9 or n2 <= 1e-9 or abs(e1[0] * e2[0] + e1[1] * e2[1]) > 1e-3 * n1 * n2:
            return None
    return pts


def _paint(col: _Collector, subpaths, matrix, state, *, stroke: bool, fill: bool) -> None:
    width = round(state["w"] * col.scale(matrix), 3)
    for path in subpaths:
        if len(path["pts"]) < 2:
            continue
        uv = col.pts(matrix, path["pts"])
        corners = _corners(uv) if path.get("rect") or path["closed"] else None
        if corners is not None:
            uv = corners
            col.add(
                {
                    "kind": "rect",
                    "corners": uv,
                    "stroke": state["stroke"] if stroke else None,
                    "fill": state["fill"] if fill else None,
                    "width": width if stroke else None,
                }
            )
            continue
        if stroke:
            col.add(
                {
                    "kind": "line",
                    "points": uv,
                    "closed": path["closed"],
                    "curved": path["curved"],
                    "stroke": state["stroke"],
                    "width": width,
                }
            )
        if fill and len(uv) >= 3:
            col.add({"kind": "fill", "points": uv, "curved": path["curved"], "fill": state["fill"]})


def _walk(col: _Collector, stream_owner, resources, reader, matrix, depth: int) -> None:
    """Interpret the path operators of a content stream (and its Form XObjects)."""
    from pypdf.generic import ContentStream

    contents = stream_owner.get_contents() if hasattr(stream_owner, "get_contents") else None
    if contents is None:
        contents = ContentStream(stream_owner, reader)
    elif not isinstance(contents, ContentStream):
        contents = ContentStream(contents, reader)
    state = {"ctm": matrix, "w": 1.0, "stroke": "#000000", "fill": "#000000"}
    stack: list[dict] = []
    subpaths: list[dict] = []
    current: dict | None = None
    start = (0.0, 0.0)
    for operands, operator in contents.operations:
        col.tick()
        op = operator
        try:
            if op == b"q":
                stack.append(dict(state))
            elif op == b"Q":
                if stack:
                    state = stack.pop()
            elif op == b"cm" and len(operands) == 6:
                state["ctm"] = _mul(tuple(float(v) for v in operands), state["ctm"])
            elif op == b"w" and operands:
                state["w"] = float(operands[0])
            elif op in (b"g", b"rg", b"k", b"sc", b"scn"):
                state["fill"] = _color(op, operands)
            elif op in (b"G", b"RG", b"K", b"SC", b"SCN"):
                state["stroke"] = _color(op, operands)
            elif op == b"m" and len(operands) == 2:
                start = (float(operands[0]), float(operands[1]))
                current = {"pts": [start], "closed": False, "curved": False}
                subpaths.append(current)
            elif op == b"l" and len(operands) == 2 and current is not None:
                current["pts"].append((float(operands[0]), float(operands[1])))
            elif op in (b"c", b"v", b"y") and current is not None:
                values = [float(v) for v in operands]
                p0 = current["pts"][-1]
                if op == b"c" and len(values) == 6:
                    p1, p2, p3 = values[0:2], values[2:4], values[4:6]
                elif op == b"v" and len(values) == 4:
                    p1, p2, p3 = p0, values[0:2], values[2:4]
                elif op == b"y" and len(values) == 4:
                    p1, p2, p3 = values[0:2], values[2:4], values[2:4]
                else:
                    continue
                current["pts"].extend(_bezier(p0, p1, p2, p3))
                current["curved"] = True
            elif op == b"h" and current is not None:
                current["closed"] = True
                current = {"pts": [start], "closed": False, "curved": False}
                subpaths.append(current)
            elif op == b"re" and len(operands) == 4:
                x, y, w, h = (float(v) for v in operands)
                subpaths.append(
                    {
                        "pts": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                        "closed": True,
                        "curved": False,
                        "rect": True,
                    }
                )
                start = (x, y)
                current = {"pts": [start], "closed": False, "curved": False}
                subpaths.append(current)
            elif op in (b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"n"):
                if op in (b"s", b"b", b"b*") and current is not None:
                    current["closed"] = True
                if op != b"n":
                    closing = op in (b"f", b"F", b"f*")
                    for path in subpaths:
                        if closing:
                            path["closed"] = True
                    _paint(
                        col,
                        subpaths,
                        state["ctm"],
                        state,
                        stroke=op in (b"S", b"s", b"B", b"B*", b"b", b"b*"),
                        fill=op not in (b"S", b"s"),
                    )
                subpaths, current = [], None
            elif op == b"Do" and operands and depth < MAX_FORM_DEPTH:
                xobjects = (resources or {}).get("/XObject")
                xobj = xobjects.get_object().get(operands[0]) if xobjects else None
                xobj = xobj.get_object() if xobj is not None else None
                if xobj is not None and xobj.get("/Subtype") == "/Form":
                    form_m = tuple(float(v) for v in xobj.get("/Matrix", (1, 0, 0, 1, 0, 0)))
                    inner = xobj.get("/Resources")
                    _walk(
                        col,
                        xobj,
                        inner.get_object() if inner is not None else resources,
                        reader,
                        _mul(form_m, state["ctm"]),
                        depth + 1,
                    )
        except (TypeError, ValueError, IndexError, AttributeError, KeyError):
            continue  # a malformed operator is skipped, as viewers do


def _texts(col: _Collector, page) -> None:
    def visit(text, cm, tm, _font, size):
        col.tick()
        if not text or not text.strip():
            return
        m = _mul(tuple(float(v) for v in tm), tuple(float(v) for v in cm))
        u, v = col.page.uv(*_apply(m, 0.0, 0.0))
        ux, uy = col.page.uv(*_apply(m, 1.0, 0.0))
        hx, hy = col.page.uv(*_apply(m, 0.0, 1.0))
        rotation = math.degrees(math.atan2(-(uy - v), ux - u))
        height = math.hypot(hx - u, hy - v) * float(size or 0.0)
        col.add(
            {
                "kind": "text",
                "text": text.strip(),
                "u": u,
                "v": v,
                "height": height,
                "rotation": rotation,
            }
        )

    page.extract_text(visitor_text=visit)


def _check_args(page: int, kinds, limit: int):
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError(f"page: a page number from 1, got {page!r}.")
    chosen = list(KINDS if kinds is None else kinds)
    unknown = sorted(set(chosen) - set(KINDS))
    if unknown or not chosen:
        raise ValueError(f"kinds: choose from {list(KINDS)}, got {chosen!r}.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_OBJECTS:
        raise ValueError(f"limit: 1-{MAX_OBJECTS} objects (MAX_OBJECTS), got {limit!r}.")
    return chosen


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _r(value: float) -> float:
    return round(value, 3) + 0.0


def read_pdf(
    path: str | Path,
    *,
    page: int = 1,
    align=None,
    region=None,
    kinds=None,
    limit: int = DEFAULT_LIMIT,
) -> dict:
    """See the module docstring. ``ValueError`` names the argument or the limit."""
    chosen = _check_args(page, kinds, limit)
    if not available():
        _refuse_missing()
    transform = overlay_transform(align) if align is not None else None
    window = None
    if region is not None:
        from backends.region_render import normalize_region

        window = normalize_region(region)
    target = Path(path)
    size = target.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"path: {size} bytes is over MAX_BYTES={MAX_BYTES}; nothing was read.")
    with open(target, "rb") as handle:
        if not handle.read(1024).lstrip(b"\xef\xbb\xbf").startswith(b"%PDF-"):
            raise ValueError(f"path: '{target.name}' is not a PDF (no %PDF- header).")
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    started = time.monotonic()
    try:
        reader = PdfReader(str(target))
        pages = len(reader.pages)
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"path: '{target.name}' does not parse as a PDF ({exc}).") from exc
    if pages > MAX_PAGES:
        raise ValueError(f"path: {pages} pages is over MAX_PAGES={MAX_PAGES}; nothing was read.")
    if page > pages:
        raise ValueError(f"page: the PDF has {pages} page(s), got page={page}.")
    pdf_page = reader.pages[page - 1]
    geometry = _Page(pdf_page)
    col = _Collector(geometry, chosen, limit, started + MAX_SECONDS, transform, window)
    truncated = None
    try:
        if "text" in col.kinds:
            _texts(col, pdf_page)
        if col.kinds - {"text"}:
            resources = pdf_page.get("/Resources")
            _walk(
                col,
                pdf_page,
                resources.get_object() if resources is not None else None,
                reader,
                (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                0,
            )
    except _Stop as stop:
        truncated = stop.args[0]
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"path: page {page} does not parse ({exc}).") from exc

    out_items = col.items
    counts = {kind: 0 for kind in chosen}
    for row in out_items:
        counts[row["kind"]] += 1
    result: dict = {
        "ok": True,
        "page": page,
        "pages": pages,
        "page_mm": list(geometry.size),
        "rotate": geometry.rotate,
        "units": "wcs" if transform else "page_mm",
        "counts": counts,
        "objects": out_items,
        "seconds": round(time.monotonic() - started, 3),
    }
    if transform:
        fit = overlay_fit(align)
        fit["mm_to_units"] = round(fit.pop("units_per_px"), 6)
        fit["rotation_deg"] = round(fit["rotation_deg"], 4) + 0.0
        for key in ("rms", "max"):
            if key in fit:
                fit[key] = round(fit[key], 3)
        if "residuals" in fit:
            fit["residuals"] = [round(e, 3) for e in fit["residuals"]]
        result["fit"] = fit
    if truncated:
        result["truncated"] = {
            "reason": truncated,
            "limit": limit if truncated == "limit" else MAX_SECONDS,
            "note": (
                f"stopped at {len(col.items)} objects (limit={limit}); narrow kinds or region"
                if truncated == "limit"
                else f"stopped after MAX_SECONDS={MAX_SECONDS} of reading"
            ),
        }
    return result
