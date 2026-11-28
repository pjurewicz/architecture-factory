"""Reading a calibrated scan: a gridded crop, colour features, a comparison (round 5, B4/B5).

Every mode takes ``align``, ``view_screenshot``'s ``overlay_align`` (pairs
``[u, v, x, y]``, image pixel with origin top-left and v down -> WCS point),
and answers in drawing units. Nothing here writes the drawing; the scan is a
file the caller has already passed through ``security.validate_path``.

Pixel convention: pixel ``(i, j)`` (column, row) covers ``[i, i + 1) x [j, j + 1)``
in the ``(u, v)`` the calibration uses, so its centre is ``(i + 0.5, j + 0.5)``.

* :func:`grid_crop` - the scan resampled into a WCS-aligned window with a
  labelled coordinate grid in drawing units drawn over it (PNG).
* :func:`find_rects` - axis-aligned filled rectangles of one colour (wall
  poché): row runs of the colour merged downwards while their ends stay within
  a pixel, so an L or a T becomes its arms.
* :func:`find_blobs` - 8-connected components of one colour (symbols, dots,
  rings): centroid, size and pixel count.
* :func:`compare` - the drawing rasterised (``view_render``) at the scan's
  resolution, resampled into the scan's pixels through ``align``, and the two
  line masks compared within a pixel tolerance.

Pillow reads the file; numpy does the rest (no scipy). Without Pillow every
mode that opens a file is refused with the capability ``scan_image``.
"""

from __future__ import annotations

import io
import math
from importlib.util import find_spec

from backends.base import UnsupportedCapabilityError
from backends.region_render import (
    OVERLAY_MAX_SOURCE_PX,
    check_max_px,
    normalize_region,
    overlay_transform,
)

#: Cap of the rows a detection returns (``limit``), and its default.
MAX_LIMIT, DEFAULT_LIMIT = 1000, 100
#: A colour mask with more row runs than this is noise, not features.
MAX_RUNS = 400_000
#: A scan pixel is ink when some channel differs from the paper's by more than
#: this (0-255): orange notes, red dimensions and grey poche count, not only
#: dark lines (round 6; a luminance threshold lost them).
SOURCE_INK_DISTANCE = 48
#: A source pixel with ink all round it within this radius is inside a fill.
FILL_CORE_PX = 3
#: A rendered pixel is ink when some channel is at most this (white paper).
INK_CHANNEL = 191
#: rects / walls: a ``skip`` pixel counts as the fill when the fill lies on
#: both sides of it within this many pixels (a label over a wall, not beside it).
SKIP_REACH_PX = 32
#: Compare: a line pixel counts as matched within this many pixels.
COMPARE_TOLERANCE_PX = 2
#: Compare: the regions of largest difference returned.
COMPARE_REGIONS = 8
#: Grid mode never enlarges the scan more than this by default.
MAX_AUTO_ZOOM = 4.0


def pillow_available() -> bool:
    return find_spec("PIL") is not None


def _require_pillow() -> None:
    if not pillow_available():
        raise UnsupportedCapabilityError(
            "scan_image",
            "scan_read needs Pillow to read the image file, and it is not installed. "
            'Install it with: pip install Pillow (or pip install -e ".[pdf]"). '
            "mode='map' (pure arithmetic on the calibration) works without it.",
        )


def parse_color(color) -> tuple[int, int, int]:
    """``'#rrggbb'`` / ``'rrggbb'`` / ``[r, g, b]`` -> (r, g, b); ``ValueError`` names ``color``."""
    if color is None:
        raise ValueError("color: give the colour to find, '#rrggbb' or [r, g, b].")
    if isinstance(color, str):
        text = color.strip().lstrip("#")
        if len(text) == 6:
            try:
                return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
            except ValueError:
                pass
        raise ValueError(f"color: expected '#rrggbb', got {color!r}.")
    try:
        r, g, b = (int(v) for v in color)
    except (TypeError, ValueError):
        raise ValueError(f"color: expected [r, g, b] or '#rrggbb', got {color!r}.") from None
    if not all(0 <= v <= 255 for v in (r, g, b)):
        raise ValueError(f"color: each channel must be 0-255, got {color!r}.")
    return r, g, b


def calibrate(spec, image_size) -> dict:
    """``align`` from one of two shortcuts (round 6), as ``{align, how, ...}``.

    * ``{"sheet": "A3", "scale": 50, "origin": [x, y]}`` - the scan is the whole
      page at 1:N: its pixel box is the sheet (orientation from the image's
      aspect), the page's bottom-left corner at ``origin`` (default 0, 0);
      ``dpi`` and ``aspect_delta_pct`` say how well the page fits.
    * ``{"points": [[u1, v1], [u2, v2]], "length": L, "at": [x, y], "angle": a}``
      - two scan pixels a known length apart: the first lands on ``at``
      (default 0, 0), the second ``length`` along ``angle`` degrees (default
      the direction the two pixels have in the image: no rotation).

    ``image_size`` (width, height) is needed by the sheet form only.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"calibrate: a dict (sheet + scale, or points + length), got {spec!r}.")

    def xy(value, where):
        try:
            x, y = (float(v) for v in value)
        except (TypeError, ValueError):
            raise ValueError(f"calibrate.{where}: [x, y], got {value!r}.") from None
        return x, y

    if "sheet" in spec:
        from engineering.standards.papers import paper_size_mm

        extra = set(spec) - {"sheet", "scale", "origin"}
        if extra:
            raise ValueError(f"calibrate: unknown key(s) {sorted(extra)} for the sheet form.")
        if image_size is None:
            raise ValueError("calibrate: the sheet form reads the scan's size; give image.")
        w, h = image_size
        try:
            scale = float(spec.get("scale", 1))
        except (TypeError, ValueError):
            scale = 0.0
        if not scale > 0:
            raise ValueError(
                f"calibrate.scale: the denominator N of 1:N, > 0, got {spec.get('scale')!r}."
            )
        try:
            pw, ph = paper_size_mm(str(spec["sheet"]), "landscape" if w >= h else "portrait")
        except (KeyError, ValueError) as exc:
            raise ValueError(f"calibrate.sheet: {exc}") from None
        x0, y0 = xy(spec.get("origin", (0, 0)), "origin")
        align = [[0, h, x0, y0], [w, 0, x0 + pw * scale, y0 + ph * scale]]
        return {
            "align": align,
            "how": f"page = {spec['sheet']} {pw:g} x {ph:g} mm at 1:{scale:g}",
            "dpi": round(w / (pw / 25.4), 2),
            "aspect_delta_pct": round(abs((w / h) / (pw / ph) - 1.0) * 100.0, 2),
        }
    if "points" in spec:
        extra = set(spec) - {"points", "length", "at", "angle"}
        if extra:
            raise ValueError(f"calibrate: unknown key(s) {sorted(extra)} for the points form.")
        pts = spec["points"]
        if not isinstance(pts, list | tuple) or len(pts) != 2:
            raise ValueError(f"calibrate.points: two pixels [[u1, v1], [u2, v2]], got {pts!r}.")
        (u1, v1), (u2, v2) = (xy(p, "points") for p in pts)
        try:
            length = float(spec.get("length"))
        except (TypeError, ValueError):
            length = 0.0
        if not length > 0 or math.hypot(u2 - u1, v2 - v1) < 1e-9:
            raise ValueError("calibrate: points two different pixels, length > 0 in drawing units.")
        x, y = xy(spec.get("at", (0, 0)), "at")
        angle = spec.get("angle")
        a = math.atan2(-(v2 - v1), u2 - u1) if angle is None else math.radians(float(angle))
        align = [[u1, v1, x, y], [u2, v2, x + length * math.cos(a), y + length * math.sin(a)]]
        return {
            "align": [[round(c, 6) for c in pair] for pair in align],
            "how": f"{length:g} units between the two pixels",
            "units_per_px": round(length / math.hypot(u2 - u1, v2 - v1), 6),
        }
    raise ValueError("calibrate: give sheet + scale, or points + length.")


def image_size(path: str) -> tuple[int, int]:
    """The scan's (width, height) in pixels, read from its header."""
    _require_pillow()
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as source:
            return source.size
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise ValueError(f"image: {path} does not exist") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"image: {path} is not an image Pillow can read") from exc


def page_align(size, box) -> list[list[float]]:
    """The align that fits a scan of ``size`` px into the paper ``box`` page to
    page, as ``view_screenshot(layout=)`` does (uniform scale, centred)."""
    w, h = size
    x0, y0, x1, y1 = box
    k = min((x1 - x0) / w, (y1 - y0) / h)
    left, top = (x0 + x1 - w * k) / 2.0, (y0 + y1 + h * k) / 2.0
    return [[0, 0, left, top], [w, h, left + w * k, top - h * k]]


class Calibration:
    """``align`` as the similarity model = a·(u - i·v) + b and its inverse."""

    def __init__(self, align):
        self.a, self.b = overlay_transform(align)
        self.upp = abs(self.a)
        self.rotation = math.degrees(math.atan2(self.a.imag, self.a.real))

    def to_wcs(self, u, v):
        z = self.a * (u - 1j * v) + self.b
        return z.real, z.imag

    def to_px(self, x, y):
        p = (complex(x, y) - self.b) / self.a
        return p.real, -p.imag

    def box_to_wcs(self, u0, v0, u1, v1) -> tuple[float, float, float, float]:
        pts = [self.to_wcs(u, v) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def box_to_px(self, x0, y0, x1, y1) -> tuple[float, float, float, float]:
        pts = [self.to_px(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        us, vs = [p[0] for p in pts], [p[1] for p in pts]
        return min(us), min(vs), max(us), max(vs)


def _open(path: str):
    """The file as an RGB PIL image; ``ValueError`` names ``image``."""
    _require_pillow()
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as source:
            source.load()
            return source.convert("RGB")
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise ValueError(f"image: {path} does not exist") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"image: {path} is not an image Pillow can read") from exc


def _window_px(cal: Calibration, size, region, region_px) -> tuple[int, int, int, int]:
    """The pixel window [u0, v0, u1, v1) (ints, clipped to the scan) to read."""
    width, height = size
    if region is not None and region_px is not None:
        raise ValueError("region and region_px are one window: give one of them.")
    if region_px is not None:
        try:
            u0, v0, u1, v1 = normalize_region(region_px)
        except ValueError as exc:
            raise ValueError(str(exc).replace("region", "region_px", 1)) from None
    elif region is not None:
        u0, v0, u1, v1 = cal.box_to_px(*normalize_region(region))
    else:
        u0, v0, u1, v1 = 0, 0, width, height
    box = (
        max(0, math.floor(u0)),
        max(0, math.floor(v0)),
        min(width, math.ceil(u1)),
        min(height, math.ceil(v1)),
    )
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        where = "region_px" if region_px is not None else "region"
        raise ValueError(
            f"{where}: the window lies outside the scan ({width} x {height} px); "
            f"the scan covers WCS {[round(v, 3) for v in cal.box_to_wcs(0, 0, width, height)]}."
        )
    return box


def _pixels(path: str, cal: Calibration, region, region_px):
    """(RGB array of the window, (u0, v0), k original px per array px, scan size)."""
    import numpy as np

    image = _open(path)
    size = image.size
    u0, v0, u1, v1 = _window_px(cal, size, region, region_px)
    crop = image.crop((u0, v0, u1, v1))
    k = 1.0
    if max(crop.size) > OVERLAY_MAX_SOURCE_PX:
        k = max(crop.size) / OVERLAY_MAX_SOURCE_PX
        crop = crop.resize(
            (max(1, round(crop.size[0] / k)), max(1, round(crop.size[1] / k))), resample=2
        )
        k = (u1 - u0) / crop.size[0]
    return np.asarray(crop, dtype=np.int16), (u0, v0), k, size


def _scan_report(path, cal: Calibration, size, origin, k, arr) -> dict:
    u0, v0 = origin
    h, w = arr.shape[:2]
    return {
        "path": str(path),
        "pixels": list(size),
        "window_px": [u0, v0, round(u0 + w * k, 3), round(v0 + h * k, 3)],
        "units_per_px": round(cal.upp, 6),
        "rotation_deg": round(cal.rotation, 4) + 0.0,
        **({"reduced": round(k, 4)} if k != 1.0 else {}),
    }


def _limit(limit) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit: a whole number 1-{MAX_LIMIT}, got {limit!r}.")
    return limit


def _size_bounds(size, cal: Calibration, k: float) -> tuple[float, float]:
    """``size`` [min, max] in drawing units (either may be None); default min 3 px."""
    lo = hi = None
    if size is not None:
        if isinstance(size, str | bytes) or not isinstance(size, list | tuple):
            raise ValueError(f"size: expected [min, max] in drawing units, got {size!r}.")
        if len(size) not in (1, 2):
            raise ValueError(f"size: expected [min, max] in drawing units, got {size!r}.")
        lo, hi = (list(size) + [None])[:2]
    try:
        lo = 3.0 * cal.upp * k if lo is None else float(lo)
        hi = math.inf if hi is None else float(hi)
    except (TypeError, ValueError):
        raise ValueError(f"size: expected [min, max] in drawing units, got {size!r}.") from None
    if lo < 0 or hi < lo:
        raise ValueError(f"size: need 0 <= min <= max, got {size!r}.")
    return lo, hi


def _mask(arr, color, tolerance):
    import numpy as np

    rgb = parse_color(color)
    if isinstance(tolerance, bool) or not isinstance(tolerance, int | float):
        raise ValueError(f"tolerance: a number 0-255, got {tolerance!r}.")
    if not 0 <= tolerance <= 255:
        raise ValueError(f"tolerance: a number 0-255, got {tolerance!r}.")
    diff = np.abs(arr - np.array(rgb, dtype=np.int16)).max(axis=2)
    return diff <= tolerance, rgb


def _runs(mask):
    """Row runs of ``mask``: arrays (row, start, end) with end exclusive, row-major."""
    import numpy as np

    h, w = mask.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    edges = np.diff(padded, axis=1)
    rs, cs = np.nonzero(edges == 1)
    re, ce = np.nonzero(edges == -1)
    if len(rs) > MAX_RUNS:
        raise ValueError(
            f"color/tolerance: {len(rs)} separate runs of that colour - this is noise, "
            "not features; narrow the tolerance or the region."
        )
    # nonzero is row-major, and each row's starts and ends alternate, so they pair
    return rs, cs, ce


def _feature(cal: Calibration, origin, k, c0, r0, c1, r1, count) -> dict:
    """A detected box in array px [c0, c1) x [r0, r1) as a WCS row."""
    u0, v0 = origin
    c0, r0, c1, r1, count = int(c0), int(r0), int(c1), int(r1), float(count)
    pu0, pv0, pu1, pv1 = u0 + c0 * k, v0 + r0 * k, u0 + c1 * k, v0 + r1 * k
    cx, cy = cal.to_wcs((pu0 + pu1) / 2.0, (pv0 + pv1) / 2.0)
    return {
        "center": [round(cx, 3), round(cy, 3)],
        "size": [round((pu1 - pu0) * cal.upp, 3), round((pv1 - pv0) * cal.upp, 3)],
        "box": [round(v, 3) for v in cal.box_to_wcs(pu0, pv0, pu1, pv1)],
        "px": [round(pu0, 2), round(pv0, 2), round(pu1, 2), round(pv1, 2)],
        "pixels": int(count),
        "fill": round(count / max(1, (c1 - c0) * (r1 - r0)), 3),
    }


def _bounded(rows: list[dict], limit: int, out: dict, key: str) -> dict:
    out["count"] = len(rows)
    out[key] = rows[:limit]
    if len(rows) > limit:
        out["truncated"] = True
    return out


def _detect_head(path, align, region, region_px, color, tolerance, size, limit, skip=None):
    cal = Calibration(align)
    limit = _limit(limit)
    arr, origin, k, scan_size = _pixels(path, cal, region, region_px)
    mask, rgb = _mask(arr, color, tolerance)
    if skip:  # round 7: colours that cross the fill (dimensions, hatching) count as it
        if (
            isinstance(skip, str)
            or not isinstance(skip, list | tuple)
            or (len(skip) == 3 and all(isinstance(c, int) for c in skip))
        ):
            skip = [skip]
        over = mask.copy()
        for other in skip:
            over &= False
            over |= _mask(arr, other, tolerance)[0]
            mask |= over & _between(mask, SKIP_REACH_PX)
    lo, hi = _size_bounds(size, cal, k)
    head = {
        "ok": True,
        "scan": _scan_report(path, cal, scan_size, origin, k, arr),
        "color": "#{:02x}{:02x}{:02x}".format(*rgb),
        "tolerance": tolerance,
        "size_limits": [round(lo, 3), None if math.isinf(hi) else round(hi, 3)],
    }
    if abs(math.remainder(cal.rotation, 90.0)) > 0.05:
        head["note"] = (
            "the scan is turned against WCS: sizes are along the image axes and "
            "box is the WCS bounding box of the turned rectangle"
        )
    return cal, mask, origin, k, lo, hi, head, limit


def _reach(mask, axis: int, reverse: bool):
    """Per pixel, the distance to the nearest ``mask`` pixel before it along
    ``axis`` (after it when ``reverse``); a large number where there is none."""
    import numpy as np

    m = np.flip(mask, axis) if reverse else mask
    n = m.shape[axis]
    idx = np.arange(n).reshape((-1, 1) if axis == 0 else (1, -1))
    last = np.maximum.accumulate(np.where(m, idx, -(10**9)), axis=axis)
    dist = idx - last
    return np.flip(dist, axis) if reverse else dist


def _between(mask, reach: int):
    """Pixels with ``mask`` on both sides within ``reach`` px, along a row or a
    column: where a crossing line or label cuts a fill, not where it overhangs."""
    out = None
    for axis in (0, 1):
        both = (_reach(mask, axis, False) <= reach) & (_reach(mask, axis, True) <= reach)
        out = both if out is None else out | both
    return out


def _boxes(mask) -> list[list[int]]:
    """Row runs merged downwards while their ends stay within a pixel:
    [c0, r0, c1, r1, count] in array px, so an L or a T becomes its arms."""
    rows, starts, ends = _runs(mask)
    done: list[list[int]] = []  # [c0, r0, c1, r1, count, last start, last end]
    open_: list[list[int]] = []
    i, n = 0, len(rows)
    while i < n:
        row = rows[i]
        j = i
        while j < n and rows[j] == row:
            j += 1
        still: list[list[int]] = []
        pool = [r for r in open_ if r[3] == row]  # rects that reached the previous row
        for s, e in zip(starts[i:j].tolist(), ends[i:j].tolist(), strict=True):
            hit = None
            for rect in pool:
                if abs(rect[5] - s) <= 1 and abs(rect[6] - e) <= 1:
                    hit = rect
                    break
            if hit is None:
                still.append([s, row, e, row + 1, e - s, s, e])
            else:
                pool.remove(hit)
                hit[0], hit[2] = min(hit[0], s), max(hit[2], e)
                hit[3] = row + 1
                hit[4] += e - s
                still.append(hit)
        kept = {id(r) for r in still}
        done.extend(r for r in open_ if id(r) not in kept)
        open_ = still
        i = j
    done.extend(open_)
    return [r[:5] for r in done]


def _wall_bands(boxes, mask, gap_px: float) -> list[dict]:
    """Boxes as bands along their long side, collinear ones joined (round 7).

    A band is {h, a0, a1, c, t, count, gaps}: horizontal or not, its run
    [a0, a1) along the axis, its centre line c across and thickness t, in
    array px. Two bands of one direction whose centre lines and thicknesses
    agree within 15 % (2 px at least) are one band when the stretch between
    them is filled in ``mask`` (a junction the run merge cut) or no longer
    than ``gap_px``; such a gap is kept in ``gaps`` (an opening candidate).
    """
    bands = []
    for c0, r0, c1, r1, count in boxes:
        h = (c1 - c0) >= (r1 - r0)
        a0, a1, b0, b1 = (c0, c1, r0, r1) if h else (r0, r1, c0, c1)
        bands.append(
            {"h": h, "a0": a0, "a1": a1, "c": (b0 + b1) / 2.0, "t": float(b1 - b0),
             "count": count, "gaps": []}
        )  # fmt: skip
    merged = True
    while merged:
        merged = False
        bands.sort(key=lambda b: (b["h"], round(b["c"]), b["a0"]))
        out: list[dict] = []
        for band in bands:
            for prev in reversed(out):
                if prev["h"] != band["h"]:
                    continue
                tol = max(2.0, 0.15 * max(prev["t"], band["t"]))
                if abs(prev["c"] - band["c"]) > tol or abs(prev["t"] - band["t"]) > tol:
                    continue
                first, second = (prev, band) if prev["a0"] <= band["a0"] else (band, prev)
                gap = second["a0"] - first["a1"]
                filled = gap <= 0
                if not filled:
                    lo = max(0, int(round(band["c"] - band["t"] / 4)))
                    hi = max(lo + 1, int(round(band["c"] + band["t"] / 4)))
                    sl = (
                        mask[lo:hi, first["a1"] : second["a0"]]
                        if band["h"]
                        else mask[first["a1"] : second["a0"], lo:hi]
                    )
                    filled = sl.size > 0 and float(sl.mean()) >= 0.8
                if not filled and gap > gap_px:
                    continue
                la, lb = prev["a1"] - prev["a0"], band["a1"] - band["a0"]
                prev["c"] = (prev["c"] * la + band["c"] * lb) / (la + lb)
                prev["t"] = (prev["t"] * la + band["t"] * lb) / (la + lb)
                prev["gaps"] += band["gaps"] + ([] if filled else [(first["a1"], second["a0"])])
                prev["a0"], prev["a1"] = min(prev["a0"], band["a0"]), max(prev["a1"], band["a1"])
                prev["count"] += band["count"]
                merged = True
                break
            else:
                out.append(band)
        bands = out
    return bands


def _band_box(b) -> tuple[float, float, float, float]:
    """A band as [c0, r0, c1, r1] in array px."""
    lo, hi = b["c"] - b["t"] / 2.0, b["c"] + b["t"] / 2.0
    return (b["a0"], lo, b["a1"], hi) if b["h"] else (lo, b["a0"], hi, b["a1"])


def find_rects(
    path,
    align,
    *,
    color,
    tolerance=40,
    size=None,
    region=None,
    region_px=None,
    limit=100,
    skip=None,
    gap=None,
):
    """Axis-aligned filled rectangles of ``color``; see the module docstring.

    ``skip`` colours count as ``color``; with ``gap`` (drawing units) collinear
    rectangles of one thickness are joined across gaps up to it."""
    cal, mask, origin, k, lo, hi, out, limit = _detect_head(
        path, align, region, region_px, color, tolerance, size, limit, skip
    )
    unit = cal.upp * k
    boxes = _boxes(mask)
    if gap is not None:
        gap_px = _nonnegative(gap, "gap") / unit
        boxes = []
        for b in _wall_bands(_boxes(mask), mask, gap_px):
            c0, r0, c1, r1 = _band_box(b)
            boxes.append([round(c0), round(r0), round(c1), round(r1), b["count"]])
    found = []
    for c0, r0, c1, r1, count in boxes:
        w, h = (c1 - c0) * unit, (r1 - r0) * unit
        if lo <= w <= hi and lo <= h <= hi:
            found.append(_feature(cal, origin, k, c0, r0, c1, r1, count))
    found.sort(key=lambda f: -f["pixels"])
    return _bounded(found, limit, out, "rects")


def _nonnegative(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not value >= 0:
        raise ValueError(f"{name}: a number of drawing units >= 0, got {value!r}.")
    return float(value)


def find_walls(
    path,
    align,
    *,
    color,
    tolerance=40,
    size=None,
    region=None,
    region_px=None,
    limit=100,
    skip=None,
    gap=None,
    origin=None,
    step=None,
):
    """Wall poche of ``color`` read as axes and thicknesses (round 7).

    The mask (``skip`` colours counted as ``color``) is closed over 2 px cuts
    and opened by the minimum thickness (``size[0]``, default 3 px), so lines,
    hatching and lettering drop out; its rectangles become bands joined along
    their axes (:func:`_wall_bands`, gaps up to ``gap`` kept as openings) and each
    band end that touches a crossing band is carried onto that band's axis, so
    L, T and X meet as ``arch_plan_from_spec`` joins them. Rows are spec walls
    {id, axis, thickness} and openings {wall, offset, width} (kind not read);
    ``origin`` is subtracted from every point, ``step`` rounds (default 1).
    """
    cal, mask, at, k, lo, hi, out, limit = _detect_head(
        path, align, region, region_px, color, tolerance, size, limit, skip
    )
    unit = cal.upp * k
    gap_px = 0.0 if gap is None else _nonnegative(gap, "gap") / unit
    step = 1.0 if step is None else _nonnegative(step, "grid")
    if step == 0:
        raise ValueError("grid: the walls rounding step must be above zero.")
    ox, oy = (0.0, 0.0) if origin is None else _xy(origin, "origin")
    r = max(0, int((lo / unit - 1) // 2))
    mask = _erode(_dilate(mask, 2), 2)
    if r:
        mask = _dilate(_erode(mask, r), r)
    bands, dropped = [], 0
    for b in _wall_bands(_boxes(mask), mask, gap_px):
        if not lo <= b["t"] * unit <= hi or b["a1"] - b["a0"] < b["t"]:
            dropped += 1  # thinner / thicker than size, or a pier no longer than thick
            continue
        bands.append(b)
    ends = []  # each band end carried onto the axis of a crossing band it touches
    for b in bands:
        new = [float(b["a0"]), float(b["a1"])]
        for i, e in enumerate((b["a0"], b["a1"])):
            probe = (e, e + b["t"]) if i == 0 else (e - b["t"], e)
            box = _band_box({**b, "a0": probe[0], "a1": probe[1]})
            best = None
            for other in bands:
                if other["h"] == b["h"]:
                    continue
                ob = _band_box(other)
                if (
                    all((box[j] <= ob[j + 2] + 2 and ob[j] <= box[j + 2] + 2) for j in (0, 1))
                    and abs(other["c"] - e) <= b["t"] + other["t"]
                ):
                    if best is None or abs(other["c"] - e) < abs(best - e):
                        best = other["c"]
            if best is not None:
                new[i] = best
        ends.append(new)

    def wcs(b, a):
        c, v = (a, b["c"]) if b["h"] else (b["c"], a)
        x, y = cal.to_wcs(at[0] + c * k, at[1] + v * k)
        return x - ox, y - oy

    def rnd(v):
        q = round(v / step) * step
        return int(q) if float(q).is_integer() else round(q, 3)

    walls, openings = [], []
    order = sorted(range(len(bands)), key=lambda i: -(ends[i][1] - ends[i][0]) * bands[i]["t"])
    for n, i in enumerate(order, 1):
        b, (a0, a1) = bands[i], ends[i]
        p, q = wcs(b, a0), wcs(b, a1)
        if (q[0], q[1]) < (p[0], p[1]):
            p, q = q, p
        wid = f"W{n}"
        walls.append(
            {"id": wid, "axis": [[rnd(p[0]), rnd(p[1])], [rnd(q[0]), rnd(q[1])]],
             "thickness": rnd(b["t"] * unit)}
        )  # fmt: skip
        for g0, g1 in b["gaps"]:
            s = sorted(math.dist(p, wcs(b, g)) for g in (g0, g1))
            openings.append({"wall": wid, "offset": rnd(s[0]), "width": rnd(s[1] - s[0])})
    out = _bounded(walls, limit, out, "walls")
    out["openings"] = [o for o in openings if any(w["id"] == o["wall"] for w in out["walls"])]
    out["dropped"] = dropped
    if origin is not None:
        out["origin"] = [ox, oy]
    return out


def _xy(value, name: str) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: expected [x, y], got {value!r}.") from None
    return x, y


def find_blobs(
    path, align, *, color, tolerance=40, size=None, region=None, region_px=None, limit=100
):
    """8-connected components of ``color``: centroid, size, pixel count."""
    import numpy as np

    cal, mask, origin, k, lo, hi, out, limit = _detect_head(
        path, align, region, region_px, color, tolerance, size, limit
    )
    rows, starts, ends = _runs(mask)
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rl, sl, el = rows.tolist(), starts.tolist(), ends.tolist()
    prev_lo = prev_hi = 0  # the previous row's runs [prev_lo, prev_hi)
    i = 0
    while i < n:
        row = rl[i]
        j = i
        while j < n and rl[j] == row:
            j += 1
        if prev_hi > prev_lo and rl[prev_lo] == row - 1:
            p = prev_lo
            for q in range(i, j):
                while p < prev_hi and el[p] < sl[q]:  # ends before q starts - 1
                    p += 1
                t = p
                while t < prev_hi and sl[t] <= el[q]:  # 8-connected overlap
                    ra, rb = find(t), find(q)
                    if ra != rb:
                        parent[rb] = ra
                    t += 1
        prev_lo, prev_hi = i, j
        i = j
    if n == 0:
        return _bounded([], limit, out, "blobs")
    labels = np.array([find(x) for x in range(n)])
    _, comp = np.unique(labels, return_inverse=True)
    m = int(comp.max()) + 1
    length = (ends - starts).astype(np.float64)
    count = np.bincount(comp, weights=length, minlength=m)
    # sum of column centres over a run [s, e): length * (s + e) / 2
    su = np.bincount(comp, weights=length * (starts + ends) / 2.0, minlength=m)
    sv = np.bincount(comp, weights=length * (rows + 0.5), minlength=m)
    c0 = np.full(m, np.iinfo(np.int64).max)
    r0 = c0.copy()
    c1 = np.full(m, -1)
    r1 = c1.copy()
    np.minimum.at(c0, comp, starts)
    np.minimum.at(r0, comp, rows)
    np.maximum.at(c1, comp, ends)
    np.maximum.at(r1, comp, rows + 1)
    unit = cal.upp * k
    found = []
    for c in range(m):
        w, h = (c1[c] - c0[c]) * unit, (r1[c] - r0[c]) * unit
        if not (lo <= w <= hi and lo <= h <= hi):
            continue
        row = _feature(cal, origin, k, int(c0[c]), int(r0[c]), int(c1[c]), int(r1[c]), count[c])
        u = origin[0] + float(su[c] / count[c]) * k
        v = origin[1] + float(sv[c] / count[c]) * k
        x, y = cal.to_wcs(u, v)
        row["center"] = [round(x, 3), round(y, 3)]  # the centroid, not the box centre
        found.append(row)
    found.sort(key=lambda f: (-f["pixels"], f["center"]))
    return _bounded(found, limit, out, "blobs")


def _nice_step(span: float, target: int = 10) -> float:
    raw = span / target
    power = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * power >= raw:
            return m * power
    return 10 * power


def _fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


#: Grid colours to choose from, the one farthest from the scan's own ink wins.
GRID_COLOURS = ((230, 0, 0), (0, 90, 255), (200, 0, 200), (0, 160, 0), (0, 170, 190))


def _grid_colour(image, color) -> tuple[int, int, int]:
    """``color`` when given, else the :data:`GRID_COLOURS` entry the scan's ink
    is farthest from (round 6: a red grid over red dimension lines)."""
    import numpy as np

    if color is not None:
        return parse_color(color)
    small = image.copy()
    small.thumbnail((256, 256))
    arr = np.asarray(small, dtype=np.int16).reshape(-1, 3)
    paper = np.median(arr, axis=0)
    ink = arr[np.abs(arr - paper).max(axis=1) > SOURCE_INK_DISTANCE]
    if not len(ink):
        return GRID_COLOURS[0]

    def near(c):  # the share of ink within reach of this colour
        return float((np.abs(ink - np.array(c)).max(axis=1) < 110).mean())

    return min(GRID_COLOURS, key=near)


def grid_crop(
    path, align, *, region=None, region_px=None, zoom=None, max_px=1568, grid=None, color=None
):
    """The window resampled into WCS axes with a labelled grid; ``png`` + report."""
    cal = Calibration(align)
    max_px = check_max_px(max_px)
    image = _open(path)
    u0, v0, u1, v1 = _window_px(cal, image.size, region, region_px)
    if region is not None:
        x0, y0, x1, y1 = normalize_region(region)
    else:
        x0, y0, x1, y1 = cal.box_to_wcs(u0, v0, u1, v1)
    span_px = max(x1 - x0, y1 - y0) / cal.upp  # the window's long side in scan pixels
    fit = max_px / span_px
    if zoom is None:
        zoom = min(MAX_AUTO_ZOOM, fit)
    else:
        if isinstance(zoom, bool) or not isinstance(zoom, int | float) or not zoom > 0:
            raise ValueError(f"zoom: output pixels per scan pixel, > 0, got {zoom!r}.")
    note = None
    if zoom > fit:
        note = f"zoom {zoom:g} would exceed max_px {max_px}; drawn at {fit:.4g}"
        zoom = fit
    ppu = zoom / cal.upp  # output px per drawing unit
    width = max(1, round((x1 - x0) * ppu))
    height = max(1, round((y1 - y0) * ppu))
    # output pixel (X, Y) is WCS (x0 + X/ppu, y1 - Y/ppu); back into the scan
    a_inv = 1.0 / cal.a
    # (u - i v) = a_inv * (x + i y - b)
    base = a_inv * (complex(x0, y1) - cal.b)
    dx = a_inv * complex(1.0 / ppu, 0.0)
    dy = a_inv * complex(0.0, -1.0 / ppu)
    coeffs = (dx.real, dy.real, base.real, -dx.imag, -dy.imag, -base.imag)
    from PIL import Image, ImageDraw

    out = image.transform(
        (width, height), Image.AFFINE, coeffs, resample=Image.BILINEAR, fillcolor=(235, 235, 235)
    ).convert("RGBA")
    if grid is None:
        step = _nice_step(max(x1 - x0, y1 - y0))
        while step * ppu < 40:
            step = _nice_step(step * 2.5, 1)
    else:
        if isinstance(grid, bool) or not isinstance(grid, int | float) or not grid > 0:
            raise ValueError(f"grid: the step in drawing units, > 0, got {grid!r}.")
        step = float(grid)
        if (x1 - x0) / step > 400 or (y1 - y0) / step > 400:
            raise ValueError(f"grid: {step:g} gives over 400 lines across the window.")
    minor = next((step / n for n in (5, 2) if step / n * ppu >= 12), None)
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(layer)

    def lines(s, colour):
        xs = [i * s for i in range(math.ceil(x0 / s), math.floor(x1 / s) + 1)]
        ys = [j * s for j in range(math.ceil(y0 / s), math.floor(y1 / s) + 1)]
        for x in xs:
            px = round((x - x0) * ppu)
            pen.line([(px, 0), (px, height)], fill=colour, width=1)
        for y in ys:
            py = round((y1 - y) * ppu)
            pen.line([(0, py), (width, py)], fill=colour, width=1)
        return xs, ys

    rgb = _grid_colour(image.crop((u0, v0, u1, v1)), color)
    if minor:
        lines(minor, (*rgb, 60))
    xs, ys = lines(step, (*rgb, 170))
    font = _font(12)
    for x in xs:
        label = _fmt(x)
        px = round((x - x0) * ppu) + 2
        box = pen.textbbox((px, 1), label, font=font)
        pen.rectangle(box, fill=(255, 255, 255, 200))
        pen.text((px, 1), label, fill=(*rgb, 255), font=font)
    for y in ys:
        # round 6: centred on its line, not above it (read 50-100 mm off)
        label = f"{_fmt(y)} -"
        py = round((y1 - y) * ppu)
        left, top, right, bottom = pen.textbbox((0, 0), label, font=font, anchor="lm")
        pen.rectangle((2 + left, py + top, 2 + right, py + bottom), fill=(255, 255, 255, 200))
        pen.text((2, py), label, fill=(*rgb, 255), font=font, anchor="lm")
    picture = Image.alpha_composite(out, layer).convert("RGB")
    buf = io.BytesIO()
    picture.save(buf, format="PNG")
    report = {
        "png": buf.getvalue(),
        "applied": True,
        "px": [width, height],
        "region": [round(v, 6) for v in (x0, y0, x0 + width / ppu, y1)],
        "px_per_unit": round(ppu, 6),
        "zoom": round(zoom, 4),
        "grid": {"step": step, "minor": minor},
        "scan": {
            "path": str(path),
            "pixels": list(image.size),
            "window_px": [u0, v0, u1, v1],
            "units_per_px": round(cal.upp, 6),
            "rotation_deg": round(cal.rotation, 4) + 0.0,
        },
        "read": f"lines in {'#{:02x}{:02x}{:02x}'.format(*rgb)} every grid.step, labelled "
        "in drawing units (x on top, y at the left, centred on its line); pixel (X, Y) "
        "is WCS (x0 + X/ppu, y1 - Y/ppu)",
    }
    if note:
        report["note"] = note
    if zoom < 1.0:
        pieces = math.ceil(1.0 / zoom)
        report["warning"] = (
            f"the scan is drawn at {zoom:.2f} of its resolution (max_px {max_px} on the "
            f"long side): small text may not read - split region into {pieces} windows "
            "along its long side, or raise max_px"
        )
    return report


# ── snap ───────────────────────────────────────────────────────────────────

#: Snap: the window searched round each pixel, and the shortest run that is a line.
SNAP_RADIUS_PX, SNAP_MIN_RUN_PX = 25, 8


def _bands(mask, axis: int, min_run: int):
    """Lines along ``axis`` (1: horizontal rows, 0: vertical columns) of ``mask``:
    ``[(centre, start, end)]`` - a band of neighbouring rows (columns) holding a
    run of at least ``min_run`` ink pixels, its centre and that run's ends."""
    import numpy as np

    m = mask if axis == 1 else mask.T
    hits = []
    for i, row in enumerate(m):
        best, start = None, None
        for j, on in enumerate([*row.tolist(), False]):
            if on and start is None:
                start = j
            elif not on and start is not None:
                if j - start >= min_run and (best is None or j - start > best[1] - best[0]):
                    best = (start, j)
                start = None
        if best:
            hits.append((i, *best))
    bands, group = [], []
    for hit in hits:
        if group and (hit[0] != group[-1][0] + 1 or abs(hit[1] - group[-1][1]) > 3):
            bands.append(group)
            group = []
        group.append(hit)
    if group:
        bands.append(group)
    return [
        (
            float(np.mean([g[0] for g in b])) + 0.5,
            float(np.median([g[1] for g in b])),
            float(np.median([g[2] for g in b])),
        )
        for b in bands
    ]


def snap_points(path, align, *, pixels, limit=100) -> dict:
    """The nearest line corner or line end to each scan pixel, in WCS (round 6).

    Within :data:`SNAP_RADIUS_PX` of the pixel the scan's ink (off the paper
    colour) is read as horizontal and vertical lines of the image (bands at
    least :data:`SNAP_MIN_RUN_PX` long, their centre lines): a corner is where
    a horizontal and a vertical one meet, an end where one stops. A slanted
    line has neither; the row then says ``found: false``.
    """
    import numpy as np

    cal = Calibration(align)
    image = _open(path)
    pts = list(pixels or ())
    if not pts:
        raise ValueError("pixels: give [[u, v], ...], the scan pixels to snap.")
    if len(pts) > _limit(limit):
        raise ValueError(f"pixels: at most {limit} at once, got {len(pts)}.")
    full = np.asarray(image, dtype=np.int16)
    paper = np.median(
        full[:: max(1, full.shape[0] // 256), :: max(1, full.shape[1] // 256)].reshape(-1, 3),
        axis=0,
    )
    r = SNAP_RADIUS_PX
    rows = []
    for i, p in enumerate(pts):
        try:
            u, v = (float(c) for c in p)
        except (TypeError, ValueError):
            raise ValueError(f"pixels[{i}]: expected [u, v], got {p!r}.") from None
        c0, r0 = max(0, int(u) - r), max(0, int(v) - r)
        c1, r1 = min(full.shape[1], int(u) + r + 1), min(full.shape[0], int(v) + r + 1)
        if c1 <= c0 or r1 <= r0:
            rows.append({"pixel": [u, v], "found": False, "reason": "outside the scan"})
            continue
        ink = np.abs(full[r0:r1, c0:c1] - paper).max(axis=2) > SOURCE_INK_DISTANCE
        horiz, vert = _bands(ink, 1, SNAP_MIN_RUN_PX), _bands(ink, 0, SNAP_MIN_RUN_PX)
        cands = []
        for hy, hs, he in horiz:
            for vx, vs, ve in vert:
                if hs - 3 <= vx <= he + 3 and vs - 3 <= hy <= ve + 3:
                    cands.append(("corner", vx, hy))
        for hy, hs, he in horiz:
            cands += [("end", hs, hy), ("end", he, hy)]
        for vx, vs, ve in vert:
            cands += [("end", vx, vs), ("end", vx, ve)]
        # a run cut by the window edge does not end there, and an arm of a
        # corner ends at the corner: neither is an end
        h, w = ink.shape
        corners = [c for c in cands if c[0] == "corner"]
        cands = corners + [
            c for c in cands
            if c[0] == "end" and 0.5 < c[1] < w - 0.5 and 0.5 < c[2] < h - 0.5
            and all(math.hypot(c[1] - k[1], c[2] - k[2]) > 4 for k in corners)
        ]  # fmt: skip
        if not cands:
            rows.append({"pixel": [u, v], "found": False, "reason": "no line corner or end near"})
            continue
        kind, cu, cv = min(cands, key=lambda c: math.hypot(c0 + c[1] - u, r0 + c[2] - v))
        pu, pv = c0 + cu, r0 + cv
        x, y = cal.to_wcs(pu, pv)
        rows.append(
            {
                "pixel": [u, v],
                "found": True,
                "kind": kind,
                "px": [round(pu, 2), round(pv, 2)],
                "point": [round(x, 3), round(y, 3)],
                "distance_px": round(math.hypot(pu - u, pv - v), 2),
            }
        )
    return {
        "ok": True,
        "snapped": rows,
        "radius_px": r,
        "method": "ink off the paper colour as the image's horizontal and vertical "
        "lines (centre lines of bands at least "
        f"{SNAP_MIN_RUN_PX} px long); corner = two meeting, end = one stopping",
    }


# ── compare ────────────────────────────────────────────────────────────────


def compare_window(path, align, *, region=None, region_px=None) -> dict:
    """Step 1 of a compare: the scan window and the render the drawing needs.

    Returns ``{"render_region": WCS box, "render_max_px": int, ...}`` for
    ``view_render``; :func:`compare` takes it back with the render.
    """
    cal = Calibration(align)
    arr, origin, k, size = _pixels(path, cal, region, region_px)
    h, w = arr.shape[:2]
    u0, v0 = origin
    box = cal.box_to_wcs(u0, v0, u0 + w * k, v0 + h * k)
    long_units = max(box[2] - box[0], box[3] - box[1])
    render_px = min(4096, max(64, round(long_units / (cal.upp * k))))
    return {
        "cal": cal,
        "arr": arr,
        "origin": origin,
        "k": k,
        "size": size,
        "path": path,
        "render_region": list(box),
        "render_max_px": render_px,
    }


def _dilate(mask, r: int):

    out = mask.copy()
    for axis in (0, 1):
        grown = out.copy()
        for d in range(1, r + 1):
            grown[(slice(d, None),) if axis == 0 else (slice(None), slice(d, None))] |= out[
                (slice(None, -d),) if axis == 0 else (slice(None), slice(None, -d))
            ]
            grown[(slice(None, -d),) if axis == 0 else (slice(None), slice(None, -d))] |= out[
                (slice(d, None),) if axis == 0 else (slice(None), slice(d, None))
            ]
        out = grown
    return out


def _erode(mask, r: int):
    return ~_dilate(~mask, r)


def compare(window: dict, render: dict, *, max_px: int = 1568) -> dict:
    """The source's ink against the drawing's, in the scan's pixels.

    The source is every pixel off the paper colour (the window's median), split
    into ``fills`` (inside an area of ink) and ``strokes``; the drawing was
    rendered with ``classify`` (lines, text, fills in their own colour), so the
    answer scores each class apart besides the totals.
    """
    import numpy as np
    from PIL import Image

    from backends.region_render import CLASS_COLORS

    cal, arr, (u0, v0), k = window["cal"], window["arr"], window["origin"], window["k"]
    h, w = arr.shape[:2]
    paper = np.median(arr.reshape(-1, 3), axis=0)
    source = np.abs(arr - paper).max(axis=2) > SOURCE_INK_DISTANCE
    src_fill = _dilate(_erode(source, FILL_CORE_PX), FILL_CORE_PX) & source
    src_stroke = source & ~src_fill

    with Image.open(io.BytesIO(render["png"])) as png:
        ink_img = np.asarray(png.convert("RGB"), dtype=np.int16)
    ink = ink_img.min(axis=2) <= INK_CHANNEL
    kind = np.where(ink, ink_img.argmax(axis=2), -1)  # 0 lines, 1 text, 2 fills
    rx0, _ry0, _rx1, ry1 = render["region"]
    rppu = render["px_per_unit"]
    rh, rw = ink.shape
    drawn_kind = np.full((h, w), -1, dtype=np.int8)
    cols = np.arange(w)
    for r_start in range(0, h, 256):
        rr = np.arange(r_start, min(h, r_start + 256))
        uu = u0 + (cols[None, :] + 0.5) * k
        vv = v0 + (rr[:, None] + 0.5) * k
        z = cal.a * (uu - 1j * vv) + cal.b
        ci = np.floor((z.real - rx0) * rppu).astype(np.int64)
        ri = np.floor((ry1 - z.imag) * rppu).astype(np.int64)
        ok = (ci >= 0) & (ci < rw) & (ri >= 0) & (ri < rh)
        block = np.full(ok.shape, -1, dtype=np.int8)
        block[ok] = kind[ri[ok], ci[ok]]
        drawn_kind[rr[0] : rr[-1] + 1] = block
    drawn = drawn_kind >= 0

    tol = COMPARE_TOLERANCE_PX
    near_drawn, near_source = _dilate(drawn, tol), _dilate(source, tol)
    # a dense drawing hatch lies within tol of the furniture and dimensions under
    # it: a source stroke counts as drawn only near a drawing line or text, and
    # what only a drawing fill reaches is reported apart (round 7)
    near_strokes = _dilate(drawn & (drawn_kind != 2), tol)
    covered = (src_stroke & near_strokes) | (src_fill & near_drawn)
    by_fill_only = source & near_drawn & ~covered
    missing = source & ~covered  # in the source, nothing drawn near it
    extra = drawn & ~near_source  # drawn, nothing in the source near it
    n_src, n_drw = int(source.sum()), int(drawn.sum())

    def pct(part, whole):
        return None if whole == 0 else round(100.0 * part / whole, 1)

    classes = {}
    for i, name in enumerate(CLASS_COLORS):
        mine = drawn_kind == i
        n = int(mine.sum())
        classes[name] = {"drawn_px": n, "on_source_pct": pct(int((mine & near_source).sum()), n)}
    source_parts = {
        name: {
            "px": int(part.sum()),
            "covered_pct": pct(int((part & covered).sum()), int(part.sum())),
        }
        for name, part in (("strokes", src_stroke), ("fills", src_fill))
    }

    cell = max(16, math.ceil(max(h, w) / 16))
    gh, gw = math.ceil(h / cell), math.ceil(w / cell)

    def cells(m):
        padded = np.zeros((gh * cell, gw * cell), dtype=np.int64)
        padded[:h, :w] = m
        return padded.reshape(gh, cell, gw, cell).sum(axis=(1, 3))

    miss_c, extra_c = cells(missing), cells(extra)
    score = miss_c + extra_c
    order = np.argsort(-score, axis=None, kind="stable")
    regions = []
    for flat in order[:COMPARE_REGIONS]:
        gy, gx = divmod(int(flat), gw)
        if score[gy, gx] == 0:
            break
        c0, r0 = gx * cell, gy * cell
        c1, r1 = min(w, c0 + cell), min(h, r0 + cell)
        regions.append(
            {
                "box": [
                    round(v, 3)
                    for v in cal.box_to_wcs(u0 + c0 * k, v0 + r0 * k, u0 + c1 * k, v0 + r1 * k)
                ],
                "missing_px": int(miss_c[gy, gx]),
                "extra_px": int(extra_c[gy, gx]),
            }
        )

    # the picture: source grey, missing red, extra blue, drawn lines dark green
    pic = np.full((h, w, 3), 255, dtype=np.uint8)
    pic[source] = (190, 190, 190)
    pic[drawn & near_source] = (0, 130, 0)
    pic[missing] = (230, 0, 0)
    pic[extra] = (0, 70, 255)
    image = Image.fromarray(pic)
    scale = min(MAX_AUTO_ZOOM, max_px / max(image.size))
    if scale > 1:
        scale = math.floor(scale)  # whole pixels, so a 1 px line stays 1 px wide
    if scale != 1:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image = image.resize(size, resample=Image.NEAREST)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "png": buf.getvalue(),
        "applied": True,
        "ok": True,
        "source_covered_pct": pct(n_src - int(missing.sum()), n_src),
        "source_covered_by_fills_only_pct": pct(int(by_fill_only.sum()), n_src),
        "drawing_on_source_pct": pct(n_drw - int(extra.sum()), n_drw),
        "source_line_px": n_src,
        "drawing_line_px": n_drw,
        "source": source_parts,
        "drawing": classes,
        "regions": regions,
        "scan": _scan_report(window["path"], cal, window["size"], (u0, v0), k, arr),
        "px": list(image.size),
        "method": {
            "frame": "drawing rendered on white at the scan's resolution, resampled into "
            "the scan's pixels through align",
            "source_line_px": f"scan pixel with a channel more than {SOURCE_INK_DISTANCE} off "
            f"the paper colour {[int(c) for c in paper]}",
            "drawing_line_px": f"render pixel with a channel <= {INK_CHANNEL}",
            "classes": "drawing: text = TEXT / MTEXT / ATTRIB / MULTILEADER, fills = HATCH / "
            "SOLID, lines = the rest (dimensions too); source: fills = ink with ink all round "
            f"within {FILL_CORE_PX} px, strokes = the rest",
            "covered": "a source stroke near a drawing line or text, a source fill near any "
            "drawing ink; source_covered_by_fills_only_pct is the source ink only a drawing "
            "fill reaches (a matching hatch, or furniture under one)",
            "text": "the render's font is not the scan's, so text on_source_pct stays low "
            "even for lettering placed right; judge text by regions, not by the score",
            "tolerance_px": tol,
            "regions": f"cells of {cell} scan px, ranked by missing + extra",
            "picture": "grey source, green matched drawing, red source not drawn, "
            "blue drawn not in source",
        },
    }
