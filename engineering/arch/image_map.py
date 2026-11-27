"""Image pixels <-> drawing points through a scan's calibration, read only.

``align`` is ``view_screenshot``'s ``overlay_align``: pairs ``[u, v, x, y]``,
image pixel (origin top-left, v down) -> drawing point. The agent calibrates
once and uses the same list here and under the overlay. Two pairs fix the
similarity exactly, so nothing checks them: the result says ``controlled:
false`` and asks for a third pair. Three or more are fitted by least squares
and the result carries the residuals.

:func:`image_map` answers ``to_drawing`` for ``pixels`` and ``to_image`` for
``points``, rounded; ``measure`` adds the length and angle of each segment
between consecutive results, in drawing units and degrees CCW from +X.
Nothing reads the image file and nothing is drawn.
"""

from __future__ import annotations

import math

from backends.region_render import overlay_fit, overlay_transform

MAX_POINTS = 2000
UNCONTROLLED = (
    "two pairs fit exactly, so nothing checks them: add a third pair far from both "
    "(e.g. the opposite corner of a room) to get residuals"
)


def _pairs(values, where: str) -> list[tuple[float, float]]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{where}: expected [[a, b], ...], got {values!r}.")
    if len(values) > MAX_POINTS:
        raise ValueError(f"{where}: at most {MAX_POINTS} points, got {len(values)}.")
    out = []
    for i, p in enumerate(values):
        try:
            a, b = (float(v) for v in p)
        except (TypeError, ValueError):
            raise ValueError(f"{where}[{i}]: expected [a, b], got {p!r}.") from None
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError(f"{where}[{i}]: must be finite, got {p!r}.")
        out.append((a, b))
    return out


def _segments(points: list[list[float]]) -> list[dict]:
    rows = []
    for (x1, y1), (x2, y2) in zip(points, points[1:], strict=False):
        rows.append(
            {
                "length": round(math.hypot(x2 - x1, y2 - y1), 3),
                "angle": round(math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0, 3),
            }
        )
    return rows


def image_map(align, *, pixels=None, points=None, measure: bool = False) -> dict:
    """See the module docstring. Refusals name their argument; nothing is written."""
    a, b = overlay_transform(align)
    px = _pairs(pixels, "pixels")
    pts = _pairs(points, "points")
    if not px and not pts:
        raise ValueError(
            "image_map: give pixels to map onto the drawing, points to map back, or both."
        )
    fit = overlay_fit(align)
    fit["units_per_px"] = round(fit["units_per_px"], 6)
    fit["rotation_deg"] = round(fit["rotation_deg"], 4) + 0.0
    if fit["controlled"]:
        fit["rms"] = round(fit["rms"], 3)
        fit["max"] = round(fit["max"], 3)
        fit["residuals"] = [round(e, 3) for e in fit["residuals"]]
    else:
        fit["note"] = UNCONTROLLED
    out: dict = {"ok": True, "fit": fit}
    if px:
        drawn = [a * complex(u, -v) + b for u, v in px]
        out["to_drawing"] = [[round(z.real, 3), round(z.imag, 3)] for z in drawn]
        if measure:
            out["drawing_segments"] = _segments(out["to_drawing"])
    if pts:
        back = [(complex(x, y) - b) / a for x, y in pts]
        out["to_image"] = [[round(p.real, 2), round(-p.imag, 2)] for p in back]
        if measure:
            out["drawing_segments_of_points"] = _segments([list(p) for p in pts])
    return out
