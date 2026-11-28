"""Region render: a picture of a window of the drawing, at a stated scale (1.8-07).

``view_screenshot(region=...)`` and ``view_zoom_and_screenshot`` share this
renderer: it draws the backend's current space, so the same DXF gives the
same pixels.

What the report promises, and why it is exact:

* ``px`` is read back from the PNG's own IHDR, never assumed.
* The figure is sized to the region's aspect with the long side ``max_px``.
  Rounding a side to whole pixels would leave x and y at two scales, so the
  shorter axis of the region is widened (by under one pixel) until both share
  one ``px_per_unit``; ``region`` is the window actually drawn, and
  ``requested_region`` the one asked for.
* The axes fill the figure (no ``bbox_inches="tight"``, no matplotlib margin),
  so pixel ``(0, 0)`` is ``(x0, y1)`` and a drawing point maps to
  ``((x - x0) * ppu, (y1 - y) * ppu)``.

Without a region the headless engine frames what the renderer drew (its
autoscaled extents) and says ``framing: "extents"``.
"""

from __future__ import annotations

import io
import math
import os
from collections.abc import Callable, Sequence

#: The long side of a model-facing image: larger images are downscaled by the
#: client anyway, and every pixel past this costs tokens and shows nothing.
DEFAULT_MAX_PX = 1568
MIN_MAX_PX = 64
MAX_MAX_PX = 4096
_DPI = 100
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

Region = tuple[float, float, float, float]


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) out of a PNG's IHDR chunk; ``ValueError`` for anything else."""
    if len(data) < 24 or data[:8] != _PNG_SIGNATURE:
        raise ValueError("not a PNG image")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def check_max_px(max_px) -> int:
    """``max_px`` as an int within [MIN_MAX_PX, MAX_MAX_PX], or ``ValueError`` naming it."""
    if isinstance(max_px, bool) or not isinstance(max_px, int | float) or max_px != int(max_px):
        raise ValueError(f"max_px must be a whole number of pixels, got {max_px!r}")
    value = int(max_px)
    if not MIN_MAX_PX <= value <= MAX_MAX_PX:
        raise ValueError(
            f"max_px must be between {MIN_MAX_PX} and {MAX_MAX_PX} pixels, got {value}"
        )
    return value


def normalize_region(region) -> Region | None:
    """``[x0, y0, x1, y1]`` in any corner order -> (xmin, ymin, xmax, ymax).

    Refuses (``ValueError`` naming ``region``) anything but four finite numbers
    spanning a positive width and height: a zero-area window has no scale.
    """
    if region is None:
        return None
    if isinstance(region, str | bytes) or not isinstance(region, Sequence):
        raise ValueError(f"region must be [x0, y0, x1, y1] in drawing units, got {region!r}")
    if len(region) != 4:
        raise ValueError(
            f"region must be [x0, y0, x1, y1] in drawing units: four numbers, got {len(region)}"
        )
    try:
        values = [float(v) for v in region]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"region must hold four numbers, got {list(region)!r}") from exc
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"region must hold four finite numbers, got {values!r}")
    x0, x1 = sorted((values[0], values[2]))
    y0, y1 = sorted((values[1], values[3]))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        raise ValueError(
            f"region {values!r} has zero width or height; a window needs two "
            "corners that differ in both x and y"
        )
    return (x0, y0, x1, y1)


#: A scan's long side is reduced to this before drawing: the output is at most
#: MAX_MAX_PX wide, so more source pixels only cost memory.
OVERLAY_MAX_SOURCE_PX = 4096


def overlay_transform(align) -> tuple[complex, complex]:
    """(a, b) with model = a·(u - i·v) + b, from ``[[u, v, x, y], ...]``.

    ``(u, v)`` is an image pixel (origin top-left, v down), ``(x, y)`` the WCS
    point it must land on. Two pairs fix a similarity (uniform scale, rotation,
    shift; no shear, no mirror) exactly; three or more fit it by least squares
    (:func:`overlay_fit` reports the residuals). ``ValueError`` names
    ``overlay_align``.
    """
    pixels, points = _calibration(align)
    n = len(pixels)
    pm, zm = sum(pixels) / n, sum(points) / n
    spread = sum(abs(p - pm) ** 2 for p in pixels)
    a = sum((p - pm).conjugate() * (z - zm) for p, z in zip(pixels, points, strict=True)) / spread
    return a, zm - a * pm


def overlay_fit(align) -> dict:
    """How well ``align`` fits one similarity: ``pairs``, ``rms`` / ``max`` (drawing
    units) and each pair's ``residuals``; two pairs fit exactly and say so."""
    a, b = overlay_transform(align)
    pixels, points = _calibration(align)
    out: dict = {"pairs": len(pixels), "units_per_px": abs(a)}
    out["rotation_deg"] = math.degrees(math.atan2(a.imag, a.real))
    if len(pixels) == 2:
        out["controlled"] = False
        return out
    errors = [abs(a * p + b - z) for p, z in zip(pixels, points, strict=True)]
    out["controlled"] = True
    out["rms"] = math.sqrt(sum(e * e for e in errors) / len(errors))
    out["max"] = max(errors)
    out["residuals"] = errors
    return out


def _calibration(align) -> tuple[list[complex], list[complex]]:
    """``overlay_align`` as image points (u - i·v) and drawing points (x + i·y)."""
    if (
        isinstance(align, str | bytes)
        or not isinstance(align, Sequence)
        or len(align) < 2
        or any(
            isinstance(p, str | bytes) or not isinstance(p, Sequence) or len(p) != 4 for p in align
        )
    ):
        raise ValueError(
            "overlay_align must be at least two calibration pairs [[u, v, x, y], [u, v, x, y]] "
            f"(image pixel -> drawing point), got {align!r}"
        )
    try:
        rows = [[float(v) for v in p] for p in align]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"overlay_align must hold numbers, got {align!r}") from exc
    if not all(math.isfinite(v) for row in rows for v in row):
        raise ValueError(f"overlay_align must hold finite numbers, got {align!r}")
    pixels = [complex(u, -v) for u, v, _x, _y in rows]
    points = [complex(x, y) for _u, _v, x, y in rows]
    if len(set(pixels)) < 2 or len(set(points)) < 2:
        raise ValueError(
            "overlay_align: the calibration points must differ both in the image "
            f"and in the drawing, got {align!r}"
        )
    return pixels, points


def check_opacity(opacity) -> float:
    """``overlay_opacity`` as a float in [0, 1], or ``ValueError`` naming it."""
    try:
        value = float(opacity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"overlay_opacity must be a number in 0-1, got {opacity!r}") from exc
    if isinstance(opacity, bool) or not 0.0 <= value <= 1.0:
        raise ValueError(f"overlay_opacity must be between 0 and 1, got {opacity!r}")
    return value


def load_overlay(path: str, align, opacity, *, fit_box: Region | None = None) -> dict:
    """The scan at ``path`` read and calibrated; ``ValueError`` names what is wrong.

    ``fit_box`` (a sheet's paper window) replaces ``align``: the image is
    scaled uniformly to fit the box and centred - a page scan on a sheet of
    the same format lands page on page, without calibration points.
    """
    if fit_box is None:
        a, b = overlay_transform(align)
    alpha = check_opacity(opacity)
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as source:
            width, height = source.size
            image = source.convert("RGBA")
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise ValueError(f"overlay_image: {path} does not exist") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"overlay_image: {path} is not an image Pillow can read") from exc
    if max(width, height) > OVERLAY_MAX_SOURCE_PX:
        image.thumbnail((OVERLAY_MAX_SOURCE_PX, OVERLAY_MAX_SOURCE_PX))
    fit = {}
    if fit_box is not None:
        x0, y0, x1, y1 = fit_box
        scale = min((x1 - x0) / width, (y1 - y0) / height)
        a = complex(scale, 0.0)
        b = complex((x0 + x1 - width * scale) / 2.0, (y0 + y1 + height * scale) / 2.0)
        page, scan = (x1 - x0) / (y1 - y0), width / height
        fit = {"fit": "page", "aspect_delta_pct": round(abs(scan / page - 1.0) * 100.0, 1)}
    corners = [
        a * complex(u, -v) + b for u, v in ((0, 0), (width, 0), (width, height), (0, height))
    ]
    xs, ys = [c.real for c in corners], [c.imag for c in corners]
    return {
        "image": image,
        "a": a,
        "b": b,
        "report": {
            "path": str(path),
            "pixels": [width, height],
            "opacity": alpha,
            "units_per_px": abs(a),
            "rotation_deg": math.degrees(math.atan2(a.imag, a.real)),
            "bounds": [min(xs), min(ys), max(xs), max(ys)],
            **fit,
        },
    }


def _draw_overlay(ax, overlay: dict) -> None:
    """The scan under everything ezdxf drew (whose z-order starts at 0)."""
    import numpy as np
    from matplotlib.transforms import Affine2D

    a, b = overlay["a"], overlay["b"]
    width, height = overlay["report"]["pixels"]
    matrix = Affine2D(np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag], [0, 0, 1]]))
    ax.imshow(
        np.asarray(overlay["image"]),
        origin="upper",
        # Pixel edges in (u, -v): the array may be reduced, the extent is the original.
        extent=(0, width, -height, 0),
        transform=matrix + ax.transData,
        alpha=overlay["report"]["opacity"],
        interpolation="bilinear",
        zorder=-1,
    )


#: Subtracted from an IMAGE's matplotlib z-order, so every raster lies under
#: every vector (whose z counts up from 0) and images keep their own order.
IMAGE_Z_OFFSET = 1_000_000


def image_frame(doc) -> int:
    """AutoCAD's IMAGEFRAME from the drawing's RASTERVARIABLES: 0 none, 1 shown
    and plotted, 2 shown, not plotted; 1 (AutoCAD's default) when absent."""
    variables = doc.rootdict.get("ACAD_IMAGE_VARS")
    return 1 if variables is None else int(variables.dxf.get("frame", 1))


def image_source(doc, filename: str) -> tuple[str | None, str]:
    """(path, "") of the IMAGEDEF's file, or (None, why) - every candidate is
    read through ``security.validate_path``; a refused one is never opened."""
    from security import validate_path

    name = str(filename or "").replace("\\", os.sep)
    if not name.strip():
        return None, "has no file name"
    candidates = [name] if os.path.isabs(name) else []
    if doc.filename:
        folder = os.path.dirname(os.path.abspath(str(doc.filename)))
        candidates += [os.path.join(folder, name), os.path.join(folder, os.path.basename(name))]
    why = "was not found"
    for candidate in candidates:
        try:
            path = validate_path(candidate)
        except Exception:  # noqa: BLE001 - a refused path is a missing file, never read
            why = "is refused by the path policy"
            continue
        if path.is_file():
            return str(path), ""
    return None, why


def _draw_image(frontend, entity, properties, *, show_frame: bool, warnings: list) -> None:
    """An IMAGE's pixels through ezdxf's pipeline (so a viewport transforms and
    clips them), faded as AutoCAD fades; a missing file is ezdxf's frame with
    the file name, and a warning."""
    import numpy as np
    from ezdxf.addons.drawing.backend import ImageData
    from ezdxf.addons.drawing.frontend import (
        UniversalFrontend,
        _blend_image_towards,
        _multiply_alpha,
    )
    from ezdxf.colors import RGB
    from ezdxf.math import Matrix44, Vec2
    from ezdxf.npshapes import NumpyPoints2d
    from PIL import Image as PILImage
    from PIL import ImageEnhance, UnidentifiedImageError

    image_def = entity.image_def
    filename = image_def.dxf.filename if image_def is not None else ""
    path, why = (
        (None, "has no IMAGEDEF") if image_def is None else image_source(entity.doc, filename)
    )
    loaded = None
    if path is not None:
        try:
            with PILImage.open(path) as source:
                loaded = source.convert("RGBA")
        except (UnidentifiedImageError, OSError):
            why = "is not an image Pillow can read"
    if loaded is None:
        warnings.append(
            f"IMAGE {entity.dxf.handle}: {filename or '<no file name>'} {why}; "
            "drawn as a frame with its file name"
        )
        if image_def is None:
            return
        document_dir, frontend.ctx.document_dir = frontend.ctx.document_dir, None
        try:  # ezdxf's missing-image rendering: the frame and the file name
            UniversalFrontend.draw_image_entity(frontend, entity, properties)
        finally:
            frontend.ctx.document_dir = document_dir
        return
    if entity.dxf.flags & entity.SHOW_IMAGE:
        width, height = loaded.size
        if max(width, height) > OVERLAY_MAX_SOURCE_PX:
            loaded.thumbnail((OVERLAY_MAX_SOURCE_PX, OVERLAY_MAX_SOURCE_PX))
        sx, sy = width / loaded.width, height / loaded.height
        # The same adjustments, in the same order, as ezdxf's own IMAGE path.
        if entity.dxf.contrast != 50:
            loaded = ImageEnhance.Contrast(loaded).enhance(entity.dxf.contrast / 50)
        if entity.dxf.fade:
            paper = RGB.from_hex(frontend.ctx.current_layout_properties.background_color[:7])
            loaded = _blend_image_towards(loaded, entity.dxf.fade / 100, paper)
        if entity.dxf.brightness != 50:
            amount = entity.dxf.brightness / 50 - 1
            toward = RGB(255, 255, 255) if amount > 0 else RGB(0, 0, 0)
            loaded = _blend_image_towards(loaded, abs(amount), toward)
        if not entity.dxf.flags & entity.USE_TRANSPARENCY:
            loaded.putalpha(255)
        if entity.transparency:
            loaded = _multiply_alpha(loaded, 1.0 - entity.transparency)
        boundary = [Vec2(p.x / sx, p.y / sy) for p in entity.pixel_boundary_path()]
        frontend.pipeline.draw_image(
            ImageData(
                image=np.array(loaded),
                transform=Matrix44.scale(sx, sy, 1) @ entity.get_wcs_transform(),
                pixel_boundary_path=NumpyPoints2d(boundary),
                use_clipping_boundary=entity.dxf.flags & entity.USE_CLIPPING_BOUNDARY,
                remove_outside=entity.dxf.clip_mode == 0,
            ),
            properties,
        )
    if show_frame:
        points = [v.vec2 for v in entity.boundary_path_wcs()]
        frontend.pipeline.draw_solid_lines(list(zip(points, points[1:], strict=False)), properties)


def frame(region: Region, max_px: int) -> tuple[int, int, Region, float]:
    """(width_px, height_px, drawn region, px_per_unit) for ``region``.

    The long side is ``max_px``; the drawn region contains ``region`` and is
    widened on one axis by under a pixel so that x and y share one scale.
    """
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    if rw >= rh:
        width, height = max_px, max(1, round(max_px * rh / rw))
    else:
        width, height = max(1, round(max_px * rw / rh)), max_px
    ppu = min(width / rw, height / rh)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w, half_h = width / ppu / 2.0, height / ppu / 2.0
    return width, height, (cx - half_w, cy - half_h, cx + half_w, cy + half_h), ppu


def _agg_figure():
    """A Figure on the Agg canvas - no pyplot, safe in a worker thread."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(dpi=_DPI)
    FigureCanvasAgg(fig)
    return fig


def _inside(point, region: Region) -> bool:
    x0, y0, x1, y1 = region
    return x0 <= point[0] <= x1 and y0 <= point[1] <= y1


#: ezdxf hands matplotlib a lineweight in millimetres and matplotlib reads every
#: linewidth as points. Unscaled, 0.50 mm was stroked 0.50 pt (0.18 mm), and the
#: one-pixel floor (0.72 pt at 100 dpi) drew 0.13 to 0.70 mm at one width.
MM_TO_PT = 72.0 / 25.4
#: The thinnest pen a vector plot strokes: ISO 128's 0.13 mm. A raster keeps
#: ezdxf's floor of one pixel.
VECTOR_MIN_LINEWEIGHT_MM = 0.13


#: ``background`` values of ``draw_layout``: ``auto`` is white paper under a plot
#: style that maps ink (monochrome, grey, screening, pens) and the layout's own
#: background otherwise; ``transparent`` draws no background at all.
BACKGROUNDS = ("auto", "white", "black", "transparent")


def screen_color(rgb: str, screen: int) -> str:
    """``#rrggbb`` at ``screen`` % ink intensity: 0 % is white paper, 100 % unchanged."""
    channels = [int(rgb[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - (255 - c) * screen / 100):02x}" for c in channels)


def _file_pens(ctb) -> dict[int, tuple[tuple[int, int, int] | None, bool, int, float | None]]:
    """``{aci: (rgb or None, grayscale, screen, lineweight mm or None)}`` of a loaded
    ctb, only for the pens that change anything."""
    pens = {}
    for aci in range(1, 256):
        style = ctb[aci]
        pen = (style.color, style.grayscale, style.screen, ctb.get_lineweight(aci))
        if pen != (None, False, 100, None):
            pens[aci] = pen
    return pens


def _pen_report(pens: dict) -> list[dict]:
    """The pens a table sets, grouped by (screen, lineweight), for the result."""
    from engineering.standards.papers import ALL_PENS

    groups: dict[tuple, list[int]] = {}
    for aci, (_rgb, _gray, screen, lineweight) in sorted(pens.items()):
        if screen != 100 or lineweight is not None:
            groups.setdefault((screen, lineweight), []).append(aci)
    return [
        {
            "aci": "all" if acis == [ALL_PENS] else acis,
            "screen": screen,
            "lineweight_mm": lineweight,
        }
        for (screen, lineweight), acis in groups.items()
    ]


def draw_layout(
    doc,
    layout,
    out,
    *,
    plot: bool,
    vector: bool = False,
    plot_style: str | None = None,
    background: str = "auto",
    warnings: list | None = None,
    classify: bool = False,
) -> dict:
    """Draw ``layout`` through the matplotlib backend ``out``, the way it plots.

    ``classify`` (scan_read compare, round 6) draws every entity in its class's
    colour instead (:data:`CLASS_COLORS`), so one render
    tells lines, text and fills apart; a WIPEOUT or IMAGE keeps its own.

    ``plot`` as passed (a PDF) leaves out every layer whose plot flag is off,
    as AutoCAD plots. An IMAGE is drawn with its pixels under the vectors and
    its frame as IMAGEFRAME says; a file it cannot read is a frame with the
    file name and a line in ``warnings``.

    Lineweights are drawn at their size in points (on a raster, at least one
    pixel). With ``plot`` (a PDF, or any paper layout) the plot style table is
    applied - ``plot_style`` when given, else the layout's own: a known table as
    ``CTB_RENDERING`` and ``CTB_PENS`` say, any other name loaded from a ctb file
    ezdxf finds, else the object colours with ``exact: false``. Model space
    rendered as a screenshot uses object colours, as AutoCAD's screen does.

    A pen is chosen by the entity's resolved ACI (ByLayer / ByBlock followed -
    ezdxf's own lookup reads only an entity's direct colour, and never screens);
    its colour, grey, screening and lineweight are applied in the render
    context, so a wipeout or a text mask keeps the paper colour. A pen's
    lineweight is drawn as the table says, under the 0.13 mm vector floor too.
    Returns ``{name, ink, exact}``, plus ``pens`` when the table sets any.
    """
    import ezdxf
    from ezdxf.addons import acadctb
    from ezdxf.addons.drawing import RenderContext
    from ezdxf.addons.drawing.config import BackgroundPolicy, Configuration
    from ezdxf.addons.drawing.frontend import UniversalFrontend, _draw_viewports
    from ezdxf.addons.drawing.pipeline import color_to_monochrome
    from ezdxf.entities import Viewport
    from ezdxf.filemanagement import find_support_file
    from ezdxf.tools.text_layout import LayoutError

    from engineering.standards.papers import ALL_PENS, CTB_PENS, CTB_RENDERING

    if background not in BACKGROUNDS:
        raise ValueError(f"background must be one of {list(BACKGROUNDS)}, got {background!r}")
    export = plot
    frame_mode = image_frame(doc)
    show_frame = frame_mode == 1 or (frame_mode == 2 and not export)
    if warnings is None:
        warnings = []
    if plot_style is not None:
        name, plot = str(plot_style).strip(), True
    else:
        name = str(layout.get_plot_style_filename() or "").strip()
        plot = plot or not layout.is_modelspace
    ink, exact, pens = "object", True, {}
    if plot and name:
        key = name.lower()
        ink, exact = CTB_RENDERING.get(key, ("object", False))
        pens = {aci: (None, False, s, lw) for aci, (s, lw) in CTB_PENS.get(key, {}).items()}
        found = find_support_file(name, ezdxf.options.support_dirs)
        if key not in CTB_RENDERING and key.endswith(".ctb") and os.path.isfile(found):
            ink, exact, pens = "file", True, _file_pens(acadctb.load(found))
    mapping = {"monochrome": lambda c: "#000000", "grayscale": color_to_monochrome}.get(ink)
    floor = VECTOR_MIN_LINEWEIGHT_MM if vector else None

    class _PlotContext(RenderContext):
        def _pen(self, entity, resolved_layer):
            if not pens:
                return None
            if not entity.dxf.hasattr("true_color"):  # a true colour has no pen
                pen = pens.get(self.resolve_pen(entity, resolved_layer=resolved_layer))
                if pen is not None:
                    return pen
            return pens.get(ALL_PENS)

        def resolve_color(self, entity, *, resolved_layer=None):
            color = super().resolve_color(entity, resolved_layer=resolved_layer)
            rgb, alpha = color[:7], color[7:]
            pen = self._pen(entity, resolved_layer)
            if pen is not None and pen[0] is not None:
                rgb = "#{:02x}{:02x}{:02x}".format(*pen[0])
            if pen is not None and pen[1]:
                rgb = color_to_monochrome(rgb)
            if mapping:
                rgb = mapping(rgb)
            if pen is not None and pen[2] != 100:
                rgb = screen_color(rgb, pen[2])
            return rgb + alpha

        def resolve_lineweight(self, entity, *, resolved_layer=None):
            pen = self._pen(entity, resolved_layer)
            if pen is not None and pen[3] is not None:
                return pen[3]
            lineweight = super().resolve_lineweight(entity, resolved_layer=resolved_layer)
            return max(lineweight, floor) if floor else lineweight

    pen_weights = [pen[3] for pen in pens.values() if pen[3] is not None]
    if background == "auto":
        # a mapped plot is ink on white paper, model space included
        policy = BackgroundPolicy.WHITE if mapping or pens else BackgroundPolicy.DEFAULT
    else:
        policy = {
            "white": BackgroundPolicy.WHITE,
            "black": BackgroundPolicy.BLACK,
            "transparent": BackgroundPolicy.OFF,
        }[background]

    class _Frontend(UniversalFrontend):
        def draw_simple_mtext(self, mtext, properties):
            # ezdxf draws an MTEXT background mask (BACKGROUNDMASK, bg_fill) only
            # on its formatted-MTEXT path; on the plain path the mask was lost
            # and a hatch showed through the text.
            if mtext.dxf.get("bg_fill", 0):
                try:
                    return self.draw_complex_mtext(mtext, properties)
                except LayoutError:
                    pass
            return super().draw_simple_mtext(mtext, properties)

        def draw_composite_entity(self, entity, properties):
            # ezdxf draws a DIMENSION's block outside any block reference, so a
            # ByBlock part (DIMCLRD 0, AutoCAD's ISO-25) drew in the layout's
            # default colour and a layer-0 part on layer 0. AutoCAD resolves
            # both from the dimension, as from an INSERT.
            if entity.dxftype() != "DIMENSION":
                return super().draw_composite_entity(entity, properties)
            self.ctx.push_state(properties)
            try:
                return super().draw_composite_entity(entity, properties)
            finally:
                self.ctx.pop_state()

        def draw_image_entity(self, entity, properties):
            # ezdxf reads an image only from a saved drawing's folder, without
            # the path policy, and always draws its frame.
            _draw_image(self, entity, properties, show_frame=show_frame, warnings=warnings)

        def draw_entities(self, entities, *, filter_func=None):
            # ezdxf draws a layout's viewports after all of its paper-space
            # objects, whatever the sheet says. AutoCAD ignores draw order
            # between a viewport and paper space and follows "Plot paperspace
            # last": set, the viewports go first and a paper-space note or
            # WIPEOUT lies on top of the model seen through them; clear,
            # ezdxf's order is right.
            if not paperspace_last(layout):
                return super().draw_entities(entities, filter_func=filter_func)
            entities = [e for e in entities if filter_func is None or filter_func(e)]
            _draw_viewports(self, [e for e in entities if isinstance(e, Viewport)])
            super().draw_entities([e for e in entities if not isinstance(e, Viewport)])

    config = Configuration(
        lineweight_scaling=MM_TO_PT,
        # the floor is applied to object lineweights above; this one lets a pen through
        min_lineweight=min([floor, *pen_weights]) * MM_TO_PT if floor else None,
        background_policy=policy,
    )
    # every ezdxf pen "use object"; export_mode drops the no-plot layers
    ctx = _PlotContext(doc, ctb=acadctb.new_ctb(), export_mode=export)
    if layout.is_modelspace:  # AutoCAD's MSLTSCALE: model patterns x the annotation scale
        ctx.linetype_scale *= model_linetype_factor(doc)
    psltscale = bool(doc.header.get("$PSLTSCALE", 1))
    frontend = _Frontend(ctx, _glyph_pipeline(out, psltscale), config=config)
    if classify:
        frontend.push_property_override_function(_class_colour)
    frontend.draw_layout(layout, finalize=True)
    result = {"name": name or None, "ink": ink, "exact": exact}
    report = _pen_report(pens)
    if report:
        result["pens"] = report
    return result


#: ``classify``'s colours: one pure channel per class, so a pixel's dominant
#: channel names its class (antialiasing only adds white).
CLASS_COLORS = {"lines": "#ff0000", "text": "#00ff00", "fills": "#0000ff"}
TEXT_TYPES = frozenset({"TEXT", "MTEXT", "ATTRIB", "MULTILEADER"})
FILL_TYPES = frozenset({"HATCH", "SOLID", "TRACE", "MPOLYGON"})


def entity_class(dxftype: str) -> str | None:
    """``lines`` / ``text`` / ``fills``, None for what keeps its own colour."""
    if dxftype in ("WIPEOUT", "IMAGE", "VIEWPORT"):
        return None
    if dxftype in TEXT_TYPES:
        return "text"
    return "fills" if dxftype in FILL_TYPES else "lines"


def _class_colour(entity, properties) -> None:
    kind = entity_class(entity.dxftype())
    if kind is not None:
        properties.color = CLASS_COLORS[kind]


#: PLOTSETTINGS flag "Plot paperspace last" (DXF 70, bit 512).
PAPERSPACE_LAST = 512


def model_linetype_factor(doc) -> float:
    """What model space multiplies linetype patterns by: the annotation
    scale's denominator (CANNOSCALE 1:50 -> 50) under MSLTSCALE 1, else 1."""
    from backends.contracts.settings import parse_scale

    vardict = doc.rootdict.get("AcDbVariableDictionary")
    if vardict is None:
        return 1.0
    ms = vardict.get("MSLTSCALE")
    if ms is not None and str(ms.dxf.value).strip() == "0":
        return 1.0
    entry = vardict.get("CANNOSCALE")
    if entry is None:
        return 1.0
    try:
        _, paper, drawing = parse_scale(str(entry.dxf.value))
    except ValueError:
        return 1.0
    return drawing / paper if paper > 0 else 1.0


def paperspace_last(layout) -> bool:
    """Whether AutoCAD draws ``layout``'s viewports under its paper-space objects."""
    if layout.is_modelspace:
        return False
    # Unset reads as ezdxf's default 688 but is written as 0 (LAYOUT export
    # rebuilds the flags from ``get(..., 0)``): 0 is what AutoCAD opens.
    return bool(int(layout.dxf_layout.dxf.get("plot_layout_flags", 0)) & PAPERSPACE_LAST)


def _glyph_pipeline(out, psltscale: bool = True):
    """ezdxf's 2D pipeline, with text glyphs filled by the font's own winding.

    ezdxf fills every path set after re-nesting its contours by bounding box,
    which a hatch needs (DXF loops have any orientation) and a glyph does not:
    a font already orients its contours for the nonzero rule. Re-nested, a
    glyph built from overlapping contours lost half its counter - the slash of
    Ø / ⌀ / ∅ pokes out of the ring, so the ring's inside was taken for an
    outer contour and filled, and the sign drew as a heavier blob than the
    digits beside it.
    """
    from ezdxf.addons.drawing.pipeline import BackendStage2d, RenderPipeline2d
    from ezdxf.npshapes import to_matplotlib_path
    from matplotlib.patches import PathPatch

    class _Stage(BackendStage2d):
        def __init__(self, pipeline, backend, converter):
            super().__init__(backend, converter)
            self.pipeline = pipeline

        def draw_filled_paths(self, paths, properties):
            if not self.pipeline.drawing_text:
                return super().draw_filled_paths(paths, properties)
            if not paths:
                return None
            props = self.converter(properties)
            self.backend.ax.add_patch(
                PathPatch(
                    to_matplotlib_path(paths),  # no detect_holes: the font's winding
                    color=props.color,
                    linewidth=0,
                    fill=True,
                    zorder=self.backend._get_z(),
                )
            )
            return None

        def draw_solid_lines(self, lines, properties):
            # A zero-length segment - every dot of a dotted hatch pattern
            # (AR-SAND, AR-CONC) - plots as a dot of its pen width, as
            # AutoCAD plots it. ezdxf scatters it as a fixed 0.1 pt² marker
            # with matplotlib's 1 pt edge, whatever the lineweight: a ~0.45 mm
            # ring on every page, so a sand hatch at 1:20 plotted as dark
            # blots over its labels while the screenshot showed fine dust.
            lines = list(lines)
            dots = [start for start, end in lines if start.isclose(end)]
            if not dots:
                return super().draw_solid_lines(lines, properties)
            rest = [(start, end) for start, end in lines if not start.isclose(end)]
            if rest:
                super().draw_solid_lines(rest, properties)
            props = self.converter(properties)
            width = self.backend.get_lineweight(props)  # points
            self.backend.ax.scatter(
                [p.x for p in dots],
                [p.y for p in dots],
                s=width * width,
                c=props.color,
                marker="o",
                linewidths=0,
                zorder=self.backend._get_z(),
            )
            return None

        def draw_image(self, image_data, properties):
            # A raster under every vector, whatever its draw order: a scan
            # drawn last would hide the walls and dimensions over it.
            ax = self.backend.ax
            first = len(ax.images)
            super().draw_image(image_data, properties)
            for image in ax.images[first:]:
                image.set_zorder(image.get_zorder() - IMAGE_Z_OFFSET)
            # ezdxf lets the image's frame set the extents; it may be off now.
            corners = image_data.transform.transform_vertices(
                (p.x, p.y) for p in image_data.pixel_boundary_path.vertices()
            )
            ax.update_datalim([(c.x, c.y) for c in corners])

    class _Pipeline(RenderPipeline2d):
        drawing_text = False

        def get_vp_ltype_scale(self):
            # ezdxf applies a viewport's linetype pattern to paper coordinates
            # but scales it by 1 / viewport scale, as if it were model ones: at
            # 1:20 with LTSCALE 15 a HIDDEN dash was 1500 mm of paper and every
            # dashed line plotted solid. AutoCAD's rule: PSLTSCALE 1 sizes the
            # pattern in paper units, 0 in model units (times the scale).
            return 1.0 if psltscale else self.current_vp_scale

        def build_render_pipeline(self):
            clipping = super().build_render_pipeline()
            stage = clipping
            while not isinstance(stage.next_stage, BackendStage2d):
                stage = stage.next_stage
            stage.next_stage = _Stage(self, self.backend, self.get_backend_properties)
            return clipping

        def draw_text(self, *args, **kwargs):
            self.drawing_text = True
            try:
                super().draw_text(*args, **kwargs)
            finally:
                self.drawing_text = False

        def draw_viewport(self, vp, layout_ctx, bbox_cache=None):
            # ezdxf draws a viewport's model space in storage order, so a
            # DRAWORDER in the SORTENTSTABLE that model space honours was lost
            # on a sheet: a hatch sent behind a WIPEOUT plotted on top of it.
            # Same order as Frontend.draw_layout gives model space itself.
            from ezdxf import reorder
            from ezdxf.addons.drawing.pipeline import filter_vp_entities

            if vp.doc is None:
                return
            try:
                limits = vp.get_modelspace_limits()
            except ValueError:  # modelspace limits not detectable
                return
            msp = vp.doc.modelspace()
            mapping = list(msp.get_redraw_order())
            entities = list(reorder.ascending(msp, mapping)) if mapping else msp
            if self.enter_viewport(vp):
                self.draw_entities(
                    layout_ctx.from_viewport(vp),
                    filter_vp_entities(entities, limits, bbox_cache),
                )
                self.exit_viewport()

    return _Pipeline(out)


_TEXT_TYPES = ("TEXT", "MTEXT", "ATTRIB", "ATTDEF")


def render_font(font: str) -> str:
    """The font file the headless renderer draws ``font`` with (ezdxf's font
    manager: the file itself, its TrueType twin, the fallback face narrowed to
    the font's width by ``shx_substitute``, else the fallback face)."""
    from ezdxf.fonts import fonts

    from .shx_substitute import substitute_for

    manager = fonts.font_manager
    if font and manager.has_font(font):
        return font
    if (narrowed := substitute_for(font)) and manager.has_font(narrowed):
        return narrowed
    twin = fonts.SHX_FONTS.get((font or "").upper())
    if twin and manager.has_font(twin):
        return twin
    return manager.fallback_font_name()


def font_substitutions(doc) -> dict[str, str]:
    """``{style font: font drawn with}`` for the text styles the drawing's
    lettering uses whose font this machine lacks (AutoCAD's SHX / TTF faces
    are not shipped): widths in a render then differ from AutoCAD's."""
    used = {
        str(e.dxf.get("style", "Standard")).lower()
        for e in doc.entitydb.values()
        if e.is_alive and e.dxftype() in _TEXT_TYPES
    }
    out: dict[str, str] = {}
    for style in doc.styles:
        if style.dxf.name.lower() not in used:
            continue
        font = style.dxf.get("font", "") or "txt"
        drawn = render_font(font)
        if drawn != font:
            out[font] = drawn
    return out


def render_layout(
    doc,
    layout,
    *,
    region: Region | None = None,
    max_px: int = DEFAULT_MAX_PX,
    overlay_handles: bool = False,
    max_labels: int = 40,
    label_anchor: Callable | None = None,
    overlay: dict | None = None,
    background: str = "auto",
    classify: bool = False,
) -> dict:
    """Render ``layout`` of ``doc`` to PNG; ``region`` None frames the extents.

    Raises ``ImportError`` when matplotlib is missing; the caller decides how
    to say so. ``region`` must already be normalised (``normalize_region``).
    ``overlay`` (``load_overlay``) is a calibrated scan drawn under the CAD
    geometry; without a region the framing then covers the scan too.
    """
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    max_px = check_max_px(max_px)
    fig = _agg_figure()
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    out = MatplotlibBackend(ax, adjust_figure=False)
    warnings: list[str] = []
    plot_style = draw_layout(
        doc, layout, out, plot=False, background=background, warnings=warnings, classify=classify
    )

    if region is None:
        (ex0, ex1), (ey0, ey1) = ax.get_xlim(), ax.get_ylim()
        if not ex1 - ex0 > 0:
            ex0, ex1 = ex0 - 0.5, ex0 + 0.5
        if not ey1 - ey0 > 0:
            ey0, ey1 = ey0 - 0.5, ey0 + 0.5
        if overlay is not None:
            ox0, oy0, ox1, oy1 = overlay["report"]["bounds"]
            if any(True for _ in layout):
                ex0, ey0, ex1, ey1 = min(ex0, ox0), min(ey0, oy0), max(ex1, ox1), max(ey1, oy1)
            else:  # nothing drawn yet: the autoscaled (0, 1) box is not content
                ex0, ey0, ex1, ey1 = ox0, oy0, ox1, oy1
        target: Region = (ex0, ey0, ex1, ey1)
        framing = "extents"
    else:
        target = region
        framing = "region"

    width, height, drawn, ppu = frame(target, max_px)
    fig.set_size_inches(width / _DPI, height / _DPI)
    ax.set_aspect("auto")
    if overlay is not None:
        _draw_overlay(ax, overlay)
    ax.set_xlim(drawn[0], drawn[2])
    ax.set_ylim(drawn[1], drawn[3])

    labels: dict[str, list[float]] = {}
    total = 0
    truncated = False
    if overlay_handles and label_anchor is not None:
        anchored = []
        for ent in layout:
            anchor = label_anchor(ent)
            if anchor is not None and _inside(anchor, drawn):
                anchored.append((ent, anchor))
        total = len(anchored)
        limit = max(0, int(max_labels))
        truncated = total > limit
        for ent, anchor in anchored[:limit]:
            handle = str(ent.dxf.get("handle", "?"))
            labels[handle] = [anchor[0], anchor[1]]
            ax.annotate(
                handle,
                anchor,
                color="#d62728",
                fontsize=7,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "#d62728",
                    "linewidth": 0.4,
                    "alpha": 0.85,
                },
            )
        if truncated:
            ax.text(
                0.5,
                0.98,
                f"handles shown for {len(labels)} of {total} entities",
                transform=ax.transAxes,
                color="#d62728",
                fontsize=9,
                ha="center",
                va="top",
            )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=ax.get_facecolor())
    png = buf.getvalue()
    got_w, got_h = png_size(png)
    return {
        "png": png,
        "applied": True,
        "framing": framing,
        "px": [got_w, got_h],
        "region": [float(v) for v in drawn],
        "requested_region": None if region is None else [float(v) for v in region],
        "px_per_unit": got_w / (drawn[2] - drawn[0]),
        "space": str(getattr(layout, "name", "Model")),
        "plot_style": plot_style,
        "labels": labels,
        "labelled": len(labels),
        "total": total,
        "truncated": truncated,
        **({} if overlay is None else {"overlay": overlay["report"]}),
        **({"warnings": warnings} if warnings else {}),
        **({"fonts_substituted": subs} if (subs := font_substitutions(doc)) else {}),
    }


def render_dxf_file(path: str, *, region: Region | None, max_px: int = DEFAULT_MAX_PX) -> dict:
    """Render the model space of the DXF at ``path``."""
    import ezdxf

    doc = ezdxf.readfile(path)
    return render_layout(doc, doc.modelspace(), region=region, max_px=max_px)


def downscale_png(png: bytes, max_px: int) -> bytes:
    """``png`` with its long side brought down to ``max_px`` (aspect kept); never enlarged."""
    width, height = png_size(png)
    if max(width, height) <= max_px:
        return png
    from PIL import Image

    scale = max_px / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = Image.open(io.BytesIO(png))
    resized = image.resize(size, Image.LANCZOS)
    out = io.BytesIO()
    resized.save(out, format="PNG")
    return out.getvalue()
