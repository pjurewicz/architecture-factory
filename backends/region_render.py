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
) -> dict:
    """Draw ``layout`` through the matplotlib backend ``out``, the way it plots.

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
    ctx = _PlotContext(doc, ctb=acadctb.new_ctb())  # every ezdxf pen "use object"
    psltscale = bool(doc.header.get("$PSLTSCALE", 1))
    _Frontend(ctx, _glyph_pipeline(out, psltscale), config=config).draw_layout(layout, finalize=True)
    result = {"name": name or None, "ink": ink, "exact": exact}
    report = _pen_report(pens)
    if report:
        result["pens"] = report
    return result


#: PLOTSETTINGS flag "Plot paperspace last" (DXF 70, bit 512).
PAPERSPACE_LAST = 512


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


def render_layout(
    doc,
    layout,
    *,
    region: Region | None = None,
    max_px: int = DEFAULT_MAX_PX,
    overlay_handles: bool = False,
    max_labels: int = 40,
    label_anchor: Callable | None = None,
) -> dict:
    """Render ``layout`` of ``doc`` to PNG; ``region`` None frames the extents.

    Raises ``ImportError`` when matplotlib is missing; the caller decides how
    to say so. ``region`` must already be normalised (``normalize_region``).
    """
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    max_px = check_max_px(max_px)
    fig = _agg_figure()
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    out = MatplotlibBackend(ax, adjust_figure=False)
    plot_style = draw_layout(doc, layout, out, plot=False)

    if region is None:
        (ex0, ex1), (ey0, ey1) = ax.get_xlim(), ax.get_ylim()
        if not ex1 - ex0 > 0:
            ex0, ex1 = ex0 - 0.5, ex0 + 0.5
        if not ey1 - ey0 > 0:
            ey0, ey1 = ey0 - 0.5, ey0 + 0.5
        target: Region = (ex0, ey0, ex1, ey1)
        framing = "extents"
    else:
        target = region
        framing = "region"

    width, height, drawn, ppu = frame(target, max_px)
    fig.set_size_inches(width / _DPI, height / _DPI)
    ax.set_aspect("auto")
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
