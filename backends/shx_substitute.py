"""SHX lettering in a headless render: the fallback face narrowed to the SHX's width.

A style whose font is ``isocp.shx`` is drawn by ezdxf with ``isocp.ttf``, an
Autodesk file no machine here has, so it falls back to the machine's default
face (DejaVuSans; DejaVuSansCondensed on CI). On mixed Polish plan text that
face's mean advance is 0.82 cap heights against 0.67 for ISO 3098 lettering:
23 % wider. ``text_overlap`` then reported collisions AutoCAD would not show,
and agents shrank their lettering to fit the preview - spoiling the DXF for a
renderer.

:func:`install` keeps one rule for every reader: for each :data:`SHX_METRICS`
row whose own file this machine lacks, it writes a copy of the fallback face
scaled horizontally to the row's mean advance (outlines and advances, nothing
else) into a cache folder, and routes ``ezdxf.fonts.fonts.make_font`` and
``resolve_font_face`` there. The renderer (screenshots, PDFs), ``disassemble``
(the TEXT / ATTRIB boxes of ``text_overlap`` and ``viewport_clip``) and
``mtext_size`` (their MTEXT boxes) then measure one width. Glyph shapes stay
the fallback's: this is a width, not a lookalike.

The mean advance is a **declared metric, not a font**: ``isocp.shx`` is
Autodesk's and not here, so ISO 3098 type B lettering's mean advance is
measured on osifont (an ISO 3098 typeface, not shipped) over :data:`SAMPLE`.
A machine that has the real file uses it; nothing is substituted.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

#: Plan lettering the metric is measured over: capitals, lower case, digits,
#: Polish letters and the marks a plan carries.
SAMPLE = (
    "SALON KUCHNIA Łazienka 12,45 m2 Gniazdo H=30CM WC szafa 0123456789 "
    "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

#: SHX / TTF file (lower case) -> mean advance over :data:`SAMPLE` in cap
#: heights (the cap height is the ``H`` glyph's top). 0.668 is osifont's
#: (ISO 3098 type B; DejaVuSans measures 0.819).
#: ISOCPEUR is ISOCP's TrueType twin and shares its widths.
SHX_METRICS: dict[str, float] = {
    "isocp.shx": 0.668,
    "isocpeur.ttf": 0.668,
}

#: Where the narrowed faces are written (per machine; rebuilt when missing).
CACHE_DIR = Path(tempfile.gettempdir()) / "acadmcp-fonts"

_lock = threading.Lock()
#: requested font (lower case) -> substitute file name, filled by :func:`install`.
_ROUTES: dict[str, str] = {}
_installed = False


def _key(font_name: str) -> str:
    """``ISOCP`` / ``isocp`` / ``C:\\fonts\\isocp.shx`` -> ``isocp.shx``."""
    name = Path(str(font_name).replace("\\", "/")).name.lower()
    return name if "." in name else f"{name}.shx"


def substitute_for(font_name: str) -> str | None:
    """The narrowed face :func:`install` routes ``font_name`` to, if any."""
    return _ROUTES.get(_key(font_name)) if font_name else None


def mean_advance(path: Path | str) -> float:
    """A TrueType face's mean advance over :data:`SAMPLE`, in cap heights."""
    from fontTools.ttLib import TTFont

    font = TTFont(str(path))
    cmap, hmtx = font.getBestCmap(), font["hmtx"]
    cap = font["glyf"][cmap[ord("H")]].yMax  # OS/2 sCapHeight is often absent
    widths = [hmtx[cmap[ord(c)]][0] for c in SAMPLE if ord(c) in cmap]
    return sum(widths) / len(widths) / cap


def _narrowed(source: Path, factor: float, target: Path, family: str) -> None:
    """Write ``source`` scaled by ``factor`` along x to ``target`` (atomically)."""
    from fontTools.ttLib import TTFont

    font = TTFont(str(source))
    glyf, hmtx = font["glyf"], font["hmtx"]
    order = font.getGlyphOrder()
    for name in order:  # simple glyphs first: a composite's bounds read its parts
        glyph = glyf[name]
        if not glyph.isComposite() and glyph.numberOfContours > 0:
            glyph.coordinates.scale((factor, 1.0))
            glyph.coordinates.toInt()
            glyph.recalcBounds(glyf)
    for name in order:
        glyph = glyf[name]
        if glyph.isComposite():
            for component in glyph.components:
                component.x = round(component.x * factor)
            glyph.recalcBounds(glyf)
        advance, _lsb = hmtx[name]
        hmtx[name] = (round(advance * factor), getattr(glyph, "xMin", 0))
    for table in ("fpgm", "prep", "cvt ", "hdmx", "LTSH", "VDMX", "kern", "GPOS"):
        if table in font:
            del font[table]
    for record in font["name"].names:
        if record.nameID in (1, 3, 4, 6, 16):
            record.string = family if record.nameID != 6 else family.replace(" ", "-")
    tmp = target.with_suffix(f".{os.getpid()}.tmp")
    font.save(str(tmp))
    os.replace(tmp, target)


def install() -> dict[str, str]:
    """Route every :data:`SHX_METRICS` font this machine lacks to its narrowed face.

    Idempotent; returns ``{requested font: substitute file}``. A failure (no
    writable temp folder, no fallback face) logs and leaves ezdxf as it was.
    """
    global _installed
    with _lock:
        if _installed:
            return dict(_ROUTES)
        _installed = True
        try:
            _install()
        except Exception as exc:  # a render must never fail over its font
            logger.warning("SHX width substitute not installed: %s", exc)
            _ROUTES.clear()
        return dict(_ROUTES)


def _install() -> None:
    from ezdxf.fonts import fonts
    from ezdxf.fonts.font_manager import get_ttf_font_face

    manager = fonts.font_manager
    fallback = manager.fallback_font_name()
    source = manager._font_cache.get(fallback, fallback).file_path
    native = mean_advance(source)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    routes: dict[str, str] = {}
    for font, advance in SHX_METRICS.items():
        twin = fonts.SHX_FONTS.get(font.upper(), font) if font.endswith(".shx") else font
        if manager.has_font(font) or manager.has_font(twin):
            continue  # the real face is here
        factor = round(advance / native, 3)
        name = f"acadmcp-{Path(fallback).stem}-x{factor:g}.ttf".lower()
        target = CACHE_DIR / name
        if not target.exists():
            _narrowed(Path(source), factor, target, f"ACADMCP {Path(fallback).stem} x{factor:g}")
        if not manager.has_font(name):
            manager._font_cache.add_entry(target, get_ttf_font_face(target))
        routes[font] = name
    if not routes:
        return
    _ROUTES.update(routes)
    make_font, resolve_font_face = fonts.make_font, fonts.resolve_font_face

    def routed_make_font(font_name, cap_height, width_factor=1.0):
        return make_font(substitute_for(font_name) or font_name, cap_height, width_factor)

    def routed_resolve_font_face(font_name, order="tsl"):
        return resolve_font_face(substitute_for(font_name) or font_name, order)

    fonts.make_font = routed_make_font
    fonts.resolve_font_face = routed_resolve_font_face
