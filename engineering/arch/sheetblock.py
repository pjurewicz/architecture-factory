"""An architect's title block and a numbered notes box, from a spec, in a sheet corner.

``titleblock_apply`` is ISO 7200's fixed field set. An architectural sheet
carries its own: project, client, address, stage, scale, date, author, the
drawing's title and number - in the office's order and words. Round 3's agent
drew about 150 entities by hand for one. :func:`draw_title_block` takes the
rows as given and draws a two-column block (label, value) into a corner of the
sheet's ISO 5457 frame, or at a point, and a numbered NOTES box stacked on it
(above it in a bottom corner, below it in a top one) at the same width.

``fields`` is a mapping or a list of ``[label, value]`` pairs; a mapping key
from :data:`FIELD_LABELS` is lettered in ``lang`` (``investor`` ->
``INWESTOR``), any other key as written. A value wraps at its spaces to the
value column; its row grows. The ``title`` row's value is lettered larger.

Sizes are paper millimetres (times ``scale`` in model space) and this
module's drafting choices, not a standard. Lines go on ``TITLEBLOCK``, text
on ``TEXT`` - the sheet layers ``titleblock_apply`` uses - and every TEXT
carries an ``ACADMCP_ANNOT`` record (tool ``arch_title_block``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engineering.arch.lang import LANGS
from engineering.mech.primitives import Line, Poly, Text

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

TOOL = "arch_title_block"
CORNERS = ("bottom_right", "bottom_left", "top_right", "top_left")

LABEL_HEIGHT = 1.8
VALUE_HEIGHT = 2.5
TITLE_HEIGHT = 3.5
LABEL_WIDTH = 35.0
ROW_HEIGHT = 7.0
PAD = 1.5
PITCH = 1.5  # line pitch, in text heights

FIELD_LABELS: dict[str, dict[str, str]] = {
    "project": {"en": "PROJECT", "pl": "TEMAT", "tr": "PROJE"},
    "investor": {"en": "CLIENT", "pl": "INWESTOR", "tr": "İŞVEREN"},
    "address": {"en": "ADDRESS", "pl": "ADRES", "tr": "ADRES"},
    "stage": {"en": "STAGE", "pl": "ETAP", "tr": "AŞAMA"},
    "scale": {"en": "SCALE", "pl": "SKALA", "tr": "ÖLÇEK"},
    "date": {"en": "DATE", "pl": "DATA", "tr": "TARİH"},
    "designer": {"en": "DESIGNER", "pl": "PROJEKTANT", "tr": "TASARIM"},
    "author": {"en": "DRAWN BY", "pl": "OPRACOWAŁ", "tr": "ÇİZEN"},
    "checked": {"en": "CHECKED", "pl": "SPRAWDZIŁ", "tr": "KONTROL"},
    "title": {"en": "DRAWING TITLE", "pl": "TYTUŁ RYSUNKU", "tr": "PAFTA ADI"},
    "drawing_no": {"en": "DRAWING NO.", "pl": "NR RYSUNKU", "tr": "PAFTA NO"},
    "revision": {"en": "REVISION", "pl": "REWIZJA", "tr": "REVİZYON"},
}
NOTES_TITLE = {"en": "NOTES", "pl": "UWAGI", "tr": "NOTLAR"}
MAX_ROWS = 30
MAX_NOTES = 40


def _text(value, where: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{where}: expected text or a number, got {value!r}.")
    text = " ".join(str(value).split())
    if not text:
        raise ValueError(f"{where}: must not be empty.")
    return text


def parse_fields(fields, lang: str) -> list[tuple[str, str, bool]]:
    """``(label, value, is_title)`` per row, refused by path."""
    if isinstance(fields, dict):
        pairs = list(fields.items())
        keyed = True
    elif isinstance(fields, (list, tuple)):
        pairs, keyed = [], False
        for i, row in enumerate(fields):
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError(f"fields[{i}]: expected [label, value], got {row!r}.")
            pairs.append((row[0], row[1]))
    else:
        raise ValueError("fields: expected {key: value} or [[label, value], ...].")
    if len(pairs) > MAX_ROWS:
        raise ValueError(f"fields: at most {MAX_ROWS} rows, got {len(pairs)}.")
    out = []
    for i, (key, value) in enumerate(pairs):
        where = f"fields.{key}" if keyed else f"fields[{i}]"
        name = _text(key, f"{where} (label)")
        known = FIELD_LABELS.get(name.lower()) if keyed else None
        label = known[lang] if known else name
        out.append((label, _text(value, where), keyed and name.lower() == "title"))
    return out


def _wrap(text: str, width: float, height: float) -> list[str]:
    """``text`` broken at its spaces so each line fits ``width`` (a word wider
    than the column stays whole on its own line)."""
    from engineering.annotation import text_width

    lines: list[str] = []
    for word in text.split():
        if lines and text_width(f"{lines[-1]} {word}", height) <= width:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines or [""]


def block_prims(rows, notes, width: float, lang: str, *, notes_on_top: bool):
    """``(prims, height of the title block, height of the notes box)`` with the
    title block's top-left corner at (0, 0), +y up, paper mm."""
    value_w = width - LABEL_WIDTH - 2 * PAD
    prims: list = []
    y = 0.0
    laid = []
    for label, value, title in rows:
        h = TITLE_HEIGHT if title else VALUE_HEIGHT
        lines = _wrap(value, value_w, h)
        row_h = max(ROW_HEIGHT, len(lines) * PITCH * h + 2 * PAD)
        laid.append((label, lines, h, y, row_h))
        y -= row_h
    block_h = -y
    for label, lines, h, top, row_h in laid:
        prims.append(Text((PAD, top - PAD - LABEL_HEIGHT), label, LABEL_HEIGHT, 0.0, "text"))
        first = top - row_h / 2.0 + (len(lines) - 1) * PITCH * h / 2.0 - h / 2.0
        for j, line in enumerate(lines):
            prims.append(Text((LABEL_WIDTH + PAD, first - j * PITCH * h), line, h, 0.0, "text"))
        if top < 0:
            prims.append(Line((0.0, top), (width, top)))
    if rows:
        prims.insert(0, Poly(((0, 0), (width, 0), (width, -block_h), (0, -block_h)), True))
        prims.insert(1, Line((LABEL_WIDTH, 0.0), (LABEL_WIDTH, -block_h)))
    notes_h = 0.0
    if notes:
        items: list[tuple[str, list[str]]] = []
        for i, note in enumerate(notes, start=1):
            number = f"{i}."
            items.append((number, _wrap(note, width - 10.0 - 2 * PAD, VALUE_HEIGHT)))
        lines = sum(len(body) for _n, body in items)
        notes_h = TITLE_HEIGHT + 4 * PAD + lines * PITCH * VALUE_HEIGHT + PAD
        top = notes_h if notes_on_top else -block_h
        if not rows:
            top = 0.0
        prims.append(
            Poly(((0, top), (width, top), (width, top - notes_h), (0, top - notes_h)), True)
        )
        prims.append(
            Text((PAD, top - PAD - TITLE_HEIGHT), NOTES_TITLE[lang], TITLE_HEIGHT, 0.0, "text")
        )
        y = top - TITLE_HEIGHT - 2 * PAD
        for number, body in items:
            for j, line in enumerate(body):
                y -= PITCH * VALUE_HEIGHT
                if j == 0:
                    prims.append(Text((PAD, y), number, VALUE_HEIGHT, 0.0, "text"))
                prims.append(Text((PAD + 8.0, y), line, VALUE_HEIGHT, 0.0, "text"))
    return prims, block_h, notes_h


async def _ensure_sheet_layers(backend, names) -> None:
    """Create a missing sheet layer with the engineering scaffold's row."""
    from engineering.layers import ENGINEERING_LAYERS

    rows = {row[0]: row for row in ENGINEERING_LAYERS}
    have = {layer.name.lower() for layer in await backend.layer_list()}
    for name in names:
        if name.lower() not in have:
            _n, color, linetype, lineweight, _d = rows[name]
            await backend.layer_create(
                name=name, color=color, linetype=linetype, lineweight=lineweight
            )


def _scaled(prims, k: float):
    out = []
    for p in prims:
        if isinstance(p, Text):
            out.append(p._replace(at=(p.at[0] * k, p.at[1] * k), height=p.height * k))
        elif isinstance(p, Line):
            out.append(p._replace(p1=(p.p1[0] * k, p.p1[1] * k), p2=(p.p2[0] * k, p.p2[1] * k)))
        else:
            out.append(p._replace(points=tuple((x * k, y * k) for x, y in p.points)))
    return out


async def draw_title_block(
    backend: AutoCADBackend,
    *,
    fields=None,
    notes=None,
    layout: str | None = None,
    corner: str = "bottom_right",
    at=None,
    size: str | None = None,
    width: float = 180.0,
    scale: float = 50,
    lang: str = "en",
) -> dict:
    """Draw the block and the notes; see the module docstring. ``at`` is the
    corner point (default: that corner of the layout's ISO 5457 frame).
    Refusals fire before anything is drawn."""
    from engineering.sheet.frames import (
        SHEET_LAYER,
        SHEET_TEXT_LAYER,
        draw_sheet_prims,
        enter_layout,
        frame_metrics,
        resolve_sheet,
    )

    if lang not in LANGS:
        raise ValueError(f"lang must be one of {', '.join(LANGS)}, got {lang!r}.")
    if corner not in CORNERS:
        raise ValueError(f"corner must be one of {', '.join(CORNERS)}, got {corner!r}.")
    rows = parse_fields(fields, lang) if fields is not None else []
    if notes is not None and not isinstance(notes, (list, tuple)):
        raise ValueError("notes: expected a list of strings.")
    texts = [_text(n, f"notes[{i}]") for i, n in enumerate(notes or ())]
    if len(texts) > MAX_NOTES:
        raise ValueError(f"notes: at most {MAX_NOTES}, got {len(texts)}.")
    if not rows and not texts:
        raise ValueError("arch_title_block: give fields, notes or both.")
    if not 60.0 <= float(width) <= 400.0:
        raise ValueError(f"width: {width!r} mm is outside 60-400 mm.")
    report: dict = {}
    if at is None:
        if layout is None:
            raise ValueError("arch_title_block: give a layout (its frame's corner) or at=[x, y].")
        size, orientation, report = await resolve_sheet(backend, size, None, layout, caller=TOOL)
        x0, y0, x1, y1 = frame_metrics(size, orientation=orientation)["frame"]
        at = (x1 if corner.endswith("right") else x0, y0 if corner.startswith("bottom") else y1)
    cx, cy = float(at[0]), float(at[1])
    k = 1.0 if layout is not None else float(scale)
    bottom = corner.startswith("bottom")
    prims, block_h, notes_h = block_prims(rows, texts, float(width), lang, notes_on_top=bottom)
    # (0, 0) is the title block's top-left corner; move the chosen corner of the
    # whole stack onto `at`
    stack_top = notes_h if (bottom and rows) else 0.0
    stack_bottom = -(block_h + (0.0 if bottom or not rows else notes_h))
    if not rows:
        stack_bottom = -notes_h
    ox = cx - (float(width) * k if corner.endswith("right") else 0.0)
    oy = cy - stack_bottom * k if bottom else cy - stack_top * k
    previous = await enter_layout(backend, layout, caller=TOOL)
    try:
        await _ensure_sheet_layers(backend, (SHEET_LAYER, SHEET_TEXT_LAYER))
        handles = await draw_sheet_prims(
            backend, _scaled(prims, k), origin=(ox, oy), annotation=TOOL
        )
    finally:
        if layout is not None:
            await backend.layout_set_current(previous)
    return {
        "ok": True,
        "layout": layout or "Model",
        "corner": corner,
        "at": [cx, cy],
        "rows": [label for label, _v, _t in rows],
        "notes": len(texts),
        "handles": handles,
        "bbox": {
            "min": [ox, oy + stack_bottom * k],
            "max": [ox + float(width) * k, oy + stack_top * k],
        },
        **report,
    }
