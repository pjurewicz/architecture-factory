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


def _cell(label, value, keyed: bool, lang: str, where: str) -> tuple[str, str, bool]:
    name = _text(label, f"{where} (label)")
    known = FIELD_LABELS.get(name.lower()) if keyed else None
    return (known[lang] if known else name, _text(value, where), keyed and name.lower() == "title")


def _is_pair(row) -> bool:
    return (
        isinstance(row, (list, tuple))
        and len(row) == 2
        and not any(isinstance(v, (list, tuple, dict)) for v in row)
    )


def parse_fields(fields, lang: str) -> list[list[tuple[str, str, bool]]]:
    """One list of ``(label, value, is_title)`` cells per row, refused by path.
    A list item is ``[label, value]`` (one cell) or a list of such pairs (cells
    side by side, ``[["Date", "2026-11"], ["Stage", "PW"], ["Scale", "1:50"]]``)."""
    if isinstance(fields, dict):
        out = [[_cell(key, value, True, lang, f"fields.{key}")] for key, value in fields.items()]
    elif isinstance(fields, (list, tuple)):
        out = []
        for i, row in enumerate(fields):
            if _is_pair(row):
                out.append([_cell(row[0], row[1], False, lang, f"fields[{i}]")])
            elif isinstance(row, (list, tuple)) and row and all(_is_pair(c) for c in row):
                if len(row) > 6:
                    raise ValueError(f"fields[{i}]: at most 6 cells side by side, got {len(row)}.")
                out.append(
                    [_cell(c[0], c[1], False, lang, f"fields[{i}][{j}]") for j, c in enumerate(row)]
                )
            else:
                raise ValueError(
                    f"fields[{i}]: expected [label, value] or [[label, value], ...], got {row!r}."
                )
    else:
        raise ValueError("fields: expected {key: value} or [[label, value], ...].")
    if len(out) > MAX_ROWS:
        raise ValueError(f"fields: at most {MAX_ROWS} rows, got {len(out)}.")
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


def _heights(text_height: float) -> tuple[float, float, float]:
    """``(label, value, title)`` heights for a value height in paper mm."""
    vh = float(text_height)
    return max(min(vh, LABEL_HEIGHT), 0.72 * vh), vh, TITLE_HEIGHT / VALUE_HEIGHT * vh


def notes_prims(notes, width: float, lang: str, top: float, text_height: float = VALUE_HEIGHT):
    """``(prims, height)``: the numbered notes box with its top-left corner at (0, top)."""
    _lh, vh, th = _heights(text_height)
    items = [(f"{i}.", _wrap(n, width - 4 * vh - 2 * PAD, vh)) for i, n in enumerate(notes, 1)]
    lines = sum(len(body) for _n, body in items)
    height = th + 4 * PAD + lines * PITCH * vh + PAD
    prims: list = [Poly(((0, top), (width, top), (width, top - height), (0, top - height)), True)]
    prims.append(Text((PAD, top - PAD - th), NOTES_TITLE[lang], th, 0.0, "text"))
    y = top - th - 2 * PAD
    for number, body in items:
        for j, line in enumerate(body):
            y -= PITCH * vh
            if j == 0:
                prims.append(Text((PAD, y), number, vh, 0.0, "text"))
            prims.append(Text((PAD + 3.2 * vh, y), line, vh, 0.0, "text"))
    return prims, height


def block_prims(
    rows,
    notes,
    width: float,
    lang: str,
    *,
    notes_on_top: bool,
    text_height: float = VALUE_HEIGHT,
    label_position: str = "left",
):
    """``(prims, height of the title block, height of the notes box)`` with the
    title block's top-left corner at (0, 0), +y up, paper mm. A one-cell row
    puts its label left of the value (``label_position="left"``) or above it;
    a row of several cells splits the width and letters each label above."""
    lh, vh, th = _heights(text_height)
    k = vh / VALUE_HEIGHT
    label_w = min(LABEL_WIDTH * k, 0.4 * width)
    prims: list = []
    y = 0.0
    for r, cells in enumerate(rows):
        cell_w = width / len(cells)
        above = len(cells) > 1 or label_position == "above"
        laid, row_h = [], ROW_HEIGHT * k
        for label, value, title in cells:
            h = th if title else vh
            span = cell_w - 2 * PAD if above else width - label_w - 2 * PAD
            lines = _wrap(value, span, h)
            need = len(lines) * PITCH * h + 2 * PAD + (lh + PAD if above else 0.0)
            laid.append((label, lines, h))
            row_h = max(row_h, need)
        for c, (label, lines, h) in enumerate(laid):
            x = c * cell_w
            prims.append(Text((x + PAD, y - PAD - lh), label, lh, 0.0, "text"))
            if above:
                first = y - PAD - lh - PAD - h
                vx = x + PAD
            else:
                first = y - row_h / 2.0 + (len(lines) - 1) * PITCH * h / 2.0 - h / 2.0
                vx = label_w + PAD
                prims.append(Line((label_w, y), (label_w, y - row_h)))
            for j, line in enumerate(lines):
                prims.append(Text((vx, first - j * PITCH * h), line, h, 0.0, "text"))
            if c:
                prims.append(Line((x, y), (x, y - row_h)))
        if r:
            prims.append(Line((0.0, y), (width, y)))
        y -= row_h
    block_h = -y
    if rows:
        prims.insert(0, Poly(((0, 0), (width, 0), (width, -block_h), (0, -block_h)), True))
    notes_h = 0.0
    if notes:
        top = 0.0 if not rows else (None if notes_on_top else -block_h)
        if top is None:
            _p, notes_h = notes_prims(notes, width, lang, 0.0, text_height)
            top = notes_h
        more, notes_h = notes_prims(notes, width, lang, top, text_height)
        prims += more
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
    text_height: float = VALUE_HEIGHT,
    label_position: str = "left",
    max_height: float | None = None,
    notes_box: dict | None = None,
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
        sheet_fit_warnings,
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
    if label_position not in ("left", "above"):
        raise ValueError(f"label_position must be left or above, got {label_position!r}.")
    if not 1.0 <= float(text_height) <= 10.0:
        raise ValueError(f"text_height: {text_height!r} mm is outside 1-10 mm.")
    box = _notes_box(notes_box) if notes_box is not None else None
    if box is not None and not texts:
        raise ValueError("notes_box: give the notes it holds.")
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
    stacked = texts if box is None else []
    prims, block_h, notes_h = block_prims(
        rows,
        stacked,
        float(width),
        lang,
        notes_on_top=bottom,
        text_height=text_height,
        label_position=label_position,
    )
    if max_height is not None and block_h + notes_h > float(max_height) + 1e-6:
        raise ValueError(
            f"max_height: the block needs {block_h + notes_h:.1f} mm, over {float(max_height):g}; "
            "lower text_height, put cells side by side ([[label, value], ...] in one row), "
            "use label_position='above' or move the notes into notes_box."
        )
    note_prims: list = []
    if box is not None:
        note_prims, need = notes_prims(texts, box["width"], lang, 0.0, text_height)
        if box.get("height") is not None:
            if need > box["height"] + 1e-6:
                raise ValueError(
                    f"notes_box: the notes need {need:.1f} mm, over its height {box['height']:g}; "
                    "widen it or lower text_height."
                )
            note_prims[0] = Poly(
                ((0, 0), (box["width"], 0), (box["width"], -box["height"]), (0, -box["height"])),
                True,
            )
            need = box["height"]
        box["need"] = need
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
        if box is not None:
            bx, by = box["x"], box["y"] + box["need"] * k  # its bottom-left corner given
            handles += await draw_sheet_prims(
                backend, _scaled(note_prims, k), origin=(bx, by), annotation=TOOL
            )
    finally:
        if layout is not None:
            await backend.layout_set_current(previous)
    bbox = {
        "min": [round(ox, 3), round(oy + stack_bottom * k, 3)],
        "max": [round(ox + float(width) * k, 3), round(oy + stack_top * k, 3)],
    }
    warnings = list(report.pop("warnings", []) or [])
    if rows or box is None:
        warnings += await sheet_fit_warnings(backend, layout, (bbox["min"], bbox["max"]))
    out_box = None
    if box is not None:
        out_box = {
            "min": [box["x"], box["y"]],
            "max": [box["x"] + box["width"] * k, box["y"] + box["need"] * k],
        }
        warnings += [
            f"notes_box {w}"
            for w in await sheet_fit_warnings(backend, layout, (out_box["min"], out_box["max"]))
        ]
    return {
        "ok": True,
        "layout": layout or "Model",
        "corner": corner,
        "at": [cx, cy],
        "rows": [" | ".join(label for label, _v, _t in cells) for cells in rows],
        "notes": len(texts),
        "handles": handles,
        "bbox": bbox,
        **({"notes_bbox": out_box} if out_box else {}),
        **report,
        **({"warnings": warnings} if warnings else {}),
    }


def _notes_box(value) -> dict:
    """``{x, y, width, height?}``: the notes' own box, its bottom-left corner."""
    if not isinstance(value, dict) or not {"x", "y", "width"} <= set(value):
        raise ValueError("notes_box: expected {x, y, width, height?} (bottom-left corner).")
    extra = sorted(set(value) - {"x", "y", "width", "height"})
    if extra:
        raise ValueError(f"notes_box: unknown key(s) {extra}.")
    out = {}
    for key in ("x", "y", "width", "height"):
        v = value.get(key)
        if v is None and key == "height":
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"notes_box.{key}: expected a number, got {v!r}.")
        out[key] = float(v)
    if not 30.0 <= out["width"] <= 400.0:
        raise ValueError(f"notes_box.width: {out['width']:g} mm is outside 30-400 mm.")
    return out
