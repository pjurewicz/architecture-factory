"""An architect's title block and a numbered notes box, from a spec, in a sheet corner.

``titleblock_apply`` is ISO 7200's fixed field set. An architectural sheet
carries its own: project, client, address, stage, scale, date, author, the
drawing's title and number - in the office's order and words. Round 3's agent
drew about 150 entities by hand for one. :func:`draw_title_block` takes the
rows as given and draws a two-column block (label, value) into a corner of the
sheet's ISO 5457 frame, or at a point, and a numbered NOTES box stacked on it
(above it in a bottom corner, below it in a top one) at the same width, or in
a box of its own.

``fields`` is a mapping or a list of rows; a mapping key from
:data:`FIELD_LABELS` is lettered in ``lang`` (``investor`` -> ``INWESTOR``),
any other key as written. A row is one cell or a list of cells side by side;
a cell is ``[label, value]`` or ``{label?, value, align?, text_height?,
width?}``. A value keeps its own line breaks (``\\n``) and wraps at its spaces
only where a line is longer than its column; its row grows. ``height`` fixes
the block's height: the rows are stretched to it, and a block whose content
needs more is refused with the height it needs. The ``title`` key's value is
lettered larger.

Sizes are paper millimetres (times ``scale`` in model space) and this
module's drafting choices, not a standard. Lines go on ``TITLEBLOCK``, text
on ``TEXT`` - the sheet layers ``titleblock_apply`` uses - and every entity
carries an ``ACADMCP_ANNOT`` record (tool ``arch_title_block``, ``set`` the
call's ``name``): a rerun with the same name on the same space deletes the
previous block and notes first and reports ``replaced``. ``color`` (ACI)
colours the block's lines and text, and the notes too unless ``notes.color``
says otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engineering.arch.lang import LANGS
from engineering.errors import InvalidArgsError
from engineering.mech.primitives import Line, Poly, Text

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

TOOL = "arch_title_block"
CORNERS = ("bottom_right", "bottom_left", "top_right", "top_left")
ALIGNS = ("left", "center", "right")

LABEL_HEIGHT = 1.8
VALUE_HEIGHT = 2.5
TITLE_HEIGHT = 3.5
LABEL_WIDTH = 35.0
ROW_HEIGHT = 7.0
PAD = 1.5
PITCH = 1.5  # line pitch, in text heights
MIN_CELL = 8.0  # paper mm: the narrowest cell a width split leaves

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
CELL_KEYS = {"label", "value", "align", "text_height", "width"}
NOTES_KEYS = {
    "items", "title", "frame", "text_height", "color", "x", "y", "width", "height", "numbering",
}  # fmt: skip
#: round 7: how a note is numbered; {n} is its number
NUMBERING = ("{n}.", "{n}/", "{n})", "-", "")


class TitleBlockArgsError(InvalidArgsError, ValueError):
    """An argument of ``arch_title_block`` refused before anything is drawn
    (kind ``invalid_args``); still a ``ValueError`` for the tool's wrapper."""


def _bad(message: str) -> TitleBlockArgsError:
    return TitleBlockArgsError(message)


def text_lines(value, where: str) -> list[str]:
    """``value`` as its lines: ``\\n`` (or a typed ``\\\\n``) breaks a line,
    runs of spaces collapse, empty lines at either end go; refused when empty."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise _bad(f"{where}: expected text or a number, got {value!r}.")
    lines = [" ".join(line.split()) for line in str(value).replace("\\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        raise _bad(f"{where}: must not be empty.")
    return lines


def _text(value, where: str) -> str:
    """One line of text (a label): its lines joined by a space."""
    return " ".join(text_lines(value, where))


def _height_arg(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad(f"{where}: expected paper mm, got {value!r}.")
    if not 1.0 <= float(value) <= 20.0:
        raise _bad(f"{where}: {value!r} mm is outside 1-20 mm.")
    return float(value)


def _cell(label, value, *, keyed: bool, lang: str, where: str, style: dict | None = None) -> dict:
    name = "" if label is None or label == "" else _text(label, f"{where}.label")
    known = FIELD_LABELS.get(name.lower()) if keyed else None
    style = style or {}
    align = style.get("align")
    if align is not None and align not in ALIGNS:
        raise _bad(f"{where}.align: one of {', '.join(ALIGNS)}, got {align!r}.")
    th = style.get("text_height")
    width = style.get("width")
    if width is not None and (
        isinstance(width, bool) or not isinstance(width, (int, float)) or width <= 0
    ):
        raise _bad(f"{where}.width: paper mm > 0, got {width!r}.")
    return {
        "label": known[lang] if known else name,
        "lines": text_lines(value, f"{where}.value" if style else where),
        "title": keyed and name.lower() == "title",
        "align": align,
        "text_height": None if th is None else _height_arg(th, f"{where}.text_height"),
        "width": None if width is None else float(width),
    }


def _dict_cell(cell: dict, *, keyed_label=None, lang: str, where: str) -> dict:
    bad = sorted(set(cell) - CELL_KEYS)
    if bad:
        raise _bad(f"{where}: unknown key(s) {bad}; a cell is {{{', '.join(sorted(CELL_KEYS))}}}.")
    if "value" not in cell:
        raise _bad(f"{where}.value: required.")
    label = keyed_label if keyed_label is not None else cell.get("label")
    return _cell(
        label, cell["value"], keyed=keyed_label is not None, lang=lang, where=where, style=cell
    )


def _is_pair(row) -> bool:
    return (
        isinstance(row, (list, tuple))
        and len(row) == 2
        and not any(isinstance(v, (list, tuple, dict)) for v in row)
    )


def _any_cell(c, lang: str, where: str) -> dict:
    if isinstance(c, dict):
        return _dict_cell(c, lang=lang, where=where)
    if _is_pair(c):
        return _cell(c[0], c[1], keyed=False, lang=lang, where=where)
    raise _bad(f"{where}: expected [label, value] or {{label?, value, align?, ...}}, got {c!r}.")


def parse_fields(fields, lang: str) -> list[list[dict]]:
    """One list of cells per row, refused by path. A list item is one cell
    (``[label, value]`` or a dict) or a list of cells side by side
    (``[["Date", "2026-11"], ["Stage", "PW"], {"label": "Scale", "value": "1:50"}]``)."""
    if isinstance(fields, dict):
        out = []
        for key, value in fields.items():
            where = f"fields.{key}"
            if isinstance(value, dict):
                out.append([_dict_cell(value, keyed_label=key, lang=lang, where=where)])
            else:
                out.append([_cell(key, value, keyed=True, lang=lang, where=where)])
    elif isinstance(fields, (list, tuple)):
        out = []
        for i, row in enumerate(fields):
            where = f"fields[{i}]"
            if isinstance(row, dict) or _is_pair(row):
                out.append([_any_cell(row, lang, where)])
            elif isinstance(row, (list, tuple)) and row:
                if len(row) > 6:
                    raise _bad(f"{where}: at most 6 cells side by side, got {len(row)}.")
                out.append([_any_cell(c, lang, f"{where}[{j}]") for j, c in enumerate(row)])
            else:
                raise _bad(f"{where}: expected a cell or a list of cells, got {row!r}.")
    else:
        raise _bad("fields: expected {key: value} or [[label, value] | [cell, ...], ...].")
    if len(out) > MAX_ROWS:
        raise _bad(f"fields: at most {MAX_ROWS} rows, got {len(out)}.")
    return out


def _wrap(lines: list[str], width: float, height: float) -> list[str]:
    """Each of ``lines`` kept as a line and broken at its spaces only where it
    is wider than ``width`` (a word wider than the column stays whole)."""
    from engineering.annotation import text_width

    out: list[str] = []
    for line in lines:
        if not line:
            out.append("")
            continue
        start = len(out)
        for word in line.split():
            if len(out) > start and text_width(f"{out[-1]} {word}", height) <= width:
                out[-1] = f"{out[-1]} {word}"
            else:
                out.append(word)
    return out or [""]


def _heights(text_height: float) -> tuple[float, float, float]:
    """``(label, value, title)`` heights for a value height in paper mm."""
    vh = float(text_height)
    return max(min(vh, LABEL_HEIGHT), 0.72 * vh), vh, TITLE_HEIGHT / VALUE_HEIGHT * vh


def _ax(align: str, x0: float, x1: float) -> float:
    """The alignment point's x for a text set ``align`` between x0 and x1."""
    return {"left": x0 + PAD, "center": (x0 + x1) / 2.0, "right": x1 - PAD}[align]


def notes_prims(
    notes: list[list[str]],
    width: float,
    top: float,
    *,
    title: str | None,
    frame: bool = True,
    text_height: float = VALUE_HEIGHT,
    numbering: str = "{n}.",
):
    """``(prims, height)``: the numbered notes box with its top-left corner at
    (0, top); each note is its lines (wrapped where too long)."""
    _lh, vh, th = _heights(text_height)
    items = [
        (numbering.format(n=i), _wrap(n, width - 4 * vh - 2 * PAD, vh))
        for i, n in enumerate(notes, 1)
    ]
    lines = sum(len(body) for _n, body in items)
    head = th + 2 * PAD if title else PAD
    height = head + lines * PITCH * vh + 3 * PAD
    prims: list = []
    if frame:
        prims.append(Poly(((0, top), (width, top), (width, top - height), (0, top - height)), True))
    if title:
        prims.append(Text((PAD, top - PAD - th), title, th, 0.0, "text"))
    y = top - head
    for number, body in items:
        for j, line in enumerate(body):
            y -= PITCH * vh
            if j == 0 and number:
                prims.append(Text((PAD, y), number, vh, 0.0, "text"))
            if line:
                prims.append(Text((PAD + 3.2 * vh, y), line, vh, 0.0, "text"))
    return prims, height


def _widths(cells: list[dict], width: float, where: str) -> list[float]:
    """Each cell's width: as given, the rest shared by the cells without one."""
    given = [c["width"] for c in cells]
    fixed = sum(w for w in given if w is not None)
    free = [i for i, w in enumerate(given) if w is None]
    if len(cells) == 1:
        if given[0] is not None:
            raise _bad(
                f"{where}.width: a cell alone spans the block; width is for cells side by side."
            )
        return [width]
    if not free:
        if abs(fixed - width) > 0.05:
            raise _bad(
                f"{where}: the cell widths add up to {fixed:g} mm, not the block's {width:g}; "
                "leave one cell without width to take the rest."
            )
        return [float(w) for w in given]
    rest = (width - fixed) / len(free)
    if rest < MIN_CELL:
        raise _bad(
            f"{where}: the cell widths leave {width - fixed:g} mm for {len(free)} cell(s) "
            f"without one (at least {MIN_CELL:g} mm each)."
        )
    return [rest if w is None else float(w) for w in given]


def block_prims(
    rows: list[list[dict]],
    width: float,
    *,
    text_height: float = VALUE_HEIGHT,
    label_position: str = "left",
    height: float | None = None,
):
    """``(prims, height, wide)`` of the title block, its top-left corner at (0, 0),
    +y up, paper mm. A one-cell row puts its label left of the value
    (``label_position="left"``) or above it, a cell without a label spans its
    row; a row of several cells letters each label above. ``height`` stretches
    the rows to it (refused when the content needs more). ``wide`` names every
    line wider than its column (a word is never broken)."""
    from engineering.annotation import text_width

    lh, vh, th = _heights(text_height)
    k = vh / VALUE_HEIGHT
    label_w = min(LABEL_WIDTH * k, 0.4 * width)
    plans, needs, wide = [], [], []
    for r, cells in enumerate(rows):
        widths = _widths(cells, width, f"fields[{r}]")
        laid, row_h = [], ROW_HEIGHT * k
        for cell, cw in zip(cells, widths, strict=True):
            h = cell["text_height"] or (th if cell["title"] else vh)
            mode = (
                "none"
                if not cell["label"]
                else ("above" if len(cells) > 1 or label_position == "above" else "left")
            )
            span = (width - label_w if mode == "left" else cw) - 2 * PAD
            lines = _wrap(cell["lines"], span, h)
            if len(lines) > len(cell["lines"]):
                need = max(text_width(line, h) for line in cell["lines"])
                wide.append(
                    f"fields[{r}]: wrapped to {len(lines)} lines in a {span:.1f} mm column "
                    f"(one line needs {need:.1f} mm) - widen the block or lower text_height"
                )
            for line in lines:
                if (w := text_width(line, h)) > span + 1e-6:
                    wide.append(
                        f"fields[{r}]: {line!r} is {w:.1f} mm wide, its column {span:.1f} mm - "
                        "lower text_height, widen the block or use label_position='above'"
                    )
            need = len(lines) * PITCH * h + 2 * PAD + (lh + PAD if mode == "above" else 0.0)
            laid.append((cell, cw, mode, lines, h))
            row_h = max(row_h, need)
        plans.append(laid)
        needs.append(row_h)
    natural = sum(needs)
    if height is not None:
        if natural > height + 1e-6:
            raise _bad(
                f"height: the block needs {natural:.1f} mm, over {height:g}; lower text_height, "
                "put cells side by side ([cell, cell, ...] in one row) or use "
                "label_position='above'."
            )
        extra = (height - natural) / len(needs) if needs else 0.0
        needs = [n + extra for n in needs]
    prims: list = []
    y = 0.0
    for r, (laid, row_h) in enumerate(zip(plans, needs, strict=True)):
        x = 0.0
        for c, (cell, cw, mode, lines, h) in enumerate(laid):
            align = cell["align"] or ("center" if cell["title"] and mode == "none" else "left")
            block = (len(lines) - 1) * PITCH * h + h  # cap of the first line to the last baseline
            if mode == "left":
                prims.append(Text((x + PAD, y - PAD - lh), cell["label"], lh, 0.0, "text"))
                prims.append(Line((label_w, y), (label_w, y - row_h)))
                x0, x1, top, bottom = label_w, width, y, y - row_h
            elif mode == "above":
                prims.append(
                    Text(
                        (_ax(align, x, x + cw), y - PAD - lh), cell["label"], lh, 0.0, "text", align
                    )
                )
                x0, x1, top, bottom = x, x + cw, y - PAD - lh - PAD, y - row_h + PAD
            else:
                x0, x1, top, bottom = x, x + cw, y, y - row_h
            first = (top + bottom) / 2.0 + block / 2.0 - h
            if mode == "above":
                first = min(first, top - h)
            for j, line in enumerate(lines):
                if line:
                    at = (_ax(align, x0, x1), first - j * PITCH * h)
                    prims.append(Text(at, line, h, 0.0, "text", align))
            if c:
                prims.append(Line((x, y), (x, y - row_h)))
            x += cw
        if r:
            prims.append(Line((0.0, y), (width, y)))
        y -= row_h
    block_h = -y
    if rows:
        prims.insert(0, Poly(((0, 0), (width, 0), (width, -block_h), (0, -block_h)), True))
    return prims, block_h, wide


def _drawn_frame(backend, layout: str):
    """``((x0, y0, x1, y1), handle)`` of the frame drawn on ``layout``: the
    largest closed axis-aligned rectangle (LWPOLYLINE) smaller than the paper
    and not one of our blocks or legends; None if there is none."""
    from engineering.annotation_record import read_annotation

    doc = backend._require_doc()
    space = doc.layouts.get(layout)
    paper = space.dxf_layout.dxf
    pw, ph = float(paper.get("paper_width", 0) or 0), float(paper.get("paper_height", 0) or 0)
    best = None
    for e in space.query("LWPOLYLINE"):
        pts = [(p[0], p[1]) for p in e.get_points("xy")]
        if not e.closed or len(pts) != 4:
            continue
        if (read_annotation(e) or {}).get("tool") in (TOOL, "arch_legend"):
            continue
        xs, ys = sorted({round(x, 6) for x, _ in pts}), sorted({round(y, 6) for _, y in pts})
        if len(xs) != 2 or len(ys) != 2:
            continue
        w, h = xs[1] - xs[0], ys[1] - ys[0]
        if (
            pw
            and ph
            and min(abs(w - pw), abs(w - ph)) < 0.5
            and min(abs(h - ph), abs(h - pw)) < 0.5
        ):
            continue  # the trimmed sheet itself
        if best is None or w * h > best[0]:
            best = (w * h, (xs[0], ys[0], xs[1], ys[1]), e.dxf.handle)
    if best is None:
        return None
    return best[1], best[2]


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


async def _draw(backend, prims, *, origin, k: float, set_name: str, color: int | None = None):
    """Draw sheet primitives (paper mm times ``k``) at ``origin``; every entity
    carries the set's ``ACADMCP_ANNOT`` record. Their handles, in order."""
    from engineering.annotation_record import tag_annotation
    from engineering.sheet.frames import SHEET_LAYER, SHEET_TEXT_LAYER

    ox, oy = origin
    handles: list[str] = []
    for p in prims:
        if isinstance(p, Text):
            info = await backend.entity_create_text(
                text=p.text,
                x=ox + p.at[0] * k,
                y=oy + p.at[1] * k,
                height=p.height * k,
                rotation=p.rotation,
                layer=SHEET_TEXT_LAYER,
                color=color,
                align=None if p.align == "left" else p.align,
            )
        elif isinstance(p, Line):
            info = await backend.entity_create_line(
                ox + p.p1[0] * k,
                oy + p.p1[1] * k,
                ox + p.p2[0] * k,
                oy + p.p2[1] * k,
                layer=SHEET_LAYER,
                color=color,
            )
        else:
            info = await backend.entity_create_polyline(
                points=[[ox + x * k, oy + y * k] for x, y in p.points],
                closed=p.closed,
                layer=SHEET_LAYER,
                color=color,
            )
        await tag_annotation(backend, info.handle, TOOL, set_name)
        handles.append(info.handle)
    return handles


def _color(c, where: str) -> int | None:
    if c is None:
        return None
    if isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= 256:
        raise _bad(f"{where}: an ACI colour 0-256, got {c!r}.")
    return c


def _num(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad(f"{where}: expected a number, got {value!r}.")
    return float(value)


def parse_notes(notes, lang: str, text_height: float) -> dict | None:
    """``notes`` as ``{items, title, frame, text_height, color, apart, x, y,
    width, height}``; a list is the items with every default."""
    if notes is None:
        return None
    spec = {"items": notes} if isinstance(notes, (list, tuple)) else notes
    if not isinstance(spec, dict) or "items" not in spec:
        raise _bad("notes: expected [text, ...] or {items, title?, frame?, text_height?, ...}.")
    bad = sorted(set(spec) - NOTES_KEYS)
    if bad:
        raise _bad(f"notes: unknown key(s) {bad}; keys are {', '.join(sorted(NOTES_KEYS))}.")
    items = spec["items"]
    if not isinstance(items, (list, tuple)) or not items:
        raise _bad("notes.items: expected a non-empty list of strings.")
    if len(items) > MAX_NOTES:
        raise _bad(f"notes.items: at most {MAX_NOTES}, got {len(items)}.")
    out = {
        "items": [text_lines(n, f"notes.items[{i}]") for i, n in enumerate(items)],
        "title": NOTES_TITLE[lang],
        "frame": True,
        "text_height": text_height,
        "color": None,
        "numbering": "{n}.",
    }
    if "numbering" in spec:
        if spec["numbering"] not in NUMBERING:
            raise _bad(f"notes.numbering: one of {NUMBERING}, got {spec['numbering']!r}.")
        out["numbering"] = spec["numbering"]
    if "title" in spec:
        t = spec["title"]
        out["title"] = None if t is None or t == "" else _text(t, "notes.title")
    if "frame" in spec:
        if not isinstance(spec["frame"], bool):
            raise _bad(f"notes.frame: true or false, got {spec['frame']!r}.")
        out["frame"] = spec["frame"]
    if spec.get("text_height") is not None:
        out["text_height"] = _height_arg(spec["text_height"], "notes.text_height")
    out["color"] = _color(spec.get("color"), "notes.color")
    apart = {"x", "y"} & set(spec)
    if apart and apart != {"x", "y"}:
        raise _bad("notes: give both x and y (the box's bottom-left corner), or neither.")
    out["apart"] = bool(apart)
    if apart:
        out["x"], out["y"] = _num(spec["x"], "notes.x"), _num(spec["y"], "notes.y")
    elif spec.get("height") is not None:
        raise _bad("notes.height: only for a box of its own (give x and y).")
    out["width"] = None if spec.get("width") is None else _num(spec["width"], "notes.width")
    if out["width"] is not None and not 30.0 <= out["width"] <= 400.0:
        raise _bad(f"notes.width: {out['width']:g} mm is outside 30-400 mm.")
    out["height"] = None if spec.get("height") is None else _num(spec["height"], "notes.height")
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
    height: float | None = None,
    color: int | None = None,
    name: str = "default",
) -> dict:
    """Draw the block and the notes; see the module docstring. ``at`` is the
    corner point (default: that corner of the layout's ISO 5457 frame).
    Refusals fire before anything is drawn or replaced."""
    from engineering.annotation_record import replace_set
    from engineering.sheet.frames import (
        SHEET_LAYER,
        SHEET_TEXT_LAYER,
        enter_layout,
        frame_metrics,
        resolve_sheet,
        sheet_fit_warnings,
    )

    if lang not in LANGS:
        raise _bad(f"lang must be one of {', '.join(LANGS)}, got {lang!r}.")
    if corner not in CORNERS:
        raise _bad(f"corner must be one of {', '.join(CORNERS)}, got {corner!r}.")
    if not isinstance(name, str) or not name.strip():
        raise _bad(f"name: a non-empty string, got {name!r}.")
    rows = parse_fields(fields, lang) if fields is not None else []
    if not 1.0 <= float(text_height) <= 10.0:
        raise _bad(f"text_height: {text_height!r} mm is outside 1-10 mm.")
    color = _color(color, "color")
    nspec = parse_notes(notes, lang, float(text_height))
    if not rows and nspec is None:
        raise _bad("arch_title_block: give fields, notes or both.")
    if not 30.0 <= float(width) <= 400.0:
        raise _bad(f"width: {width!r} mm is outside 30-400 mm.")
    if label_position not in ("left", "above"):
        raise _bad(f"label_position must be left or above, got {label_position!r}.")
    if height is not None and not rows:
        raise _bad("height: the title block's own height; give fields (notes take notes.height).")
    width = float(width)
    prims, block_h, wide = block_prims(
        rows,
        width,
        text_height=text_height,
        label_position=label_position,
        height=None if height is None else _num(height, "height"),
    )
    note_prims, notes_h, stacked = [], 0.0, False
    if nspec is not None:
        nw = nspec["width"] or width
        stacked = not nspec["apart"]
        if stacked and nspec["width"] is not None and rows and abs(nw - width) > 1e-6:
            raise _bad("notes.width: notes stacked on the block take its width; give x and y.")
        note_prims, notes_h = notes_prims(
            nspec["items"],
            nw,
            0.0,
            title=nspec["title"],
            frame=nspec["frame"],
            text_height=nspec["text_height"],
            numbering=nspec["numbering"],
        )
        if nspec["height"] is not None:
            if notes_h > nspec["height"] + 1e-6:
                raise _bad(
                    f"notes.height: the notes need {notes_h:.1f} mm, over {nspec['height']:g}; "
                    "widen the box or lower notes.text_height."
                )
            notes_h = nspec["height"]
            if nspec["frame"]:
                note_prims[0] = Poly(((0, 0), (nw, 0), (nw, -notes_h), (0, -notes_h)), True)
        nspec["w"] = nw
    report: dict = {}
    if at is None:
        if layout is None:
            raise _bad("arch_title_block: give a layout (its frame's corner) or at=[x, y].")
        drawn = await backend._async(_drawn_frame, backend, layout, readonly=True)
        if drawn is not None:  # round 7: the frame on the sheet, not ISO 5457's
            (x0, y0, x1, y1), handle = drawn
            report = {"anchor": {"frame": handle}}
        else:
            size, orientation, report = await resolve_sheet(
                backend, size, None, layout, caller=TOOL
            )
            x0, y0, x1, y1 = frame_metrics(size, orientation=orientation)["frame"]
            report["anchor"] = "iso_5457"
        at = (x1 if corner.endswith("right") else x0, y0 if corner.startswith("bottom") else y1)
    cx, cy = float(at[0]), float(at[1])
    k = 1.0 if layout is not None else float(scale)
    bottom = corner.startswith("bottom")
    # (0, 0) is the title block's top-left corner (or the notes' when there is
    # no block); the stacked notes sit above it in a bottom corner, below in a top one
    if not rows:
        stack_top, stack_bottom, notes_top = 0.0, -notes_h, 0.0
    elif stacked:
        notes_top = notes_h if bottom else -block_h
        stack_top = notes_h if bottom else 0.0
        stack_bottom = -block_h if bottom else -(block_h + notes_h)
    else:
        stack_top, stack_bottom, notes_top = 0.0, -block_h, 0.0
    ox = cx - (width * k if corner.endswith("right") else 0.0)
    oy = cy - stack_bottom * k if bottom else cy - stack_top * k
    previous = await enter_layout(backend, layout, caller=TOOL)
    try:
        replaced = await replace_set(backend, layout, TOOL, name)
        await _ensure_sheet_layers(backend, (SHEET_LAYER, SHEET_TEXT_LAYER))
        handles = await _draw(backend, prims, origin=(ox, oy), k=k, set_name=name, color=color)
        if nspec is not None:
            if stacked:
                norigin = (ox, oy + notes_top * k)
            else:
                norigin = (nspec["x"], nspec["y"] + notes_h * k)  # its bottom-left corner given
            handles += await _draw(
                backend,
                note_prims,
                origin=norigin,
                k=k,
                set_name=name,
                color=color if nspec["color"] is None else nspec["color"],
            )
    finally:
        if layout is not None:
            await backend.layout_set_current(previous)
    bbox = {
        "min": [round(ox, 3), round(oy + stack_bottom * k, 3)],
        "max": [round(ox + width * k, 3), round(oy + stack_top * k, 3)],
    }
    wrapped = [w for w in wide if ": wrapped to " in w]  # round 7: said, not warned
    warnings = list(report.pop("warnings", []) or []) + [w for w in wide if w not in wrapped]
    if rows or stacked:
        warnings += await sheet_fit_warnings(backend, layout, (bbox["min"], bbox["max"]))
    out_box = None
    if nspec is not None and not stacked:
        out_box = {
            "min": [nspec["x"], nspec["y"]],
            "max": [nspec["x"] + nspec["w"] * k, nspec["y"] + notes_h * k],
        }
        warnings += [
            f"notes {w}"
            for w in await sheet_fit_warnings(backend, layout, (out_box["min"], out_box["max"]))
        ]
    return {
        "ok": True,
        "name": name,
        "layout": layout or "Model",
        "corner": corner,
        "at": [cx, cy],
        "rows": [" | ".join(c["label"] for c in cells) for cells in rows],
        "height": round(block_h, 3),
        "notes": len(nspec["items"]) if nspec else 0,
        **({"wrapped": wrapped} if wrapped else {}),
        "handles": handles,
        "replaced": replaced,
        "bbox": bbox,
        **({"notes_bbox": out_box} if out_box else {}),
        **report,
        **({"warnings": warnings} if warnings else {}),
    }
