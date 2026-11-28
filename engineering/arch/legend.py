"""A legend built from what the drawing uses: a sample and a description per row.

:func:`draw_legend` reads model space once - every block an INSERT uses and
every hatch pattern (pattern, angle and scale) - and draws one row each: a
sample cell and its description, materials first, then symbols (``order``
names rows to put first), under a title (``title=None``: none), unframed
unless ``frame``. A sample is the real thing: the same block, scaled to fit its
cell but never above its plotted size, and the same hatch pattern at the same
printed density.

Descriptions come from the server's own tables, in ``lang``: an
``arch_services`` block names its kind, a catalogue block its item, a wall
hatch its material (``WALL_HATCH_TABLE`` by pattern, angle and density).
Anything else is named by its block or pattern name, and ``labels`` overrides
or (``null``) drops any row by that key. A label keeps its line breaks
(``\\n``) and its row grows to them. An ``extra`` row with ``blocks`` or
``keys`` shows several samples side by side (hot and cold water).

On a paper layout (``layout``) every size is paper millimetres; in model
space it is paper millimetres times ``scale``, as the plan's other lettering.
The frame, title, descriptions and sample outlines go on ``A-SYMB-T-N`` (or
``layer``, in ``color``); a sample keeps its source's layer so it prints as on
the plan. Every entity carries an ``ACADMCP_ANNOT`` record (tool
``arch_legend``, ``set`` the call's ``name``), so a legend never lists another
one's samples and a rerun with the same name on the same space deletes the
previous one first (``replaced``).
"""

from __future__ import annotations

import fnmatch
import math
from typing import TYPE_CHECKING

from engineering.arch import services as elec
from engineering.arch.catalogue import _BY_NAME, BLOCK_PREFIX, CATALOGUE_NAMES, LABEL_PL
from engineering.arch.lang import LANGS
from engineering.arch.layers import ARCH_ROLE_LAYER
from engineering.arch.materials import TABLE_PLOT_SCALE, WALL_HATCH_TABLE
from engineering.errors import InvalidArgsError

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

TOOL = "arch_legend"
LAYER = ARCH_ROLE_LAYER["symbol"]

#: Paper millimetres: this module's drafting choices, not a standard.
TITLE_HEIGHT = 3.5  # at TEXT_HEIGHT; scales with text_height
TEXT_HEIGHT = 2.5
ROW_GAP = 2.0  # between two rows
LINE_PITCH = 1.5  # a label's line pitch, in text heights
CELL_W, CELL_H = 15.0, 6.0
SAMPLE_GAP = 2.0  # between two samples of one row
PAD = 2.0
ROWS_PER_COLUMN = 20

TITLE_AUTO = "auto"  # the title in `lang`
TITLE = {"en": "LEGEND", "pl": "OZNACZENIA", "tr": "LEJANT"}


class LegendArgsError(InvalidArgsError, ValueError):
    """An ``arch_legend`` argument refused before anything is drawn (kind
    ``invalid_args``); still a ``ValueError`` for the tool's wrapper."""


MATERIAL_LABELS: dict[str, dict[str, str]] = {
    "brick": {"en": "brick wall", "pl": "mur z cegły", "tr": "tuğla duvar"},
    "aac": {
        "en": "aerated concrete (AAC) wall",
        "pl": "ściana z betonu komórkowego",
        "tr": "gazbeton duvar",
    },
    "concrete": {"en": "concrete", "pl": "beton", "tr": "beton"},
    "reinforced_concrete": {"en": "reinforced concrete", "pl": "żelbet", "tr": "betonarme"},
    "gypsum_board": {"en": "gypsum board", "pl": "płyta g-k", "tr": "alçı pano"},
    "stone": {"en": "stone", "pl": "kamień", "tr": "taş"},
    "timber": {"en": "timber", "pl": "drewno", "tr": "ahşap"},
}


def _material(pattern: str, angle: float, scale: float, plot_scale: float) -> str | None:
    """The wall material whose hatch this is at ``plot_scale``, or None."""
    for name, row in WALL_HATCH_TABLE.items():
        if name not in MATERIAL_LABELS:
            continue
        want = row["scale"] * plot_scale / TABLE_PLOT_SCALE
        if (
            row["pattern"].upper() == pattern.upper()
            and abs((row["angle"] - angle + 180.0) % 360.0 - 180.0) < 1e-6
            and abs(want - scale) <= 1e-6 * max(1.0, want)
        ):
            return name
    return None


def _catalogue_item(block: str) -> str | None:
    """``ARCH_DOUBLE_BED`` / ``ARCH_DOUBLE_BED_1800x2000`` -> ``double_bed``."""
    if not block.upper().startswith(BLOCK_PREFIX) or block.upper().startswith(elec.BLOCK_PREFIXES):
        return None
    rest = block[len(BLOCK_PREFIX) :].lower()
    for name in sorted(CATALOGUE_NAMES, key=len, reverse=True):
        if rest == name or rest.startswith(f"{name}_"):
            return name
    return None


def _elec_kind(block: str) -> str | None:
    return elec.kind_of_block(block)


def _catalogue_label(item: str, lang: str) -> str:
    row = _BY_NAME[item]
    return {"tr": row.label_tr, "pl": LABEL_PL[item]}.get(lang, row.label_en)


def _extents(doc, name: str):
    from ezdxf import bbox

    box = bbox.extents(x for x in doc.blocks.get(name) if x.dxftype() != "ATTDEF")
    return (box.extmin.x, box.extmin.y, box.extmax.x, box.extmax.y) if box.has_data else None


def _run_style(e) -> dict:
    return {
        "layer": e.dxf.layer,
        "linetype": e.dxf.get("linetype", "BYLAYER"),
        "lineweight": e.dxf.get("lineweight", -1),
        "ltscale": float(e.dxf.get("ltscale", 1.0)),
        "color": e.dxf.get("color", 256),
    }


def _read(backend) -> dict:
    """Items model space uses, one group each (a legend's own samples excluded):
    ``{key: row}`` with ``kind`` block / hatch / line."""
    from engineering.annotation_record import APP_ID, read_annotation

    doc = backend._require_doc()
    records = {id(e): p for e, p in elec.read_records(doc)}
    wall_hatch = ARCH_ROLE_LAYER["wall_hatch"].upper()
    groups: dict[str, dict] = {}
    for e in doc.modelspace():
        if e.has_xdata(APP_ID) and TOOL in str(e.get_xdata(APP_ID)):
            continue
        kind = e.dxftype()
        record = records.get(id(e))
        if kind == "INSERT":
            name = e.dxf.name
            block = doc.blocks.get(name)
            if name.startswith("*") or block is None or block.block.is_xref:
                continue
            item = record["item"] if record and record["kind"] == "device" else _elec_kind(name)
            if item is not None:
                key, order, off = item, 1, False
            elif (item := _catalogue_item(name)) is not None:
                key, order, off = item, 2, True
            else:
                key, order, off = name, 3, False
            row = groups.setdefault(
                key.upper(),
                {
                    "kind": "block",
                    "key": key,
                    "order": 10 + order,
                    "item": item,
                    "default_off": off,
                    "count": 0,
                    "names": [],
                    "layer": e.dxf.layer,
                    "block": name,
                },
            )
            row["count"] += 1
            if name not in row["names"]:
                row["names"].append(name)
            if off and name.upper() == f"{BLOCK_PREFIX}{item.upper()}":
                row["block"] = name  # the nominal size samples the item
        elif kind == "HATCH":
            solid = bool(e.dxf.solid_fill)
            pattern = "SOLID" if solid else e.dxf.pattern_name.upper()
            angle = 0.0 if solid else round(float(e.dxf.pattern_angle), 6)
            pscale = 1.0 if solid else round(float(e.dxf.pattern_scale), 6)
            poche = solid and e.dxf.layer.upper() == wall_hatch
            own = (read_annotation(e) or {}).get("legend")  # round 7: a zone names itself
            key = f"{pattern}@{angle:g}@{pscale:g}" + (f"@{own}" if own else "")
            row = groups.setdefault(
                key,
                {
                    "kind": "hatch",
                    "key": own or pattern,
                    **({"legend": own} if own else {}),
                    "pattern": pattern,
                    "angle": angle,
                    "scale": pscale,
                    "order": 0,
                    "default_off": poche,
                    "count": 0,
                    "names": [pattern],
                    "layer": e.dxf.layer,
                },
            )
            row["count"] += 1
        elif kind in ("LWPOLYLINE", "LINE") and record and record["kind"] == "run":
            item = record["item"]
            row = groups.setdefault(
                item.upper(),
                {
                    "kind": "line",
                    "key": item,
                    "item": item,
                    "order": 9,
                    "default_off": False,
                    "count": 0,
                    "names": [item],
                    **_run_style(e),
                },
            )
            row["count"] += 1
    for row in groups.values():
        if row["kind"] == "block":
            row["extents"] = _extents(doc, row["block"])
    return {"groups": list(groups.values())}


def _match(row: dict, patterns) -> bool:
    names = [row["key"], *row.get("names", ())]
    return any(fnmatch.fnmatchcase(n.upper(), str(p).upper()) for p in patterns for n in names)


def _globs(value, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise LegendArgsError(f"{where}: expected a list of names or patterns (* and ?).")
    return list(value)


def legend_rows(
    found: dict,
    *,
    lang: str,
    scale: float,
    labels: dict | None,
    only=None,
    exclude=None,
    layers=None,
) -> tuple[list[dict], list[str]]:
    """``(rows in order, keys left out by default)``: materials and other
    hatches, runs, then blocks; ``labels`` overrides a label or drops a row
    (``None``) by key or block / pattern name, case-insensitively."""
    only, exclude, layers = (
        _globs(only, "only"),
        _globs(exclude, "exclude"),
        _globs(layers, "layers"),
    )
    overrides = {str(k).upper(): v for k, v in (labels or {}).items()}
    groups = found["groups"]
    unknown = sorted(k for k in overrides if not any(_match(r, [k]) for r in groups))
    if unknown:
        raise LegendArgsError(
            f"labels: no item, block or hatch pattern {unknown} in model space; the keys "
            "are the items, block names and pattern names the drawing uses."
        )
    rows, skipped = [], []
    for row in groups:
        row = dict(row)
        if row["kind"] == "hatch" and row.get("legend"):
            row["label"], row["order"] = row["legend"], 0
        elif row["kind"] == "hatch":
            material = _material(row["pattern"], row["angle"], row["scale"], scale)
            row["label"] = MATERIAL_LABELS[material][lang] if material else row["pattern"]
            row["order"] = 0 if material else 1
        elif row["kind"] == "line":
            row["label"] = elec.label(row["item"], lang)
        elif row["order"] == 11:
            row["label"] = elec.label(row["item"], lang)
        elif row["order"] == 12:
            row["label"] = _catalogue_label(row["item"], lang)
        else:
            row["label"] = row["key"]
        if only:
            if not _match(row, only):
                continue
        elif row["default_off"]:
            skipped.append(row["key"])
            continue
        if exclude and _match(row, exclude):
            continue
        if layers and not any(
            fnmatch.fnmatchcase(str(row.get("layer", "")).upper(), p.upper()) for p in layers
        ):
            continue
        hit = next((k for k in overrides if _match(row, [k])), None)
        if hit is not None:
            if overrides[hit] is None:
                continue
            row["label"] = _label_text(overrides[hit], f"labels.{hit}")
        rows.append(row)
    rows.sort(key=lambda r: (r["order"], r["label"].casefold()))
    return rows, sorted(set(skipped))


_EXTRA_KEYS = {"label", "block", "pattern", "kind", "blocks", "keys"}
#: round 7: kind "line" draws a line in a layer's or its own style
_LINE_KEYS = {"layer", "linetype", "lineweight", "color"}
_SAMPLE_KEYS = ("block", "pattern", "kind", "blocks", "keys")
MAX_SAMPLES = 4


def _label_text(value, where: str) -> str:
    """A label with its own line breaks (``\\n``) kept, spaces collapsed."""
    from engineering.arch.sheetblock import text_lines

    return "\n".join(text_lines(value, where))


def _block_sample(doc, name, where: str) -> dict:
    name = str(name)
    if name not in doc.blocks:
        raise LegendArgsError(f"{where}: no block {name!r} in the drawing.")
    return {"kind": "block", "block": name, "extents": _extents(doc, name), "layer": LAYER}


def _extra_rows(backend, extra, k: float, model_rows: list[dict]) -> tuple[list[dict], set]:
    """``(extra rows, ids of model rows they take over)``: a sample from a
    block, a hatch pattern or an ``arch_services`` kind, several samples side
    by side (``blocks`` / ``keys`` - a model row named in ``keys`` moves into
    this one), or none; checked by path. Our own symbol block is defined when
    the drawing lacks it."""
    if extra is None:
        return [], set()
    if not isinstance(extra, (list, tuple)):
        raise LegendArgsError(
            "extra: expected [{label, block? | pattern? | kind? | blocks? | keys?}]."
        )
    doc = backend._require_doc()
    out, taken = [], set()
    for i, item in enumerate(extra):
        where = f"extra[{i}]"
        if not isinstance(item, dict) or "label" not in item:
            raise LegendArgsError(
                f"{where}: expected {{label, block? | pattern? | kind? | blocks? | keys?}}."
            )
        allowed = _EXTRA_KEYS | (_LINE_KEYS if item.get("kind") == "line" else set())
        bad = sorted(set(item) - allowed)
        if bad or len(set(item) & set(_SAMPLE_KEYS)) > 1:
            raise LegendArgsError(
                f"{where}: give label and at most one of {', '.join(_SAMPLE_KEYS)}"
                f" ({', '.join(sorted(_LINE_KEYS))} with kind 'line')."
            )
        label = _label_text(item["label"], f"{where}.label")
        row = {
            "key": label,
            "label": label,
            "count": 0,
            "order": 20 + i,
            "layer": LAYER,
            "names": [],
            "extra": True,
        }
        for many in ("blocks", "keys"):
            if many in item:
                names = item[many]
                if (
                    not isinstance(names, (list, tuple))
                    or not names
                    or len(names) > MAX_SAMPLES
                    or not all(isinstance(n, str) for n in names)
                ):
                    raise LegendArgsError(f"{where}.{many}: 1-{MAX_SAMPLES} names side by side.")
        if "blocks" in item:
            samples = [
                _block_sample(doc, n, f"{where}.blocks[{j}]") for j, n in enumerate(item["blocks"])
            ]
            row.update(kind="block", samples=samples)
        elif "keys" in item:
            samples = []
            for j, key in enumerate(item["keys"]):
                hit = [r for r in model_rows if _match(r, [key])]
                if not hit:
                    raise LegendArgsError(
                        f"{where}.keys[{j}]: no item, block or hatch pattern {key!r} in model "
                        "space; the keys are the items, block names and pattern names it uses."
                    )
                for r in hit:
                    if id(r) not in taken:
                        taken.add(id(r))
                        samples.append(r)
                        row["count"] += r["count"]
                        row["names"] += [r["key"], *r.get("names", ())]
            row.update(kind=samples[0]["kind"], samples=samples[: MAX_SAMPLES * 2])
        elif "block" in item:
            row.update(_block_sample(doc, item["block"], f"{where}.block"))
        elif "pattern" in item:
            row.update(kind="hatch", pattern=str(item["pattern"]).upper(), angle=0.0, scale=k)
        elif "kind" in item:
            kind = item["kind"]
            if kind == "line":
                row.update(kind="line", **_line_style(doc, item, where))
            elif kind in elec.RUN_KINDS:
                run = elec.RUN_KINDS[kind]
                row.update(
                    kind="line",
                    item=kind,
                    linetype=run.linetype,
                    lineweight=run.lineweight,
                    ltscale=0.0,
                    color=256,
                )
            elif kind in elec.KINDS:
                shape = "switch" if kind == "switch_frame" else kind
                row.update(kind="block", block=elec.block_name(shape, k), define=shape)
            else:
                raise elec.unknown_kind(kind, f"{where}.kind")
        else:
            row["kind"] = "text"
        out.append(row)
    return out, taken


def _line_style(doc, item: dict, where: str) -> dict:
    """An extra line row's style: its layer's (which must exist) unless it names
    its own linetype, lineweight or colour."""
    from backends.base import parse_lineweight

    layer = item.get("layer", LAYER)
    if not isinstance(layer, str) or (layer != LAYER and layer not in doc.layers):
        raise LegendArgsError(f"{where}.layer: no layer {layer!r} in the drawing.")
    linetype = item.get("linetype", "BYLAYER")
    if not isinstance(linetype, str) or not linetype.strip():
        raise LegendArgsError(f"{where}.linetype: a linetype name, got {linetype!r}.")
    try:
        weight = parse_lineweight(item.get("lineweight", -1))
    except ValueError as exc:
        raise LegendArgsError(f"{where}.lineweight: {exc}") from None
    return {
        "layer": layer,
        "linetype": linetype,
        "lineweight": weight,
        "color": _color_arg(item.get("color", 256), f"{where}.color"),
        "ltscale": 0.0,
        "item": None,
    }


def _ordered(rows: list[dict], order) -> list[dict]:
    """``rows`` with those ``order`` names (keys, block / pattern names,
    labels; ``*`` ``?``) first, in that order; the rest keep theirs."""
    patterns = _globs(order, "order")
    for i, p in enumerate(patterns):
        if not any(_match(r, [p]) or _match({"key": r["label"]}, [p]) for r in rows):
            raise LegendArgsError(
                f"order[{i}]: {p!r} names no row; the rows are "
                f"{', '.join(dict.fromkeys(r['key'] for r in rows))}."
            )

    def rank(r):
        for i, p in enumerate(patterns):
            if _match(r, [p]) or _match({"key": r["label"]}, [p]):
                return i
        return len(patterns)

    return sorted(rows, key=rank)  # stable: the default order within a rank


def _color_arg(value, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 256:
        raise LegendArgsError(f"{where}: an ACI colour 0-256, got {value!r}.")
    return value


async def draw_legend(
    backend: AutoCADBackend,
    *,
    at,
    layout: str | None = None,
    scale: float = 50,
    lang: str = "en",
    labels: dict | None = None,
    only=None,
    exclude=None,
    layers=None,
    extra=None,
    order=None,
    columns: int | None = None,
    max_height: float | None = None,
    title: str | None = TITLE_AUTO,
    frame: bool = False,
    text_height: float = TEXT_HEIGHT,
    row_gap: float = ROW_GAP,
    color: int | None = None,
    layer: str | None = None,
    name: str = "default",
) -> dict:
    """Draw the legend with its top-left corner at ``at``; see the module docstring."""
    from engineering.annotation import text_width
    from engineering.annotation_record import replace_set, tag_annotation
    from engineering.arch.draw import ensure_arch_layers
    from engineering.sheet.frames import enter_layout, sheet_fit_warnings

    if lang not in LANGS:
        raise LegendArgsError(f"lang must be one of {', '.join(LANGS)}, got {lang!r}.")
    if labels is not None and not isinstance(labels, dict):
        raise LegendArgsError("labels: expected {item, block or pattern name: text or null}.")
    k = float(scale)
    if not math.isfinite(k) or k <= 0:
        raise LegendArgsError("scale: the plot-scale denominator must be > 0.")
    if columns is not None and (
        isinstance(columns, bool) or not isinstance(columns, int) or columns < 1
    ):
        raise LegendArgsError(f"columns: a whole number >= 1, got {columns!r}.")
    if max_height is not None and not (isinstance(max_height, (int, float)) and max_height > 0):
        raise LegendArgsError(f"max_height: paper mm > 0, got {max_height!r}.")
    if isinstance(text_height, bool) or not isinstance(text_height, (int, float)):
        raise LegendArgsError(f"text_height: paper mm, got {text_height!r}.")
    if not 1.0 <= float(text_height) <= 10.0:
        raise LegendArgsError(f"text_height: {text_height!r} mm is outside 1-10 mm.")
    if isinstance(row_gap, bool) or not isinstance(row_gap, (int, float)) or not 0 <= row_gap <= 20:
        raise LegendArgsError(f"row_gap: paper mm 0-20, got {row_gap!r}.")
    if not isinstance(frame, bool):
        raise LegendArgsError(f"frame: true or false, got {frame!r}.")
    if not isinstance(name, str) or not name.strip():
        raise LegendArgsError(f"name: a non-empty string, got {name!r}.")
    if layer is not None and (not isinstance(layer, str) or not layer.strip()):
        raise LegendArgsError(f"layer: a layer name, got {layer!r}.")
    color = _color_arg(color, "color")
    text_layer = layer or LAYER
    heading = TITLE[lang] if title == TITLE_AUTO else (_text_or_none(title, "title"))
    x0, y0 = (float(at[0]), float(at[1]))
    found = await backend._async(_read, backend, readonly=True)
    rows, skipped = legend_rows(
        found, lang=lang, scale=k, labels=labels, only=only, exclude=exclude, layers=layers
    )
    model_rows = rows
    if extra is not None and any(isinstance(e, dict) and "keys" in e for e in extra):
        model_rows, _skip = legend_rows(
            found, lang=lang, scale=k, labels=labels, only=["*"], layers=layers
        )
    extras, taken = _extra_rows(backend, extra, k, model_rows)
    taken_keys = {(r["kind"], r["key"], r["label"]) for r in model_rows if id(r) in taken}
    rows = [r for r in rows if (r["kind"], r["key"], r["label"]) not in taken_keys] + extras
    if not rows:
        raise LegendArgsError(
            "arch_legend: no row left to draw - model space uses no block, hatch or run "
            "the filters keep, and no extra row was given."
            + (
                f" Left out by default: {', '.join(skipped)} (name them in only)."
                if skipped
                else ""
            )
        )
    if order is not None:
        rows = _ordered(rows, order)
    th = float(text_height)
    title_h = TITLE_HEIGHT * th / TEXT_HEIGHT
    pitch = LINE_PITCH * th
    gap = float(row_gap)
    for row in rows:
        if "samples" not in row:
            row["samples"] = [] if row["kind"] == "text" else [row]
        row["lines"] = row["label"].split("\n")
        row["h"] = max(CELL_H, (len(row["lines"]) - 1) * pitch + th)
    u = 1.0 if layout is not None else k  # drawing units per paper mm
    header = title_h + 2 * PAD if heading else 0.0
    warnings: list[str] = []
    if columns is not None:
        per = math.ceil(len(rows) / columns)
        columns_rows = [rows[i : i + per] for i in range(0, len(rows), per)]
    elif max_height is not None:
        room = float(max_height) - header - 2 * PAD
        columns_rows, col, used = [], [], 0.0
        for row in rows:
            need = row["h"] + (gap if col else 0.0)
            if col and used + need > room + 1e-9:
                columns_rows.append(col)
                col, used, need = [], 0.0, row["h"]
            col.append(row)
            used += need
        columns_rows.append(col)
    else:
        columns_rows = [rows[i : i + ROWS_PER_COLUMN] for i in range(0, len(rows), ROWS_PER_COLUMN)]

    def sample_w(r):
        n = len(r["samples"])
        return n * CELL_W + max(n - 1, 0) * SAMPLE_GAP

    def col_h(col):
        return sum(r["h"] for r in col) + gap * (len(col) - 1)

    sample_ws = [max(sample_w(r) for r in col) for col in columns_rows]
    widths = [
        sw + 2 * PAD + max(text_width(line, th) for r in col for line in r["lines"]) + PAD
        for col, sw in zip(columns_rows, sample_ws, strict=True)
    ]
    width = sum(widths) + PAD
    if heading:
        width = max(width, text_width(heading, title_h) + 2 * PAD)
    height = header + PAD + max(col_h(c) for c in columns_rows) + PAD
    if max_height is not None and height > float(max_height) + 1e-6:
        warnings.append(
            f"max_height: {len(rows)} rows in {len(columns_rows)} column(s) need {height:g} mm, "
            f"over {float(max_height):g}; add columns, drop rows (exclude) or raise it"
        )

    def p(x, y):  # paper mm from the top-left corner -> drawing units
        return x0 + x * u, y0 - y * u

    warnings += await sheet_fit_warnings(backend, layout, (p(0, height), p(width, 0)))
    for row in rows:
        for s in row["samples"]:
            if s.get("define") and s["block"] not in {b.name for b in await backend.block_list()}:
                spec = elec.block_spec(s["define"], k)
                await backend.block_define(
                    spec["name"], spec["entities"], spec["attdefs"], 0.0, 0.0, False, False
                )
                s["extents"] = await backend._async(
                    lambda n=s["block"]: _extents(backend._require_doc(), n), readonly=True
                )
            if s["kind"] == "line" and s["linetype"].upper() not in (
                "BYLAYER",
                "BYBLOCK",
                "CONTINUOUS",
            ):
                await backend.linetype_load(s["linetype"])

    previous = await enter_layout(backend, layout, caller=TOOL)
    handles: list[str] = []
    drawn: list[dict] = []
    try:
        replaced = await replace_set(backend, layout, TOOL, name)
        await ensure_arch_layers(
            backend, (text_layer, *{s["layer"] for r in rows for s in r["samples"]})
        )

        async def keep(info):
            await tag_annotation(backend, info.handle, TOOL, name)
            handles.append(info.handle)
            return info

        if frame:
            outline = [p(0, 0), p(width, 0), p(width, height), p(0, height)]
            await keep(
                await backend.entity_create_polyline(
                    points=outline, closed=True, layer=text_layer, color=color
                )
            )
            if heading:
                await keep(
                    await backend.entity_create_line(
                        *p(0, header), *p(width, header), layer=text_layer, color=color
                    )
                )
        if heading:
            await keep(
                await backend.entity_create_text(
                    heading, *p(PAD, PAD + title_h), title_h * u, layer=text_layer, color=color
                )
            )
        left = 0.0
        for col, col_width, sw in zip(columns_rows, widths, sample_ws, strict=True):
            top = header + PAD
            for row in col:
                mid = top + row["h"] / 2.0
                cy0 = mid - CELL_H / 2.0
                for j, s in enumerate(row["samples"]):
                    cx0 = left + PAD + j * (CELL_W + SAMPLE_GAP)
                    cell = [p(cx0, cy0), p(cx0 + CELL_W, cy0), p(cx0 + CELL_W, cy0 + CELL_H)]
                    cell.append(p(cx0, cy0 + CELL_H))
                    await keep(await _sample(backend, s, cell, k, u))
                    if s["kind"] == "hatch":
                        await keep(
                            await backend.entity_create_polyline(
                                points=cell, closed=True, layer=text_layer, color=color
                            )
                        )
                # the label's lines centred on the row: cap of the first to the last baseline
                first = mid - ((len(row["lines"]) - 1) * pitch + th) / 2.0 + th
                for i, line in enumerate(row["lines"]):
                    if not line:
                        continue
                    tx, ty = p(left + PAD + sw + PAD, first + i * pitch)
                    await keep(
                        await backend.entity_create_text(
                            line, tx, ty, th * u, layer=text_layer, color=color
                        )
                    )
                drawn.append(
                    {
                        "kind": row["kind"],
                        "key": row["key"],
                        "label": row["label"],
                        "count": row["count"],
                        **(
                            {
                                "samples": [
                                    s["key"] if "key" in s else s["block"] for s in row["samples"]
                                ]
                            }
                            if len(row["samples"]) > 1
                            else {}
                        ),
                    }
                )
                top += row["h"] + gap
            left += col_width
    finally:
        if layout is not None:
            await backend.layout_set_current(previous)
    result = {
        "ok": True,
        "name": name,
        "layout": layout or "Model",
        "rows": drawn,
        "handles": handles,
        "replaced": replaced,
        "bbox": {"min": list(p(0, height)), "max": list(p(width, 0))},
        "layer": text_layer,
        "backend": backend.name,
    }
    bare = sorted(
        {r["key"] for r in rows if r["kind"] == "hatch" and r["label"] == r.get("pattern")}
    )
    if bare:
        warnings.append(
            f"labels: {', '.join(bare)} read as their pattern name; "
            "give labels={pattern: text} (a zone tool's hatch names itself)"
        )
    if skipped:
        result["left_out"] = {
            "keys": skipped,
            "why": "catalogue furniture and wall poche are left out by default; name them in only",
        }
    if warnings:
        result["warnings"] = warnings
    return result


def _text_or_none(value, where: str) -> str | None:
    if value is None or value == "":
        return None
    from engineering.arch.sheetblock import _text

    try:
        return _text(value, where)
    except ValueError as exc:
        raise LegendArgsError(str(exc)) from None


async def _sample(backend, row: dict, cell, plot_scale: float, u: float):
    """The row's sample inside ``cell`` (four corners, drawing units)."""
    (x0, y0), (x1, _), (_, y1), _ = cell
    lo_x, hi_x, lo_y, hi_y = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    if row["kind"] == "hatch":
        printed = row["scale"] * u / plot_scale  # the same density on the sheet
        return await backend.entity_create_hatch(
            "SOLID" if row["pattern"] == "SOLID" else row["pattern"],
            [list(c) for c in cell],
            scale=printed,
            angle=row["angle"],
            layer=row["layer"],
        )
    if row["kind"] == "line":
        return await backend._async(
            _line_sample, backend, row, lo_x, hi_x, (lo_y + hi_y) / 2.0, u, plot_scale
        )
    extents = row.get("extents")
    true = u / plot_scale  # the block's plotted size
    if extents is None:
        s, bx, by = true, 0.0, 0.0
    else:
        bw, bh = max(extents[2] - extents[0], 1e-9), max(extents[3] - extents[1], 1e-9)
        s = min(true, 0.9 * (hi_x - lo_x) / bw, 0.9 * (hi_y - lo_y) / bh)
        bx, by = (extents[0] + extents[2]) / 2.0, (extents[1] + extents[3]) / 2.0
    cx, cy = (lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0
    return await backend.block_insert(
        row["block"], cx - s * bx, cy - s * by, s, s, 0.0, None, row["layer"]
    )


def _line_sample(backend, row, lo_x, hi_x, y, u, plot_scale):
    """A line across the cell in the run's own layer, linetype and weight; its
    dashes as long on the sheet as on the plan."""
    from backends.ezdxf_backend import _entity_info_dxf

    doc = backend._require_doc()
    ltscale = row["ltscale"] * u / plot_scale
    if not ltscale and row["linetype"].upper() in doc.linetypes and row["kind"] == "line":
        run = elec.RUN_KINDS.get(row.get("item"))
        pattern = sum(abs(v) for v in doc.linetypes.get(row["linetype"]).pattern_tags.compile())
        if run and run.dash and pattern:
            lt = float(doc.header.get("$LTSCALE", 1.0) or 1.0)
            ltscale = run.dash * u / pattern / lt
    attribs = {
        "layer": row["layer"],
        "linetype": row["linetype"],
        "lineweight": row["lineweight"],
        "color": row["color"],
    }
    if ltscale:
        attribs["ltscale"] = ltscale
    pad = 0.1 * (hi_x - lo_x)
    line = backend._msp().add_line((lo_x + pad, y), (hi_x - pad, y), dxfattribs=attribs)
    backend._mark_dirty()
    return _entity_info_dxf(line)
