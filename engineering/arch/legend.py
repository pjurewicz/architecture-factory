"""A legend built from what the drawing uses: a sample and a description per row.

:func:`draw_legend` reads model space once - every block an INSERT uses and
every hatch pattern (pattern, angle and scale) - and draws one row each: a
sample cell and its description, materials first, then symbols, under a
title, framed. A sample is the real thing: the same block, scaled to fit its
cell but never above its plotted size, and the same hatch pattern at the same
printed density.

Descriptions come from the server's own tables, in ``lang``: an
``arch_electrical`` block names its kind, a catalogue block its item, a wall
hatch its material (``WALL_HATCH_TABLE`` by pattern, angle and density).
Anything else is named by its block or pattern name, and ``labels`` overrides
or (``null``) drops any row by that key.

On a paper layout (``layout``) every size is paper millimetres; in model
space it is paper millimetres times ``scale``, as the plan's other lettering.
The frame, title, descriptions and sample outlines go on ``A-SYMB-T-N``; a
sample keeps its source's layer so it prints as on the plan. Every entity
carries an ``ACADMCP_ANNOT`` record (tool ``arch_legend``), and a second
legend never lists the first one's samples.
"""

from __future__ import annotations

import fnmatch
import math
from typing import TYPE_CHECKING

from engineering.arch import electrical as elec
from engineering.arch.catalogue import _BY_NAME, BLOCK_PREFIX, CATALOGUE_NAMES, LABEL_PL
from engineering.arch.lang import LANGS
from engineering.arch.layers import ARCH_ROLE_LAYER
from engineering.arch.materials import TABLE_PLOT_SCALE, WALL_HATCH_TABLE

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

TOOL = "arch_legend"
LAYER = ARCH_ROLE_LAYER["symbol"]

#: Paper millimetres: this module's drafting choices, not a standard.
TITLE_HEIGHT = 3.5
TEXT_HEIGHT = 2.5
ROW_PITCH = 8.0
CELL_W, CELL_H = 15.0, 6.0
PAD = 2.0
ROWS_PER_COLUMN = 20

TITLE = {"en": "LEGEND", "pl": "LEGENDA", "tr": "LEJANT"}
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
    if not block.upper().startswith(BLOCK_PREFIX) or block.upper().startswith(elec.BLOCK_PREFIX):
        return None
    rest = block[len(BLOCK_PREFIX) :].lower()
    for name in sorted(CATALOGUE_NAMES, key=len, reverse=True):
        if rest == name or rest.startswith(f"{name}_"):
            return name
    return None


def _elec_kind(block: str) -> str | None:
    upper = block.upper()
    if not upper.startswith(elec.BLOCK_PREFIX):
        return None
    for kind in sorted(elec.KINDS, key=len, reverse=True):
        if upper[len(elec.BLOCK_PREFIX) :].startswith(f"{kind.upper()}_"):
            return kind
    return None


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
    from engineering.annotation_record import APP_ID

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
            key = f"{pattern}@{angle:g}@{pscale:g}"
            row = groups.setdefault(
                key,
                {
                    "kind": "hatch",
                    "key": pattern,
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
        raise ValueError(f"{where}: expected a list of names or patterns (* and ?).")
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
        raise ValueError(
            f"labels: no item, block or hatch pattern {unknown} in model space; the keys "
            "are the items, block names and pattern names the drawing uses."
        )
    rows, skipped = [], []
    for row in groups:
        row = dict(row)
        if row["kind"] == "hatch":
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
            row["label"] = " ".join(str(overrides[hit]).split())
        rows.append(row)
    rows.sort(key=lambda r: (r["order"], r["label"].casefold()))
    return rows, sorted(set(skipped))


_EXTRA_KEYS = {"label", "block", "pattern", "kind"}


def _extra_rows(backend, extra, k: float) -> list[dict]:
    """``extra`` rows: a sample from a block, a hatch pattern or an
    ``arch_electrical`` kind, or none; checked by path. Our own symbol block
    is defined when the drawing lacks it."""
    if extra is None:
        return []
    if not isinstance(extra, (list, tuple)):
        raise ValueError("extra: expected [{label, block? | pattern? | kind?}].")
    doc = backend._require_doc()
    out = []
    for i, item in enumerate(extra):
        where = f"extra[{i}]"
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise ValueError(f"{where}: expected {{label, block? | pattern? | kind?}}.")
        bad = sorted(set(item) - _EXTRA_KEYS)
        if bad or len(set(item) & {"block", "pattern", "kind"}) > 1:
            raise ValueError(f"{where}: give label and at most one of block, pattern, kind.")
        row = {
            "key": item["label"],
            "label": " ".join(item["label"].split()),
            "count": 0,
            "order": 20 + i,
            "layer": LAYER,
            "names": [],
            "extra": True,
        }
        if "block" in item:
            name = str(item["block"])
            if name not in doc.blocks:
                raise ValueError(f"{where}.block: no block {name!r} in the drawing.")
            row.update(kind="block", block=name, extents=_extents(doc, name))
        elif "pattern" in item:
            row.update(kind="hatch", pattern=str(item["pattern"]).upper(), angle=0.0, scale=k)
        elif "kind" in item:
            kind = item["kind"]
            if kind in elec.RUN_KINDS:
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
                raise ValueError(f"{where}.kind: {kind!r} is not an arch_electrical kind.")
        else:
            row["kind"] = "text"
        out.append(row)
    return out


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
    columns: int | None = None,
    max_height: float | None = None,
) -> dict:
    """Draw the legend with its top-left corner at ``at``; see the module docstring."""
    from engineering.annotation import text_width
    from engineering.annotation_record import tag_annotation
    from engineering.arch.draw import ensure_arch_layers
    from engineering.sheet.frames import enter_layout, sheet_fit_warnings

    if lang not in LANGS:
        raise ValueError(f"lang must be one of {', '.join(LANGS)}, got {lang!r}.")
    if labels is not None and not isinstance(labels, dict):
        raise ValueError("labels: expected {item, block or pattern name: text or null}.")
    k = float(scale)
    if not math.isfinite(k) or k <= 0:
        raise ValueError("scale: the plot-scale denominator must be > 0.")
    if columns is not None and (
        isinstance(columns, bool) or not isinstance(columns, int) or columns < 1
    ):
        raise ValueError(f"columns: a whole number >= 1, got {columns!r}.")
    if max_height is not None and not (isinstance(max_height, (int, float)) and max_height > 0):
        raise ValueError(f"max_height: paper mm > 0, got {max_height!r}.")
    x0, y0 = (float(at[0]), float(at[1]))
    found = await backend._async(_read, backend, readonly=True)
    rows, skipped = legend_rows(
        found, lang=lang, scale=k, labels=labels, only=only, exclude=exclude, layers=layers
    )
    rows += _extra_rows(backend, extra, k)
    if not rows:
        raise ValueError(
            "arch_legend: no row left to draw - model space uses no block, hatch or run "
            "the filters keep, and no extra row was given."
            + (
                f" Left out by default: {', '.join(skipped)} (name them in only)."
                if skipped
                else ""
            )
        )
    u = 1.0 if layout is not None else k  # drawing units per paper mm
    header = TITLE_HEIGHT + 3 * PAD
    warnings: list[str] = []
    if columns is not None:
        per = math.ceil(len(rows) / columns)
    elif max_height is not None:
        per = max(1, int((float(max_height) - header - PAD + 1e-9) // ROW_PITCH))
    else:
        per = ROWS_PER_COLUMN
    columns_rows = [rows[i : i + per] for i in range(0, len(rows), per)]
    widths = [
        CELL_W + 2 * PAD + max(text_width(r["label"], TEXT_HEIGHT) for r in col) + PAD
        for col in columns_rows
    ]
    title = TITLE[lang]
    width = max(sum(widths) + PAD, text_width(title, TITLE_HEIGHT) + 2 * PAD)
    height = TITLE_HEIGHT + 2 * PAD + ROW_PITCH * len(columns_rows[0]) + PAD
    if max_height is not None and height > float(max_height) + 1e-6:
        warnings.append(
            f"max_height: {len(rows)} rows in {len(columns_rows)} column(s) need {height:g} mm, "
            f"over {float(max_height):g}; add columns, drop rows (exclude) or raise it"
        )

    def p(x, y):  # paper mm from the top-left corner -> drawing units
        return x0 + x * u, y0 - y * u

    warnings += await sheet_fit_warnings(backend, layout, (p(0, height), p(width, 0)))
    for row in rows:
        if row.get("define") and row["block"] not in {b.name for b in await backend.block_list()}:
            spec = elec.block_spec(row["define"], k)
            await backend.block_define(
                spec["name"], spec["entities"], spec["attdefs"], 0.0, 0.0, False, False
            )
            row["extents"] = await backend._async(
                lambda n=row["block"]: _extents(backend._require_doc(), n), readonly=True
            )
        if row["kind"] == "line" and row["linetype"].upper() not in (
            "BYLAYER",
            "BYBLOCK",
            "CONTINUOUS",
        ):
            await backend.linetype_load(row["linetype"])

    previous = await enter_layout(backend, layout, caller=TOOL)
    handles: list[str] = []
    drawn: list[dict] = []
    try:
        await ensure_arch_layers(backend, (LAYER, *{r["layer"] for r in rows}))

        async def keep(info):
            await tag_annotation(backend, info.handle, TOOL)
            handles.append(info.handle)
            return info

        frame = [p(0, 0), p(width, 0), p(width, height), p(0, height)]
        await keep(await backend.entity_create_polyline(points=frame, closed=True, layer=LAYER))
        rule = TITLE_HEIGHT + 2 * PAD
        await keep(await backend.entity_create_line(*p(0, rule), *p(width, rule), layer=LAYER))
        await keep(
            await backend.entity_create_text(
                title, *p(PAD, PAD + TITLE_HEIGHT), TITLE_HEIGHT * u, layer=LAYER
            )
        )
        left = 0.0
        for col, col_width in zip(columns_rows, widths, strict=True):
            for i, row in enumerate(col):
                top = rule + PAD + i * ROW_PITCH
                cx0, cy0 = left + PAD, top
                cell = [p(cx0, cy0), p(cx0 + CELL_W, cy0), p(cx0 + CELL_W, cy0 + CELL_H)]
                cell.append(p(cx0, cy0 + CELL_H))
                if row["kind"] != "text":
                    await keep(await _sample(backend, row, cell, k, u))
                if row["kind"] == "hatch":
                    await keep(
                        await backend.entity_create_polyline(points=cell, closed=True, layer=LAYER)
                    )
                tx, ty = p(cx0 + CELL_W + PAD, cy0 + CELL_H / 2.0 + TEXT_HEIGHT / 2.0)
                await keep(
                    await backend.entity_create_text(
                        row["label"], tx, ty, TEXT_HEIGHT * u, layer=LAYER
                    )
                )
                drawn.append(
                    {
                        "kind": row["kind"],
                        "key": row["key"],
                        "label": row["label"],
                        "count": row["count"],
                    }
                )
            left += col_width
    finally:
        if layout is not None:
            await backend.layout_set_current(previous)
    result = {
        "ok": True,
        "layout": layout or "Model",
        "rows": drawn,
        "handles": handles,
        "bbox": {"min": list(p(0, height)), "max": list(p(width, 0))},
        "layer": LAYER,
        "backend": backend.name,
    }
    if skipped:
        result["left_out"] = {
            "keys": skipped,
            "why": "catalogue furniture and wall poche are left out by default; name them in only",
        }
    if warnings:
        result["warnings"] = warnings
    return result


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
