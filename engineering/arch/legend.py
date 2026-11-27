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
    if not block.upper().startswith(BLOCK_PREFIX):
        return None
    rest = block[len(BLOCK_PREFIX) :].lower()
    for name in sorted(CATALOGUE_NAMES, key=len, reverse=True):
        if rest == name or rest.startswith(f"{name}_"):
            return name
    return None


def _block_label(block: str, lang: str) -> tuple[int, str]:
    """``(order, description)``: electrical, then catalogue, then other blocks."""
    upper = block.upper()
    if upper.startswith(elec.BLOCK_PREFIX):
        for kind in sorted(elec.KINDS, key=len, reverse=True):
            if upper[len(elec.BLOCK_PREFIX) :].startswith(f"{kind.upper()}_"):
                return 1, elec.label(kind, lang)
    item = _catalogue_item(block)
    if item is not None:
        row = _BY_NAME[item]
        return 2, {"tr": row.label_tr, "pl": LABEL_PL[item]}.get(lang, row.label_en)
    return 3, block


def _read(backend) -> dict:
    """Blocks and hatch patterns model space uses (a legend's own samples excluded)."""
    from ezdxf import bbox

    from engineering.annotation_record import APP_ID

    doc = backend._require_doc()
    blocks: dict[str, dict] = {}
    hatches: dict[tuple, dict] = {}
    for e in doc.modelspace():
        if e.has_xdata(APP_ID) and TOOL in str(e.get_xdata(APP_ID)):
            continue
        kind = e.dxftype()
        if kind == "INSERT":
            name = e.dxf.name
            block = doc.blocks.get(name)
            if name.startswith("*") or block is None or block.block.is_xref:
                continue
            row = blocks.setdefault(name, {"count": 0, "layer": e.dxf.layer})
            row["count"] += 1
            if "extents" not in row:
                box = bbox.extents(x for x in block if x.dxftype() not in ("ATTDEF",))
                row["extents"] = (
                    (box.extmin.x, box.extmin.y, box.extmax.x, box.extmax.y)
                    if box.has_data
                    else None
                )
        elif kind == "HATCH":
            solid = bool(e.dxf.solid_fill)
            key = (
                "SOLID" if solid else e.dxf.pattern_name.upper(),
                0.0 if solid else round(float(e.dxf.pattern_angle), 6),
                1.0 if solid else round(float(e.dxf.pattern_scale), 6),
            )
            row = hatches.setdefault(key, {"count": 0, "layer": e.dxf.layer})
            row["count"] += 1
    return {"blocks": blocks, "hatches": hatches}


def legend_rows(found: dict, *, lang: str, scale: float, labels: dict | None) -> list[dict]:
    """The rows in order: materials and other hatches, then blocks, each with
    its ``key`` (pattern or block name), ``label`` and source; ``labels``
    overrides a label or drops a row (``None``) by key, case-insensitively."""
    overrides = {str(k).upper(): v for k, v in (labels or {}).items()}
    unknown = sorted(
        k
        for k in overrides
        if k not in {b.upper() for b in found["blocks"]}
        and k not in {h[0] for h in found["hatches"]}
    )
    if unknown:
        raise ValueError(
            f"labels: no block or hatch pattern {unknown} in model space; the keys are "
            "the block names and pattern names the drawing uses."
        )
    rows: list[dict] = []
    for (pattern, angle, pscale), row in sorted(found["hatches"].items()):
        material = _material(pattern, angle, pscale, scale)
        text = MATERIAL_LABELS[material][lang] if material else pattern
        rows.append(
            {
                "kind": "hatch",
                "key": pattern,
                "order": 0 if material else 1,
                "label": text,
                "pattern": pattern,
                "angle": angle,
                "scale": pscale,
                **row,
            }
        )
    for name, row in sorted(found["blocks"].items()):
        order, text = _block_label(name, lang)
        rows.append({"kind": "block", "key": name, "order": 10 + order, "label": text, **row})
    out = []
    for row in sorted(rows, key=lambda r: (r["order"], r["label"].casefold())):
        if row["key"].upper() in overrides:
            if overrides[row["key"].upper()] is None:
                continue
            row["label"] = " ".join(str(overrides[row["key"].upper()]).split())
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
) -> dict:
    """Draw the legend with its top-left corner at ``at``; see the module docstring."""
    from engineering.annotation import text_width
    from engineering.annotation_record import tag_annotation
    from engineering.arch.draw import ensure_arch_layers
    from engineering.sheet.frames import enter_layout

    if lang not in LANGS:
        raise ValueError(f"lang must be one of {', '.join(LANGS)}, got {lang!r}.")
    if labels is not None and not isinstance(labels, dict):
        raise ValueError("labels: expected {block or pattern name: text or null}.")
    k = float(scale)
    if not math.isfinite(k) or k <= 0:
        raise ValueError("scale: the plot-scale denominator must be > 0.")
    x0, y0 = (float(at[0]), float(at[1]))
    found = await backend._async(_read, backend, readonly=True)
    rows = legend_rows(found, lang=lang, scale=k, labels=labels)
    if not rows:
        raise ValueError(
            "arch_legend: model space uses no block and no hatch pattern; nothing to explain."
        )
    u = 1.0 if layout is not None else k  # drawing units per paper mm
    columns = [rows[i : i + ROWS_PER_COLUMN] for i in range(0, len(rows), ROWS_PER_COLUMN)]
    widths = [
        CELL_W + 2 * PAD + max(text_width(r["label"], TEXT_HEIGHT) for r in col) + PAD
        for col in columns
    ]
    title = TITLE[lang]
    width = max(sum(widths) + PAD, text_width(title, TITLE_HEIGHT) + 2 * PAD)
    height = TITLE_HEIGHT + 2 * PAD + ROW_PITCH * len(columns[0]) + PAD

    previous = await enter_layout(backend, layout, caller=TOOL)
    handles: list[str] = []
    drawn: list[dict] = []
    try:
        await ensure_arch_layers(backend, (LAYER, *{r["layer"] for r in rows}))

        async def keep(info):
            await tag_annotation(backend, info.handle, TOOL)
            handles.append(info.handle)
            return info

        def p(x, y):  # paper mm from the top-left corner -> drawing units
            return x0 + x * u, y0 - y * u

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
        for col, col_width in zip(columns, widths, strict=True):
            for i, row in enumerate(col):
                top = rule + PAD + i * ROW_PITCH
                cx0, cy0 = left + PAD, top
                cell = [p(cx0, cy0), p(cx0 + CELL_W, cy0), p(cx0 + CELL_W, cy0 + CELL_H)]
                cell.append(p(cx0, cy0 + CELL_H))
                sample = await _sample(backend, row, cell, k, u)
                await keep(sample)
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
    return {
        "ok": True,
        "layout": layout or "Model",
        "rows": drawn,
        "handles": handles,
        "bbox": {"min": list(p(0, height)), "max": list(p(width, 0))},
        "layer": LAYER,
        "backend": backend.name,
    }


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
        row["key"], cx - s * bx, cy - s * by, s, s, 0.0, None, row["layer"]
    )
