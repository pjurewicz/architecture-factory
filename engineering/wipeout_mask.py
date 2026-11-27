"""The ``wipeout_mask`` critique focus: an entity a block's WIPEOUT hides (round 4).

A catalogue block (a built-in wardrobe, a bathtub) carries a WIPEOUT so the
plan's lines under its footprint do not show through. Inserted *after* LED
strips and symbols, its mask hid them: nothing reported it, ``drawing_critique``
stayed silent and only a 300 dpi PDF showed the gap. This focus reads, per
space (model space, every paper layout), every INSERT whose block definition
holds a WIPEOUT, takes the mask polygon in WCS (the block's own transform),
and reports every visible entity of that space drawn *before* the INSERT in
its draw order (``entities_in_redraw_order``, so DRAWORDER counts) that the
polygon covers: its bounding box's centre, or a corner of it, lies inside.
Code ``hidden_by_wipeout``, one issue per masking INSERT, a warning, never
repaired - ``draw_order(where=...)`` or ``arch_catalogue_insert``'s default
``send_to_back`` is the fix.

Not tested: a WIPEOUT drawn directly in a space (not in a block) and a mask
inside a nested block; an entity on a frozen or off layer; the INSERT's own
contents; entities of the same block (a second INSERT of one wardrobe beside
another is tested like any entity). Coverage: one ``checked`` row per space
with the number of masks tested; ``domain_absent`` when no block masks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .plan_spec import Issue
from .qa_coverage import DOCUMENT, domain_row, ledger_of

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

FOCUS = "wipeout_mask"
CODE = "hidden_by_wipeout"
MAX_ITEMS = 20


def _inside(pt, ring) -> bool:
    x, y = pt
    inside = False
    n = len(ring)
    for i in range(n):
        (x1, y1), (x2, y2) = ring[i], ring[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _masks(doc) -> dict[str, list[list[tuple[float, float]]]]:
    """Block name -> its WIPEOUT polygons in block coordinates."""
    out: dict[str, list] = {}
    for block in doc.blocks:
        if block.name.startswith("*"):
            continue
        rings = []
        for e in block:
            if e.dxftype() == "WIPEOUT":
                ring = [(v.x, v.y) for v in e.boundary_path_wcs()]
                if len(ring) >= 3:
                    rings.append(ring)
        if rings:
            out[block.name.upper()] = rings
    return out


def _scan(doc) -> dict[str, dict]:
    """``{space: {n, hits: [(insert handle, block, [(handle, type)])]}}``."""
    from ezdxf import bbox

    masks = _masks(doc)
    if not masks:
        return {}
    hidden_layers = {
        lyr.dxf.name.upper() for lyr in doc.layers if lyr.is_frozen() or not lyr.is_on()
    }
    found: dict[str, dict] = {}
    for layout in doc.layouts:
        order = list(layout.entities_in_redraw_order())
        row = {"n": 0, "hits": []}
        for i, ref in enumerate(order):
            if ref.dxftype() != "INSERT" or ref.dxf.name.upper() not in masks:
                continue
            row["n"] += 1
            m = ref.matrix44()
            rings = [
                [tuple(m.transform((x, y, 0)))[:2] for x, y in r]
                for r in masks[ref.dxf.name.upper()]
            ]
            hit = []
            for e in order[:i]:
                if e.dxftype() in ("VIEWPORT",) or e.dxf.get("invisible", 0):
                    continue
                if str(e.dxf.get("layer", "0")).upper() in hidden_layers:
                    continue
                try:
                    box = bbox.extents([e], fast=True)
                except Exception:  # noqa: BLE001 - an unreadable box is not a finding
                    continue
                if not box.has_data:
                    continue
                (x0, y0), (x1, y1) = box.extmin.vec2, box.extmax.vec2
                probes = [((x0 + x1) / 2, (y0 + y1) / 2), (x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                if any(_inside(p, r) for r in rings for p in probes):
                    hit.append((e.dxf.handle, e.dxftype()))
            if hit:
                row["hits"].append((ref.dxf.handle, ref.dxf.name, hit))
        found[layout.name] = row
    return found


async def check_wipeout_mask(backend: AutoCADBackend, shared: dict) -> list[Issue]:
    from .qa_scan import _async_readonly

    found = await _async_readonly(backend, lambda: _scan(backend._require_doc()))
    if not found:
        domain_row(shared, FOCUS, present=False, n=0)
        return []
    ledger = ledger_of(shared)
    issues: list[Issue] = []
    for space, row in found.items():
        ledger.checked(FOCUS, space or DOCUMENT, row["n"])
        for handle, block, hit in row["hits"]:
            shown = ", ".join(f"{t} {h}" for h, t in hit[:3])
            more = f" and {len(hit) - 3} more" if len(hit) > 3 else ""
            issues.append(
                Issue(
                    severity="warning",
                    focus=FOCUS,  # type: ignore[arg-type]
                    message=(
                        f"INSERT {handle} ({block}) on '{space}' masks {len(hit)} "
                        f"entit{'y' if len(hit) == 1 else 'ies'} drawn before it: {shown}{more}. "
                        f"Send it back: draw_order(handles=['{handle}']) or "
                        "draw_order(where={block: ...})."
                    ),
                    handles=[handle, *(h for h, _t in hit[:MAX_ITEMS])],
                    detail={
                        "code": CODE,
                        "layout": space,
                        "insert": handle,
                        "block": block,
                        "count": len(hit),
                        "hidden": [h for h, _t in hit[:MAX_ITEMS]],
                        "repairable": False,
                    },
                )
            )
    return issues


check_wipeout_mask.needs_shared = True  # type: ignore[attr-defined]
