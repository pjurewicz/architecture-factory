"""ISO 7200 title blocks for every ISO 5457 sheet from A4 to A0.

SOURCE
------
Width            ISO 7200:2004 -- the title block is 180 mm wide and sits in the
                 lower right-hand corner of the drawing space, against the
                 frame. 180 mm is exactly the drawing-space width of a portrait
                 A4, which is why one block fits every size.
Data fields      ISO 7200:2004 -- the mandatory fields are the legal owner
                 (`company`), the identification number (`drawing_no`), the
                 date of issue (`date`), the segment/sheet number (`sheet`) and
                 the title (`title`). The rest carried here are optional
                 fields: creator (`drawn_by`), approval person (`checked_by`),
                 version (`revision`), and the supplementary item data this
                 shop draws (`part_no`, `material`, `scale`, `units`). ISO
                 7200's full list of optional fields is NOT transcribed -- only
                 the fields this block has a cell for.
Projection angle The truncated-cone symbol of ISO 5456-2 / ISO 128-30. ISO's
                 own figure is paywalled and was NOT read here; the arrangement
                 is transcribed from reproductions of it, measured, and named
                 in `projection_symbol_prims`.

The row heights (20 / 15 / 15 / 10 mm) and the cell splits are this
repository's layout, not ISO 7200 values: the standard fixes the *data*, not
the ruling. They are exactly the ones `titleblock_apply_iso_a3` has drawn since
v1.0, so the A3 sheet this server makes has not moved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from engineering.mech.primitives import Circle, Line, Poly, Prim, Pt, Text
from engineering.sheet.frames import (
    draw_sheet_prims,
    enter_layout,
    frame_metrics,
    frame_prims,
)

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

__all__ = [
    "LABEL_HEIGHT",
    "LABEL_PAD_X",
    "LABEL_PAD_Y",
    "PROJECTION_ANGLES",
    "ROW1_H",
    "ROW2_H",
    "ROW3_H",
    "ROW4_H",
    "ROW4_VALUE_PAD_Y",
    "TB_HEIGHT",
    "TB_WIDTH",
    "TITLEBLOCK_FIELDS",
    "TITLEBLOCK_RULE_COUNT",
    "TITLE_HEIGHT",
    "TITLE_TEXT_INDEX",
    "VALUE_FIELDS",
    "VALUE_HEIGHT",
    "VALUE_PAD_X",
    "VALUE_PAD_Y",
    "TitleBlockMetadata",
    "apply_titleblock",
    "projection_symbol_prims",
    "titleblock_origin",
    "titleblock_prims",
    "value_text_indices",
]

TB_WIDTH = 180.0
TB_HEIGHT = 60.0

ROW1_H = 20.0  # title
ROW2_H = 15.0  # drawing_no | revision | sheet
ROW3_H = 15.0  # part_no | material | scale
ROW4_H = 10.0  # drawn_by | checked_by | date | company

LABEL_HEIGHT = 2.0
VALUE_HEIGHT = 3.5
TITLE_HEIGHT = 5.0
LABEL_PAD_X = 1.5
LABEL_PAD_Y = 1.5
VALUE_PAD_X = 3.0
VALUE_PAD_Y = 5.0
ROW4_VALUE_PAD_Y = 1.5

#: Text heights of advance per character assumed when fitting a text into its
#: cell. Deliberately wider than the 0.6 the centring rule uses: measured on
#: ezdxf's rendering of the `txt` style, real titles run 0.72-0.90 (capitals
#: with Polish diacritics at the top), so 0.6 would still let them overflow.
FIT_ADVANCE = 0.9
#: The ISO 3098-0 nominal heights a title-block text may be lowered to, largest
#: first. 1.8 mm is the smallest the series has; below it the block refuses.
ISO_3098_HEIGHTS = (5.0, 3.5, 2.5, 1.8)

PROJECTION_ANGLES = ("first", "third")
PROJECTION_CONE_LENGTH = 8.0
PROJECTION_R = 4.0
PROJECTION_SMALL_R = 2.0
PROJECTION_GAP = 4.0

#: Field -> the label printed in its cell, in the order the cells are drawn.
VALUE_FIELDS: tuple[tuple[str, str], ...] = (
    ("drawing_no", "DWG NO"),
    ("revision", "REV"),
    ("sheet", "SHEET"),
    ("part_no", "PART NO"),
    ("material", "MATERIAL"),
    ("scale", "SCALE"),
    ("drawn_by", "DRAWN"),
    ("checked_by", "CHECKED"),
    ("date", "DATE"),
    ("company", "COMPANY"),
)

#: Every ISO 7200 data field this block carries. `units` has no cell of its own
#: (the sheet is millimetres and the drawing says so), but it round-trips in the
#: payload so a reader is never told the drawing is unitless.
TITLEBLOCK_FIELDS: tuple[str, ...] = (
    "title",
    "drawing_no",
    "part_no",
    "material",
    "scale",
    "units",
    "drawn_by",
    "checked_by",
    "date",
    "sheet",
    "revision",
    "company",
)

#: The block frame plus the ten rules, at the head of the primitive tuple.
TITLEBLOCK_RULE_COUNT = 11
TITLE_TEXT_INDEX = 11


@dataclass
class TitleBlockMetadata:
    """ISO 7200 data fields. Strings, written verbatim -- no transformation."""

    title: str
    drawing_no: str
    part_no: str = ""
    material: str = ""
    scale: str = "1:1"
    units: str = "mm"
    drawn_by: str = ""
    checked_by: str = ""
    date: str = ""
    sheet: str = "1/1"
    revision: str = "A"
    company: str = "Anka-Makine"


def titleblock_origin(size: str, *, orientation: str = "landscape") -> Pt:
    """Lower-left corner of the title block, sheet-local."""
    metrics = frame_metrics(size, orientation=orientation)
    x0, y0, x1, _y1 = metrics["frame"]
    if x1 - x0 < TB_WIDTH - 1e-9:
        raise ValueError(
            f"{metrics['size']} {orientation}: the drawing area is {x1 - x0:.1f} mm wide, "
            f"narrower than the ISO 7200 title block ({TB_WIDTH:.0f} mm)"
        )
    return (x1 - TB_WIDTH, y0)


def projection_symbol_prims(at: Pt, angle: str = "first") -> tuple[Prim, ...]:
    """The ISO 5456-2 / ISO 128-30 projection-angle symbol, centred on `at`.

    ARRANGEMENT -- transcribed, not derived. The symbol is a truncated cone
    drawn in two views: the cone in elevation (a trapezoid) and the end view
    (two concentric circles). Which side the end view sits on is what makes the
    symbol mean *first* or *third* angle, and it is a convention that has to be
    read off the standard's figure: the projection rule alone cannot pick it,
    because the same picture is a legal drawing of the mirrored object seen
    from the other side.

    ISO 5456-2's own figure is behind ISO's paywall and was not read. Two
    independent reproductions of it were downloaded and measured instead, and
    they agree on every landmark used here:

    * FreeCAD's ISO 5457 sheet template, first-angle symbol
      (`src/Mod/TechDraw/Templates/ISO/A3_Landscape_ISO5457_advanced.svg`,
      ids `first_angle_trapezoid` / `first_angle_*_circle`): the trapezoid is
      `m 389,222 -10,2.5 v 5 l 10,2.5 z` -- a 5-unit side at x = 379 and a
      10-unit side at x = 389 -- and the circles (r 2.5 and r 5) are at
      cx = 396. Short side LEFT; end view to the RIGHT of the cone.
    * Wikimedia Commons `Convention placement vues dessin technique.svg`, which
      draws both symbols side by side, labelled FR and US. FR (first angle):
      trapezoid 35.856 units tall at x = 13.343 and 66.135 at x = 81.869,
      circles at x = 148.005. US (third angle): trapezoid 35.856 at
      x = 380.275 and 66.135 at x = 448.801, circles at x = 314.140.

    So, in both symbols alike, the trapezoid's SHORT side is on the left; only
    the end view changes side:

        first angle   cone, then the circles to its RIGHT
        third angle   the circles, then the cone to their RIGHT

    which is the same thing as the rule the standard is usually quoted by --
    the short side points AWAY from the circles in first angle and TOWARDS
    them in third. `tests/test_sheet_titleblock.py` pins the ordering against
    the measured figures above, so this cannot be "fixed" back by eye.

    The proportions (large circle diameter 8, small 4, cone 8 mm long) are this
    module's; ISO 5456-2's proportion table is not transcribed. They keep the
    1:2 small-to-large ratio both reference figures' circles have.
    """
    if angle not in PROJECTION_ANGLES:
        raise ValueError(f"projection angle must be one of {PROJECTION_ANGLES}, got {angle!r}")
    cx, cy = float(at[0]), float(at[1])
    half = (PROJECTION_CONE_LENGTH + PROJECTION_GAP + 2 * PROJECTION_R) / 2.0
    # The end view sits right of the cone in first angle, left of it in third.
    sign = 1.0 if angle == "first" else -1.0
    circles_x = (half - PROJECTION_R) * sign
    # The trapezoid itself is the same picture in both symbols: short side left.
    cone_left = -half if angle == "first" else half - PROJECTION_CONE_LENGTH
    cone_right = cone_left + PROJECTION_CONE_LENGTH
    cone = Poly(
        points=(
            (cx + cone_left, cy - PROJECTION_SMALL_R),
            (cx + cone_right, cy - PROJECTION_R),
            (cx + cone_right, cy + PROJECTION_R),
            (cx + cone_left, cy + PROJECTION_SMALL_R),
        ),
        closed=True,
    )
    axis = Line(p1=(cx - half - 1.0, cy), p2=(cx + half + 1.0, cy), role="center")
    return (
        cone,
        Circle(center=(cx + circles_x, cy), radius=PROJECTION_R),
        Circle(center=(cx + circles_x, cy), radius=PROJECTION_SMALL_R),
        axis,
    )


def value_text_indices() -> dict[str, int]:
    """Field -> its index in the tuple `titleblock_prims` returns.

    The cells are drawn label-then-value, starting right after the title, so
    the value of the i-th field is at TITLE_TEXT_INDEX + 2i + 2.
    """
    return {field: TITLE_TEXT_INDEX + 2 * i + 2 for i, (field, _label) in enumerate(VALUE_FIELDS)}


def titleblock_prims(
    size: str,
    fields,
    *,
    orientation: str = "landscape",
    projection: str | None = "first",
) -> tuple[Prim, ...]:
    """The title block as primitive descriptions, sheet-local.

    Order is pinned and read back by `value_text_indices`: the block frame, the
    ten rules, the title, then (label, value) per cell in `VALUE_FIELDS` order,
    then the four projection-symbol primitives when `projection` is not None.
    """
    meta = fields if isinstance(fields, TitleBlockMetadata) else TitleBlockMetadata(**fields)
    if projection is not None and projection not in PROJECTION_ANGLES:
        raise ValueError(
            f"projection must be one of {PROJECTION_ANGLES} or None, got {projection!r}"
        )
    tb_x0, tb_y0 = titleblock_origin(size, orientation=orientation)
    tb_x1, tb_y1 = tb_x0 + TB_WIDTH, tb_y0 + TB_HEIGHT
    row4_top = tb_y0 + ROW4_H
    row3_top = row4_top + ROW3_H
    row2_top = row3_top + ROW2_H
    r2_split1 = tb_x0 + TB_WIDTH * 0.5
    r2_split2 = tb_x0 + TB_WIDTH * 0.75
    r3_split1 = tb_x0 + TB_WIDTH / 3.0
    r3_split2 = tb_x0 + TB_WIDTH * 2.0 / 3.0
    r4_split1 = tb_x0 + TB_WIDTH * 0.25
    r4_split2 = tb_x0 + TB_WIDTH * 0.50
    r4_split3 = tb_x0 + TB_WIDTH * 0.75

    prims: list[Prim] = [
        Poly(
            points=((tb_x0, tb_y0), (tb_x1, tb_y0), (tb_x1, tb_y1), (tb_x0, tb_y1)),
            closed=True,
        ),
        Line(p1=(tb_x0, row4_top), p2=(tb_x1, row4_top)),
        Line(p1=(tb_x0, row3_top), p2=(tb_x1, row3_top)),
        Line(p1=(tb_x0, row2_top), p2=(tb_x1, row2_top)),
        Line(p1=(r2_split1, row3_top), p2=(r2_split1, row2_top)),
        Line(p1=(r2_split2, row3_top), p2=(r2_split2, row2_top)),
        Line(p1=(r3_split1, row4_top), p2=(r3_split1, row3_top)),
        Line(p1=(r3_split2, row4_top), p2=(r3_split2, row3_top)),
        Line(p1=(r4_split1, tb_y0), p2=(r4_split1, row4_top)),
        Line(p1=(r4_split2, tb_y0), p2=(r4_split2, row4_top)),
        Line(p1=(r4_split3, tb_y0), p2=(r4_split3, row4_top)),
    ]

    # The title shares row 1 with the projection symbol, so it is fitted into
    # and kept inside the part of the row the symbol leaves free.
    symbol_x = tb_x0 + 15.0
    title_left = tb_x0 + VALUE_PAD_X
    if projection is not None:
        symbol_half = (PROJECTION_CONE_LENGTH + PROJECTION_GAP + 2 * PROJECTION_R) / 2.0
        title_left = symbol_x + symbol_half + 1.0 + VALUE_PAD_X
    title_right = tb_x1 - VALUE_PAD_X
    title_h = _series_height({"title": (meta.title, title_right - title_left)}, TITLE_HEIGHT)
    centred_x = tb_x0 + TB_WIDTH * 0.5 - (len(meta.title) * title_h * 0.3)
    title_x = max(title_left, min(centred_x, title_right - FIT_ADVANCE * title_h * len(meta.title)))
    title_y = row2_top + (ROW1_H - title_h) / 2.0
    prims.append(Text(at=(title_x, title_y), text=meta.title, height=title_h))

    # field -> (cell left edge, cell right edge, the row's top rule, the value's baseline)
    cells: dict[str, tuple[float, float, float, float]] = {
        "drawing_no": (tb_x0, r2_split1, row2_top, row3_top + VALUE_PAD_Y),
        "revision": (r2_split1, r2_split2, row2_top, row3_top + VALUE_PAD_Y),
        "sheet": (r2_split2, tb_x1, row2_top, row3_top + VALUE_PAD_Y),
        "part_no": (tb_x0, r3_split1, row3_top, row4_top + VALUE_PAD_Y),
        "material": (r3_split1, r3_split2, row3_top, row4_top + VALUE_PAD_Y),
        "scale": (r3_split2, tb_x1, row3_top, row4_top + VALUE_PAD_Y),
        "drawn_by": (tb_x0, r4_split1, row4_top, tb_y0 + ROW4_VALUE_PAD_Y),
        "checked_by": (r4_split1, r4_split2, row4_top, tb_y0 + ROW4_VALUE_PAD_Y),
        "date": (r4_split2, r4_split3, row4_top, tb_y0 + ROW4_VALUE_PAD_Y),
        "company": (r4_split3, tb_x1, row4_top, tb_y0 + ROW4_VALUE_PAD_Y),
    }
    # Every value shares one height: the largest at which all of them fit.
    value_h = _series_height(
        {
            f: (getattr(meta, f), right - left - 2 * VALUE_PAD_X)
            for f, (left, right, _, _) in cells.items()
        },
        VALUE_HEIGHT,
    )
    for field, label in VALUE_FIELDS:
        left, _right, row_top, value_y = cells[field]
        value = getattr(meta, field)
        prims.append(
            Text(
                at=(left + LABEL_PAD_X, row_top - LABEL_HEIGHT - LABEL_PAD_Y),
                text=label,
                height=LABEL_HEIGHT,
            )
        )
        prims.append(Text(at=(left + VALUE_PAD_X, value_y), text=value, height=value_h))

    if projection is not None:
        prims.extend(projection_symbol_prims((symbol_x, row2_top + ROW1_H / 2.0), projection))
    return tuple(prims)


def _series_height(texts: dict[str, tuple[str, float]], nominal: float) -> float:
    """The largest ISO 3098 height up to `nominal` at which every text fits its
    width, `texts` being field -> (text, width), at FIT_ADVANCE per character.

    The height steps down rather than the width factor because
    `entity_create_text` draws in the current text style and carries no width
    factor of its own. A text that does not fit even at the smallest height is
    refused by field name, before anything is drawn.
    """
    for height in (h for h in ISO_3098_HEIGHTS if h <= nominal):
        if all(FIT_ADVANCE * height * len(text) <= width for text, width in texts.values()):
            return height
    smallest = ISO_3098_HEIGHTS[-1]
    too_long = [
        f"{field} ({len(text)} characters, at most {int(width / (FIT_ADVANCE * smallest))})"
        for field, (text, width) in texts.items()
        if FIT_ADVANCE * smallest * len(text) > width
    ]
    raise ValueError(
        f"title block: too long to fit its cell even at {smallest} mm: {', '.join(too_long)}"
    )


async def apply_titleblock(
    backend: AutoCADBackend,
    *,
    size: str = "A3",
    metadata: TitleBlockMetadata,
    origin: Pt = (0.0, 0.0),
    orientation: str = "landscape",
    projection: str | None = "first",
    frame: bool = True,
    zones: bool = False,
    marks: bool = False,
    layout: str | None = None,
) -> dict:
    """Draw the ISO 5457 frame (optionally) and the ISO 7200 title block.

    `projection=None`, `zones=False` and `marks=False` reproduce exactly what
    `titleblock_apply_iso_a3` has always drawn. Every refusal -- an unknown
    size, an unknown orientation, an unknown projection angle, a missing layout
    -- fires before the first entity reaches the drawing.
    """
    metrics = frame_metrics(size, orientation=orientation)
    tb_prims = titleblock_prims(size, metadata, orientation=orientation, projection=projection)
    sheet_prims = (
        frame_prims(size, orientation=orientation, zones=zones, marks=marks) if frame else ()
    )
    previous = await enter_layout(backend, layout, caller="titleblock_apply")
    try:
        frame_handles = await draw_sheet_prims(backend, sheet_prims, origin=origin)
        tb_handles = await draw_sheet_prims(backend, tb_prims, origin=origin)
    finally:
        if layout is not None:
            await backend.layout_set_current(previous)

    ox, oy = float(origin[0]), float(origin[1])
    w, h = metrics["sheet"]
    symbol_start = TITLE_TEXT_INDEX + 1 + 2 * len(VALUE_FIELDS)
    return {
        "ok": True,
        "size": metrics["size"],
        "orientation": orientation,
        "layout": layout or "Model",
        "projection": projection,
        "outer_border": frame_handles[0] if frame else None,
        "inner_border": frame_handles[1] if frame else None,
        "frame_handles": frame_handles,
        "titleblock_lines": tb_handles[:TITLEBLOCK_RULE_COUNT],
        "title_text": tb_handles[TITLE_TEXT_INDEX],
        "value_texts": {f: tb_handles[i] for f, i in value_text_indices().items()},
        "projection_symbol": tb_handles[symbol_start:],
        "metadata": asdict(metadata),
        "text_heights": {
            "title": tb_prims[TITLE_TEXT_INDEX].height,
            "values": tb_prims[value_text_indices()["drawing_no"]].height,
        },
        "bbox": {"min": [ox, oy], "max": [ox + w, oy + h]},
    }
