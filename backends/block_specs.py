# backends/block_specs.py
"""Typed primitive specs for ``block_define``, validated before any write.

The backend consumes the normalised specs, so a malformed request is refused
before anything is written. Every error names the offending entry
(``entities[i]`` / ``attdefs[i]``) and key.
"""

from __future__ import annotations

import math
import re

from backends.base import TEXT_ALIGN, parse_lineweight
from engineering.validation import short_repr
from security import illegal_symbol_name_chars

ALLOWED_TYPES = (
    "line",
    "circle",
    "arc",
    "polyline",
    "donut",
    "text",
    "solid",
    "wipeout",
    "hatch",
)
#: Keys every primitive may carry, as on an ``entity_batch_create`` item. Unset,
#: a primitive is ByBlock (the INSERT supplies colour, linetype, lineweight).
_COMMON = ("type", "layer", "color", "linetype", "lineweight")
#: The TEXT justifications every create shares (``backends.base.TEXT_ALIGN``).
TEXT_ALIGNMENTS = tuple(TEXT_ALIGN)
#: Checked with ``fullmatch`` — ``$`` alone would accept a trailing newline, which
#: ezdxf then strips on write, turning two "distinct" tags into one duplicate.
ATTDEF_TAG_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,30}$")

#: The keys are ``entity_create_*``'s (and so ``entity_batch_create``'s): one
#: vocabulary for a primitive wherever it is drawn.
_REQUIRED = {
    "line": ("x1", "y1", "x2", "y2"),
    "circle": ("cx", "cy", "radius"),
    "arc": ("cx", "cy", "radius", "start_angle", "end_angle"),
    "polyline": ("points",),
    "text": ("text", "x", "y", "height"),
    "solid": ("points",),
    "wipeout": ("points",),
    "hatch": (),
    "donut": ("cx", "cy", "outer_radius"),
}
#: Keys a primitive may carry besides ``type``, ``layer`` and its required ones.
_OPTIONAL = {
    "line": (),
    "circle": (),
    "arc": (),
    "polyline": ("closed", "bulges", "width"),
    "text": ("rotation", "align"),
    "solid": (),
    "wipeout": (),
    "hatch": ("boundary_points", "circle", "holes", "pattern", "scale", "angle"),
    "donut": ("inner_radius",),
}
_ATTDEF_KEYS = (
    "tag",
    "prompt",
    "default",
    "x",
    "y",
    "height",
    "rotation",
    "align",
    "invisible",
)
_POSITIVE = {"radius", "height", "outer_radius"}
#: DXF symbol-table names (layers, blocks) are at most 255 characters.
MAX_NAME_LENGTH = 255


def _scalar(value, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where} must be a number, got {short_repr(value)}")
    try:
        number = float(value)
    except OverflowError as exc:  # a 400-digit JSON integer is an int, not a float
        raise TypeError(f"{where} must be finite") from exc
    if not math.isfinite(number):
        raise TypeError(f"{where} must be finite")
    if positive and number <= 0:
        raise TypeError(f"{where} must be > 0, got {number}")
    return number


def _number(spec: dict, key: str, where: str) -> float:
    return _scalar(spec.get(key), f"{where}: {key!r}", positive=key in _POSITIVE)


def _points(
    spec: dict, where: str, *, minimum: int, maximum: int | None = None
) -> list[tuple[float, float]]:
    raw = spec.get("points")
    if not isinstance(raw, (list, tuple)) or len(raw) < minimum:
        raise TypeError(f"{where}: 'points' must be a list of at least {minimum} [x, y] pairs")
    if maximum is not None and len(raw) > maximum:
        raise TypeError(f"{where}: 'points' must have at most {maximum} pairs")
    out: list[tuple[float, float]] = []
    for i, pt in enumerate(raw):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise TypeError(f"{where}: points[{i}] must be an [x, y] pair")
        out.append(
            (
                _scalar(pt[0], f"{where}: points[{i}].x"),
                _scalar(pt[1], f"{where}: points[{i}].y"),
            )
        )
    return out


def _align(spec: dict, where: str) -> str:
    align = spec.get("align", "left")
    if align not in TEXT_ALIGNMENTS:
        raise TypeError(
            f"{where}: 'align' must be one of {TEXT_ALIGNMENTS}, got {short_repr(align)}"
        )
    return align


def _layer(spec: dict, where: str) -> str:
    layer = spec.get("layer", "0")
    if not isinstance(layer, str) or not layer.strip():
        raise TypeError(f"{where}: 'layer' must be a non-empty string")
    layer = layer.strip()
    if len(layer) > MAX_NAME_LENGTH:
        raise TypeError(
            f"{where}: 'layer' is {len(layer)} characters; a DXF layer name is at most "
            f"{MAX_NAME_LENGTH}"
        )
    # Gate the DXF name rule here rather than letting ezdxf raise DXFValueError
    # on `add_*(dxfattribs={"layer": ...})` mid-loop, after the block exists and
    # earlier primitives are already written (and, with overwrite, the old
    # definition is already gone).
    offenders = illegal_symbol_name_chars(layer)
    if offenders:
        shown = ", ".join(repr(c) for c in offenders)
        raise TypeError(f"{where}: 'layer' is not a legal DXF layer name (contains {shown})")
    return layer


def _one_line(value, where: str) -> str:
    """A single-line DXF string (TEXT content, ATTDEF prompt/default).

    ezdxf's DXFAttr fixer (``fix_one_line_text``) silently strips line breaks
    and a trailing ``^`` on write, logged at DEBUG only, so the declared value
    and the value read back would differ. Refuse it here, before any write, using ezdxf's
    own ``is_valid_one_line_text`` rule plus the other C0 controls DXF text
    cannot carry.
    """
    if not isinstance(value, str):
        raise TypeError(f"{where} must be a string")
    controls = sorted({c for c in value if ord(c) < 32})
    if controls:
        shown = ", ".join(repr(c) for c in controls)
        raise TypeError(
            f"{where} must be a single line without control characters (contains {shown})"
        )
    if value.endswith("^"):
        raise TypeError(f"{where} must not end with '^' (DXF strips a trailing caret)")
    return value


def _is_pair(value) -> bool:
    return isinstance(value, (list, tuple))


def _disc(value, where: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TypeError(f"{where} must be [cx, cy, radius]")
    return [
        _scalar(value[0], f"{where}[0]"),
        _scalar(value[1], f"{where}[1]"),
        _scalar(value[2], f"{where}[2]", positive=True),
    ]


def _pattern(name, where: str) -> str:
    """A predefined hatch pattern in any case; ezdxf would fill an unknown one as ANSI31."""
    from ezdxf.tools.pattern import ISO_PATTERN

    key = str(name).strip().upper()
    if key != "SOLID" and key not in ISO_PATTERN:
        raise TypeError(f"{where}: 'pattern' {short_repr(name)} is not a predefined hatch pattern")
    return key


def _properties(spec: dict, where: str) -> dict:
    """``color`` (ACI 0-256), ``linetype`` and ``lineweight`` as written; unset is ByBlock."""
    out: dict = {"color": 0, "linetype": "ByBlock", "lineweight": -2}
    if spec.get("color") is not None:
        color = spec["color"]
        if isinstance(color, bool) or not isinstance(color, int) or not 0 <= color <= 256:
            raise TypeError(f"{where}: 'color' must be an ACI 0-256, got {short_repr(color)}")
        out["color"] = color
    if spec.get("linetype") is not None:
        linetype = spec["linetype"]
        if not isinstance(linetype, str) or not linetype.strip():
            raise TypeError(f"{where}: 'linetype' must be a non-empty string")
        out["linetype"] = linetype.strip()
    if spec.get("lineweight") is not None:
        try:
            out["lineweight"] = parse_lineweight(spec["lineweight"])
        except ValueError as exc:
            raise TypeError(f"{where}: 'lineweight': {exc}") from None
    return out


def _refuse_unknown_keys(spec: dict, allowed: set[str], where: str, what: str) -> None:
    """A key nobody reads is refused by name, never dropped (backlog W1-T2-03)."""
    unknown = sorted(str(key) for key in spec if key not in allowed)
    if unknown:
        raise TypeError(
            f"{where}: unknown key(s) {', '.join(map(repr, unknown))}; {what} takes "
            f"{', '.join(sorted(allowed))}"
        )


def validate_block_name(name) -> str:
    """``block_define``'s ``name`` must be a string before anything reads it.

    A non-string reached ``(name or "").strip()`` and died with an
    ``AttributeError`` instead of a refusal (backlog W2-T3-07).
    """
    if not isinstance(name, str):
        raise TypeError(f"block_define: 'name' must be a string, got {type(name).__name__}")
    return name


def validate_entity_specs(entities) -> list[dict]:
    """Normalise ``block_define`` primitive specs; refuse the whole list on the first defect."""
    if not isinstance(entities, (list, tuple)):
        # A dict, a string or a generator used to be refused as "empty".
        raise TypeError(
            f"block_define: 'entities' must be a list of objects, got {type(entities).__name__}"
        )
    if not entities:
        raise ValueError(
            "block_define: 'entities' is empty — a block that draws nothing is refused"
        )
    out: list[dict] = []
    for i, spec in enumerate(entities):
        where = f"entities[{i}]"
        if not isinstance(spec, dict):
            raise TypeError(f"{where}: must be an object")
        kind = spec.get("type")
        if kind not in ALLOWED_TYPES:
            raise TypeError(
                f"{where}: 'type' must be one of {ALLOWED_TYPES}, got {short_repr(kind)}"
            )
        _refuse_unknown_keys(
            spec, {*_COMMON, *_REQUIRED[kind], *_OPTIONAL[kind]}, where, f"a {kind}"
        )
        for key in _REQUIRED[kind]:
            if key not in spec:
                raise TypeError(f"{where}: {kind} requires {key!r}")
        norm: dict = {"type": kind, "layer": _layer(spec, where), **_properties(spec, where)}
        if kind == "line":
            for key in ("x1", "y1", "x2", "y2"):
                norm[key] = _number(spec, key, where)
        elif kind == "circle":
            for key in ("cx", "cy", "radius"):
                norm[key] = _number(spec, key, where)
        elif kind == "arc":
            for key in ("cx", "cy", "radius", "start_angle", "end_angle"):
                norm[key] = _number(spec, key, where)
        elif kind == "polyline":
            pts = _points(spec, where, minimum=2)
            closed = spec.get("closed", False)
            if not isinstance(closed, bool):
                raise TypeError(f"{where}: 'closed' must be a boolean")
            bulges = spec.get("bulges")
            if bulges is not None:
                if not isinstance(bulges, (list, tuple)) or len(bulges) != len(pts):
                    raise TypeError(f"{where}: 'bulges' must list one value per point")
                bulges = [_scalar(b, f"{where}: bulges[{j}]") for j, b in enumerate(bulges)]
            width = _scalar(spec.get("width", 0.0), f"{where}: 'width'")
            if width < 0:
                raise TypeError(f"{where}: 'width' must be >= 0, got {width}")
            norm.update(
                points=pts, closed=closed, bulges=list(bulges) if bulges else None, width=width
            )
        elif kind == "donut":
            # AutoCAD's DONUT: a closed two-arc LWPOLYLINE with a constant width
            # (inner_radius 0 is a filled dot)
            outer = _number(spec, "outer_radius", where)
            inner = _scalar(spec.get("inner_radius", 0.0), f"{where}: 'inner_radius'")
            if not 0 <= inner < outer:
                raise TypeError(
                    f"{where}: 'inner_radius' must be >= 0 and < outer_radius {outer}, got {inner}"
                )
            norm.update(cx=_number(spec, "cx", where), cy=_number(spec, "cy", where))
            norm.update(inner_radius=inner, outer_radius=outer)
        elif kind == "text":
            norm.update(
                text=_one_line(spec.get("text"), f"{where}: 'text'"),
                x=_number(spec, "x", where),
                y=_number(spec, "y", where),
                height=_number(spec, "height", where),
                rotation=_scalar(spec.get("rotation", 0.0), f"{where}: 'rotation'"),
                align=_align(spec, where),
            )
        elif kind == "solid":
            norm["points"] = _points(spec, where, minimum=3, maximum=4)
        elif kind == "hatch":
            # a polygon (boundary_points, entity_batch_create's key) or a disc
            # (circle [cx, cy, r]); holes are islands, each a polygon or a disc
            if ("boundary_points" in spec) == ("circle" in spec):
                raise TypeError(
                    f"{where}: hatch takes boundary_points (3+) or circle [cx, cy, r], one of them"
                )
            if "boundary_points" in spec:
                norm["boundary"] = _points(
                    {"points": spec["boundary_points"]}, f"{where}: boundary_points", minimum=3
                )
            else:
                norm["boundary"] = _disc(spec["circle"], f"{where}: circle")
            holes = spec.get("holes") or []
            if not isinstance(holes, (list, tuple)):
                raise TypeError(f"{where}: 'holes' must be a list of polygons or [cx, cy, r]")
            norm["holes"] = [
                _disc(hole, f"{where}: holes[{j}]")
                if isinstance(hole, (list, tuple)) and len(hole) == 3 and not _is_pair(hole[0])
                else _points({"points": hole}, f"{where}: holes[{j}]", minimum=3)
                for j, hole in enumerate(holes)
            ]
            norm["pattern"] = _pattern(spec.get("pattern", "SOLID"), where)
            norm["scale"] = _scalar(spec.get("scale", 1.0), f"{where}: 'scale'", positive=True)
            norm["angle"] = _scalar(spec.get("angle", 0.0), f"{where}: 'angle'")
        elif kind == "wipeout":
            # A closed polygon (its last point joins the first); it masks what a
            # layout draws before the INSERT, and the primitives listed after it.
            norm["points"] = _points(spec, where, minimum=3)
        out.append(norm)
    return out


def validate_attdef_specs(attdefs) -> list[dict]:
    """Normalise ATTDEF specs: tag pattern, uniqueness, numeric fields, flags."""
    if attdefs is None:
        return []
    if not isinstance(attdefs, (list, tuple)):
        raise TypeError("block_define: 'attdefs' must be a list")
    out: list[dict] = []
    seen: set[str] = set()
    for i, spec in enumerate(attdefs):
        where = f"attdefs[{i}]"
        if not isinstance(spec, dict):
            raise TypeError(f"{where}: must be an object")
        _refuse_unknown_keys(spec, set(_ATTDEF_KEYS), where, "an ATTDEF")
        tag = spec.get("tag")
        if not isinstance(tag, str) or not ATTDEF_TAG_RE.fullmatch(tag):
            raise TypeError(
                f"{where}: 'tag' must match {ATTDEF_TAG_RE.pattern}, got {short_repr(tag)}"
            )
        if tag in seen:
            raise TypeError(f"{where}: duplicate tag {tag!r}")
        seen.add(tag)
        prompt = _one_line(spec.get("prompt", tag), f"{where}: 'prompt'")
        default = _one_line(spec.get("default", ""), f"{where}: 'default'")
        invisible = spec.get("invisible", False)
        if not isinstance(invisible, bool):
            raise TypeError(f"{where}: 'invisible' must be a boolean")
        out.append(
            {
                "tag": tag,
                "prompt": prompt,
                "default": default,
                "x": _number(spec, "x", where),
                "y": _number(spec, "y", where),
                "height": _number(spec, "height", where),
                "rotation": _scalar(spec.get("rotation", 0.0), f"{where}: 'rotation'"),
                "align": _align(spec, where),
                "invisible": invisible,
            }
        )
    return out


def validate_base_point(base_x, base_y) -> tuple[float, float]:
    """``block_define``'s base point as finite floats; ``TypeError`` names the key.

    Only ``float()`` was applied before, so ``float("nan")`` was written into
    the BLOCK record headlessly and would reach AutoCAD as a NaN VARIANT.
    """
    return (
        _scalar(base_x, "block_define: 'base_x'"),
        _scalar(base_y, "block_define: 'base_y'"),
    )


def referenced_layers(entities: list[dict]) -> list[str]:
    """The distinct layers ``block_define``'s primitives reference, other than ``0``.

    DXF and AutoCAD layer tables are case-insensitive, so the set is keyed on
    the folded name and keeps the first spelling seen: a request naming
    ``Alpha`` and ``ALPHA`` wants *one* layer. Deduping case-sensitively let
    ezdxf write ``ALPHA`` and then raise ``DXFTableEntryError`` on ``Alpha``
    mid-call.
    """
    seen: dict[str, str] = {}
    for spec in entities:
        layer = spec["layer"]
        if layer != "0" and layer.lower() not in seen:
            seen[layer.lower()] = layer
    return sorted(seen.values())


def validate_attribute_values(attributes, where: str = "attributes") -> dict[str, str]:
    """``{TAG: value}`` for ``block_insert`` / ``block_set_attributes`` as the
    strings that will be written, refused before any write.

    ``str(value)`` used to be applied to whatever arrived, so a JSON ``null``
    became the text ``None``, ``true`` became ``True`` and an object became its
    Python repr — silently. A string is taken as is (one line,
    the DXF rule ``_one_line`` enforces); an int or a finite float is written
    with ``str`` (``101`` → ``"101"``, ``2.5`` → ``"2.5"``); ``bool``, ``None``,
    lists and objects are refused by tag. ``None`` for the whole mapping means
    "no attributes".
    """
    if attributes is None:
        return {}
    if not isinstance(attributes, dict):
        raise TypeError(f"{where} must be an object of {{TAG: value}} pairs")
    out: dict[str, str] = {}
    spelled: dict[str, str] = {}
    for tag, value in attributes.items():
        if not isinstance(tag, str) or not tag.strip():
            raise TypeError(f"{where}: every tag must be a non-empty string")
        label = f"{where}[{short_repr(tag)}]"
        # Tags are matched upper-case: AutoCAD stores ATTDEF tags upper-case and
        # ATTDEF_TAG_RE requires it, so `{"tag": ...}` was silently dropped
        # against an ATTDEF `TAG` (backlog W1-T1-05).
        key = tag.strip().upper()
        if key in spelled:
            raise TypeError(
                f"{where}: {short_repr(spelled[key])} and {short_repr(tag)} name the same "
                "attribute (tags are not case-sensitive)"
            )
        spelled[key] = tag
        if isinstance(value, str):
            out[key] = _one_line(value, label)
        elif isinstance(value, bool) or value is None or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be a string or a number, got {type(value).__name__}")
        else:
            if isinstance(value, float) and not math.isfinite(value):
                raise TypeError(f"{label} must be finite")
            out[key] = str(value)
    return out


def solid_vertices(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """DXF SOLID draws vtx0→vtx1→vtx3→vtx2, so a quad given in polygon order
    (a, b, c, d) is stored as (a, b, d, c). Triangles are stored as given."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) == 4:
        return [pts[0], pts[1], pts[3], pts[2]]
    return pts
