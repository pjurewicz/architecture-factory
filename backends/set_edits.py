"""Set-based edits: ``entity_modify_where``, ``layer_rename``, ``layer_merge`` (1.8-12).

The rival's layer clean-up took 11 calls; ours took 124, because the verbs did
not exist and ``entity_batch_modify`` wants one row per entity. This module
holds the engine-neutral half of the three verbs - the argument grammar and
the headless matching and reassignment.

**where** - every key optional, at least one required (an empty filter would
select every entity in the space; say ``{"type": "*"}`` to mean that):

- ``layer``: a name or a list of names; ``*`` and ``?`` are the only
  wildcards, case-insensitive (layer names are);
- ``type``: a DXF type or a list (``LINE``, ``INSERT``, ``DIMENSION``),
  same wildcards;
- ``color``: an ACI number 0-256 (256 = ByLayer, 0 = ByBlock);
- ``linetype``: a name or a list, same wildcards (``ByLayer`` matches an
  entity that follows its layer);
- ``lineweight``: anything ``parse_lineweight`` takes (``"ByLayer"``, 25,
  ``0.25``, ``-3``);
- ``region``: ``[x0, y0, x1, y1]`` in WCS, x0 <= x1 and y0 <= y1; an entity
  matches when its box meets the region (closed intervals, the rule the
  preview's region fingerprint uses); an entity with no finite box never
  matches;
- ``block``: an INSERT's block name or a list, same wildcards (an anonymous
  dynamic-block reference is matched by its ``*U`` name).

**set** - at least one of ``layer`` (created when missing and reported),
``color`` (0-256; a true colour is dropped with it), ``linetype`` (loaded
when missing), ``lineweight`` (``parse_lineweight``), ``visible`` (bool), and
the type-specific ``pattern`` / ``pattern_scale`` / ``pattern_angle`` (HATCH),
``text_style`` / ``text_height`` (TEXT, MTEXT), ``width_factor`` (TEXT), which
leave every other matched type as it is (``TYPED_KEYS``).

**space** - ``model``, ``paper`` (every paper layout) or ``all``. Block
definitions are never searched by ``entity_modify_where``; ``layer_merge``
and ``layer_rename`` reach into them because a layer's name lives there too.

An entity on a locked layer is not written - AutoCAD's own rule, which the
headless engine keeps - and is
reported in ``skipped_locked``, never counted as matched.

Refusals are by name and before anything is written: a malformed ``where``
or ``set`` is ``invalid_args``; ``layer_rename`` / ``layer_merge`` refusing
a layer (``0``, ``DEFPOINTS``, an xref-dependent one, a missing source, an
existing rename target) are ``refused``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

from fastmcp.exceptions import ToolError

from backends.base import parse_lineweight
from engineering.errors import InvalidArgsError

__all__ = [
    "HANDLE_SAMPLE",
    "PROTECTED_LAYERS",
    "SPACES",
    "SetSpec",
    "Where",
    "changes_payload",
    "check_layer_name",
    "check_origin",
    "glob_matches",
    "hatch_origin",
    "parse_set",
    "parse_space",
    "parse_where",
    "region_meets",
    "set_hatch_origin",
]

#: How many handles a result lists before it says ``truncated``.
HANDLE_SAMPLE = 20
#: The spaces ``entity_modify_where`` searches.
SPACES = ("model", "paper", "all")
#: Layers no rename or merge may take from: AutoCAD refuses both.
PROTECTED_LAYERS = frozenset({"0", "defpoints"})

WHERE_KEYS = ("layer", "type", "color", "linetype", "lineweight", "region", "block")
SET_KEYS = (
    "layer",
    "color",
    "linetype",
    "lineweight",
    "visible",
    # round 3: written only on the types that have them (HATCH; TEXT / MTEXT,
    # width_factor TEXT only), every other match left as it is
    "pattern",
    "pattern_scale",
    "pattern_angle",
    "text_style",
    "text_height",
    "width_factor",
)
#: The type-specific ``set`` keys and the DXF types that carry them.
TYPED_KEYS = {
    "pattern": ("HATCH",),
    "pattern_scale": ("HATCH",),
    "pattern_angle": ("HATCH",),
    "text_style": ("TEXT", "MTEXT"),
    "text_height": ("TEXT", "MTEXT"),
    "width_factor": ("TEXT",),
}


def _invalid(path: str, message: str) -> InvalidArgsError:
    return InvalidArgsError(f"invalid_args: {path}: {message}")


# ── globs ──────────────────────────────────────────────────────────────────


def _names(value, path: str) -> tuple[str, ...]:
    """A name or a non-empty list of names, each a non-empty string."""
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)) or not items:
        raise _invalid(path, f"expected a name or a non-empty list of names, got {value!r}")
    out = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise _invalid(f"{path}[{index}]", f"expected a non-empty name, got {item!r}")
        out.append(item.strip())
    return tuple(dict.fromkeys(out))


def _glob_regex(patterns: Iterable[str]) -> re.Pattern:
    parts = []
    for pattern in patterns:
        body = "".join(
            ".*" if char == "*" else "." if char == "?" else re.escape(char) for char in pattern
        )
        parts.append(f"(?:{body})")
    return re.compile(r"\A(?:" + "|".join(parts) + r")\Z", re.IGNORECASE | re.DOTALL)


def glob_matches(patterns: tuple[str, ...], value: str) -> bool:
    """True when ``value`` matches one of ``patterns`` (``*`` / ``?``, any case)."""
    return _glob_regex(patterns).match(str(value or "")) is not None


# ── where / set / space ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Where:
    """A parsed ``where``. ``None`` means "not filtered on"."""

    layers: tuple[str, ...] | None = None
    types: tuple[str, ...] | None = None
    color: int | None = None
    linetypes: tuple[str, ...] | None = None
    lineweight: int | None = None
    region: tuple[float, float, float, float] | None = None
    blocks: tuple[str, ...] | None = None

    def to_dict(self) -> dict:
        out: dict = {}
        if self.layers is not None:
            out["layer"] = list(self.layers)
        if self.types is not None:
            out["type"] = list(self.types)
        if self.color is not None:
            out["color"] = self.color
        if self.linetypes is not None:
            out["linetype"] = list(self.linetypes)
        if self.lineweight is not None:
            out["lineweight"] = self.lineweight
        if self.region is not None:
            out["region"] = list(self.region)
        if self.blocks is not None:
            out["block"] = list(self.blocks)
        return out


def _color(value, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 256:
        raise _invalid(path, f"expected an ACI number 0-256 (256 = ByLayer), got {value!r}")
    return int(value)


def _lineweight(value, path: str) -> int:
    try:
        code = parse_lineweight(value)
    except ValueError as exc:
        raise _invalid(path, str(exc)) from None
    if code is None:
        raise _invalid(path, "expected a lineweight, got None")
    return code


def _region(value, path: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise _invalid(path, f"expected [x0, y0, x1, y1], got {value!r}")
    numbers = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _invalid(f"{path}[{index}]", f"expected a number, got {item!r}")
        if not math.isfinite(float(item)):
            raise _invalid(f"{path}[{index}]", f"must be finite, got {item!r}")
        numbers.append(float(item))
    x0, y0, x1, y1 = numbers
    if x0 > x1 or y0 > y1:
        raise _invalid(path, f"x0 <= x1 and y0 <= y1 are required, got {value!r}")
    return x0, y0, x1, y1


def _mapping(raw, path: str, keys: tuple[str, ...]) -> dict:
    if not isinstance(raw, dict):
        raise _invalid(path, f"expected an object, got {type(raw).__name__}")
    unknown = sorted(str(k) for k in raw if k not in keys)
    if unknown:
        raise _invalid(path, f"unknown key(s) {', '.join(unknown)}; known: {', '.join(keys)}")
    given = {k: v for k, v in raw.items() if v is not None}
    if not given:
        raise _invalid(path, f"name at least one of {', '.join(keys)}")
    return given


def parse_where(raw) -> Where:
    """A validated :class:`Where`; refused by path (``invalid_args``)."""
    given = _mapping(raw, "where", WHERE_KEYS)
    return Where(
        layers=_names(given["layer"], "where.layer") if "layer" in given else None,
        types=tuple(t.upper() for t in _names(given["type"], "where.type"))
        if "type" in given
        else None,
        color=_color(given["color"], "where.color") if "color" in given else None,
        linetypes=_names(given["linetype"], "where.linetype") if "linetype" in given else None,
        lineweight=_lineweight(given["lineweight"], "where.lineweight")
        if "lineweight" in given
        else None,
        region=_region(given["region"], "where.region") if "region" in given else None,
        blocks=_names(given["block"], "where.block") if "block" in given else None,
    )


@dataclass(frozen=True)
class SetSpec:
    """A parsed ``set``. ``None`` means "left as it is"."""

    layer: str | None = None
    color: int | None = None
    linetype: str | None = None
    lineweight: int | None = None
    visible: bool | None = None
    pattern: str | None = None
    pattern_scale: float | None = None
    pattern_angle: float | None = None
    text_style: str | None = None
    text_height: float | None = None
    width_factor: float | None = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in SET_KEYS if getattr(self, k) is not None}


def parse_set(raw) -> SetSpec:
    """A validated :class:`SetSpec`; refused by path (``invalid_args``)."""
    given = _mapping(raw, "set", SET_KEYS)
    layer = None
    if "layer" in given:
        if not isinstance(given["layer"], str) or not given["layer"].strip():
            raise _invalid("set.layer", f"expected a layer name, got {given['layer']!r}")
        layer = check_layer_name(given["layer"], "set.layer", invalid_args=True)
    linetype = None
    if "linetype" in given:
        if not isinstance(given["linetype"], str) or not given["linetype"].strip():
            raise _invalid("set.linetype", f"expected a linetype name, got {given['linetype']!r}")
        linetype = given["linetype"].strip()
    visible = None
    if "visible" in given:
        if not isinstance(given["visible"], bool):
            raise _invalid("set.visible", f"expected true or false, got {given['visible']!r}")
        visible = given["visible"]
    typed: dict = {}
    for key in ("pattern", "text_style"):
        if key in given:
            if not isinstance(given[key], str) or not given[key].strip():
                raise _invalid(f"set.{key}", f"expected a name, got {given[key]!r}")
            typed[key] = given[key].strip()
    for key in ("pattern_scale", "text_height", "width_factor", "pattern_angle"):
        if key in given:
            value = given[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _invalid(f"set.{key}", f"expected a number, got {value!r}")
            if key != "pattern_angle" and not float(value) > 0:
                raise _invalid(f"set.{key}", f"must be > 0, got {value!r}")
            typed[key] = float(value)
    return SetSpec(
        **typed,
        layer=layer,
        color=_color(given["color"], "set.color") if "color" in given else None,
        linetype=linetype,
        lineweight=_lineweight(given["lineweight"], "set.lineweight")
        if "lineweight" in given
        else None,
        visible=visible,
    )


def resolve_layer(spec: SetSpec, existing: Iterable[str]) -> tuple[SetSpec, bool]:
    """``(spec, missing)``: ``spec.layer`` spelt as the layer table stores it
    (names ignore case, so writing ``A-WALL`` onto ``a-wall`` would read back
    as a change forever), and whether the layer still has to be created."""
    if spec.layer is None:
        return spec, False
    names = {str(name).lower(): str(name) for name in existing}
    stored = names.get(spec.layer.lower())
    if stored is None:
        return spec, True
    return replace(spec, layer=stored), False


def parse_space(raw) -> str:
    space = str(raw or "").strip().lower()
    if space not in SPACES:
        raise _invalid("space", f"expected one of {', '.join(SPACES)}, got {raw!r}")
    return space


def check_layer_name(name, path: str, *, invalid_args: bool = False) -> str:
    """``name`` if a layer may carry it; refused by name otherwise.

    The DXF symbol-table rule (``security.sanitize_symbol_name``) plus ``|``,
    the xref-dependent separator no local layer may carry.
    """
    from security import sanitize_symbol_name

    try:
        cleaned = sanitize_symbol_name(str(name), kind="layer")
        if "|" in cleaned:
            raise ToolError("layer name rejected: '|' marks an xref-dependent layer.")
    except ToolError as exc:
        if invalid_args:
            raise _invalid(path, str(exc)) from None
        raise ToolError(f"{path}: {exc}") from None
    return cleaned


def region_meets(box, region) -> bool:
    """True when ``box`` and ``region`` share a point (closed intervals)."""
    if box is None:
        return False
    x0, y0, x1, y1 = box
    rx0, ry0, rx1, ry1 = region
    return not (x1 < rx0 or x0 > rx1 or y1 < ry0 or y0 > ry1)


def changes_payload(*, modified: int = 0, **counts: int) -> dict:
    """The ``changes`` block every set edit carries (1.8-10's shape).

    These verbs add and delete no entity, so ``added`` / ``deleted`` are empty
    and the set is ``complete``; ``counts`` carries what moved
    (``modified`` entities, plus layer-table rows such as ``layers_deleted``).
    """
    return {
        "added": [],
        "deleted": [],
        "counts": {"added": 0, "deleted": 0, "modified": int(modified), **counts},
        "complete": True,
    }


# ── the headless engine ────────────────────────────────────────────────────


def ezdxf_spaces(doc, space: str):
    """``(name, layout)`` for every layout ``space`` covers, model first."""
    if space in ("model", "all"):
        yield "Model", doc.modelspace()
    if space in ("paper", "all"):
        for name in doc.layouts.names_in_taborder():
            if name.lower() != "model":
                yield name, doc.layouts.get(name)


def ezdxf_locked_layers(doc) -> set[str]:
    """Lower-cased names of the locked layers."""
    return {str(layer.dxf.name).lower() for layer in doc.layers if layer.is_locked()}


def ezdxf_matches(entity, where: Where) -> bool:
    """True when ``entity`` passes every filter of ``where``."""
    dxftype = entity.dxftype().upper()
    if where.types is not None and not glob_matches(where.types, dxftype):
        return False
    if where.layers is not None and not glob_matches(where.layers, entity.dxf.get("layer", "0")):
        return False
    if where.blocks is not None:
        if dxftype != "INSERT" or not glob_matches(where.blocks, entity.dxf.get("name", "")):
            return False
    if where.color is not None and int(entity.dxf.get("color", 256)) != where.color:
        return False
    if where.linetypes is not None and not glob_matches(
        where.linetypes, entity.dxf.get("linetype", "ByLayer")
    ):
        return False
    if where.lineweight is not None and int(entity.dxf.get("lineweight", -1)) != where.lineweight:
        return False
    if where.region is not None:
        from engineering.understand.snapshot import _box

        if not region_meets(_box(entity), where.region):
            return False
    return True


def ezdxf_select(doc, where: Where, space: str) -> tuple[list, dict]:
    """``(entities, skipped_locked)``: what ``where`` matches in ``space``,
    sorted by handle, with the entities on locked layers set aside. A paper
    layout's own overall viewport (id 1) is the sheet, not drawn content, and
    AutoCAD does not let a selection pick it - so neither does this."""
    locked = ezdxf_locked_layers(doc)
    found = []
    skipped: dict[str, int] = {}
    for _name, layout in ezdxf_spaces(doc, space):
        for entity in layout:
            if entity.dxftype() == "VIEWPORT" and int(entity.dxf.get("id", 0)) == 1:
                continue
            if not ezdxf_matches(entity, where):
                continue
            layer = str(entity.dxf.get("layer", "0"))
            if layer.lower() in locked:
                skipped[layer] = skipped.get(layer, 0) + 1
                continue
            found.append(entity)
    found.sort(key=lambda e: int(str(e.dxf.handle), 16))
    return found, skipped


def skipped_payload(skipped: dict[str, int]) -> dict:
    return {"count": sum(skipped.values()), "layers": sorted(skipped)}


def ezdxf_differs(entity, spec: SetSpec) -> bool:
    """True when writing ``spec`` would change ``entity``."""
    dxf = entity.dxf
    if spec.layer is not None and str(dxf.get("layer", "0")) != spec.layer:
        return True
    if spec.color is not None and (
        int(dxf.get("color", 256)) != spec.color or dxf.hasattr("true_color")
    ):
        return True
    if spec.linetype is not None and str(dxf.get("linetype", "ByLayer")).lower() != (
        spec.linetype.lower()
    ):
        return True
    if spec.lineweight is not None and int(dxf.get("lineweight", -1)) != spec.lineweight:
        return True
    if spec.visible is not None and bool(dxf.get("invisible", 0)) == spec.visible:
        return True
    return any(old != new for old, new in _typed_changes(entity, spec))


def _typed_changes(entity, spec: SetSpec):
    """``(now, wanted)`` for each type-specific key ``spec`` sets on ``entity``'s type."""
    kind, dxf = entity.dxftype(), entity.dxf
    if kind == "HATCH":
        if spec.pattern is not None:
            yield str(dxf.get("pattern_name", "")).upper(), spec.pattern
        solid = (spec.pattern or str(dxf.get("pattern_name", ""))).upper() == "SOLID"
        if spec.pattern_scale is not None and not solid:
            yield round(float(dxf.get("pattern_scale", 1.0)), 9), round(spec.pattern_scale, 9)
        if spec.pattern_angle is not None and not solid:
            yield round(float(dxf.get("pattern_angle", 0.0)), 9), round(spec.pattern_angle, 9)
    if kind in ("TEXT", "MTEXT"):
        if spec.text_style is not None:
            yield str(dxf.get("style", "Standard")).lower(), spec.text_style.lower()
        if spec.text_height is not None:
            field = "height" if kind == "TEXT" else "char_height"
            yield round(float(dxf.get(field, 0.0)), 9), round(spec.text_height, 9)
    if kind == "TEXT" and spec.width_factor is not None:
        yield round(float(dxf.get("width", 1.0)), 9), round(spec.width_factor, 9)


def ezdxf_dimension_follow(entity, old_layer: str, old_color: int) -> None:
    """A DIMENSION's ``*D`` block made to follow a new layer or colour.

    The headless renderer writes the block's lines, arrows and text on the
    dimension's layer and in its colour as they were when it drew them, so a
    dimension moved to another layer kept the old one's colour until a
    dimstyle change redrew it (round 3). What followed the dimension then
    follows it now; a colour its style sets (DIMCLRD / E / T) and the
    ``Defpoints`` points stay as they are.
    """
    if entity.dxftype() != "DIMENSION":
        return
    block = entity.get_geometry_block()
    if block is None:
        return
    layer, color = str(entity.dxf.get("layer", "0")), int(entity.dxf.get("color", 256))
    for part in block:
        if layer.lower() != old_layer.lower() and (
            str(part.dxf.get("layer", "0")).lower() == old_layer.lower()
        ):
            part.dxf.layer = layer
        if color != old_color and int(part.dxf.get("color", 256)) == old_color:
            part.dxf.color = color


def check_origin(value, name: str = "origin") -> tuple[float, float]:
    """A hatch pattern origin as ``(x, y)``, or ``invalid_args`` naming ``name``."""
    ok = (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        and all(math.isfinite(v) for v in value)
    )
    if not ok:
        raise InvalidArgsError(f"invalid_args: {name}: expected [x, y] numbers, got {value!r}")
    return float(value[0]), float(value[1])


def hatch_origin(hatch) -> tuple[float, float]:
    """The point a hatch's pattern is anchored at (AutoCAD's HATCH origin), WCS.

    DXF stores no origin: it is how far the first pattern line's base point
    sits from where the predefined pattern, at the hatch's own scale and angle,
    puts it - ``(0, 0)`` unless something moved it. A solid fill, or a pattern
    that is not a predefined one, has none to read: ``(0, 0)``.
    """
    from ezdxf.math import Vec2
    from ezdxf.tools import pattern as patterns

    lines = hatch.pattern.lines if hatch.pattern is not None else []
    measurement = hatch.doc.header.get("$MEASUREMENT", 1) if hatch.doc else 1
    table = patterns.ISO_PATTERN if measurement else patterns.IMPERIAL_PATTERN
    definition = table.get(str(hatch.dxf.pattern_name))
    if not lines or definition is None:
        return 0.0, 0.0
    home = Vec2(definition[0][1]).rotate_deg(float(hatch.dxf.pattern_angle))
    home = home * float(hatch.dxf.pattern_scale)
    at = lines[0].base_point - home
    return at.x, at.y


def set_hatch_origin(hatch, origin: tuple[float, float]) -> None:
    """Move the pattern's anchor to ``origin`` (WCS) by shifting every line's base point.

    The phase of the lines or dots moves; the boundary and the pattern's scale
    and angle do not. Every edit that rescales or rotates the pattern
    (``set_pattern_scale`` scales the base points about WCS ``(0, 0)``, taking
    an origin with it) reads :func:`hatch_origin` first and puts it back here.
    """
    from ezdxf.math import Vec2

    here = hatch_origin(hatch)
    shift = Vec2(origin[0] - here[0], origin[1] - here[1])
    if shift.magnitude < 1e-12:
        return
    for line in hatch.pattern.lines:
        line.base_point = line.base_point + shift


def ezdxf_write(entity, spec: SetSpec) -> None:
    """Write ``spec`` onto ``entity`` (a POLYLINE's vertices follow its layer,
    a DIMENSION's block its layer and colour)."""
    dxf = entity.dxf
    old_layer, old_color = str(dxf.get("layer", "0")), int(dxf.get("color", 256))
    if spec.layer is not None:
        dxf.layer = spec.layer
        if entity.dxftype() == "POLYLINE":
            for vertex in entity.vertices:
                vertex.dxf.layer = spec.layer
            if entity.seqend is not None:
                entity.seqend.dxf.layer = spec.layer
    if spec.color is not None:
        dxf.color = spec.color
        dxf.discard("true_color")  # a true colour overrides the ACI it was given
        dxf.discard("color_name")
    if spec.linetype is not None:
        dxf.linetype = spec.linetype
    if spec.lineweight is not None:
        dxf.lineweight = spec.lineweight
    if spec.visible is not None:
        dxf.invisible = 0 if spec.visible else 1
    ezdxf_dimension_follow(entity, old_layer, old_color)
    _write_typed(entity, spec)


def _write_typed(entity, spec: SetSpec) -> None:
    """The type-specific keys: a hatch's fill, a text's style, height, width."""
    kind, dxf = entity.dxftype(), entity.dxf
    if kind == "HATCH":
        scale = spec.pattern_scale or float(dxf.get("pattern_scale", 1.0) or 1.0)
        angle = spec.pattern_angle
        angle = float(dxf.get("pattern_angle", 0.0)) if angle is None else angle
        color = int(dxf.get("color", 256))
        origin = hatch_origin(entity)  # a rescale or rotation must not move the anchor
        if spec.pattern == "SOLID":
            entity.set_solid_fill(color=color, style=int(dxf.get("hatch_style", 0)))
        elif spec.pattern is not None:
            entity.set_pattern_fill(spec.pattern, color=color, scale=scale, angle=angle)
        elif str(dxf.get("pattern_name", "")).upper() != "SOLID":
            if spec.pattern_scale is not None:
                entity.set_pattern_scale(scale)
            if spec.pattern_angle is not None:
                entity.set_pattern_angle(angle)
        if entity.has_pattern_fill:
            set_hatch_origin(entity, origin)
    if kind in ("TEXT", "MTEXT"):
        if spec.text_style is not None:
            dxf.style = spec.text_style
            if kind == "TEXT" and spec.width_factor is None:
                # as entity_edit_text: a TEXT given a style takes its width factor
                style = entity.doc.styles.get(spec.text_style) if entity.doc else None
                if style is not None:
                    dxf.width = float(style.dxf.get("width", 1.0) or 1.0)
        if spec.text_height is not None:
            setattr(dxf, "height" if kind == "TEXT" else "char_height", spec.text_height)
    if kind == "TEXT" and spec.width_factor is not None:
        dxf.width = spec.width_factor


def ezdxf_containers(doc) -> dict[str, tuple[str, str]]:
    """Block-record handle -> ``("layout" | "block", name)``."""
    out: dict[str, tuple[str, str]] = {}
    for layout in doc.layouts:
        out[str(layout.block_record_handle)] = ("layout", layout.name)
    for block in doc.blocks:
        record = block.block_record
        if block.is_any_layout:
            continue
        out[str(record.dxf.handle)] = ("block", block.name)
    return out


def ezdxf_container_of(doc, entity, containers) -> tuple[str, str] | None:
    """The layout or block definition ``entity`` lives in, through its owners
    (an ATTRIB's INSERT, a VERTEX's POLYLINE); None when the chain breaks."""
    seen = 0
    current = entity
    while current is not None and seen < 16:
        owner = str(current.dxf.get("owner", "") or "")
        if owner in containers:
            return containers[owner]
        current = doc.entitydb.get(owner) if owner else None
        seen += 1
    return None


def ezdxf_layer_references(doc, keys: set[str]):
    """Every live entity whose ``layer`` is one of ``keys`` (lower-cased),
    from the entity database: layouts, block definitions, ATTRIBs, VERTEXes."""
    for entity in list(doc.entitydb.values()):
        if not entity.is_alive or entity.dxftype() == "LAYER":
            continue
        if entity.dxf.hasattr("layer") and str(entity.dxf.layer).lower() in keys:
            yield entity


def tally(doc, entities, containers) -> tuple[dict[str, int], list[str]]:
    """``(by_space, blocks_touched)`` for the entities about to be reassigned."""
    by_space: dict[str, int] = {}
    blocks: set[str] = set()
    for entity in entities:
        where = ezdxf_container_of(doc, entity, containers)
        if where is None:
            by_space["(unowned)"] = by_space.get("(unowned)", 0) + 1
        elif where[0] == "layout":
            by_space[where[1]] = by_space.get(where[1], 0) + 1
        else:
            blocks.add(where[1])
            by_space["(block definitions)"] = by_space.get("(block definitions)", 0) + 1
    return by_space, sorted(blocks)


BLOCK_NOTE = (
    "A block definition was edited: every INSERT of {names} shows the change, in every "
    "layout and in every block that nests it."
)


def block_note(blocks: list[str]) -> str | None:
    if not blocks:
        return None
    shown = ", ".join(blocks[:10]) + (f" (+{len(blocks) - 10} more)" if len(blocks) > 10 else "")
    return BLOCK_NOTE.format(names=shown)


def merge_target(target) -> str:
    if not isinstance(target, str) or not target.strip():
        raise _invalid("target", f"expected a layer name, got {target!r}")
    return target.strip()


def merge_sources(sources, target: str) -> tuple[str, ...]:
    """The validated, de-duplicated source names of a merge (refused by name)."""
    names = [sources] if isinstance(sources, str) else sources
    if not isinstance(names, (list, tuple)) or not names:
        raise _invalid("sources", f"expected a non-empty list of layer names, got {sources!r}")
    out: list[str] = []
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            raise _invalid(f"sources[{index}]", f"expected a layer name, got {name!r}")
        out.append(name.strip())
    unique = tuple(dict.fromkeys(out))
    if len({n.lower() for n in unique}) != len(unique):
        raise _invalid("sources", "names a layer twice (layer names ignore case)")
    if target.lower() in {n.lower() for n in unique}:
        raise ToolError(
            f"layer_merge: refused - the target {target!r} is also a source; a layer "
            "cannot merge into itself."
        )
    for name in unique:
        if name.lower() in PROTECTED_LAYERS:
            raise ToolError(
                f"layer_merge: refused - layer {name!r} cannot be merged away (AutoCAD keeps "
                "0 and DEFPOINTS in every drawing)."
            )
        if "|" in name:
            raise ToolError(
                f"layer_merge: refused - {name!r} is an xref-dependent layer; it belongs to "
                "the referenced file."
            )
    return unique


def refuse_missing(tool: str, missing: list[str], existing: Iterable[str]) -> None:
    if not missing:
        return
    known = sorted(existing, key=str.lower)
    shown = ", ".join(known[:20]) + (" ..." if len(known) > 20 else "")
    raise ToolError(
        f"{tool}: refused - no layer named {', '.join(repr(m) for m in missing)} in the "
        f"drawing; nothing was changed. Layers: {shown}"
    )


def rename_checks(old, new, existing: Iterable[str]) -> tuple[str, str]:
    """``(old as stored, new)``, or a refusal by name. ``existing`` is the
    layer table's names."""
    if not isinstance(old, str) or not old.strip():
        raise _invalid("old", f"expected a layer name, got {old!r}")
    if not isinstance(new, str) or not new.strip():
        raise _invalid("new", f"expected a layer name, got {new!r}")
    old = old.strip()
    names = {name.lower(): name for name in existing}
    if old.lower() in PROTECTED_LAYERS:
        raise ToolError(
            f"layer_rename: refused - layer {old!r} cannot be renamed (AutoCAD keeps 0 and "
            "DEFPOINTS under those names)."
        )
    if "|" in old:
        raise ToolError(
            f"layer_rename: refused - {old!r} is an xref-dependent layer; rename it in the "
            "referenced file."
        )
    if old.lower() not in names:
        refuse_missing("layer_rename", [old], names.values())
    new = check_layer_name(new, "layer_rename")
    if new.lower() in PROTECTED_LAYERS:
        raise ToolError(f"layer_rename: refused - {new!r} is a reserved layer name.")
    if new.lower() == old.lower():
        raise ToolError(
            f"layer_rename: refused - {old!r} and {new!r} differ only in case and layer "
            "names ignore case; nothing to rename."
        )
    if new.lower() in names:
        raise ToolError(
            f"layer_rename: refused - layer {names[new.lower()]!r} already exists; "
            "layer_merge moves one layer's entities onto another."
        )
    return names[old.lower()], new
