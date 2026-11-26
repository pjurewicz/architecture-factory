"""Creating geometry.

Split out of the single 1854-line ``AutoCADBackend`` ABC in v1.5.0 (M7).
``AutoCADBackend`` composes every contract in this package, so importing
``backends.base.AutoCADBackend`` is unchanged for callers.
"""

from __future__ import annotations

import functools
import inspect
import typing
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from backends.capability import capability

if TYPE_CHECKING:
    from backends.base import EntityInfo


@functools.cache
def batch_create_adapter():
    """The pydantic adapter of one ``entity_batch_create`` item.

    A union discriminated by ``type``: one model per type, built from that
    type's ``entity_create_*`` signature on this contract (the rectangle's from
    ``batch_create_spec``: the polyline's minus ``points`` / ``closed``, plus
    ``x1, y1, x2, y2``), each with ``lineweight`` and no other field. The tool
    validates every item with it before drawing and advertises its JSON
    schema, so the documented syntax and the accepted one cannot drift apart.
    """
    from typing import Annotated, Literal, Union

    from pydantic import ConfigDict, Field, TypeAdapter, create_model

    def params(target: str) -> dict:
        creator = getattr(EntityCreationContract, f"entity_create_{target}")
        hints = typing.get_type_hints(creator, localns={"EntityInfo": object})
        return {
            name: (hints[name], ... if p.default is p.empty else p.default)
            for name, p in inspect.signature(creator).parameters.items()
            if name != "self"
        }

    lineweight = (int | float | str | None, None)  # parse_lineweight names the spellings
    shapes = {target: params(target) for target in EntityCreationContract.BATCH_CREATE_TYPES}
    for alias, target in EntityCreationContract.BATCH_CREATE_ALIASES.items():
        # batch_create_spec rewrites x1, y1, x2, y2 into the target's points / closed
        base = {k: v for k, v in shapes[target].items() if k not in ("points", "closed")}
        shapes[alias] = {key: (float, ...) for key in ("x1", "y1", "x2", "y2")} | base
    models = [
        create_model(
            kind,
            __config__=ConfigDict(extra="forbid"),
            type=(Literal[kind], ...),
            lineweight=lineweight,
            **fields,
        )
        for kind, fields in shapes.items()
    ]
    return TypeAdapter(Annotated[Union[tuple(models)], Field(discriminator="type")])  # noqa: UP007


def inline_json_schema(schema: dict) -> dict:
    """A pydantic schema made self-contained so it can sit under a property.

    ``$ref`` into ``$defs`` is inlined, ``title`` dropped, an optional
    argument's ``anyOf [X, null]`` read as ``X`` and the OpenAPI
    ``discriminator`` removed: what stays is plain JSON Schema, a ``oneOf``
    whose branches tell themselves apart by their ``const`` ``type``.
    """
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
        options = node.get("anyOf")
        if isinstance(options, list) and {"type": "null"} in options and "default" in node:
            # an optional argument: its default stands for "not given", so null is noise
            kept = [option for option in options if option != {"type": "null"}]
            rest = {key: value for key, value in node.items() if key != "anyOf"}
            node = rest | (kept[0] if len(kept) == 1 else {"anyOf": kept})
        return {
            key: (walk(value) if key != "properties" else {k: walk(v) for k, v in value.items()})
            for key, value in node.items()
            if key not in ("$defs", "title", "discriminator")
        }

    return walk(schema)


class EntityCreationContract(ABC):
    # ``xdata={app: values}`` (v1.7.0, B17): these six create members write
    # XDATA in the create call itself - one lock, one undo step, no extra COM
    # round trip per entity - validated by
    # ``backends/xdata_specs.py::prepare_xdata`` before anything is created.
    # The mechanical generators tag every entity they draw this way.

    @abstractmethod
    async def entity_create_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        z1: float = 0.0,
        z2: float = 0.0,
        layer: str | None = None,
        color: int | None = None,
        linetype: str | None = None,
        xdata: dict[str, list] | None = None,
    ) -> EntityInfo: ...

    @abstractmethod
    async def entity_create_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        layer: str | None = None,
        color: int | None = None,
        xdata: dict[str, list] | None = None,
    ) -> EntityInfo: ...

    @abstractmethod
    async def entity_create_arc(
        self,
        cx: float,
        cy: float,
        radius: float,
        start_angle: float,
        end_angle: float,
        layer: str | None = None,
        color: int | None = None,
        xdata: dict[str, list] | None = None,
    ) -> EntityInfo: ...

    @abstractmethod
    async def entity_create_polyline(
        self,
        points: list[list[float]],
        closed: bool = False,
        layer: str | None = None,
        color: int | None = None,
        xdata: dict[str, list] | None = None,
        bulges: list[float] | None = None,
    ) -> EntityInfo:
        """``bulges`` (v1.8.0): one per vertex, WCS counter-clockwise positive,
        set in the same create call; validated by ``backends/vertices.py``
        before anything is drawn."""
        ...

    @abstractmethod
    async def entity_create_text(
        self,
        text: str,
        x: float,
        y: float,
        height: float = 2.5,
        rotation: float = 0.0,
        layer: str | None = None,
        color: int | None = None,
        xdata: dict[str, list] | None = None,
        align: str = "left",
    ) -> EntityInfo:
        """``align`` (1.8.0): ``"left"`` puts the baseline's left end at (x, y),
        ``"center"`` its middle (AutoCAD's Center justification)."""
        ...

    @abstractmethod
    async def entity_create_mtext(
        self,
        text: str,
        x: float,
        y: float,
        width: float = 100.0,
        height: float = 2.5,
        rotation: float = 0.0,
        layer: str | None = None,
        color: int | None = None,
    ) -> EntityInfo: ...

    @abstractmethod
    async def entity_create_table(
        self,
        x: float,
        y: float,
        rows: list[list[str]],
        headers: list[str] | None = None,
        column_widths: list[float] | None = None,
        row_height: float = 7.0,
        text_height: float = 2.5,
        title: str | None = None,
        layer: str = "TEXT",
        xdata: dict[str, list] | None = None,
    ) -> EntityInfo:
        """``xdata`` (v1.8.0) is written in the create call: on the native
        TABLE live, and on every child of the headless composite (its lines and
        cell texts), so a reader finds all of the table by its tag rather than
        only the handle the result names."""
        ...

    @abstractmethod
    async def leader_create_mleader(
        self,
        points: list[list[float]],
        text: str,
        text_height: float = 2.5,
        landing_gap: float = 1.0,
        arrow_size: float = 2.5,
        layer: str = "DIM",
    ) -> EntityInfo: ...

    @abstractmethod
    async def entity_create_hatch(
        self,
        pattern: str,
        boundary_points: list[list[float]],
        scale: float = 1.0,
        angle: float = 0.0,
        layer: str | None = None,
        color: int | None = None,
        xdata: dict[str, list] | None = None,
    ) -> EntityInfo: ...

    @abstractmethod
    async def entity_create_spline(
        self,
        fit_points: list[list[float]],
        layer: str | None = None,
        color: int | None = None,
    ) -> EntityInfo: ...

    @abstractmethod
    async def entity_create_ellipse(
        self,
        cx: float,
        cy: float,
        major_x: float,
        major_y: float,
        ratio: float = 0.5,
        layer: str | None = None,
        color: int | None = None,
    ) -> EntityInfo: ...

    @abstractmethod
    async def entity_create_point(
        self,
        x: float,
        y: float,
        layer: str | None = None,
        color: int | None = None,
    ) -> EntityInfo: ...

    @abstractmethod
    async def entity_create_block_ref(
        self,
        name: str,
        x: float,
        y: float,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        rotation: float = 0.0,
        layer: str | None = None,
    ) -> EntityInfo: ...

    # ── bulk tools (1.8-23) ──────────────────────────────────────────────────
    #
    # Concrete: the base implementation is the loop ``server.py`` ran up to
    # 1.8.0 - one ``entity_create_*`` / ``entity_delete`` per item - so the
    # headless results are what they were, byte for byte. ``ComBackend``
    # overrides both with ONE STA job (one guard, one refresh, per-item rows
    # carrying a ``kind``). Neither is atomic: an item that fails leaves the
    # earlier ones written; atomicity is ``cad_batch(transaction=True)``'s and,
    # for the delete tool, the server's transaction around this call.

    #: ``entity_batch_create``'s ``type`` -> the ``entity_create_*`` it calls.
    BATCH_CREATE_TYPES = (
        "line",
        "circle",
        "arc",
        "polyline",
        "text",
        "mtext",
        "point",
        "hatch",
        "spline",
        "ellipse",
    )

    #: Batch types drawn by another type's create: ``rectangle`` is the closed
    #: polyline ``entity_create_rectangle`` draws, from ``x1, y1, x2, y2``.
    BATCH_CREATE_ALIASES = {"rectangle": "polyline"}

    @staticmethod
    def batch_create_spec(kind: str, spec: dict) -> dict:
        """``spec`` as its create's arguments (an alias's rewritten)."""
        if kind != "rectangle":
            return spec
        spec = dict(spec)
        try:
            x1, y1, x2, y2 = (spec.pop(key) for key in ("x1", "y1", "x2", "y2"))
        except KeyError as exc:
            raise ValueError(f"rectangle needs x1, y1, x2, y2 (missing {exc})") from None
        spec["points"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        spec["closed"] = True
        return spec

    async def entity_batch_create(self, specs: list[dict], *, progress=None) -> dict:
        """Create every spec in ``specs``. ``{created, errors, entities}``.

        A spec is ``{"type": <BATCH_CREATE_TYPES or _ALIASES>, ...that create's
        arguments}`` plus an optional ``lineweight`` (refused by name before
        the create, so a bad one draws nothing). ``errors`` rows are
        ``{index, error}`` for a missing or unknown type and
        ``{index, type, error}`` for a create that raised; ``entities`` the
        created ``EntityInfo`` as dicts, in spec order. ``progress(i, total)``
        is awaited before each spec when given. The specs are not modified.
        """
        from dataclasses import asdict

        from backends.base import lineweight_name, parse_lineweight

        results: list[dict] = []
        errors: list[dict] = []
        total = len(specs)
        for index, raw in enumerate(specs):
            if progress is not None:
                await progress(index, total)
            spec = dict(raw)
            kind = spec.pop("type", None)
            if not kind:
                errors.append({"index": index, "error": "Missing 'type' key"})
                continue
            name = str(kind).lower()
            target = self.BATCH_CREATE_ALIASES.get(name, name)
            if target not in self.BATCH_CREATE_TYPES:
                errors.append({"index": index, "error": f"Unknown type: {kind}"})
                continue
            creator = getattr(self, f"entity_create_{target}")
            try:
                # Parsed before the create, so a refused lineweight draws nothing.
                weight = parse_lineweight(spec.pop("lineweight", None))
                spec = self.batch_create_spec(name, spec)
                info = await creator(**spec)
                if weight is not None:
                    await self.entity_set_properties(info.handle, lineweight=weight)
                    info.lineweight_name = lineweight_name(weight)
                results.append(asdict(info))
            except Exception as exc:
                errors.append({"index": index, "type": kind, "error": str(exc)})
        return {"created": len(results), "errors": errors, "entities": results}

    async def entity_delete_many(
        self, handles: list[str], *, progress=None, stop_on_error: bool = False
    ) -> dict:
        """Delete every handle in ``handles``. ``{deleted, errors}``.

        ``errors`` rows are ``{handle, error}`` (the live engine adds
        ``kind``). ``stop_on_error`` stops at the first failure - for a caller
        that rolls the whole call back anyway. ``progress(i, total)`` is
        awaited before each handle when given.
        """
        deleted = 0
        errors: list[dict] = []
        total = len(handles)
        for index, handle in enumerate(handles):
            if progress is not None:
                await progress(index, total)
            try:
                await self.entity_delete(handle)
                deleted += 1
            except Exception as exc:
                errors.append({"handle": handle, "error": str(exc)})
                if stop_on_error:
                    break
        return {"deleted": deleted, "errors": errors}

    # ── hatch depth (M8 / F4) ────────────────────────────────────────────────

    @abstractmethod
    async def hatch_set_gradient(
        self,
        handle: str,
        color1: list[int],
        color2: list[int],
        rotation: float = 0.0,
        centered: float = 0.0,
        one_color: bool = False,
        tint: float = 0.0,
        name: str = "LINEAR",
    ) -> dict:
        """Replace a hatch's fill with a two-colour gradient. ``{ok, handle, gradient}``."""
        ...

    @abstractmethod
    async def hatch_edit(
        self,
        handle: str,
        pattern: str = "",
        scale: float | None = None,
        angle: float | None = None,
        color: int | None = None,
        style: str = "",
    ) -> dict:
        """Edit an existing hatch in place. Returns ``{ok, handle, changed}``.

        ``changed`` names exactly the attributes that moved: an omitted
        parameter must leave its attribute alone, because a partial edit that
        quietly resets the rest is data loss. ``style`` is the island style —
        ``normal``, ``outer`` or ``ignore``.
        """
        ...

    @abstractmethod
    async def hatch_add_boundary(self, handle: str, edges: list[dict]) -> dict:
        """Add one boundary path built from typed edges.

        Returns ``{ok, handle, path_count, edge_types}``. Edges are
        ``{"type": "line"|"arc"|"ellipse", ...}``. Typed edges exist because a
        boundary API that only accepts vertex lists silently straightens every
        curve it is given.
        """
        ...

    # ── annotation objects (M8 / F15) ────────────────────────────────────────

    @capability(  # noqa: B027 — @capability supplies the body
        "wipeout",
        reason=(
            "ActiveX has no wipeout constructor — ModelSpace.AddWipeout does not exist "
            "(verified against AutoCAD 2026); WIPEOUT is a command only. Use the headless "
            "backend (AUTOCAD_MCP_BACKEND=ezdxf), or run WIPEOUT in AutoCAD directly."
        ),
    )
    async def entity_create_wipeout(
        self, points: list[list[float]], layer: str | None = None
    ) -> dict:
        """Mask the drawing behind a closed polygon. Returns ``{ok, handle, points}``.

        Refuses fewer than three points: a zero-area mask masks nothing while
        reporting success.
        """
        ...

    @capability(  # noqa: B027 — @capability supplies the body
        "revcloud",
        reason=(
            "ActiveX exposes no revision-cloud member and driving the REVCLOUD command "
            "blind against a live drawing is unverified. Use the headless backend "
            "(AUTOCAD_MCP_BACKEND=ezdxf), or run REVCLOUD in AutoCAD directly."
        ),
    )
    async def entity_create_revcloud(
        self,
        points: list[list[float]],
        segment_length: float,
        layer: str | None = None,
        closed: bool = True,
    ) -> dict:
        """Draw a revision cloud along a path. Returns ``{ok, handle, segments}``.

        A revision cloud *is* a polyline whose every segment carries an arc
        bulge; a segment length longer than the path yields no arcs at all, so
        that case is refused rather than returned as a rectangle.
        """
        ...
