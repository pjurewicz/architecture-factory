"""``ACADMCP_ANNOT``: the feature record on the lettering an annotation generator writes.

A level mark's ``±0.00``, a title-block field, a revision row, a sheet zone
letter, a parts-list cell and a callout note are text this server drew as part
of a symbol or an annotation, not a measurement typed in place of a DIMENSION.
Each such TEXT / MTEXT carries one small record, so a reader tells our
lettering from hand-typed text by the record and never by its layer or its
content:

    {"v": 1, "kind": "annotation", "tool": "arch_symbol", "id": "4A"}

``tool`` names the generator; ``id`` is the entity's own handle when it was
written, so two records are never equal (``text_overlap`` treats entities that
carry the same ``ACADMCP_*`` record as one item) and a copy is recognisable by
an ``id`` that is not its handle. The chunking is the mechanical codec's
(``engineering/mech/xdata.py``).

A generator that redraws its result in place (``arch_title_block``,
``arch_legend``) tags every entity it writes - lines, inserts and hatches too -
and adds ``"set": <name>``; :func:`set_handles` finds that set in one space so
a rerun deletes it before drawing. A record without ``set`` reads as
``"default"``.
"""

from __future__ import annotations

from engineering.mech.xdata import PAYLOAD_VERSION, to_values

APP_ID = "ACADMCP_ANNOT"
KINDS: tuple[str, ...] = ("annotation",)


DEFAULT_SET = "default"


def annotation_values(
    tool: str, handle: str, set_name: str | None = None, legend: str | None = None
) -> list[str]:
    """The chunk strings ``entity_set_xdata(handle, APP_ID, values)`` takes;
    ``legend`` is the label ``arch_legend`` gives the entity's row (a hatch)."""
    payload = {"v": PAYLOAD_VERSION, "kind": "annotation", "tool": str(tool), "id": str(handle)}
    if set_name is not None:
        payload["set"] = str(set_name)
    if legend:
        payload["legend"] = str(legend)
    return to_values(payload, app_id=APP_ID, kinds=KINDS)


async def tag_annotation(
    backend, handle: str, tool: str, set_name: str | None = None, legend: str | None = None
) -> None:
    """Write the record onto one entity the generator ``tool`` just drew."""
    await backend.entity_set_xdata(
        handle, APP_ID, annotation_values(tool, handle, set_name, legend)
    )


def read_annotation(entity) -> dict | None:
    """The entity's record, or None (no record, or one that does not decode)."""
    from engineering.mech.xdata import decode

    if not entity.has_xdata(APP_ID):
        return None
    try:
        return decode(list(entity.get_xdata(APP_ID)), app_id=APP_ID, kinds=KINDS)
    except ValueError:
        return None


def set_handles(space, tool: str, set_name: str) -> list[str]:
    """Handles of the entities in ``space`` (a layout) that ``tool`` drew as
    the set ``set_name`` (a record without ``set`` is ``"default"``)."""
    out = []
    for e in space:
        record = read_annotation(e)
        if (
            record
            and record.get("tool") == tool
            and str(record.get("set", DEFAULT_SET)) == set_name
        ):
            out.append(e.dxf.handle)
    return out


async def replace_set(backend, layout: str | None, tool: str, set_name: str) -> int:
    """Delete ``tool``'s set ``set_name`` from ``layout`` (None: model space);
    the number deleted."""

    def _find():
        doc = backend._require_doc()
        space = doc.modelspace() if layout is None else doc.layouts.get(layout)
        return set_handles(space, tool, set_name)

    handles = await backend._async(_find, readonly=True)
    if handles:
        await backend.entity_delete_many(handles)
    return len(handles)
