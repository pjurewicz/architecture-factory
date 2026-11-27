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
"""

from __future__ import annotations

from engineering.mech.xdata import PAYLOAD_VERSION, to_values

APP_ID = "ACADMCP_ANNOT"
KINDS: tuple[str, ...] = ("annotation",)


def annotation_values(tool: str, handle: str) -> list[str]:
    """The chunk strings ``entity_set_xdata(handle, APP_ID, values)`` takes."""
    payload = {"v": PAYLOAD_VERSION, "kind": "annotation", "tool": str(tool), "id": str(handle)}
    return to_values(payload, app_id=APP_ID, kinds=KINDS)


async def tag_annotation(backend, handle: str, tool: str) -> None:
    """Write the record onto one entity the generator ``tool`` just drew."""
    await backend.entity_set_xdata(handle, APP_ID, annotation_values(tool, handle))
