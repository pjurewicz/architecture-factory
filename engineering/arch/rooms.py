"""Rooms: labels that carry a measured area, and a reader for plans that are not ours.

A room's area is the floor its walls enclose. `label_room` never takes it from
the caller: it reads the wall faces off the drawing, finds the planar face the
label point lies in (`engineering.arch.faces`) and writes *that* area into the
label and into the ``room`` record of ``ACADMCP_ARCH`` - so a room schedule is
a read of the drawing, and the number on the sheet is the number the geometry
gives.

A door or a window cuts its wall, so the two rooms either side of a door would
read as one face through the gap. The opening records on the drawing close it:
for each opening, the wall's two face lines are continued across the gap - in
memory only, never drawn - which is the convention a net floor area is measured
to (the wall face, not the reveal). A plan without our records has no such
lines, and `rooms_detect` says so rather than guessing where a door was.

`rooms_detect` runs the same finder on any lines and lightweight polylines -
a foreign plan - and reports each face with a confidence, the way `pid_graph`
does: 1.0 when every edge of the face lies on our wall layer, 0.6 when any
edge is a plain line from somewhere else. It never modifies the drawing. What
it cannot read - an arc, an old-style POLYLINE, walls inside a block
reference - is listed in ``skipped`` with its handle, never dropped and never
approximated, so a room it could not close is a room the answer says it
missed, not a wrong area.

DECLARED, not standard:

* The label text heights (3.5 mm name, 2.5 mm number and area, on paper) are
  two values of the ISO 3098-1 series; which line gets which is this module's
  choice. The line pitch of 1.6 x the text height is this module's choice too.
* ``align="center"`` centres every line on the label point and the whole
  stack on it vertically; ``fit=True`` shrinks the stack (never grows it) so
  its estimated extent fits inside the room around the point, down to
  ``MIN_FIT_HEIGHT`` on paper. The width of a line is estimated, not measured:
  ``CHAR_WIDTH_FACTOR`` (0.9 text heights per character, generous for
  capitals) times its length; the result reports the factor applied and
  whether the label fits at it.
* ``mask=True`` (the default) draws a WIPEOUT under the label on its layer,
  so walls, hatches and grid lines drawn before it do not run through the
  text. Its rectangle is the texts' measured extent plus ``MASK_MARGIN``
  (0.25 x the smallest line height) all round, clipped to the room's face so
  it never hides a wall; a room with no face (an open room) gets the bare
  rectangle. The texts stay three TEXTs, the name carrying the room record.
* ``MIN_ROOM_WIDTH``: a face whose mean width (2 x area / perimeter) is under
  600 mm is taken for a wall body or an opening reveal, not a room. That is
  this reader's heuristic, not a building rule - a 200 mm wall ring measures
  200, a 900 mm corridor 5 m long measures 763.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from engineering.arch.faces import DEFAULT_TOL, Face, face_containing, planar_faces, tagged_faces
from engineering.arch.lang import LANGS, fmt_area_m2
from engineering.arch.layers import ARCH_ROLE_LAYER
from engineering.arch.model import (
    APP_ID,
    Opening,
    Room,
    RoomSeparator,
    Stair,
    Wall,
    WallPolygon,
    decode,
    from_payload,
    point_in_outline,
    room_from_dict,
    segment_of,
    to_payload,
    to_values,
)
from engineering.measure import polygon_area_perimeter
from engineering.mech.primitives import Pt, Text

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

__all__ = [
    "CONFIDENCE_FOREIGN",
    "CONFIDENCE_OURS",
    "DEFAULT_MIN_AREA",
    "ALIGNS",
    "LINE_PITCH",
    "MIN_ROOM_WIDTH",
    "NAME_HEIGHT",
    "TEXT_HEIGHT",
    "WALL_LAYER_KEYWORDS",
    "detect_rooms",
    "fit_factor",
    "label_room",
    "mean_width",
    "opening_closures",
    "read_records",
    "read_segments",
    "room_label_mask",
    "room_label_prims",
    "rooms_detect",
]

NAME_HEIGHT = 3.5  # mm on paper (ISO 3098-1 series value)
TEXT_HEIGHT = 2.5  # mm on paper (ISO 3098-1 series value)
LINE_PITCH = 1.6  # baseline step, x the height of the line being placed
MIN_FIT_HEIGHT = 1.8  # mm on paper: the smallest ISO 3098-1 series value fit shrinks to
#: Character advance in text heights, deliberately generous: room names are capitals,
#: and over-estimating only makes a fitted label smaller, never spill out of the room.
CHAR_WIDTH_FACTOR = 0.9
FIT_MARGIN = 0.9  # the share of the free width / height a fitted label may fill
MASK_MARGIN = 0.25  # the label mask's margin round the texts, x the smallest line height
ALIGNS: tuple[str, ...] = ("left", "center")
MIN_ROOM_WIDTH = 600.0  # mm; the wall-body heuristic above
DEFAULT_MIN_AREA = 1.0e6  # mm^2 = 1 m^2
CONFIDENCE_OURS = 1.0
CONFIDENCE_FOREIGN = 0.6
#: A layer whose name contains one of these (any case) is read by default.
WALL_LAYER_KEYWORDS = ("WALL", "DUVAR")
#: Entity types that bound a region with a curve; the finder reads straight
#: segments only, so these are reported as skipped, never approximated.
CURVE_TYPES = frozenset({"ARC", "CIRCLE", "ELLIPSE", "SPLINE"})
#: Polylines whose ``points`` are the vertices: LWPOLYLINE (POLYLINE is the
#: name AutoCAD's object model gives the same entity). A headless POLYLINE is the old-style one - it arrives with no
#: ``points`` at all, and is skipped by that test.
LIGHT_POLYLINE_TYPES = frozenset({"LWPOLYLINE", "POLYLINE"})
#: Old-style polylines, whose vertices do not reach this reader as ``points``.
#: Skipped, by name.
HEAVY_POLYLINE_TYPES = frozenset({"2DPOLYLINE", "3DPOLYLINE", "POLYFACEMESH", "POLYGONMESH"})
#: Block references (INSERT; BLOCKREFERENCE is AutoCAD's object-model name).
BLOCK_REFERENCE_TYPES = frozenset({"INSERT", "BLOCKREFERENCE", "MINSERTBLOCK"})
#: Annotation and fills: they sit on wall layers (a poche hatch, a wall tag)
#: without bounding anything, so they are passed over without a skipped entry.
#: Every DIMENSION kind (AutoCAD's object model names them ROTATEDDIMENSION, ...) is
#: matched by name below.
NON_BOUNDARY_TYPES = frozenset(
    {
        "TEXT",
        "MTEXT",
        "ATTDEF",
        "ATTRIBUTEDEFINITION",
        "ATTRIB",
        "POINT",
        "HATCH",
        "LEADER",
        "MLEADER",
        "MULTILEADER",
        "TOLERANCE",
        "FCF",
    }
)
_ARC_REASON = "the face finder reads straight segments only and does not approximate an arc"


def _check_lang(lang: str) -> str:
    if lang not in LANGS:
        raise ValueError(
            f"lang: {lang!r} is not a label language; languages are {', '.join(LANGS)}"
        )
    return lang


def _positive(value, where: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: expected a number, got {value!r}") from None
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{where}: must be a finite number greater than zero, got {value!r}")
    return number


def _label_lines(
    room: Room, lang: str, factor: float, with_area: bool = True
) -> list[tuple[str, float]]:
    lines = [(room.name, NAME_HEIGHT * factor)]
    if room.number:
        lines.append((str(room.number), TEXT_HEIGHT * factor))
    if with_area:
        lines.append((fmt_area_m2(room.area, lang), TEXT_HEIGHT * factor))
    return lines


def _block_height(lines) -> float:
    """From the top of the first line to the last baseline."""
    return lines[0][1] + sum(LINE_PITCH * height for _text, height in lines[1:])


def _check_align(align: str) -> str:
    if align not in ALIGNS:
        raise ValueError(
            f"align: {align!r} is not an alignment; alignments are {', '.join(ALIGNS)}"
        )
    return align


def room_label_prims(
    room: Room,
    *,
    lang: str = "en",
    scale=50,
    align: str = "left",
    shrink: float = 1.0,
    with_area: bool = True,
) -> tuple[Text, ...]:
    """Name, number (when there is one) and measured area, stacked down
    (``with_area=False``: no area line, for a room with no closed face).

    Heights are paper millimetres times the plot-scale denominator (at 1:50 the
    3.5 mm name is 175 drawing units high), times ``shrink``. ``align="left"``:
    the first baseline starts at ``room.at`` and each next one is 1.6 x its own
    height lower. ``align="center"``: every line is centred on ``room.at``'s x
    and the stack's middle is at its y.
    """
    _check_lang(lang)
    _check_align(align)
    factor = _positive(scale, "scale") * _positive(shrink, "shrink")
    x, y = float(room.at[0]), float(room.at[1])
    lines = _label_lines(room, lang, factor, with_area)
    baseline = y
    if align == "center":
        baseline = y + _block_height(lines) / 2.0 - lines[0][1]
    prims: list[Text] = []
    for index, (text, height) in enumerate(lines):
        if index:
            baseline -= LINE_PITCH * height
        prims.append(
            Text(
                at=(x, baseline),
                text=text,
                height=height,
                rotation=0.0,
                role="room",
                align=align,
            )
        )
    return tuple(prims)


def room_label_mask(boxes, face: Face | None = None, *, margin: float) -> dict:
    """The polygons a label's WIPEOUTs cover: the texts' extent plus ``margin``, inside the room.

    ``boxes`` are the texts' measured extents, ``((x0, y0), (x1, y1))`` each.
    With a ``face`` the rectangle is clipped to its outer loop, so the mask
    never hides a wall face - one polygon per piece left. A hole of the face
    (a column) under the rectangle cannot be left visible by a WIPEOUT clipped
    this way, so then there is no mask and ``omitted`` says why.
    ``{"polygons": [[[x, y], ...], ...], "clipped": bool, "omitted": str | None}``.
    """
    from ezdxf.math import Vec2
    from ezdxf.math.clipping import ConcaveClippingPolygon2d

    x0 = min(box[0][0] for box in boxes) - margin
    y0 = min(box[0][1] for box in boxes) - margin
    x1 = max(box[1][0] for box in boxes) + margin
    y1 = max(box[1][1] for box in boxes) + margin
    rect = [Vec2(x0, y0), Vec2(x1, y0), Vec2(x1, y1), Vec2(x0, y1)]
    if face is None:
        return {"polygons": [[[p.x, p.y] for p in rect]], "clipped": False, "omitted": None}
    for hole in face.holes:
        if ConcaveClippingPolygon2d([Vec2(p) for p in hole]).clip_polygon(rect):
            return {
                "polygons": [],
                "clipped": False,
                "omitted": "a hole of the room face (a column) lies under the label, and a "
                "WIPEOUT over it would hide it; move the label point to show a mask",
            }
    pieces = ConcaveClippingPolygon2d([Vec2(p) for p in face.loop]).clip_polygon(rect)
    polygons = []
    for piece in pieces:
        points = list(piece)
        if len(points) > 1 and points[0].isclose(points[-1]):
            points.pop()
        if len(points) >= 3:
            polygons.append([[p.x, p.y] for p in points])
    clipped = (
        len(polygons) != 1
        or len(polygons[0]) != 4
        or any(not any(Vec2(q).isclose(r) for r in rect) for q in polygons[0])
    )
    return {"polygons": polygons, "clipped": clipped, "omitted": None}


async def _text_extents(backend: AutoCADBackend, prims: tuple[Text, ...]) -> list:
    """Each text's extent as the backend will draw it, measured before it is drawn.

    The texts are laid out in a scratch ezdxf document with the drawing's
    current text style font (the style ``entity_create_text`` gives a new TEXT)
    and measured with ``ezdxf.bbox`` - the measurement ``entity_get`` reports
    for the drawn text. Nothing is written to the drawing.
    """
    import ezdxf
    from ezdxf import bbox
    from ezdxf.enums import TextEntityAlignment

    style = next((row for row in await backend.textstyle_list() if row["current"]), None)
    scratch = ezdxf.new()
    if style and style["font"]:
        scratch.styles.new("LABEL", dxfattribs={"font": style["font"]})
    msp = scratch.modelspace()
    boxes = []
    for prim in prims:
        text = msp.add_text(
            prim.text,
            dxfattribs={
                "style": "LABEL" if style and style["font"] else "Standard",
                "height": prim.height,
                "rotation": prim.rotation,
                "insert": prim.at,
            },
        )
        if prim.align != "left":
            text.set_placement(prim.at, align=TextEntityAlignment[prim.align.upper()])
        box = bbox.extents([text])
        boxes.append(((box.extmin.x, box.extmin.y), (box.extmax.x, box.extmax.y)))
    return boxes


async def _draw_label_prims(
    backend: AutoCADBackend, prims: tuple[Text, ...], face: Face | None, *, mask: bool
) -> dict:
    """The label: (``mask``) its WIPEOUTs first, then its texts.

    The mask is created before the texts, so it sits under them both in the
    entity order and in the draw order - a viewer that ignores the
    SORTENTSTABLE still shows the texts on top. The first WIPEOUT in a drawing
    without wipeout variables sets ``WIPEOUTFRAME`` to 0 (no frame on screen
    or on a plot); a drawing that already sets it keeps its choice. ``handles``
    are the texts, the name first (the room record goes on it).
    """
    from engineering.mech.draw import (  # lazy: mech.draw imports the arch layer set
        _ensure_layers,
        draw_prims,
    )

    shape = None
    masks: list[str] = []
    if mask:
        shape = room_label_mask(
            await _text_extents(backend, prims),
            face,
            margin=MASK_MARGIN * min(p.height for p in prims),
        )
        layer = ARCH_ROLE_LAYER["room"]
        await _ensure_layers(backend, [layer])
        for polygon in shape["polygons"]:
            made = await backend.entity_create_wipeout(polygon, layer)
            if not made.get("ok"):
                raise ValueError(f"room label mask: {made.get('error')}")
            masks.append(made["handle"])
    drawn = await draw_prims(backend, prims, role_layer=ARCH_ROLE_LAYER)
    return {
        "handles": list(drawn["handles"]),
        "mask": None
        if shape is None
        else {"handles": masks, "clipped": shape["clipped"], "omitted": shape["omitted"]},
    }


def _chord(loops, point: Pt, axis: int) -> tuple[float, float]:
    """The free run through ``point`` along ``axis`` (0 = x) before any edge of ``loops``."""
    lo, hi = -math.inf, math.inf
    other = 1 - axis
    for loop in loops:
        for i in range(len(loop)):
            p, q = loop[i], loop[(i + 1) % len(loop)]
            if (p[other] > point[other]) == (q[other] > point[other]):
                continue
            k = (point[other] - p[other]) / (q[other] - p[other])
            hit = p[axis] + k * (q[axis] - p[axis])
            if hit <= point[axis]:
                lo = max(lo, hit)
            else:
                hi = min(hi, hit)
    return lo, hi


def fit_factor(room: Room, face: Face, *, lang: str, scale, align: str) -> dict:
    """How much to shrink the label so its estimated extent fits the face around ``room.at``.

    ``{"factor", "fits", "estimated": True}``: ``factor`` is at most 1 and at
    least ``MIN_FIT_HEIGHT / NAME_HEIGHT``; ``fits`` is False when even the
    smallest factor leaves the estimate outside the room.
    """
    factor = _positive(scale, "scale")
    lines = _label_lines(room, lang, factor)
    width = max(CHAR_WIDTH_FACTOR * height * len(text) for text, height in lines)
    block = _block_height(lines)
    x, y = float(room.at[0]), float(room.at[1])
    loops = (face.loop, *face.holes)
    left, right = _chord(loops, (x, y), 0)
    below, above = _chord(loops, (x, y), 1)
    if align == "center":
        limits = (2.0 * min(x - left, right - x) / width, 2.0 * min(y - below, above - y) / block)
    else:
        # the stack runs from the name's top, one name height above the point,
        # down to the last baseline
        top = lines[0][1]
        limits = ((right - x) / width, (above - y) / top, (y - below) / (block - top))
    wanted = min(1.0, *(FIT_MARGIN * limit for limit in limits))
    floor = MIN_FIT_HEIGHT / NAME_HEIGHT
    return {"factor": max(wanted, floor), "fits": wanted >= floor, "estimated": True}


def _face_offsets(wall: Wall) -> tuple[float, float]:
    """Signed distances of the wall's two faces from its axis, + to the left."""
    t = float(wall.thickness)
    if wall.justification == "left":
        return (0.0, t)
    if wall.justification == "right":
        return (-t, 0.0)
    return (-t / 2.0, t / 2.0)


def opening_closures(walls, openings) -> tuple[tuple[tuple[Pt, Pt], ...], tuple[dict, ...]]:
    """The wall-face lines continued across every opening, and what could not be closed.

    For an opening at ``offset`` of ``width`` on its wall, the segment of the
    axis it sits on gives the direction ``u`` and the left normal ``n``; each
    face line runs from ``start + n*d`` to ``start + u*width + n*d`` for the
    face's signed offset ``d``. An opening whose wall has no record is returned
    in ``omitted`` - its gap stays open - never guessed. When the wall engine
    can rebuild the network, its own cut lines are used instead: a corner
    window's gap ends on the mitre line, not square to the axis, and only the
    engine knows where that is.
    """
    from engineering.arch.walls import wall_geometry

    by_id = {wall.id: wall for wall in walls}
    closures: list[tuple[Pt, Pt]] = []
    omitted: list[dict] = []
    cut_rows: dict[str, dict] = {}
    hosted = [opening for opening in openings if opening.wall in by_id]
    if walls:
        try:
            geometry = wall_geometry(list(walls), hosted, poche=False)
            cut_rows = {cut["element"]: cut for cut in geometry.cuts}
        except ValueError:
            cut_rows = {}  # a stale record: fall back on each opening's own numbers
    for opening in openings:
        cut = cut_rows.get(opening.id)
        if cut is not None:
            closures.extend((tuple(cut["right"]), tuple(cut["left"])))
            continue
        wall = by_id.get(opening.wall)
        if wall is None:
            omitted.append(
                {
                    "element": opening.id,
                    "reason": f"host wall {opening.wall!r} has no record on this drawing; "
                    "its gap is left open",
                }
            )
            continue
        try:
            index, along = segment_of(wall, float(opening.offset))
        except ValueError as exc:
            # a stale record (its wall was shortened since): the reader is
            # read-only and answers for every other room, so it says so here
            omitted.append({"element": opening.id, "reason": f"{exc}; its gap is left open"})
            continue
        axis = [(float(p[0]), float(p[1])) for p in wall.axis]
        p0, p1 = axis[index], axis[(index + 1) % len(axis)]
        length = math.dist(p0, p1)
        ux, uy = (p1[0] - p0[0]) / length, (p1[1] - p0[1]) / length
        nx, ny = -uy, ux
        sx, sy = p0[0] + ux * along, p0[1] + uy * along
        ex, ey = sx + ux * float(opening.width), sy + uy * float(opening.width)
        for d in _face_offsets(wall):
            closures.append(((sx + nx * d, sy + ny * d), (ex + nx * d, ey + ny * d)))
    return tuple(closures), tuple(omitted)


def is_wall_outline(face: Face, outlines) -> bool:
    """True when ``face`` is the inside of a wall outline (a wall polygon is solid wall).

    The face finder sees a wall polygon as one more closed face; it is one when
    its area is the outline's (within 0.1 %) and its centroid lies inside it.
    Only ``wall_polygon`` records are passed in: a plain closed polyline on a
    wall layer may be the inner face of a double-line wall, i.e. a room, and a
    solid one (a column, a wall body) is already thinner than
    ``MIN_ROOM_WIDTH``.
    """
    for outline in outlines:
        ring = [(float(x), float(y)) for x, y in outline]
        area, _perimeter = polygon_area_perimeter([(x, y, 0.0) for x, y in ring], True)
        if area > 0.0 and abs(face.area - area) <= 1e-3 * area:
            if face_containing(
                (Face(loop=tuple(ring), area=area, centroid=face.centroid),), face.centroid
            ):
                return True
    return False


def mean_width(face: Face) -> float:
    """2 x area / perimeter: the thickness of a long thin face, holes included."""
    perimeter = sum(
        polygon_area_perimeter([(x, y, 0.0) for x, y in ring], True)[1]
        for ring in (face.loop, *face.holes)
    )
    return 2.0 * face.area / perimeter if perimeter > 0.0 else 0.0


def detect_rooms(
    segments,
    *,
    min_area: float = DEFAULT_MIN_AREA,
    tol: float = DEFAULT_TOL,
    min_width: float = MIN_ROOM_WIDTH,
) -> tuple[dict, ...]:
    """Faces of at least ``min_area`` mm^2 and ``min_width`` mean width, with a confidence.

    ``segments`` are ``(p1, p2)`` or ``(p1, p2, layer)``. A face scores 1.0
    when every edge of it (holes included) is covered by a segment on
    ``ARCH_ROLE_LAYER["wall"]`` or ``["room_separator"]``, and 0.6 otherwise - an untagged segment is a
    plain line from somewhere else. The minimum of the inputs, never raised by
    agreement: one foreign edge is enough to make the face 0.6.
    """
    min_area = _positive(min_area, "min_area")
    min_width = _positive(min_width, "min_width")
    ours_layers = {ARCH_ROLE_LAYER["wall"].casefold(), ARCH_ROLE_LAYER["room_separator"].casefold()}
    rooms: list[dict] = []
    for face, tags in tagged_faces(segments, tol=tol):
        if face.area < min_area or mean_width(face) < min_width:
            continue
        ours = all(
            any(isinstance(t, str) and t.casefold() in ours_layers for t in edge) for edge in tags
        )
        rooms.append(
            {
                "loop": face.loop,
                "holes": face.holes,
                "area": face.area,
                "centroid": face.centroid,
                "confidence": CONFIDENCE_OURS if ours else CONFIDENCE_FOREIGN,
            }
        )
    return tuple(rooms)


async def read_records(backend: AutoCADBackend, *, space=None) -> dict:
    """Every ``ACADMCP_ARCH`` record on the drawing's architectural layers.

    The shape `read_plan` (Task 4) returns - ``walls``, ``openings``, ``stairs``,
    ``rooms`` as model dataclasses - plus ``room_handles`` (room id -> the
    handle of the label that carries it). The current space (or ``space``) is read
    once, not sampled and not once per layer: a record past the 200th entity is
    still a record, and the drawing is scanned once rather than once for each
    of the twelve arch layers.
    Entities on the ``ARCH_ROLE_LAYER`` layers (any case) are read in layer-name
    order, drawing order within a layer. XDATA that is not ours or does not
    decode is skipped - this reader never guesses a record - and counted in
    ``unreadable``. A record that appears on several entities (a wall on each
    of its outlines) is kept once, by kind and id.
    """
    from engineering.sheet.bom import all_entities  # lazy: sheet imports stay out of arch

    found: dict[str, list] = {
        "walls": [],
        "openings": [],
        "stairs": [],
        "rooms": [],
        "wall_polygons": [],
        "room_separators": [],
    }
    room_handles: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    unreadable = 0
    buckets = {
        Wall: "walls",
        Opening: "openings",
        Stair: "stairs",
        Room: "rooms",
        WallPolygon: "wall_polygons",
        RoomSeparator: "room_separators",
    }
    arch_layers = {name.casefold(): name for name in ARCH_ROLE_LAYER.values()}
    ours = [
        entity
        for entity in await all_entities(backend, space=space)
        if str(entity.layer or "").casefold() in arch_layers
    ]
    ours.sort(key=lambda entity: arch_layers[str(entity.layer).casefold()])  # stable
    for entity in ours:
        raw = await backend.entity_get_xdata(entity.handle, APP_ID)
        values = (raw.get("xdata") or {}).get(APP_ID) or []
        if not values:
            continue
        try:
            obj = from_payload(decode(values))
        except (ValueError, KeyError, TypeError):
            obj = None
        bucket = buckets.get(type(obj))
        if bucket is None:
            unreadable += 1
            continue
        key = (bucket, obj.id)
        if key in seen:
            continue
        seen.add(key)
        found[bucket].append(obj)
        if bucket == "rooms":
            room_handles[obj.id] = entity.handle
    return {**found, "room_handles": room_handles, "unreadable": unreadable}


async def read_segments(backend: AutoCADBackend, layers, *, space=None) -> tuple[list, list[dict]]:
    """Straight segments on ``layers`` as ``(p1, p2, layer)``, and what was skipped.

    Read: LINE, and the lightweight polyline (LWPOLYLINE) by its ``points``. Everything else that can
    bound a room is listed in the second return value with its handle, type and
    reason - never dropped, never guessed:

    * an ARC/CIRCLE/ELLIPSE/SPLINE or a bulged polyline edge is an arc, and the
      finder reads straight segments only - it is never flattened into chords;
    * an old-style POLYLINE (``HEAVY_POLYLINE_TYPES``; a headless POLYLINE
      reports no vertices) is not read, because its vertices do not reach this
      reader;
    * a block reference is not exploded: walls inside a block are not read;
    * any other type is listed as not a line.

    Text, dimensions, leaders, points and hatches (``NON_BOUNDARY_TYPES``)
    bound nothing and are passed over without an entry.
    """
    from engineering.sheet.bom import all_entities  # lazy: sheet imports stay out of arch

    segments: list = []
    skipped: list[dict] = []

    def skip(entity, kind: str, reason: str) -> None:
        skipped.append({"handle": entity.handle, "type": kind, "reason": reason})

    for layer in layers:
        for entity in await all_entities(backend, layer=layer, space=space):
            props = entity.properties or {}
            kind = str(entity.type).upper()
            if kind in NON_BOUNDARY_TYPES or "DIMENSION" in kind:
                continue
            if kind == "LINE" and "start" in props and "end" in props:
                segments.append((tuple(props["start"][:2]), tuple(props["end"][:2]), entity.layer))
            elif kind in LIGHT_POLYLINE_TYPES and props.get("points"):
                points = [tuple(p[:2]) for p in props["points"]]
                bulges = list(props.get("bulges") or [])
                count = len(points)
                arcs = 0
                for i in range(count if props.get("closed") else count - 1):
                    if i < len(bulges) and abs(float(bulges[i])) > 1e-12:
                        arcs += 1
                        continue
                    segments.append((points[i], points[(i + 1) % count], entity.layer))
                if arcs:
                    skip(entity, kind, f"{arcs} arc segment(s): {_ARC_REASON}")
            elif kind in CURVE_TYPES:
                skip(entity, kind, f"a curve: {_ARC_REASON}")
            elif kind in HEAVY_POLYLINE_TYPES or kind == "POLYLINE":
                skip(
                    entity,
                    kind,
                    "an old-style polyline: its vertices are not read reliably by this "
                    "reader on either engine, so its edges are not counted; convert it to a "
                    "lightweight polyline (CONVERTPOLY) or explode it to lines",
                )
            elif kind in BLOCK_REFERENCE_TYPES:
                skip(
                    entity,
                    kind,
                    "a block reference: walls inside a block are not read; explode it "
                    "(block_explode) to count them",
                )
            else:
                skip(
                    entity,
                    kind,
                    "not a LINE or a lightweight polyline: the face finder does not read it",
                )
    return segments, skipped


def _wcs_loop(loop) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in loop]


async def wall_layers(backend: AutoCADBackend) -> list[str]:
    """The arch wall layer, every other layer named ``A-WALL*`` but the poche layer,
    and the room separator layer (a separator bounds a room like a wall face).

    That is where a drafter puts what the wall engine does not draw - a bay, a
    column, a wall imported from another model - as plain closed polylines.
    """
    wall = ARCH_ROLE_LAYER["wall"]
    hatch = ARCH_ROLE_LAYER["wall_hatch"].casefold()
    separator = ARCH_ROLE_LAYER["room_separator"]
    names = [wall]
    for layer in await backend.layer_list():
        name = layer.name
        folded = name.casefold()
        if folded.startswith("a-wall") and folded != hatch and folded != wall.casefold():
            names.append(name)
        elif folded == separator.casefold():
            names.append(name)
    return names


async def label_room(
    backend: AutoCADBackend,
    room_dict: dict,
    *,
    lang: str = "en",
    scale=50,
    align: str = "left",
    fit: bool = False,
    open_room: bool = False,
    mask: bool = True,
) -> dict:
    """Label the room around ``room_dict["at"]`` with its name, number and measured area.

    ``open_room=True`` (``arch_plan_from_spec``'s rooms): a point that lies in
    no closed face - an open living room - is labelled with its name and number
    only, without an area and without a room record (an unmeasured room must
    not enter a schedule total), and the result carries ``area: None`` and a
    ``warning`` with code ``room_open``. Every other refusal still refuses.

    ``mask`` (default True) draws a WIPEOUT under the label, clipped to the
    room (see the module docstring); the result's ``mask`` is
    ``{handles, clipped, omitted}``, or None with ``mask=False``.

    Refused before anything is drawn: an ``area`` in the request (areas are
    measured, never typed), an unknown ``lang``, a non-positive ``scale``, an
    ``id`` already labelled on the drawing, a point that lies in no closed face
    of the wall layer, a point inside a wall body, and a point in a face that
    already holds a room record (one face, one room: a second record would be
    counted twice by every schedule total). ``id`` defaults to the first free
    ``R<n>``.
    """
    if not isinstance(room_dict, dict):
        raise ValueError(
            f"room: expected an object with name and at, got {type(room_dict).__name__}"
        )
    if "area" in room_dict:
        raise ValueError(
            "room.area: a room's area is measured from its walls, never typed; remove it"
        )
    _check_lang(lang)
    _check_align(align)
    if not isinstance(fit, bool):
        raise ValueError(f"fit: expected true or false, got {fit!r}")
    if not isinstance(mask, bool):
        raise ValueError(f"mask: expected true or false, got {mask!r}")
    _positive(scale, "scale")
    records = await read_records(backend)
    taken = {room.id for room in records["rooms"]}
    data = dict(room_dict)
    data.setdefault("number", None)
    if not data.get("id"):
        number = len(taken) + 1
        while f"R{number}" in taken:
            number += 1
        data["id"] = f"R{number}"
    room = room_from_dict(data, "room")
    if room.id in taken:
        raise ValueError(
            f"room.id: {room.id!r} is already labelled on this drawing "
            f"(label {records['room_handles'][room.id]}); choose another id"
        )

    x, y = float(room.at[0]), float(room.at[1])
    if room.boundary is not None:
        return await _label_bounded_room(
            backend, room, records, lang=lang, scale=scale, align=align, fit=fit, mask=mask
        )

    wall_layer = ARCH_ROLE_LAYER["wall"]
    layers = await wall_layers(backend)
    segments, skipped = await read_segments(backend, layers)
    closures, omitted = opening_closures(records["walls"], records["openings"])
    faces = planar_faces(
        [*segments, *((p1, p2, wall_layer) for p1, p2 in closures)], tol=DEFAULT_TOL
    )
    face = face_containing(faces, (x, y))
    if face is None:
        ends = _open_ends_near(records, (x, y))
        where = (
            f"room.at: ({x:g}, {y:g}) lies in no closed face of the {len(segments)} "
            f"segment(s) on {', '.join(layers)}"
        )
        if not open_room:
            raise ValueError(
                f"{where}; {_open_ends_text(ends)}close the walls around it first (arch_wall), "
                "or read a plan drawn by other means with arch_rooms_detect"
            )
        return await _label_open_room(
            backend,
            room,
            where,
            ends,
            lang=lang,
            scale=scale,
            align=align,
            layers=layers,
            mask=mask,
        )
    polygon_ids = [p.id for p in records["wall_polygons"]]
    if is_wall_outline(face, [p.points for p in records["wall_polygons"]]):
        raise ValueError(
            f"room.at: ({x:g}, {y:g}) lies inside a wall polygon (one of "
            f"{', '.join(polygon_ids)}), which is solid wall, not a room"
        )
    width = mean_width(face)
    if width < MIN_ROOM_WIDTH:
        raise ValueError(
            f"room.at: ({x:g}, {y:g}) lies inside a wall body (a face {width:.0f} mm wide on "
            f"average, under the {MIN_ROOM_WIDTH:.0f} mm a room is taken to need), not in a room"
        )
    # One face is one room: a second record on it would be a second area in
    # every schedule total (a retried call after a timeout is the usual way in).
    _refuse_labelled(
        room,
        records,
        [
            r
            for r in records["rooms"]
            if (
                point_in_outline(room.at, r.boundary)
                if r.boundary
                else face_containing(faces, r.at) is face
            )
        ],
    )
    return {
        **await _draw_label(
            backend,
            replace(room, area=face.area),
            face,
            lang=lang,
            scale=scale,
            align=align,
            fit=fit,
            mask=mask,
        ),
        "area_source": "walls",
        "wall_layers": layers,
        "loop": _wcs_loop(face.loop),
        "holes": [_wcs_loop(hole) for hole in face.holes],
        "wall_segments": len(segments),
        "closures": len(closures),
        "skipped": skipped,
        "omitted": list(omitted),
    }


def _refuse_labelled(room: Room, records: dict, already: list) -> None:
    if not already:
        return
    held = ", ".join(f"{r.id} {r.name!r} (label {records['room_handles'][r.id]})" for r in already)
    raise ValueError(
        f"room.at: ({room.at[0]:g}, {room.at[1]:g}) lies in a room that is already labelled - "
        f"{held}; one face carries one room record, so its area is counted once. Delete that "
        "label to relabel the room, or place this one in another room"
    )


async def _draw_label(
    backend: AutoCADBackend, measured: Room, face: Face, *, lang, scale, align, fit, mask
) -> dict:
    """The label of a measured room and its record on the name text."""
    fitted = (
        fit_factor(measured, face, lang=lang, scale=scale, align=align)
        if fit
        else {"factor": 1.0, "fits": None, "estimated": True}
    )
    drawn = await _draw_label_prims(
        backend,
        room_label_prims(measured, lang=lang, scale=scale, align=align, shrink=fitted["factor"]),
        face,
        mask=mask,
    )
    handle = drawn["handles"][0]
    await backend.entity_set_xdata(handle, APP_ID, to_values(to_payload(measured)))
    return {
        "ok": True,
        "room": {
            "id": measured.id,
            "name": measured.name,
            "number": measured.number,
            "at": [float(measured.at[0]), float(measured.at[1])],
            "area": measured.area,
            "area_m2": measured.area / 1.0e6,
            "area_text": fmt_area_m2(measured.area, lang),
        },
        "handle": handle,
        "handles": list(drawn["handles"]),
        "mask": drawn["mask"],
        "layer": ARCH_ROLE_LAYER["room"],
        "align": align,
        "fit": {**fitted, "name_height_mm": NAME_HEIGHT * fitted["factor"]} if fit else None,
    }


async def _label_bounded_room(
    backend: AutoCADBackend, room: Room, records: dict, *, lang, scale, align, fit, mask
) -> dict:
    """A room with its own ``boundary``: the area is that polygon's, the walls are not read.

    One room per region still holds: refused when an existing room's label
    point lies inside this boundary, or this label point inside an existing
    room's boundary.
    """
    _refuse_labelled(
        room,
        records,
        [
            r
            for r in records["rooms"]
            if point_in_outline(r.at, room.boundary)
            or (r.boundary and point_in_outline(room.at, r.boundary))
        ],
    )
    loop = tuple(room.boundary)
    area, _perimeter = polygon_area_perimeter([(px, py, 0.0) for px, py in loop], True)
    face = Face(loop=loop, area=area, centroid=tuple(room.at))
    return {
        **await _draw_label(
            backend,
            replace(room, area=area),
            face,
            lang=lang,
            scale=scale,
            align=align,
            fit=fit,
            mask=mask,
        ),
        "area_source": "boundary",
        "loop": _wcs_loop(loop),
        "holes": [],
    }


def _open_ends_near(records: dict, point: Pt) -> list[dict]:
    """`open_ends` of the drawing's walls and separators, nearest to ``point`` first."""
    from engineering.arch.critique import open_ends  # lazy: critique imports the QA scan

    ends = open_ends(records["walls"], records["room_separators"])
    return sorted(ends, key=lambda row: math.dist(row["at"], point))


def _open_ends_text(ends: list[dict]) -> str:
    if not ends:
        return ""
    shown = "; ".join(
        f"{row['element']}.{row['end']} at ({row['at'][0]:g}, {row['at'][1]:g})"
        + (
            " (faces " + ", ".join(f"({fx:g}, {fy:g})" for fx, fy in row["faces"]) + ")"
            if row.get("faces")
            else ""
        )
        for row in ends[:2]
    )
    return (
        f"the nearest free end(s): {shown} - close the opening with a room_separator "
        "between the ends that bound this room, or give the room a boundary; "
    )


async def _label_open_room(
    backend: AutoCADBackend,
    room: Room,
    where: str,
    ends: list[dict],
    *,
    lang,
    scale,
    align,
    layers,
    mask,
) -> dict:
    """``label_room(open_room=True)`` for a point in no closed face: name and number, no area."""
    drawn = await _draw_label_prims(
        backend,
        room_label_prims(room, lang=lang, scale=scale, align=align, with_area=False),
        None,
        mask=mask,
    )
    message = (
        f"{where}; {room.id} {room.name!r} is labelled without an area and has no room "
        f"record, so no schedule counts it; {_open_ends_text(ends)}or close the walls"
    )
    return {
        "ok": True,
        "room": {
            "id": room.id,
            "name": room.name,
            "number": room.number,
            "at": [float(room.at[0]), float(room.at[1])],
            "area": None,
            "area_m2": None,
            "area_text": None,
        },
        "handle": drawn["handles"][0],
        "handles": list(drawn["handles"]),
        "mask": drawn["mask"],
        "layer": ARCH_ROLE_LAYER["room"],
        "align": align,
        "fit": None,
        "area_source": None,
        "wall_layers": layers,
        "loop": None,
        "holes": [],
        "warning": {"code": "room_open", "id": room.id, "message": message, "open_ends": ends},
    }


async def rooms_detect(
    backend: AutoCADBackend,
    *,
    layers=None,
    min_area: float = DEFAULT_MIN_AREA,
    tol: float = DEFAULT_TOL,
    space=None,
) -> dict:
    """Read the rooms of any plan: faces of the lines on ``layers``, measured, never drawn.

    ``layers=None`` reads every layer whose name contains WALL or DUVAR (any
    case), and the room separator layer. A named layer the drawing does not have is refused with the list of
    the ones it does. Our own opening records, when there are any, close door
    and window gaps exactly as `label_room` does; a foreign plan's gaps stay
    open, and two rooms joined by an open doorway read as one face.

    Each room reports ``labels`` - every room record whose point lies in its
    face - and ``label``, the first of them or None. A face carrying more than
    one record is listed in ``label_conflicts``, because a schedule reading the
    records would count its area once per record. What `read_segments` could
    not read is in ``skipped``, with its handle and why. ``space`` reads that
    layout instead of the current one (the critique's reader).
    """
    min_area = _positive(min_area, "min_area")
    tol = _positive(tol, "tol")
    available = [layer.name for layer in await backend.layer_list()]
    if layers is None:
        chosen = [
            name for name in available if any(key in name.upper() for key in WALL_LAYER_KEYWORDS)
        ]
        separator = ARCH_ROLE_LAYER["room_separator"].casefold()
        if chosen:  # separators bound rooms only where there are walls
            chosen += [name for name in available if name.casefold() == separator]
        if not chosen:
            raise ValueError(
                "layers: no layer name contains "
                f"{' or '.join(WALL_LAYER_KEYWORDS)}; name the layers the walls are on "
                f"(layers are {', '.join(sorted(available))})"
            )
    else:
        lookup = {name.casefold(): name for name in available}
        missing = [str(name) for name in layers if str(name).casefold() not in lookup]
        if missing:
            raise ValueError(
                f"layers: {', '.join(missing)} not in this drawing; "
                f"layers are {', '.join(sorted(available))}"
            )
        chosen = [lookup[str(name).casefold()] for name in layers]

    segments, skipped = await read_segments(backend, chosen, space=space)
    records = await read_records(backend, space=space)
    closures, omitted = opening_closures(records["walls"], records["openings"])
    wall_layer = ARCH_ROLE_LAYER["wall"]
    extra = [(p1, p2, wall_layer) for p1, p2 in closures]
    rooms = detect_rooms([*segments, *extra], min_area=min_area, tol=tol)
    outlines = [polygon.points for polygon in records["wall_polygons"]]
    rooms = tuple(
        room
        for room in rooms
        if not is_wall_outline(
            Face(
                loop=room["loop"], area=room["area"], centroid=room["centroid"], holes=room["holes"]
            ),
            outlines,
        )
    )

    out: list[dict] = []
    conflicts: list[dict] = []
    for room in rooms:
        face = Face(
            loop=room["loop"], area=room["area"], centroid=room["centroid"], holes=room["holes"]
        )
        labels = [
            {"id": r.id, "name": r.name, "number": r.number, "area": r.area}
            for r in records["rooms"]
            if face_containing((face,), r.at) is not None
        ]
        if len(labels) > 1:
            conflicts.append(
                {
                    "labels": [label["id"] for label in labels],
                    "centroid": [room["centroid"][0], room["centroid"][1]],
                    "reason": f"one face carries {len(labels)} room records; a schedule "
                    f"would count its {room['area'] / 1.0e6:.2f} m2 {len(labels)} times",
                }
            )
        out.append(
            {
                "loop": _wcs_loop(room["loop"]),
                "holes": [_wcs_loop(hole) for hole in room["holes"]],
                "area": room["area"],
                "area_m2": room["area"] / 1.0e6,
                "centroid": [room["centroid"][0], room["centroid"][1]],
                "confidence": room["confidence"],
                "label": labels[0] if labels else None,
                "labels": labels,
            }
        )
    return {
        "ok": True,
        "modified": False,
        "layers": chosen,
        "segments": len(segments),
        "closures": len(extra),
        "count": len(out),
        "rooms": out,
        "confidence_min": min((room["confidence"] for room in out), default=None),
        "label_conflicts": conflicts,
        "skipped": skipped,
        "omitted": list(omitted),
        "min_area": min_area,
        "tol": tol,
    }
