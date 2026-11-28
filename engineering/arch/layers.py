"""The ``arch`` layer set and the role -> layer map every architectural module uses.

Names follow the ISO 13567 field order the repository's ``iso13567`` set
already uses — agent (``A`` = architect), element, presentation (``E`` element
graphics, ``H`` hatching, ``T`` text/annotation), status (``N`` new, ``R`` to be
removed - a ``demolition`` wall) — with the
element field in its common short spelling (``WALL``, ``DOOR``, ``WIND``); the
fields are not padded to ISO 13567-2's fixed widths, exactly as the existing
``M-GEOMET-E-N`` rows are not.

Lineweights are ISO 128 values only (0.13 / 0.18 / 0.25 / 0.50). The one
exception is ``A-CONST-E-N`` at 0.05 mm, the scratch weight every construction
layer in this repository carries: the ``iso128`` critique skips a layer whose
name contains ``CONST``, and ``construction_left`` requires it to be empty
before finalize. Colours are a repository convention — ISO 13567 fixes names,
not colours.

``ARCH_ROLE_LAYER`` is the one place a role becomes a layer name, so no tool
hardcodes one.
"""

from __future__ import annotations

#: (name, ACI colour, linetype, lineweight mm, description) — the row shape of
#: every set in ``engineering/layers.py``.
ARCH_LAYERS: list[tuple[str, int, str, float, str]] = [
    ("0", 7, "Continuous", 0.25, "default"),
    ("A-WALL-E-N", 7, "Continuous", 0.50, "Architecture / walls, cut outline / new"),
    ("A-WALL-H-N", 8, "Continuous", 0.13, "Architecture / walls, poche hatch / new"),
    ("A-WALL-E-R", 2, "ACAD_ISO02W100", 0.35, "Architecture / walls, outline / to be removed"),
    ("A-WALL-H-R", 2, "Continuous", 0.13, "Architecture / walls, hatch / to be removed"),
    ("A-DOOR-E-N", 3, "Continuous", 0.25, "Architecture / doors, leaf and swing / new"),
    ("A-WIND-E-N", 4, "Continuous", 0.25, "Architecture / windows, frame and glazing / new"),
    ("A-STAIR-E-N", 6, "Continuous", 0.25, "Architecture / stairs / new"),
    ("A-FURN-E-N", 5, "Continuous", 0.18, "Architecture / furniture / new"),
    ("A-SANR-E-N", 5, "Continuous", 0.18, "Architecture / sanitary fixtures / new"),
    (
        "A-ELEC-E-N",
        1,
        "Continuous",
        0.18,
        "Architecture / electrical outlets, switches and devices / new",
    ),
    # installation trades (``engineering/arch/services.py``): ISO 13567 agent
    # P = plumbing, M = mechanical; never greyed by ``arch_underlay``
    (
        "P-DOMW-E-N",
        5,
        "Continuous",
        0.25,
        "Plumbing / domestic water points / new",
    ),
    ("P-SANR-E-N", 32, "Continuous", 0.25, "Plumbing / drainage points and drains / new"),
    ("M-HEAT-E-N", 6, "Continuous", 0.25, "Mechanical / heating, radiators and manifolds / new"),
    ("M-VENT-E-N", 4, "Continuous", 0.25, "Mechanical / ventilation air terminals / new"),
    ("A-GRID-E-N", 1, "ACAD_ISO04W100", 0.18, "Architecture / structural grid / new"),
    ("A-ROOM-T-N", 7, "Continuous", 0.25, "Architecture / room labels / new"),
    ("A-ROOM-E-N", 8, "Continuous", 0.13, "Architecture / room separation lines / new"),
    ("A-DIMEN-T-N", 2, "Continuous", 0.18, "Architecture / dimensions / new"),
    ("A-SYMB-T-N", 7, "Continuous", 0.25, "Architecture / symbols and marks / new"),
    ("A-OVHD-E-N", 3, "ACAD_ISO02W100", 0.18, "Architecture / overhead, above the cut plane / new"),
    (
        "A-FLOR-E-N",
        8,
        "Continuous",
        0.13,
        "Architecture / floor finish, tile and panel layout / new",
    ),
    # skirting (``arch_floor_zone(skirting=)``): one layer per type, so each
    # line keeps ByLayer properties; colours and weights a repository convention
    ("A-FLOR-SKRT-E-N", 30, "Continuous", 0.35, "Architecture / skirting, applied / new"),
    (
        "A-FLOR-SKRF-E-N",
        30,
        "ACAD_ISO02W100",
        0.25,
        "Architecture / skirting, flush (shadow gap) / new",
    ),
    ("A-ELEV-E-N", 7, "Continuous", 0.50, "Architecture / interior elevation, wall outline / new"),
    (
        "A-ELEV-FIT-E-N",
        5,
        "Continuous",
        0.25,
        "Architecture / interior elevation, openings and fittings / new",
    ),
    (
        "A-ELEV-TILE-E-N",
        8,
        "Continuous",
        0.13,
        "Architecture / interior elevation, wall tiling / new",
    ),
    (
        "A-ELEV-SRV-E-N",
        1,
        "ACAD_ISO04W100",
        0.18,
        "Architecture / interior elevation, service axes / new",
    ),
    ("A-ELEV-T-N", 7, "Continuous", 0.25, "Architecture / interior elevation, labels / new"),
    ("A-CONST-E-N", 250, "Continuous", 0.05, "Architecture / construction (scratch)"),
]

#: Architectural role -> layer. ``wall_hatch`` is where a ``HatchArea`` lands
#: when ``draw_prims`` is given this map (a HatchArea carries no role of its own).
ARCH_ROLE_LAYER: dict[str, str] = {
    "wall": "A-WALL-E-N",
    "wall_hatch": "A-WALL-H-N",
    "wall_removed": "A-WALL-E-R",
    "hatch:demolition": "A-WALL-H-R",  # a HatchArea of that material
    "door": "A-DOOR-E-N",
    "window": "A-WIND-E-N",
    "stair": "A-STAIR-E-N",
    "furniture": "A-FURN-E-N",
    "sanitary": "A-SANR-E-N",
    "electrical": "A-ELEC-E-N",
    "water": "P-DOMW-E-N",
    "drainage": "P-SANR-E-N",
    "heating": "M-HEAT-E-N",
    "ventilation": "M-VENT-E-N",
    "grid": "A-GRID-E-N",
    "room": "A-ROOM-T-N",
    "room_separator": "A-ROOM-E-N",
    "dim": "A-DIMEN-T-N",
    "symbol": "A-SYMB-T-N",
    "overhead": "A-OVHD-E-N",
    "floor": "A-FLOR-E-N",
    "skirting_applied": "A-FLOR-SKRT-E-N",
    "skirting_flush": "A-FLOR-SKRF-E-N",
    # interior elevations (``engineering/arch/elevation.py``): never an A-WALL
    # layer, or the room finder would read an elevation's outline as walls
    "elev_outline": "A-ELEV-E-N",
    "elev_fitting": "A-ELEV-FIT-E-N",
    "elev_tile": "A-ELEV-TILE-E-N",
    "elev_service": "A-ELEV-SRV-E-N",
    "elev_text": "A-ELEV-T-N",
}

#: The two roles ``engineering/layers.py::resolve_role_layer`` answers for.
ARCH_LAYER_ROLES: dict[str, str] = {"dim": "A-DIMEN-T-N", "construction": "A-CONST-E-N"}

#: name -> (colour, linetype, lineweight): what ``draw_prims`` creates a missing
#: architectural layer with.
ARCH_LAYER_DEFS: dict[str, tuple[int, str, float]] = {
    name: (color, linetype, lineweight) for name, color, linetype, lineweight, _d in ARCH_LAYERS
}
