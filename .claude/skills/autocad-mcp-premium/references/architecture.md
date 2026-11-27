# Architecture

## The plan model

A wall is an axis, a thickness, a justification and a material
(`arch_wall`); a door or window is hosted in a wall (`arch_opening`). Never
hand-draw either with lines: the engine resolves L / T / X junctions, cuts the
openings and hatches cut faces by material, and lists what it cannot place in
`omitted`.

A whole plan is one call, inside one transaction:

```text
arch_plan_from_spec(spec={"sheet": {...}, "lang": "en", "walls": [...],
                         "openings": [...], "rooms": [...], ...})
```

It returns `rooms` (`{id, number, name, area_m2, fits}`), `warnings`,
`omitted`, `plan_bbox` and `outer_bbox`; `critique` (the focuses
`arch_room_unlabelled`, `arch_wall_gap`, `arch_opening_clash`) only when it
found something. `verbose=True` adds each step's result (loops, chain
dimensions, handles); a `cad_batch` binding always sees it. A refusal names
its path (`walls[1].thickness: ...`) and leaves the drawing untouched.
Furniture, grids and symbols are separate (`arch_catalogue_insert`,
`arch_grid`, `arch_symbol`); stairs are `arch_stair`, whose Blondel check
(2R + G within 600-650 mm) is a design rule, not a building code.
Place furniture against a wall with `anchor: "back_center"` on the wall face and
`rotation` = the face's direction with the room on its left; the result's
`footprint` and `front_dir` say where it landed - no offsets by eye.
A `rooms[]` item takes `arch_room`'s label options too: `align: "center"`
centres the label on `at`, and `fit: true` fits it into a small room by
wrapping the name onto two lines first, then shrinking the name - never below
the 2.5 mm ISO 3098 floor `drawing_critique` checks; number and area keep
2.5 mm (defaults `left` / `false`, as on `arch_room`).

To change one door, one label or one wall of a drawn plan, patch it by id -
never delete and redraw the plan:

```text
arch_plan_patch(changes={"openings.D1.offset": 1200, "openings.D2.hinge": "end",
                         "rooms.R2.name": "KITCHEN", "walls.INT.thickness": 120})
```

It redraws only the walls the junction engine says moved, their openings,
the rooms whose area moved and the chains and schedules whose rows moved;
furniture, notes, your dimensions and layouts keep their handles. `null` on a
key removes it (`"openings.D2.hand": null` before giving `hinge`),
`"openings.D2": null` removes the door, `"openings+": {...}` adds one. A
missing room separator is patched in the same way - `"room_separators+": {"id":
"L1", "points": [...]}` - never by redrawing the plan. A plan
entity edited by hand refuses (`hand_edited_tagged_entity`) unless
`overwrite=True`; `dry_run=True` shows the affected set first.

## Areas are measured, never typed

A room's area is the planar face its label point lies in, measured off the
walls just drawn (`arch_room`). No tool accepts an area from you;
`arch_plan_from_spec` refuses a typed `area` by path. To read rooms from any
plan - including one you did not draw - use `arch_rooms_detect` (read-only;
confidence 1.0 for our walls, 0.6 for plain lines).

An open room (a living room open to its kitchen) has no closed face. Never
guess where it ends: `arch_plan_from_spec` labels it without an area and warns
`room_open` with the free wall ends in `open_ends` (`faces` are the end-cap
corners). Close it with a `room_separators[]` line (or `arch_room_separator`)
between those face points, or give the room its own `boundary` polygon; the
area is then measured from it. Never shoelace vertices
you read back; `analysis_measure_entity(handle)` measures real geometry and
states its own accuracy.

## Text and scale

`scale` is a denominator (50 = 1:50) and a paper text height `h` is drawn at
`h x scale` units. `lang="tr"` (Turkish) or `lang="pl"` (Polish) switches
words and the decimal comma; any other language is refused with the list.

## Schedules and dimensions

`arch_schedule` draws door / window / room schedules as real TABLEs read from
the `ACADMCP_ARCH` records; `arch_dimension_chains` adds exterior chains.
Furniture and sanitary blocks carry nominal catalogue sizes and say so.
`arch_catalogue_insert` takes `width` / `depth` / `height` (mm) within each
item's `size_range` (from `arch_catalogue_list`): the symbol is redrawn for the
footprint, not scaled, and each footprint is its own block
(`ARCH_<NAME>_<W>x<D>`) with a `SIZE` attribute `WxDxH`.

## Interior sheets: electrical symbols, legend, title block

Outlets, switches, data / TV / USB, detectors, thermostat, intercom and the
distribution board are one `arch_electrical` call with every item
(`[{kind, at, count, height}]`), never hand-made blocks: a wall device snaps
to the nearest wall face and faces the room, `count` 2 prints `2x` and
`height` 300 prints `H=30CM`, both kept horizontal. The graphics are a
declared nominal convention. `arch_legend` then builds the legend from what
model space uses (blocks and hatch patterns, each with a real sample), and
`arch_title_block` draws the label / value title block and the numbered
notes box in a corner of the sheet's frame - `titleblock_apply` stays the
ISO 7200 block.
