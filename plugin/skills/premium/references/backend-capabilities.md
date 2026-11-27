# Backend capabilities

The server runs one engine, headless (`ezdxf`): a DXF document in memory, no
AutoCAD. `system_capabilities` lists every feature with `supported` and a
reason. Ask it instead of assuming.

| Feature | headless (`ezdxf`) |
|---|---|
| DWG | `drawing_open` converts a `.dwg` through the ODA File Converter when installed (`"dwg_read"`); `drawing_save()` in place on it is refused (`"dwg"`) - use `drawing_save(path='<name>.dwg', version=)`, which also needs the ODA File Converter (`"dwg_write"`) and never re-points |
| Undo / rollback | full document snapshots |
| Checkpoints | restore in place or as a new document; `drawing_undo` steps back to them after a restart |
| Views | screenshots are rendered, never captured from a window |
| A region picture: `view_screenshot(region=[x0, y0, x1, y1], max_px=...)` | rendered |

A screenshot comes with a report: `applied`, `px`, `region` (the window drawn)
and `px_per_unit` - drawing point (x, y) is pixel ((x - x0) * ppu, (y1 - y) *
ppu). A detail costs fewer tokens as a region than as a whole sheet.

A refusal by capability is an engine boundary, not a bug: report it with its
`capability` key and the route it names (an ODA install, doing it in
AutoCAD), and continue with what the engine can do.

## Coordinates and units

Every coordinate in or out of a tool is WCS, in drawing units; angles in
degrees, counter-clockwise. An entity in a tilted plane reports
`plane_normal` and refuses an xy edit (capability `"ocs_tilted_plane"`); move
it instead. A UCS set with `ucs_set` is stored in the drawing only - no tool
reads input in it.

## Where new geometry goes

Model space, until `layout_set_current` names a sheet; to draw on a sheet,
call `layout_set_current` first.

## Quarantine

A call that outruns its deadline while its worker still writes quarantines
the document: later calls on it are refused `quarantined`; the exits that
replace the document (`drawing_open` of the saved file, `drawing_new`,
`document_close(discard=True)`) still work. What the abandoned call wrote is unknown - never
save over the last good file from a quarantined document. A read-only call
that runs long never quarantines; others answer `busy` until it ends.
`drawing_history` lists recent operations and how each ended.
