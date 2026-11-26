# Architecture Factory fork

This repository is a fork of [U-C4N/Autocad-MCP](https://github.com/U-C4N/Autocad-MCP)
(`autocad-mcp-pro`, MIT). It is the drawing module of **Architecture Factory**: AI
agents use it over MCP to produce technical drawings (plans, elevations, sections,
dimensions, title blocks) as DWG, DXF, PDF and PNG files.

The upstream project is built around a live AutoCAD on Windows, with ezdxf as a
headless fallback. This fork turns that around: **the headless engine is what we
run**, and the live AutoCAD engine is kept only because it is upstream code we do
not want to diverge from.

## Assumptions

- **No AutoCAD.** Drawings are built in memory with [ezdxf](https://ezdxf.mozman.at/).
  DWG is read and written through the ODA File Converter (`drawing_export_dwg`, and
  `drawing_open` on a `.dwg`). PDF and PNG previews come from ezdxf's own renderer.
  Nothing needs Windows or an Autodesk licence.
- **Runs on a server, Linux or macOS.** The target is a container or an Amp orb,
  not a designer's desktop. `.agents/setup` installs ODA File Converter and the
  `pdf` and `office` extras for an orb.
- **Served over HTTP to several agents at once.** The server runs with
  `--transport http` behind a bearer token (`MCP_AUTH_TOKEN`) and
  `HTTP_ALLOWED_HOSTS`. Several MCP clients may draw at the same time, so each
  session must have its own workspace (`SESSION_ISOLATION`, below).
- **The model only sees tools that can work.** Every tool definition costs prompt
  tokens and a tool that can only fail misleads the model. On a headless server
  the tools that need a live AutoCAD are hidden (`HIDE_LIVE_ONLY_TOOLS`, below).
  Use `DISCOVERY_MODE`, `TOOL_PROFILE` and `TOOL_PACKS` (see the
  [README](README.md#configuration)) to keep the catalog small; `TOOL_PACKS=core,arch`
  fits architectural drawing.
- **Upstream stays the base.** Changes go in as small, self-contained commits that
  keep upstream's tests and gates green (`scripts/check_doc_numbers.py`,
  `benchmarks.token_suite --budget`, `benchmarks.call_budget`, ruff), so upstream
  releases can still be merged in. The upstream README below this fork's note is
  left as upstream wrote it, apart from the counts the gates check.

## What differs from upstream

| Change | Setting | Why |
|---|---|---|
| Live-only tools are hidden on a headless server: `system_run_lisp`, `system_run_command`, `system_launch`, `system_preferences_get` / `_set`, `user_pick_point`, `user_select`, `system_prompt_message`. They are left out of `tools/list` and search, and a call is refused `policy_refused` with `policy: "headless_engine"` | `HIDE_LIVE_ONLY_TOOLS` (default `true`) | They can only fail without AutoCAD and cost tokens |
| Each MCP session of a headless server gets its own engine: its own open documents, active document, undo and transactions. The first session keeps the server's engine; stateless HTTP requests share it | `SESSION_ISOLATION` (default `true`), `SESSION_IDLE_TIMEOUT` (default `3600` s) | Upstream has one workspace per server, as AutoCAD has one active drawing: two agents drew into each other's documents without an error |
| A read-only browser page at `/preview`: every session, its open documents, a render of its active document that refreshes as the agent draws, its recent operations, and PDF / DXF / DWG downloads. HTTP Basic auth with its own credentials, separate from the MCP token; off until a password is set | `PREVIEW_PASSWORD`, `PREVIEW_USER` (default `preview`) | A headless server has no AutoCAD window to watch |
| Amp orb setup: ODA File Converter and the headless extras | `.agents/setup` | The server runs in Amp orbs |

Not isolated between sessions: files on disk (two sessions saving to one path
overwrite each other) and the operation journal.

## Known gaps

- `drawing_save_as` refuses a `.dwg` path headlessly; `drawing_export_dwg` is the
  DWG writer, but the refusal does not name it.
- `system_capabilities` reports `dwg: false` next to `dwg_write: true` when ODA is
  installed.
- Three `tests/test_tool_registry.py::test_drawing_open_*` tests expect DWG reading
  to be unavailable and fail where ODA is installed, upstream too.

The changelog's *Unreleased* section lists each change as it lands.
