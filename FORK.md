# Architecture Factory fork

This repository is a fork of [U-C4N/Autocad-MCP](https://github.com/U-C4N/Autocad-MCP)
(`autocad-mcp-pro`, MIT). It is the drawing module of **Architecture Factory**: AI
agents use it over MCP to produce technical drawings (plans, elevations, sections,
dimensions, title blocks) as DWG, DXF, PDF and PNG files.

The upstream project is built around a live AutoCAD on Windows, with ezdxf as a
headless fallback. This fork turns that around: **the headless engine is the only
engine**. The live AutoCAD (COM) engine is being removed in stages (see *Removing
the live engine* below); `AUTOCAD_MCP_BACKEND=com` already fails startup.

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
  tokens and a tool that can only fail misleads the model, so the tools that
  needed a live AutoCAD are removed (*Removing the live engine*, below). Use `DISCOVERY_MODE`, `TOOL_PROFILE` and `TOOL_PACKS` (see the
  [README](README.md#configuration)) to keep the catalog small; `TOOL_PACKS=core,arch`
  fits architectural drawing.
- **Upstream is cherry-picked, not merged.** Upstream adds most features to both
  engines in one commit, so once the live engine is gone a merge would bring it
  back piece by piece. Upstream commits are reviewed and taken one by one with
  `git cherry-pick -x`; the procedure is written down here when the removal is
  done. Changes still go in as small, self-contained commits that keep the gates
  green (`scripts/check_doc_numbers.py`, `benchmarks.token_suite --budget`,
  `benchmarks.call_budget`, ruff). The last commit shared with upstream is
  `6f6a538` (2026-11-06); the tag `pre-com-removal` marks the fork before the
  removal started.

## What differs from upstream

| Change | Setting | Why |
|---|---|---|
| The tools only a live AutoCAD could run are removed: `system_run_lisp`, `system_run_command`, `system_launch`, `system_preferences_get` / `_set`, `user_pick_point`, `user_select`, `system_prompt_message`, the five `solid_*` and the `view_zoom_extents` / `view_zoom_window` no-ops (`view_screenshot(region=)` frames a window). With them went the command / LISP denylist, `LISP_POLICY`, `DANGEROUS_COMMANDS_ENABLED` and `ENABLE_3D` | - | They could only fail without AutoCAD and cost tokens |
| Each MCP session of a headless server gets its own engine: its own open documents, active document, undo and transactions. The first session keeps the server's engine; stateless HTTP requests share it. An engine is released with its unsaved documents when the client ends its session, or after the idle timeout when it vanishes | `SESSION_ISOLATION` (default `true`), `SESSION_IDLE_TIMEOUT` (default `3600` s) | Upstream has one workspace per server, as AutoCAD has one active drawing: two agents drew into each other's documents without an error |
| A read-only browser page at `/preview`: every session, its open documents, a render of its active document that refreshes as the agent draws, its recent tool calls with duration and data size: argument bytes in, text and image bytes out, measured rather than converted to tokens, since every model counts tokens its own way, and PDF / DXF / DWG downloads; beside it `/preview/tools`, how many times each tool was called and failed (by error kind) across all sessions, read from the operation journal so it outlives a restart. HTTP Basic auth with its own credentials, separate from the MCP token; off until a password is set | `PREVIEW_PASSWORD`, `PREVIEW_USER` (default `preview`) | A headless server has no AutoCAD window to watch |
| Amp orb setup: ODA File Converter and the headless extras | `.agents/setup` | The server runs in Amp orbs |

Not isolated between sessions: files on disk (two sessions saving to one path
overwrite each other) and the operation journal.

## Removing the live engine

Decided 2026-11-26. In order, each stage its own commit with the suite green:

1. Backend selection: the server always builds the ezdxf engine; `auto` (the
   default) and `ezdxf` are accepted, any other value, `com` included, fails
   startup. **Done.**
2. The live-only tools (`system_run_*`, `system_launch`, `system_preferences_*`,
   `user_*`, `system_prompt_message`, `solid_*`, `view_zoom_extents` /
   `view_zoom_window`) and what exists only for them (`security_lisp.py`, the
   command denylist, `LISP_POLICY`, `DANGEROUS_COMMANDS_ENABLED`, `ENABLE_3D`,
   `HIDE_LIVE_ONLY_TOOLS`). **Done.**
3. The COM engine itself (`backends/com_backend.py` and its helpers, the live
   branches in `engineering/`, the `COM_*` / `CAD_LAUNCH` settings, `pywin32`,
   the COM tests), the error kinds only it raises, and the `accoreconsole` DWG
   route (ODA stays). **Done.**
4. One capability map instead of two. **Done** with stage 3 (the capability
   gate checks the ezdxf map alone).
5. The live smokes, `scripts/live/`, the live claims in
   `scripts/check_doc_numbers.py` and the COM-only benchmark tasks. **Done.**
   The task matrices stay as upstream has them, so benchmark commits still
   cherry-pick: a task only the live engine could run stays
   `environment_blocked`.
6. Documentation: CLAUDE.md, README, the skill, the plugin, and the tool
   docstrings that still describe the live engine. **Done.**
7. The cherry-pick procedure, with the upstream commit reviewed up to.

The package name and version scheme stay as they are.

## Known gaps

- `system_capabilities` reports `dwg: false` next to `dwg_write: true` when ODA is
  installed.
- Three `tests/test_tool_registry.py::test_drawing_open_*` tests expect DWG reading
  to be unavailable and fail where ODA is installed, upstream too.

The changelog's *Unreleased* section lists each change as it lands.
