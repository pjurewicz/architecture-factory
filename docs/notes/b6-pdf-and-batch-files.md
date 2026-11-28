# B6: vector PDF reading and batches from a file

Branch `r5-b6-pdf-batchfile`, not merged.

## What it adds

- `pdf_read(path, page=1, align=None, region=None, kinds=None, limit=2000)`:
  `arch` pack, cost `read`, `engineering/pdfread.py`. It returns one PDF page's
  `text` (string, insertion point, height, rotation), `rect` (any closed path
  with four right-angled corners, `re` or m/l/l/l/h), `line` (stroked subpaths,
  with Beziers flattened and marked `curved`) and `fill` (filled subpaths).
  Form XObjects are followed at most 8 levels deep. Clip paths, images and
  shadings are not read.
- **Coordinates.** Output is in *page millimetres*, with `u` measured from the
  left edge and `v` from the top edge, v pointing down. This is the page as a
  viewer shows it (CropBox, `/Rotate` and `/UserUnit` applied). It is the same
  convention as an image pixel in `view_screenshot(overlay_align=)` /
  `image_map`, with millimetres in place of pixels. `align` takes the same
  `[[u, v, x, y], ...]` pairs and goes through the same
  `backends/region_render.py::overlay_transform` / `overlay_fit`. Two pairs fit
  exactly (`fit.controlled: false`). Three or more pairs fit a least-squares
  similarity and report `rms` / `max` / `residuals`. `fit.mm_to_units` is the
  scale: 50 means 1:50 in mm. With `align`, every object also carries WCS
  `x, y` / `corners` / `points`, and `region` becomes a WCS window. Without
  `align`, `region` is a page-mm window.
- **Library.** pypdf (BSD-3, pure Python) is the new `pdfread` extra and is
  also in `full`. `uv.lock` adds pypdf 7.2.0. Without pypdf, the call is
  refused with capability `pdf_read`, which names the extra. The capability is
  declared in `EzdxfBackend.capabilities()` and re-evaluated with `find_spec`,
  like `xlsx_write`.
- `cad_batch(steps_path=)` and `entity_batch_create(entities_path=)` each take a
  `.json` file (`engineering/batch_file.py`) holding the list the inline
  argument would. Exactly one of the inline list or the path is allowed;
  `invalid_args` otherwise. The file's list then runs through the same code as
  the inline list: schema, semantic plan-time checks, the literal-handle
  pre-check and the step ceiling. A refusal names
  `steps_path 'x.json'[i]` / `entities_path 'x.json'[i].key`.

## uploads.py / security.py

- `security.KINDS_UPLOAD` adds `.pdf` and `.json`, and `uploads.FORMATS` maps
  them.
- Sniffing: a PDF must start with `%PDF-` at byte 0. A PNG renamed `.pdf`, or
  HTML holding `%PDF-` later in the file, is refused 422 `content_mismatch`.
  For JSON, the head must start with `[` or `{`, and after the body is on disk
  the whole file must parse (`batch_file.parse`) within 16 MiB and 64 levels of
  nesting. Otherwise it is refused 422 and nothing is kept.
- `file_upload_url` takes `.pdf` / `.json`. Its result carries a `read` hint
  naming `pdf_read` or `steps_path`. The old "rasterise a PDF first" wording
  is gone.
- Read paths go through `security.validate_path` (ALLOWED_PATHS, traversal).
  Nothing new is written, so no new `kinds=` writer exists.

## Risks

- **Parser attack surface.** pypdf parses untrusted bytes in-process. It is
  pure Python, so memory-safety bugs are unlikely, but CPU and memory
  exhaustion are possible. Mitigations:
  - `MAX_BYTES` (64 MiB) is checked before parsing.
  - `MAX_PAGES` (2000) is checked after the xref is read, before any page is
    interpreted.
  - `MAX_SECONDS` (20 s wall clock) is checked every 256 operators and
    truncates with `truncated.reason = "max_seconds"`.
  - `limit` (at most `MAX_OBJECTS` = 20000) truncates with
    `truncated.reason = "limit"`.
  - Form XObject nesting is capped at 8 levels.
  - pypdf's own decompression ceilings (`ZLIB_MAX_OUTPUT_LENGTH` and related
    limits) bound a flate bomb.

  The wall clock is cooperative: one huge content stream is still decoded
  before the first check. This runs in a worker thread (`asyncio.to_thread`),
  so the event loop stays free, but the thread cannot be killed. A subprocess
  with a hard timeout would be the next step if this is exposed widely.
- **Text extraction** goes through pypdf's `extract_text(visitor_text=)`.
  Positions are each text run's insertion point (Tm x CTM). A run is
  whatever the producer emitted; a word split across `Tj` calls arrives in
  pieces. Type3 and CID fonts without a ToUnicode map give whatever pypdf
  decodes.
- **JSON bomb.** JSON has no entity expansion, so the risk is size and depth.
  - Size: 16 MiB, checked against `stat()` before reading and again at upload.
  - Depth: 64 levels, checked without recursion. A `RecursionError` from the
    parser itself is caught as a refusal.
  - The list is then bounded by `MAX_BATCH_STEPS` (cad_batch).
    `entity_batch_create` has no item ceiling today, as before.
- **Replay.** `operation_id` replay hashes the arguments, which means the
  *path*, not the file. Editing the file and retrying with the same id replays
  the old result. This is stated in the docstring.
- **Upload expiry.** An upload is swept after `UPLOAD_RETAIN_S`. A batch file
  is read once per call, so this is harmless. A path from an expired upload
  gives `file_not_found`.

## Token cost (benchmarks.token_suite --lanes idle, before -> after)

| cell | before | after | delta |
|---|---|---|---|
| idle.default | 64,208 | 64,308 | +100 |
| idle.packs_core | 44,042 | 43,941 | -101 |
| idle.lean | 17,506 | 17,384 | -122 |
| idle.search_pinned | 1,106 | 1,087 | -19 |
| growth.packs_core_arch | 51,318 (+2.63 %) | 51,418 (+2.83 %) | +100 |

`pdf_read` costs about 200 tokens and is only on surfaces that include the
`arch` pack. To pay for the two new path parameters, `cad_batch`'s existing
parameter descriptions (`steps`, `on_error`, `expected_revision`,
`preconditions`, `operation_id`) were shortened. The detail they carried moved
into the docstring after the wire cut. The old `operation_id` text also said
"held in memory only", which has been wrong since the durable journal.

**Saving.** 75 furniture inserts as typed JSON are 9,909 characters, about
2,365 tokens by the suite's ratio counter. `entities_path` is 68 characters,
about 16 tokens. The saving is about 2,350 tokens per pass and about 4,700 for
the dry run plus the apply that agents send today. The 25k figure from the
field run had richer items, and the ratio holds: about 31 tokens per item down
to a constant.

The saving is real only when the list does not pass through the model:
produced by a script, an earlier tool result saved client-side, or a file the
user has, then uploaded with `curl` to `file_upload_url`. A model that writes
the JSON itself still pays the output tokens once, but no longer twice.
