"""M6 — inspecting and editing the viewports on a sheet.

`viewport_create` shipped in v1.4 and there was no way to see, rescale, lock or
remove what it made. Four things in the DXF viewport model make these more than
attribute pokes, and all four fail silently if ignored:

* **The main viewport is not a drafting viewport.** Every page-set-up layout
  carries an `id == 1` entity representing the tab's own pan/zoom state. Its
  `view_height` is not a scale, so writing one reports success and changes
  nothing on the sheet; deleting it leaves `layout.dxf.viewport_handle` pointing
  at a dead handle, which ezdxf's Auditor reports as zero errors — only AutoCAD
  notices. It is reported with `is_main`, never hidden: "I deleted every
  viewport and one is still there" has to be explainable.
* **R12 has no slot for either attribute.** `view_height` and `flags` are
  dropped on export, and `get_scale()` then returns a fabricated number (150.0
  measured, for a viewport that stores no scale at all). So the writers refuse
  on R12 and the reader reports `None` rather than a number it invented.
* **`flags` is a shared bitfield.** A bare `flags = VSF_LOCK_ZOOM` wipes the
  UCS-icon and grid bits the main viewport carries (`0x88020`) and still saves
  cleanly. Read-modify-write only.
* **A deleted entity stays in `entitydb`** with `dxftype()` still answering
  `"VIEWPORT"`, so a resolver that type-checks before checking liveness passes
  on a corpse.

Viewports are addressed by handle. `dxf.id` is exposed as metadata only: ezdxf's
first user viewport gets id 3 rather than 2, and ids are never renumbered.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio

SHEET = "A3-Sheet"
LOCK_BIT = 0x4000


async def _sheet(backend, name: str = SHEET):
    """A page-set-up A3 layout, which is what gives it a main viewport."""
    await backend.layout_create(name)
    layout = backend._doc.layouts.get(name)
    layout.page_setup(size=(420, 297), margins=(0, 0, 0, 0), units="mm")
    return layout


async def _r12_backend(tmp_path):
    """An R12 document opened through the backend, with one viewport on a sheet."""
    import ezdxf

    from backends.ezdxf_backend import EzdxfBackend

    doc = ezdxf.new("R12")
    doc.layouts.get("Layout1").add_viewport(
        center=(210, 148.5), size=(200, 150), view_center_point=(50, 25), view_height=300
    )
    path = tmp_path / "r12.dxf"
    doc.saveas(str(path))

    backend = EzdxfBackend()
    await backend.connect()
    await backend.drawing_open(str(path))
    return backend


# ── viewport_list ───────────────────────────────────────────────────────────


async def test_listing_reports_the_viewport_that_was_created(backend):
    await _sheet(backend)
    created = await backend.viewport_create(SHEET, 210, 148.5, 200, 150, 50, 25, scale=0.5)

    result = await backend.viewport_list(SHEET)

    assert result["ok"] is True
    row = next(vp for vp in result["viewports"] if vp["handle"] == created["handle"])
    assert row["layout"] == SHEET
    assert row["is_main"] is False
    assert row["scale"] == pytest.approx(0.5)
    assert row["view_height"] == pytest.approx(300.0)
    assert row["width"] == pytest.approx(200.0)
    assert row["locked"] is False


async def test_the_main_viewport_is_reported_and_labelled_not_hidden(backend):
    """Hiding it makes "I deleted them all and one remains" unexplainable."""
    layout = await _sheet(backend)
    main = layout.main_viewport()
    assert main is not None, "fixture assumption: page_setup creates a main viewport"

    result = await backend.viewport_list(SHEET)

    rows = {vp["handle"]: vp for vp in result["viewports"]}
    assert rows[main.dxf.handle]["is_main"] is True
    assert sum(1 for vp in rows.values() if vp["is_main"]) == 1


async def test_a_fresh_layout_has_no_phantom_viewport(backend):
    """The main viewport appears at page_setup, not at layout_create."""
    await backend.layout_create(SHEET)
    result = await backend.viewport_list(SHEET)
    assert result["viewports"] == []
    assert result["count"] == 0


async def test_listing_without_a_layout_covers_every_sheet(backend):
    await _sheet(backend, "S1")
    await _sheet(backend, "S2")
    await backend.viewport_create("S1", 100, 100, 50, 50, 0, 0)
    await backend.viewport_create("S2", 100, 100, 50, 50, 0, 0)

    result = await backend.viewport_list()

    assert {vp["layout"] for vp in result["viewports"]} >= {"S1", "S2"}


async def test_listing_a_missing_or_blank_layout_refuses(backend):
    """ezdxf's `layouts.get("")` quietly returns the active paper space."""
    for name in ("NOSUCH", "", "   "):
        result = await backend.viewport_list(name)
        assert result["ok"] is False, f"viewport_list({name!r}) must refuse"


async def test_listing_model_space_refuses(backend):
    result = await backend.viewport_list("Model")
    assert result["ok"] is False


async def test_r12_reports_no_scale_rather_than_inventing_one(tmp_path):
    """`get_scale()` returns a fabricated number when view_height is absent."""
    backend = await _r12_backend(tmp_path)
    try:
        result = await backend.viewport_list("Layout1")
        assert result["ok"] is True
        row = result["viewports"][0]
        assert row["scale"] is None
        assert row["locked"] is None
        assert "R12" in result["note"]
    finally:
        await backend.disconnect()


# ── viewport_set_scale ──────────────────────────────────────────────────────


async def test_setting_the_scale_changes_the_view_height_and_nothing_else(backend):
    await _sheet(backend)
    created = await backend.viewport_create(SHEET, 210, 148.5, 200, 150, 50, 25, scale=1.0)
    before = backend._doc.entitydb.get(created["handle"])
    center, width, height = before.dxf.center, before.dxf.width, before.dxf.height
    view_center = before.dxf.view_center_point

    result = await backend.viewport_set_scale(created["handle"], 0.5)

    assert result["ok"] is True
    assert result["scale"] == pytest.approx(0.5)
    assert result["view_height"] == pytest.approx(300.0)
    after = backend._doc.entitydb.get(created["handle"])
    assert after.dxf.center == center
    assert (after.dxf.width, after.dxf.height) == (width, height)
    assert after.dxf.view_center_point == view_center


async def test_the_reported_scale_is_read_back_not_echoed(backend):
    await _sheet(backend)
    created = await backend.viewport_create(SHEET, 210, 148.5, 200, 150, 50, 25)

    result = await backend.viewport_set_scale(created["handle"], 0.25)

    stored = backend._doc.entitydb.get(created["handle"]).get_scale()
    assert result["scale"] == pytest.approx(stored)


@pytest.mark.parametrize("bad", [0, -1, -0.5])
async def test_a_non_positive_scale_is_refused(backend, bad):
    await _sheet(backend)
    created = await backend.viewport_create(SHEET, 210, 148.5, 200, 150, 50, 25)
    result = await backend.viewport_set_scale(created["handle"], bad)
    assert result["ok"] is False


async def test_setting_the_scale_of_the_main_viewport_is_refused(backend):
    """Its view_height is the tab's pan/zoom state, not a drafting scale."""
    layout = await _sheet(backend)
    result = await backend.viewport_set_scale(layout.main_viewport().dxf.handle, 0.5)
    assert result["ok"] is False
    assert "main" in result["error"].lower()


async def test_setting_the_scale_on_r12_is_refused_rather_than_lost(tmp_path):
    backend = await _r12_backend(tmp_path)
    try:
        handle = (await backend.viewport_list("Layout1"))["viewports"][0]["handle"]
        result = await backend.viewport_set_scale(handle, 0.5)
        assert result["ok"] is False
        assert "R12" in result["error"]
    finally:
        await backend.disconnect()


async def test_a_handle_that_is_not_a_viewport_is_refused(backend):
    line = await backend.entity_create_line(0, 0, 10, 0)
    result = await backend.viewport_set_scale(line.handle, 0.5)
    assert result["ok"] is False
    assert "LINE" in result["error"]


async def test_an_unknown_handle_is_refused(backend):
    result = await backend.viewport_set_scale("DEADBEEF", 0.5)
    assert result["ok"] is False


# ── viewport_lock ───────────────────────────────────────────────────────────


async def test_locking_and_unlocking_round_trips(backend, tmp_path):
    import ezdxf

    await _sheet(backend)
    created = await backend.viewport_create(SHEET, 210, 148.5, 200, 150, 50, 25)

    assert (await backend.viewport_lock(created["handle"], True))["locked"] is True
    path = tmp_path / "locked.dxf"
    await backend.drawing_save_as(str(path))
    reloaded = ezdxf.readfile(str(path)).entitydb.get(created["handle"])
    assert reloaded.dxf.flags & LOCK_BIT

    assert (await backend.viewport_lock(created["handle"], False))["locked"] is False
    assert not (backend._doc.entitydb.get(created["handle"]).dxf.flags & LOCK_BIT)


async def test_locking_preserves_the_other_flag_bits(backend):
    """A bare assignment wipes the UCS-icon and grid bits and still saves."""
    await _sheet(backend)
    created = await backend.viewport_create(SHEET, 210, 148.5, 200, 150, 50, 25)
    entity = backend._doc.entitydb.get(created["handle"])
    entity.dxf.flags = 0x88020

    await backend.viewport_lock(created["handle"], True)

    assert backend._doc.entitydb.get(created["handle"]).dxf.flags == 0x88020 | LOCK_BIT


async def test_the_reported_lock_state_is_read_back_from_the_flags(backend):
    await _sheet(backend)
    created = await backend.viewport_create(SHEET, 210, 148.5, 200, 150, 50, 25)
    await backend.viewport_lock(created["handle"], True)

    row = next(
        vp
        for vp in (await backend.viewport_list(SHEET))["viewports"]
        if vp["handle"] == created["handle"]
    )
    assert row["locked"] is True


async def test_locking_on_r12_is_refused_rather_than_lost(tmp_path):
    backend = await _r12_backend(tmp_path)
    try:
        handle = (await backend.viewport_list("Layout1"))["viewports"][0]["handle"]
        result = await backend.viewport_lock(handle, True)
        assert result["ok"] is False
        assert "R12" in result["error"]
    finally:
        await backend.disconnect()


# ── viewport_delete ─────────────────────────────────────────────────────────


async def test_deleting_a_viewport_removes_it_from_the_layout(backend):
    await _sheet(backend)
    created = await backend.viewport_create(SHEET, 210, 148.5, 200, 150, 50, 25)

    result = await backend.viewport_delete(created["handle"])

    assert result["ok"] is True
    assert result["layout"] == SHEET
    assert result["was_main"] is False
    handles = {vp["handle"] for vp in (await backend.viewport_list(SHEET))["viewports"]}
    assert created["handle"] not in handles


async def test_deleting_the_main_viewport_needs_force(backend):
    layout = await _sheet(backend)
    main = layout.main_viewport()

    result = await backend.viewport_delete(main.dxf.handle)

    assert result["ok"] is False
    assert layout.main_viewport() is not None


async def test_forcing_the_main_viewport_away_does_not_leave_a_dangling_pointer(backend):
    """No auditor reports this; only AutoCAD does."""
    layout = await _sheet(backend)
    main = layout.main_viewport()

    result = await backend.viewport_delete(main.dxf.handle, force=True)

    assert result["ok"] is True
    assert result["was_main"] is True
    pointer = layout.dxf.get("viewport_handle", None)
    if pointer is not None:
        target = backend._doc.entitydb.get(pointer)
        assert target is not None and target.is_alive


async def test_a_deleted_viewport_handle_is_refused_afterwards(backend):
    """The corpse stays in entitydb and still answers dxftype() == VIEWPORT."""
    await _sheet(backend)
    created = await backend.viewport_create(SHEET, 210, 148.5, 200, 150, 50, 25)
    await backend.viewport_delete(created["handle"])

    result = await backend.viewport_set_scale(created["handle"], 0.5)
    assert result["ok"] is False


async def test_deleting_a_handle_that_is_not_a_viewport_is_refused(backend):
    line = await backend.entity_create_line(0, 0, 10, 0)
    result = await backend.viewport_delete(line.handle)
    assert result["ok"] is False
    assert line.handle in {e.dxf.handle for e in backend._doc.modelspace()}


# ── what the headless renderer can actually do ──────────────────────────────


async def _ink(backend, layout: str, path) -> int:
    """Dark pixels in a rendered layout — a proxy for "something was drawn"."""
    from PIL import Image

    result = await backend.drawing_export_pdf(str(path), layout=layout)
    assert result["ok"] is True, result
    image = Image.open(str(path)).convert("L")
    pixels = image.load()
    width, height = image.size
    return sum(1 for y in range(height) for x in range(width) if pixels[x, y] < 200)


async def test_the_headless_renderer_does_project_model_content_through_viewports(
    backend, tmp_path
):
    """v1.4 declared this COM-only. It is not, and has not been since ezdxf 1.4.

    The claim appeared in three places at once — the `viewport_render`
    capability, the note in every `drawing_export_pdf` response, and the README
    table — and it sent callers to the live backend for something the headless
    one already does. Understating a capability is the same class of untruth as
    overstating one: the response says something about the server that is not
    so.

    Proven by subtraction rather than by pixel-peeping: a sheet whose only
    contents are viewports renders *less* ink once the model geometry it looks
    at is gone.
    """
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    pytest.importorskip("PIL", reason="rendering needs the [pdf] extra")

    circle = await backend.entity_create_circle(100, 50, 40)
    await _sheet(backend)
    await backend.viewport_create(SHEET, 210, 148.5, 260, 180, 100, 50, scale=1.0)
    assert not [e for e in backend._doc.layouts.get(SHEET) if e.dxftype() != "VIEWPORT"], (
        "the sheet must hold nothing but viewports, or the ink could be its own"
    )

    with_model = await _ink(backend, SHEET, tmp_path / "with.png")
    await backend.entity_delete(circle.handle)
    without_model = await _ink(backend, SHEET, tmp_path / "without.png")

    assert with_model > without_model, (
        "the sheet renders the same with and without the model geometry its "
        "viewport looks at — projection really is not happening"
    )


async def test_a_viewport_draws_model_space_in_its_draw_order(backend, tmp_path):
    """ezdxf draws a viewport's model space in creation order: a hatch sent
    behind a WIPEOUT with `draw_order` covered it again on the sheet, while
    model space showed it masked. Proven by subtraction: sending the hatch
    back must uncover the wiped-out square on the sheet too."""
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    pytest.importorskip("PIL", reason="rendering needs the [pdf] extra")

    await backend.entity_create_wipeout([[80, 30], [120, 30], [120, 70], [80, 70]])
    hatch = await backend.entity_create_hatch("SOLID", [[60, 10], [140, 10], [140, 90], [60, 90]])
    await _sheet(backend)
    await backend.viewport_create(SHEET, 210, 148.5, 260, 180, 100, 50, scale=1.0)

    on_top = await _ink(backend, SHEET, tmp_path / "on_top.png")
    await backend.draw_order([hatch.handle], "send_to_back")
    behind = await _ink(backend, SHEET, tmp_path / "behind.png")

    # the wipeout is a quarter of the hatch's area
    assert behind < on_top * 0.85, (on_top, behind)


async def test_the_capability_map_no_longer_calls_viewport_rendering_com_only(backend):
    features = backend.capabilities().to_dict()["features"]
    assert features["viewport_render"]["supported"] is True


async def test_the_export_note_does_not_send_callers_to_com_for_this(backend, tmp_path):
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    await _sheet(backend)
    result = await backend.drawing_export_pdf(str(tmp_path / "sheet.png"), layout=SHEET)
    assert "COM-only" not in (result.get("note") or "")


async def test_exporting_a_layout_finds_it_whatever_case_it_is_typed_in(backend, tmp_path):
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    await _sheet(backend)
    result = await backend.drawing_export_pdf(str(tmp_path / "sheet.png"), layout=SHEET.upper())
    assert result["ok"] is True


# ── viewport_create(fit_region=) ─────────────────────────────────────────────


async def test_fit_region_sizes_the_viewport_and_names_what_else_it_shows(backend):
    """A 5000 x 2500 region at 1:50 is 100 x 50 mm of paper, plus 5 mm a side;
    the window shown is therefore 250 model units wider than the region on each
    side, and what lies there is reported, not silently printed."""
    inside = await backend.entity_create_line(1500, 2500, 5500, 4000)
    crossing = await backend.entity_create_line(5800, 3000, 6400, 3000)
    margin = await backend.entity_create_circle(6100, 3000, 50)
    await backend.entity_create_circle(9000, 3000, 50)  # outside the window
    await _sheet(backend)

    result = await backend.viewport_create(
        SHEET, 210, 148.5, scale=0.02, fit_region=[6000, 4500, 1000, 2000], margin_mm=5
    )

    assert result["ok"] is True
    assert result["width"] == pytest.approx(110.0)
    assert result["height"] == pytest.approx(60.0)
    assert result["view_center"] == pytest.approx([3500.0, 3250.0])
    assert result["view_height"] == pytest.approx(3000.0)
    fit = result["fit"]
    assert fit["region"] == pytest.approx([1000, 2000, 6000, 4500])
    assert fit["model_window"] == pytest.approx([750, 1750, 6250, 4750])
    reasons = {row["handle"]: row["reason"] for row in fit["intrusions"]}
    assert reasons == {crossing.handle: "crosses_region", margin.handle: "in_margin"}
    assert inside.handle not in reasons
    assert fit["intrusion_count"] == 2

    row = next(
        vp
        for vp in (await backend.viewport_list(SHEET))["viewports"]
        if vp["handle"] == result["handle"]
    )
    assert row["scale"] == pytest.approx(0.02)
    assert row["view_center"][:2] == pytest.approx([3500.0, 3250.0])


async def test_fit_region_measures_an_mtext_by_its_lettering_box(backend):
    """A 5 mm "ABS 2 mm" with no width is 36 x 5 mm of ink, the box
    viewport_clip measures; ezdxf.bbox wraps it at every word (14 x 23 mm),
    which put it inside the region and kept it out of the report."""
    label = await backend.entity_create_mtext("ABS 2 mm", 80, 50, width=0, height=5)
    await _sheet(backend)

    result = await backend.viewport_create(
        SHEET, 210, 148.5, scale=1.0, fit_region=[0, 0, 100, 100], margin_mm=5
    )

    rows = {row["handle"]: row for row in result["fit"]["intrusions"]}
    assert rows[label.handle]["reason"] == "crosses_region"
    x0, y0, x1, y1 = rows[label.handle]["extents"]
    assert (x0, y0, y1) == pytest.approx((80.0, 45.0, 50.0))
    assert x1 > 110.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fit_region": [0, 0, 10, 10], "width": 50},
        {"fit_region": [0, 0, 10, 0]},
        {"fit_region": [0, 0, 10, 10], "margin_mm": -1},
        {"width": 50, "height": 50, "view_center_x": 0, "view_center_y": 0, "margin_mm": 5},
        {"width": 50, "height": 50},
    ],
    ids=["mixed", "zero_area", "negative_margin", "margin_without_fit", "no_view_center"],
)
async def test_fit_region_refuses_ambiguous_arguments(backend, kwargs):
    from engineering.errors import InvalidArgsError

    await _sheet(backend)
    with pytest.raises(InvalidArgsError):
        await backend.viewport_create(SHEET, 210, 148.5, **kwargs)
    assert (await backend.viewport_list(SHEET))["count"] == 1  # only the main one


async def test_viewport_create_warns_about_linetype_scale_before_the_first_plot(backend):
    """The warning used to arrive only with drawing_export_pdf, after a PDF had
    already plotted every dashed line solid."""
    await _sheet(backend)
    plain = await backend.viewport_create(SHEET, 210, 148, 200, 150, 0, 0, 0.05)
    assert "warnings" not in plain
    backend._doc.header["$LTSCALE"] = 15.0
    scaled = await backend.viewport_create(SHEET, 100, 100, 50, 50, 0, 0, 0.05)
    (warning,) = scaled["warnings"]
    assert warning.startswith(f"viewport {scaled['handle']} ") and "15x" in warning


async def test_a_linetype_scale_set_after_the_viewport_warns_where_it_is_set(backend):
    """Round 3: LTSCALE set after viewport_create warned only on a layout render."""
    await _sheet(backend)
    viewport = await backend.viewport_create(SHEET, 210, 148, 200, 150, 0, 0, 0.05)
    assert "warnings" not in await backend.system_set_variable("LTSCALE", 1.0)
    (warning,) = (await backend.system_set_variable("LTSCALE", 50.0))["warnings"]
    assert warning.startswith(f"viewport {viewport['handle']} ") and "50x" in warning
    settings = await backend.drawing_settings({"psltscale": False, "ltscale": 1.0})
    assert any("PSLTSCALE 0" in w for w in settings["warnings"])


async def test_the_linetype_advice_is_one_setting_whatever_the_state(backend):
    """Round 5 (A8): LTSCALE 55 / PSLTSCALE 1 said "keep LTSCALE 1", PSLTSCALE 0 /
    LTSCALE 8 said "LTSCALE 1 (or LTSCALE 55)": both now name the same setting."""
    from backends.ezdxf_backend import LINETYPE_SHEET_FIX, _linetype_scale_warnings

    await backend.layout_create("A3")
    await backend.viewport_create("A3", 200, 150, 380, 270, 5000, 5000, 1 / 55)
    doc = backend._doc
    advice = []
    for ltscale, psl in ((55.0, 1), (8.0, 0)):
        doc.header["$LTSCALE"], doc.header["$PSLTSCALE"] = ltscale, psl
        (warning,) = _linetype_scale_warnings(doc, doc.layouts.get("A3"))
        advice.append(warning.split("; ", 1)[1])
    assert advice == [LINETYPE_SHEET_FIX, LINETYPE_SHEET_FIX]
    doc.header["$LTSCALE"], doc.header["$PSLTSCALE"] = 1.0, 1
    assert _linetype_scale_warnings(doc, doc.layouts.get("A3")) == []
