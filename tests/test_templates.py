"""Bundled templates: catalogue, resolution, reproducible build, drawing_new(template=)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

import server
from backends.base import UnsupportedCapabilityError
from engineering.standards.templates import (
    TEMPLATE_CATALOG,
    TEMPLATES_DIR,
    resolve_template,
    template_rows,
)

pytestmark = pytest.mark.asyncio

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("iso_a3_mech", "iso_a1_arch", "iso_a3_pid", "ansi_b_mech", "ansi_d_arch")


def _build_module():
    spec = importlib.util.spec_from_file_location(
        "build_templates", ROOT / "scripts" / "build_templates.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Ctx:
    def __init__(self, backend):
        self.lifespan_context = {"backend": backend}

    async def info(self, message):
        pass

    async def warning(self, message):
        pass

    async def report_progress(self, progress, total):
        pass


# ── catalogue ───────────────────────────────────────────────────────────────


def test_catalog_has_the_five_spec_rows():
    assert tuple(TEMPLATE_CATALOG) == NAMES
    rows = {spec.name: spec for spec in TEMPLATE_CATALOG.values()}
    assert (rows["iso_a3_mech"].paper, rows["iso_a3_mech"].layer_set) == ("ISO_A3", "mech")
    assert (rows["iso_a1_arch"].paper, rows["iso_a1_arch"].layer_set) == ("ISO_A1", "iso13567")
    assert rows["iso_a1_arch"].settings["annotation_scale"] == "1:50"
    assert (rows["iso_a3_pid"].paper, rows["iso_a3_pid"].layer_set) == ("ISO_A3", "pid")
    assert (rows["ansi_b_mech"].paper, rows["ansi_b_mech"].dimstyle) == ("ANSI_B", "ANSI")
    assert (rows["ansi_d_arch"].paper, rows["ansi_d_arch"].scale) == ("ANSI_D", "fit")
    assert [spec.title_block for spec in TEMPLATE_CATALOG.values()] == [
        True,
        False,
        True,
        False,
        False,
    ]
    assert all(spec.orientation == "landscape" for spec in TEMPLATE_CATALOG.values())
    assert {spec.textstyle for spec in TEMPLATE_CATALOG.values()} == {"ISOCP", "ROMANS"}


def test_template_rows_report_files():
    rows = template_rows()
    assert [row["name"] for row in rows] == list(NAMES)
    for row in rows:
        assert set(row["files"]) == {"dxf", "dwt"}
        assert row["files"]["dxf"]["path"].endswith(f"{row['name']}.dxf")
        assert isinstance(row["files"]["dxf"]["present"], bool)


def test_resolve_template_bundled_path_and_unknown(tmp_path):
    path, source = resolve_template("iso_a3_mech")
    assert source == "bundled" and path == str(TEMPLATES_DIR / "iso_a3_mech.dxf")
    path, source = resolve_template("ISO_A3_MECH")
    assert source == "bundled"
    explicit = tmp_path / "mine.dwt"
    assert resolve_template(str(explicit)) == (str(explicit), "path")
    assert resolve_template("sheet.dxf") == ("sheet.dxf", "path")
    with pytest.raises(
        ValueError, match="iso_a3_mech, iso_a1_arch, iso_a3_pid, ansi_b_mech, ansi_d_arch"
    ):
        resolve_template("iso_a2_mech")
    with pytest.raises(ValueError, match="template"):
        resolve_template("")


# ── the build ───────────────────────────────────────────────────────────────


async def test_build_is_deterministic_run_to_run(tmp_path):
    build = _build_module()
    first = await build.build_all(tmp_path / "one")
    second = await build.build_all(tmp_path / "two")
    assert [row["name"] for row in first] == list(NAMES)
    for name in NAMES:
        a = build.normalised_lines(tmp_path / "one" / f"{name}.dxf")
        b = build.normalised_lines(tmp_path / "two" / f"{name}.dxf")
        assert a == b, f"{name} differs between two builds"
    assert all(row["page_setup"]["ok"] for row in first)
    assert all(row["layers"] for row in first)
    assert second  # the second build is only there to be compared with the first


async def test_committed_templates_match_a_fresh_build(tmp_path):
    """The spec's reproducibility claim — the same comparison the script's --check runs."""
    build = _build_module()
    missing = [name for name in NAMES if not (TEMPLATES_DIR / f"{name}.dxf").is_file()]
    assert not missing, f"bundled templates missing: {missing} — run scripts/build_templates.py"
    await build.build_all(tmp_path)
    drifted = build.drifted_names(tmp_path)
    assert drifted == [], f"templates drifted from the script: {drifted}"


# ── drawing_new(template=) ──────────────────────────────────────────────────


async def test_drawing_new_from_a_bundled_template(backend):
    result = await server.drawing_new(template="iso_a3_mech", ctx=_Ctx(backend))
    assert result["ok"] is True
    assert result["template"] == {
        "name": "iso_a3_mech",
        "path": str(TEMPLATES_DIR / "iso_a3_mech.dxf"),
        "source": "bundled",
    }
    layers = {layer.name for layer in await backend.layer_list()}
    assert {"GEOMETRY", "DIM", "HIDDEN", "CENTER", "TITLEBLOCK"} <= layers
    row = (await backend.page_setup_list("A3"))[0]
    assert row["paper"] == "ISO_A3" and row["orientation"] == "landscape"
    assert row["plot_style"] == "monochrome.ctb" and row["scale"] == "1:1"
    sheet = backend._doc.layouts.get("A3")
    assert any(e.dxftype() == "LWPOLYLINE" for e in sheet), "the ISO 5457 frame is on the sheet"
    assert float(await backend.system_get_variable("DIMTXT")) == 2.5
    if hasattr(backend, "dimstyle_create"):  # after group S merges
        assert await backend.system_get_variable("DIMSTYLE") == "ISO-25"


async def test_drawing_new_from_the_pid_and_ansi_templates(backend):
    await server.drawing_new(template="iso_a3_pid", ctx=_Ctx(backend))
    assert "PROCESS-PIPING-MAIN" in {layer.name for layer in await backend.layer_list()}
    await server.drawing_new(template="ansi_d_arch", ctx=_Ctx(backend))
    assert "M-GEOMET-E-N" in {layer.name for layer in await backend.layer_list()}
    row = (await backend.page_setup_list("D"))[0]
    assert row["paper"] == "ANSI_D" and row["size_mm"] == [864.0, 559.0] and row["scale"] == "fit"
    assert (await backend.drawing_settings())["settings"]["decimal_separator"] == "."


#: Spec §4.1, the eight variables `_with_header_dimvars` does not fold — the
#: ones a header-only write silently loses on the headless renderer.
_PLACEMENT = ("dimtad", "dimtih", "dimtoh", "dimexo", "dimexe", "dimgap", "dimlunit", "dimlwd")


async def _effective_dimvars(backend) -> dict:
    """What the next dimension is really rendered with, read off the entity."""
    from ezdxf.entities import DimStyleOverride

    info = await backend.dimension_linear(0, 0, 100, 0, 50, 20)
    override = DimStyleOverride(backend._doc.entitydb[info.handle])
    return {name: override.get(name) for name in _PLACEMENT + ("dimtxt", "dimasz")}


async def test_templates_dimension_with_their_own_preset_headlessly(backend):
    """The build's fallback lands on the ``Standard`` dimstyle, not only the header.

    Measured before the fix: ``ansi_b_mech`` reported DIMTAD 0 / DIMTIH 1 from
    ``system_get_variable`` while the dimension it drew used dimtad=1 / dimtih=0
    (the ISO values on ``Standard``), so the same template dimensioned
    differently per engine. The header and the style must agree.
    """
    await server.drawing_new(template="ansi_b_mech", ctx=_Ctx(backend))
    ansi = await _effective_dimvars(backend)
    assert ansi == {
        "dimtad": 0,
        "dimtih": 1,
        "dimtoh": 1,
        "dimexo": 1.5,
        "dimexe": 1.5,
        "dimgap": 1.0,
        "dimlunit": 2,
        "dimlwd": -2,
        "dimtxt": 3.0,
        "dimasz": 3.0,
    }
    for name, value in ansi.items():
        header = await backend.system_get_variable(name.upper())
        assert float(header) == float(value), f"{name}: header {header} != style {value}"

    await server.drawing_new(template="iso_a3_mech", ctx=_Ctx(backend))
    iso = await _effective_dimvars(backend)
    assert (iso["dimtad"], iso["dimtih"], iso["dimtoh"]) == (1, 0, 0)
    assert (iso["dimexo"], iso["dimexe"], iso["dimgap"]) == (0.625, 1.25, 0.625)
    assert (iso["dimtxt"], iso["dimasz"]) == (2.5, 2.5)


async def test_drawing_new_unknown_template_is_refused_with_the_names(backend):
    await backend.entity_create_line(0, 0, 1, 1)
    before = (await backend.drawing_info()).entity_count
    with pytest.raises(ToolError, match="iso_a3_mech"):
        await server.drawing_new(template="iso_a2_mech", ctx=_Ctx(backend))
    assert (await backend.drawing_info()).entity_count == before, "the refusal replaced nothing"


async def test_drawing_new_missing_template_path_is_refused(backend, tmp_path):
    with pytest.raises(ToolError, match="not found"):
        await server.drawing_new(template=str(tmp_path / "gone.dxf"), ctx=_Ctx(backend))


async def test_drawing_new_from_an_explicit_dxf_path_reports_source_path(backend, tmp_path):
    await backend.layer_create("MINE", color=3)
    saved = tmp_path / "mine.dxf"
    await backend.drawing_save_as(str(saved), "dxf")
    result = await server.drawing_new(template=str(saved), ctx=_Ctx(backend))
    assert result["template"]["source"] == "path"
    assert result["template"]["path"] == str(saved.resolve())
    assert "MINE" in {layer.name for layer in await backend.layer_list()}


async def test_drawing_template_list_tool(backend):
    result = await server.drawing_template_list(ctx=_Ctx(backend))
    assert result["ok"] and result["count"] == 5
    assert [row["name"] for row in result["templates"]] == list(NAMES)
    assert result["engine"] == "ezdxf"


# ── drawing_template_save ───────────────────────────────────────────────────


async def test_template_save_dxf_headless_round_trips(backend, tmp_path):
    await backend.layer_create("MINE", color=3)
    out = tmp_path / "mine_template.dxf"
    result = await backend.drawing_template_save(str(out), name="Mine", description="my sheet")
    assert result == {
        "ok": True,
        "path": str(out),
        "format": "dxf",
        "backend": "ezdxf",
        "name": "Mine",
        "description_written": False,
        "description_note": (
            "DXF has no template description field; it is a DWG/DWT summary property "
            "(capability 'dwgprops')"
        ),
    }
    assert out.is_file()
    await server.drawing_new(template=str(out), ctx=_Ctx(backend))
    assert "MINE" in {layer.name for layer in await backend.layer_list()}


async def test_template_save_dwt_headless_is_refused_with_the_capability(backend, tmp_path):
    out = tmp_path / "mine.dwt"
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        await backend.drawing_template_save(str(out))
    assert excinfo.value.capability == "dwt_write"
    message = str(excinfo.value)
    assert ".dxf" in message
    assert not out.exists()


async def test_template_save_refuses_other_suffixes(backend, tmp_path):
    with pytest.raises(ValueError, match=r"\.dxf"):
        await backend.drawing_template_save(str(tmp_path / "mine.txt"))


async def test_a_template_is_saved_by_drawing_save(backend, tmp_path):
    with pytest.raises(ToolError):
        await server.drawing_save("../../outside.dxf", ctx=_Ctx(backend))
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        await server.drawing_save(str(tmp_path / "t.dwt"), ctx=_Ctx(backend))
    assert excinfo.value.capability == "dwt_write" and "t.dxf" in str(excinfo.value)
    assert not (tmp_path / "t.dwt").exists()
    result = await server.drawing_save(str(tmp_path / "t.dxf"), ctx=_Ctx(backend))
    assert result["ok"] and (tmp_path / "t.dxf").is_file()
