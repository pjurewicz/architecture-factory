"""External references, raster images and DWG output on the headless engine.

The three capability keys are the point of this file: a refusal the feature map
never mentions is a lie by omission, so every key raised here is asserted to be
declared in the backend's capability map.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from backends.base import AutoCADBackend, UnsupportedCapabilityError
from backends.contracts.refs import (
    DWG_VERSIONS,
    XREF_ACTIONS,
    XREF_KINDS,
    RefsContract,
)
from backends.ezdxf_backend import EzdxfBackend

TRACK_G_KEYS = ("dwg_write", "xref_live", "xlsx_write")


def _features(backend) -> dict:
    return backend.capabilities().to_dict()["features"]


def test_the_contract_is_composed_into_the_backend():
    assert issubclass(AutoCADBackend, RefsContract)
    assert issubclass(EzdxfBackend, RefsContract)
    for name in ("xref_attach", "xref_manage", "image_attach", "drawing_export_dwg"):
        assert callable(getattr(AutoCADBackend, name, None)), name


def test_the_three_keys_are_declared_in_the_capability_map():
    ezdxf_features = _features(EzdxfBackend())
    for key in TRACK_G_KEYS:
        assert key in ezdxf_features, f"{key} undeclared on ezdxf"


def test_ezdxf_does_not_declare_live_xrefs():
    ezdxf_features = _features(EzdxfBackend())
    assert ezdxf_features["xref_live"]["supported"] is False
    assert "live" in (ezdxf_features["xref_live"]["reason"] or "")


def test_ezdxf_dwg_write_is_re_evaluated_from_the_converter_not_hardcoded():
    """The same shape as viewport_render's matplotlib check: the flag follows
    the machine, and the reason names the way to change it."""
    from ezdxf.addons import odafc

    feature = _features(EzdxfBackend())["dwg_write"]
    assert feature["supported"] is bool(odafc.is_installed())
    if not feature["supported"]:
        assert "oda" in (feature["reason"] or "").lower()


def test_xlsx_write_follows_openpyxl():
    present = find_spec("openpyxl") is not None
    feature = _features(EzdxfBackend())["xlsx_write"]
    assert feature["supported"] is present
    if not present:
        assert "openpyxl" in (feature["reason"] or "")


def test_the_vocabularies_are_closed():
    assert XREF_KINDS == ("attach", "overlay")
    assert XREF_ACTIONS == ("list", "reload", "bind", "detach", "path")
    assert DWG_VERSIONS == ("R2000", "R2004", "R2007", "R2010", "R2013", "R2018")


# -- ezdxf -------------------------------------------------------------------


@pytest.fixture
def external(tmp_path: Path) -> str:
    """A real DXF on disk for the xref to point at."""
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.modelspace().add_circle((0, 0), 5)
    target = tmp_path / "BASE.dxf"
    doc.saveas(target)
    return str(target)


@pytest.mark.asyncio
async def test_attach_creates_an_xref_block_and_an_insert(backend, external):
    result = await backend.xref_attach(external, (10.0, 20.0), 2.0, 30.0, "attach")
    assert result["ok"] is True
    assert result["name"] == "BASE"
    assert result["kind"] == "attach"
    insert = await backend.entity_get(result["handle"])
    assert insert.type == "INSERT"
    assert insert.properties["block_name"] == "BASE"

    listing = await backend.xref_manage("", "list", None)
    assert [row["name"] for row in listing["xrefs"]] == ["BASE"]
    assert listing["xrefs"][0]["kind"] == "attach"
    assert Path(listing["xrefs"][0]["path"]).name == "BASE.dxf"
    assert listing["xrefs"][0]["inserts"] == 1


@pytest.mark.asyncio
async def test_overlay_is_flagged_as_an_overlay(backend, external):
    await backend.xref_attach(external, (0.0, 0.0), 1.0, 0.0, "overlay")
    listing = await backend.xref_manage("", "list", None)
    assert listing["xrefs"][0]["kind"] == "overlay"


@pytest.mark.asyncio
async def test_attaching_the_same_name_twice_is_refused_before_writing(backend, external):
    await backend.xref_attach(external, (0.0, 0.0), 1.0, 0.0, "attach")
    before = len(await backend.entity_list())
    with pytest.raises(ValueError) as excinfo:
        await backend.xref_attach(external, (0.0, 0.0), 1.0, 0.0, "attach")
    assert "BASE" in str(excinfo.value)
    assert len(await backend.entity_list()) == before


@pytest.mark.asyncio
async def test_a_missing_file_and_a_bad_kind_are_refused(backend, tmp_path):
    with pytest.raises(ValueError) as missing:
        await backend.xref_attach(str(tmp_path / "nope.dxf"), (0.0, 0.0), 1.0, 0.0, "attach")
    assert "nope.dxf" in str(missing.value)
    with pytest.raises(ValueError) as kind:
        await backend.xref_attach(str(tmp_path), (0.0, 0.0), 1.0, 0.0, "underlay")
    assert "underlay" in str(kind.value)


@pytest.mark.asyncio
async def test_path_rewrites_the_saved_path_and_detach_removes_the_block(
    backend, external, tmp_path
):
    await backend.xref_attach(external, (0.0, 0.0), 1.0, 0.0, "attach")
    moved = str(tmp_path / "moved" / "BASE.dxf")
    changed = await backend.xref_manage("BASE", "path", moved)
    assert changed["ok"] is True
    assert changed["path"] == moved
    assert (await backend.xref_manage("", "list", None))["xrefs"][0]["path"] == moved

    detached = await backend.xref_manage("BASE", "detach", None)
    assert detached["ok"] is True and detached["inserts_removed"] == 1
    assert (await backend.xref_manage("", "list", None))["xrefs"] == []


@pytest.mark.asyncio
async def test_an_xref_shown_on_a_sheet_is_counted_and_detached_with_it(backend, external):
    """Model space is not the drawing.

    An xref is most often *shown* on a paper-space sheet. Counting or deleting
    over `doc.modelspace()` alone reported 1 insert where the drawing held 2
    and then removed the definition with `safe=False`, leaving the sheet's
    INSERT with no BLOCK behind it -- which ezdxf's own recover reports as
    `FIX 103 UNDEFINED_BLOCK` and throws the geometry away.
    """
    await backend.xref_attach(external, (0.0, 0.0), 1.0, 0.0, "attach")
    doc = backend._doc
    doc.layout("Layout1").add_blockref("BASE", (10, 10))

    listing = await backend.xref_manage("", "list", None)
    assert listing["xrefs"][0]["inserts"] == 2

    detached = await backend.xref_manage("BASE", "detach", None)
    assert detached["inserts_removed"] == 2
    assert "BASE" not in doc.blocks
    assert [e for e in doc.layout("Layout1") if e.dxftype() == "INSERT"] == []


@pytest.mark.asyncio
async def test_detach_leaves_a_drawing_that_survives_a_round_trip(backend, external, tmp_path):
    """The evidence a caller actually cares about: the saved file audits clean."""
    import ezdxf.recover

    await backend.xref_attach(external, (0.0, 0.0), 1.0, 0.0, "attach")
    backend._doc.layout("Layout1").add_blockref("BASE", (10, 10))
    await backend.xref_manage("BASE", "detach", None)

    out = tmp_path / "detached.dxf"
    backend._doc.saveas(out)
    _, auditor = ezdxf.recover.readfile(str(out))
    assert [f.message for f in auditor.fixes] == []
    assert [e.message for e in auditor.errors] == []


@pytest.mark.asyncio
async def test_the_definition_is_dropped_with_the_in_use_check_on(backend, external):
    """`safe=True` is the seatbelt, not decoration.

    The broken version passed `safe=False`, the flag that *suppresses* the
    in-use check, so a missed insert became a corrupt file instead of an
    error. This pins the check: with the model-space insert gone but the
    sheet's still there -- exactly the state the old victim scan produced --
    `delete_block` refuses.
    """
    from ezdxf.lldxf.const import DXFBlockInUseError

    await backend.xref_attach(external, (0.0, 0.0), 1.0, 0.0, "attach")
    doc = backend._doc
    doc.layout("Layout1").add_blockref("BASE", (10, 10))
    msp = doc.modelspace()
    for entity in [e for e in msp if e.dxftype() == "INSERT"]:
        msp.delete_entity(entity)

    with pytest.raises(DXFBlockInUseError):
        doc.blocks.delete_block("BASE", safe=True)
    assert "BASE" in doc.blocks


@pytest.mark.asyncio
async def test_reload_and_bind_refuse_headlessly_with_the_xref_live_key(backend, external):
    await backend.xref_attach(external, (0.0, 0.0), 1.0, 0.0, "attach")
    for action in ("reload", "bind"):
        with pytest.raises(UnsupportedCapabilityError) as excinfo:
            await backend.xref_manage("BASE", action, None)
        assert excinfo.value.capability == "xref_live"
        assert "autocad" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_an_unknown_action_and_an_unknown_name_are_refused(backend, external):
    await backend.xref_attach(external, (0.0, 0.0), 1.0, 0.0, "attach")
    with pytest.raises(ValueError) as action:
        await backend.xref_manage("BASE", "unload", None)
    assert "unload" in str(action.value)
    with pytest.raises(ValueError) as name:
        await backend.xref_manage("NOPE", "detach", None)
    assert "NOPE" in str(name.value)


@pytest.mark.asyncio
async def test_image_attach_places_the_raster_at_its_pixel_aspect(backend, tmp_path):
    from PIL import Image

    png = tmp_path / "logo.png"
    Image.new("RGB", (800, 400), "white").save(png)
    result = await backend.image_attach(str(png), (5.0, 5.0), 0.1, 0.0)
    assert result["ok"] is True
    assert result["pixels"] == [800, 400]
    assert result["size_mm"] == [80.0, 40.0]
    image = await backend.entity_get(result["handle"])
    assert image.type == "IMAGE"


@pytest.mark.asyncio
async def test_a_non_image_file_is_refused_before_writing(backend, external):
    before = len(await backend.entity_list())
    with pytest.raises(ValueError) as excinfo:
        await backend.image_attach(external, (0.0, 0.0), 1.0, 0.0)
    assert "BASE.dxf" in str(excinfo.value)
    assert len(await backend.entity_list()) == before


@pytest.mark.asyncio
async def test_headless_dwg_export_refuses_with_dwg_write_when_the_converter_is_absent(
    backend, tmp_path
):
    from ezdxf.addons import odafc

    if odafc.is_installed():
        pytest.skip("the ODA File Converter is installed on this machine")
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        await backend.drawing_export_dwg(str(tmp_path / "out.dwg"), "R2018")
    assert excinfo.value.capability == "dwg_write"
    assert "oda" in str(excinfo.value).lower()
    assert not (tmp_path / "out.dwg").exists()


@pytest.mark.asyncio
async def test_an_unknown_dwg_version_is_refused_on_both_engines(backend, tmp_path):
    with pytest.raises(ValueError) as excinfo:
        await backend.drawing_export_dwg(str(tmp_path / "out.dwg"), "R14")
    message = str(excinfo.value)
    assert "R14" in message
    assert "R2018" in message
