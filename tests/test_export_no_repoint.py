"""Task 17 (B23 residue): exporting a DXF never re-points the open document.

`drawing_save_as` keeps its re-pointing semantics; `drawing_export_dxf` writes a
copy and leaves the document's path, saved flag and revision alone.
"""

from __future__ import annotations

import ezdxf
import pytest

from backends.base import UnsupportedCapabilityError
from engineering.understand.snapshot import dxf_version_of
from engineering.verification.report import document_identity_of


def test_dxf_version_of_reads_the_acadver_header(tmp_path):
    for release, code in (("R2010", "AC1024"), ("R2018", "AC1032")):  # the DXF reference's codes
        path = tmp_path / f"{release}.dxf"
        ezdxf.new(release).saveas(path)
        assert dxf_version_of(path) == code
    text = tmp_path / "note.dxf"
    text.write_text("not a drawing", encoding="utf-8")
    assert dxf_version_of(text) is None
    assert dxf_version_of(tmp_path / "absent.dxf") is None


@pytest.mark.asyncio
async def test_headless_export_writes_a_copy_and_leaves_the_document_alone(backend, tmp_path):
    await backend.entity_create_circle(0, 0, 20)
    before_info = await backend.drawing_info()
    before = await document_identity_of(backend)
    target = tmp_path / "copy.dxf"
    result = await backend.drawing_export_dxf(str(target))
    after_info = await backend.drawing_info()
    after = await document_identity_of(backend)
    assert (after_info.full_path, after_info.saved) == ("", False)
    assert (after_info.full_path, after_info.saved) == (before_info.full_path, before_info.saved)
    assert after["revision"] == before["revision"]
    assert result["repointed"] is False
    assert result["method"] == "ezdxf.write"
    assert result["dxf_version"] == "AC1024"  # drawing_new writes R2010
    assert ezdxf.readfile(target).modelspace().query("CIRCLE")[0].dxf.radius == 20.0


@pytest.mark.asyncio
async def test_headless_export_still_refuses_a_dwg_name(backend, tmp_path):
    with pytest.raises(UnsupportedCapabilityError):
        await backend.drawing_export_dxf(str(tmp_path / "copy.dwg"))


@pytest.mark.asyncio
async def test_save_as_keeps_re_pointing_the_document(backend, tmp_path):
    target = tmp_path / "saved.dxf"
    await backend.drawing_save_as(str(target), "dxf")
    info = await backend.drawing_info()
    assert (info.full_path, info.saved) == (str(target), True)
