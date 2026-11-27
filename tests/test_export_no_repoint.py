"""Task 17 (B23 residue): exporting a DXF never re-points the open document.

The backend's `drawing_save_as` keeps its re-pointing semantics; its
`drawing_export_dxf` writes a copy and leaves the document's path, saved flag
and revision alone. The one tool over both is `drawing_save(path, copy=)`.
"""

from __future__ import annotations

import ezdxf
import pytest

from backends.base import UnsupportedCapabilityError
from backends.ezdxf_backend import EzdxfBackend
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


@pytest.mark.asyncio
async def test_drawing_save_routes_by_path_copy_and_extension(monkeypatch, tmp_path):
    """The one save tool: refusals before writing, SAVEAS, a copy, DWT and DWG."""
    from fastmcp import Client

    import server

    dwg_calls = []

    async def fake_dwg(self, path, version="R2018", *, document=None):
        dwg_calls.append((path, version))
        return {"ok": True, "path": path, "version": version}

    monkeypatch.setattr(EzdxfBackend, "drawing_export_dwg", fake_dwg)
    async with Client(server.mcp) as client:

        async def save(**args):
            result = await client.call_tool("drawing_save", args, raise_on_error=False)
            return result.is_error, result.structured_content

        async def where():
            info = (await client.call_tool("drawing_info", {})).structured_content
            return info["full_path"], info["saved"]

        await client.call_tool("drawing_new", {})
        await client.call_tool("entity_create_circle", {"cx": 0, "cy": 0, "radius": 20})
        a, b = tmp_path / "a.dxf", tmp_path / "b.dxf"

        error, payload = await save(copy=True)
        assert error and payload["kind"] == "invalid_args" and "requires path" in payload["message"]
        error, payload = await save(path=str(a), version="R2018")
        assert error and payload["kind"] == "invalid_args" and not a.exists()

        assert not (await save(path=str(a)))[0]
        assert await where() == (str(a), True)  # SAVEAS re-points
        await client.call_tool("entity_create_circle", {"cx": 50, "cy": 0, "radius": 5})
        error, payload = await save(path=str(b), copy=True)
        assert not error and payload["repointed"] is False
        assert await where() == (str(a), False)  # still a.dxf, still unsaved
        assert len(ezdxf.readfile(b).modelspace().query("CIRCLE")) == 2
        assert len(ezdxf.readfile(a).modelspace().query("CIRCLE")) == 1

        error, payload = await save(path=str(tmp_path / "t.dwt"))
        assert error and payload["capability"] == "dwt_write" and "t.dxf" in payload["message"]
        assert not (tmp_path / "t.dwt").exists()

        error, payload = await save(path=str(tmp_path / "c.dwg"), version="R2013")
        assert not error and payload["repointed"] is False
        assert dwg_calls == [(str(tmp_path / "c.dwg"), "R2013")]
        assert await where() == (str(a), False)  # a DWG is always a copy


@pytest.mark.asyncio
async def test_drawing_save_creates_missing_folders_only_inside_the_allowed_paths(
    monkeypatch, tmp_path
):
    from fastmcp import Client

    import config
    import server

    allowed = tmp_path / "work"
    allowed.mkdir()
    monkeypatch.setattr(config.settings, "allowed_paths", [allowed.resolve()])
    async with Client(server.mcp) as client:
        await client.call_tool("drawing_new", {})
        await client.call_tool("entity_create_circle", {"cx": 0, "cy": 0, "radius": 20})

        async def save(**args):
            result = await client.call_tool("drawing_save", args, raise_on_error=False)
            return result.is_error, result.structured_content

        target = allowed / "a" / "b" / "part.dxf"
        assert not (await save(path=str(target), copy=True))[0]
        assert ezdxf.readfile(target).modelspace().query("CIRCLE")

        error, payload = await save(path=str(tmp_path / "elsewhere" / "part.dxf"))
        assert error and payload["kind"] == "path_refused"
        assert payload["allowed_dirs"] == [str(allowed.resolve())]
        assert not (tmp_path / "elsewhere").exists()

        # a relative path lands in the first allowed folder, not the process cwd
        error, payload = await save(path="rel/part.dxf", copy=True)
        assert not error and payload["path"] == str(allowed.resolve() / "rel" / "part.dxf")
        status = (await client.call_tool("system_status", {})).structured_content
        assert status["files"] == {
            "allowed_dirs": [str(allowed.resolve())],
            "relative_to": str(allowed.resolve()),
        }
        refused = await client.call_tool(
            "drawing_save", {"path": "/srv/x.dxf"}, raise_on_error=False
        )
        text = refused.content[0].text
        assert text.startswith("path_refused: ") and str(allowed.resolve()) in text
        assert " Hint: " in text

        error, _ = await save(path=str(allowed / "tpl" / "t.dwt"))
        assert error and not (allowed / "tpl").exists()
