"""K1 (1.8.0): ``drawing_open`` reads a DWG headlessly through a converter.

The headless engine converts a ``.dwg`` to DXF through the ODA File Converter
and refuses with capability ``dwg_read`` when it is not on the machine. These
tests stand in for the converter by patching ``dwg_read._read_with_odafc``.
"""

from __future__ import annotations

from pathlib import Path

import ezdxf
import pytest
from fastmcp import Client

import server
from backends import dwg_read
from backends import ezdxf_backend as eb
from backends.base import UnsupportedCapabilityError
from backends.ezdxf_backend import EzdxfBackend
from engineering import errors

# Marked per test: half of these are plain checks (pytest-asyncio warns on those).
asyncio_test = pytest.mark.asyncio

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def no_odafc(monkeypatch):
    monkeypatch.setattr(eb, "_odafc_available", lambda: False)


@pytest.fixture
def oda_stub(monkeypatch):
    """The ODA route stood in for: every path it is asked to convert is recorded."""
    calls: list[str] = []

    def convert(path):
        calls.append(path)
        doc = ezdxf.new("R2018")
        doc.modelspace().add_line((0, 0), (10, 0))
        return doc

    monkeypatch.setattr(eb, "_odafc_available", lambda: True)
    monkeypatch.setattr(dwg_read, "_read_with_odafc", convert)
    return calls


@pytest.fixture
def broken_odafc(monkeypatch):
    def broken(_path):
        raise RuntimeError("ODA File Converter crashed")

    monkeypatch.setattr(eb, "_odafc_available", lambda: True)
    monkeypatch.setattr(dwg_read, "_read_with_odafc", broken)


def _dwg(tmp_path: Path, name: str = "part.dwg") -> Path:
    path = tmp_path / name
    path.write_bytes(b"AC1032" + b"\x00" * 128)
    return path


# ── capability ───────────────────────────────────────────────────────────────


def test_dwg_read_is_declared_in_the_capability_map(no_odafc):
    ezdxf_row = EzdxfBackend().capabilities().to_dict()["features"]["dwg_read"]
    # No ODA here: no route.
    assert ezdxf_row["supported"] is False
    assert "oda_file_converter" in ezdxf_row["reason"]


def test_the_ezdxf_row_is_evaluated_per_call(monkeypatch, no_odafc):
    backend = EzdxfBackend()
    assert backend.capabilities().features["dwg_read"].supported is False
    monkeypatch.setattr(eb, "_odafc_available", lambda: True)
    row = backend.capabilities().features["dwg_read"]
    assert (row.supported, row.mode) == (True, "odafc")
    monkeypatch.setattr(eb, "_odafc_available", lambda: False)
    assert backend.capabilities().features["dwg_read"].supported is False


# ── the refusal ──────────────────────────────────────────────────────────────


@asyncio_test
async def test_no_converter_is_a_dwg_read_refusal_naming_the_converter(backend, tmp_path, no_odafc):
    before = (await backend.drawing_info()).name
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        await backend.drawing_open(str(_dwg(tmp_path)))
    assert excinfo.value.capability == "dwg_read"
    message = str(excinfo.value)
    assert "ODA File Converter" in message
    assert "Nothing was opened" in message
    assert (await backend.drawing_info()).name == before, "the open document is kept"


@asyncio_test
async def test_the_refusal_reaches_the_wire_typed(tmp_path, monkeypatch, no_odafc):
    monkeypatch.setenv("AUTOCAD_MCP_BACKEND", "ezdxf")
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "drawing_open", {"path": str(_dwg(tmp_path))}, raise_on_error=False
        )
    assert result.is_error is True
    payload = result.structured_content
    assert payload["kind"] == "unsupported" and payload["capability"] == "dwg_read"
    assert payload["retryable"] is False


@asyncio_test
async def test_recover_on_a_dwg_is_still_recover_unsupported(backend, tmp_path, oda_stub):
    from backends.dxf_recover import RecoverUnsupportedError

    with pytest.raises(RecoverUnsupportedError):
        await backend.drawing_open(str(_dwg(tmp_path)), recover=True)
    assert oda_stub == [], "recover never reaches the converter"


@asyncio_test
async def test_a_missing_dwg_is_file_not_found(backend, tmp_path, oda_stub):
    with pytest.raises(FileNotFoundError):
        await backend.drawing_open(str(tmp_path / "absent.dwg"))
    assert errors.classify(FileNotFoundError("x"))["kind"] == "file_not_found"


@asyncio_test
async def test_the_result_saves_as_dxf_but_not_onto_the_dwg(backend, tmp_path, oda_stub):
    await backend.drawing_open(str(_dwg(tmp_path)))
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        await backend.drawing_save()
    assert excinfo.value.capability == "dwg"
    saved = await backend.drawing_save_as(str(tmp_path / "part.dxf"), "dxf")
    assert saved["ok"] is True


# ── the ODA route ────────────────────────────────────────────────────────────


@asyncio_test
async def test_odafc_converts_the_dwg(backend, tmp_path, monkeypatch):
    oda_doc = ezdxf.new("R2013")
    oda_doc.modelspace().add_point((1, 2))
    monkeypatch.setattr(eb, "_odafc_available", lambda: True)
    monkeypatch.setattr(dwg_read, "_read_with_odafc", lambda _path: oda_doc)
    result = await backend.drawing_open(str(_dwg(tmp_path)))
    assert result["dwg_read"]["route"] == "odafc"
    assert result["dwg_read"]["dxf_version"] == "AC1027"


@asyncio_test
async def test_a_failed_odafc_is_a_typed_failure(backend, tmp_path, broken_odafc):
    before = (await backend.drawing_info()).name
    with pytest.raises(dwg_read.DwgConversionError) as excinfo:
        await backend.drawing_open(str(_dwg(tmp_path)))
    (attempt,) = excinfo.value.attempts
    assert attempt["route"] == "odafc" and "crashed" in attempt["reason"]
    assert errors.classify(excinfo.value)["kind"] == "dwg_conversion_failed"
    assert (await backend.drawing_info()).name == before


@asyncio_test
async def test_a_failed_conversion_reaches_the_wire_with_its_attempts(
    tmp_path, broken_odafc, monkeypatch
):
    monkeypatch.setenv("AUTOCAD_MCP_BACKEND", "ezdxf")
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "drawing_open", {"path": str(_dwg(tmp_path))}, raise_on_error=False
        )
    payload = result.structured_content
    assert result.is_error is True
    assert payload["kind"] == "dwg_conversion_failed"
    assert payload["category"] == "dependency" and payload["retryable"] is False
    assert payload["attempts"][0]["route"] == "odafc"
    assert server._STATE_OF_KIND["dwg_conversion_failed"] == "refused"


# ── the v9 matrix task ───────────────────────────────────────────────────────


def test_the_v9_task_is_registered_headless_only():
    from benchmarks.adapters.autocad_mcp_pro import HEADLESS_ONLY, AutoCADMCPProAdapter
    from benchmarks.tasks_v9 import DEFAULT_MATRIX, NEW_TASKS_V9, task_by_id

    assert task_by_id("dwg_headless_roundtrip", "v9").category == "reliability"
    # Appended at the end of NEW_TASKS_V9; parallel items append after it, so
    # membership, not the last position, is what this pins.
    assert "dwg_headless_roundtrip" in {task.task_id for task in NEW_TASKS_V9}
    assert "dwg_headless_roundtrip" in HEADLESS_ONLY
    assert hasattr(AutoCADMCPProAdapter, "_task_dwg_headless_roundtrip")
    assert DEFAULT_MATRIX == "v9"  # 1.8-25


async def _run_v9_task(tmp_path):
    from benchmarks.adapters.autocad_mcp_pro import AutoCADMCPProAdapter
    from benchmarks.tasks_v9 import task_by_id

    adapter = AutoCADMCPProAdapter(backend="ezdxf")
    await adapter.setup(tmp_path)
    try:
        return await adapter.run_task(task_by_id("dwg_headless_roundtrip", "v9"))
    finally:
        await adapter.cleanup()


@asyncio_test
async def test_the_v9_task_without_a_converter_is_environment_blocked(tmp_path, no_odafc):
    result = await _run_v9_task(tmp_path)
    assert result.status == "environment_blocked", result
    assert result.metrics["twin_ok"] is True
    assert result.metrics["twin"] == {"kind": "unsupported", "capability": "dwg_read"}


# ── the real converter on a slow display ────────────────────────────────────


@pytest.mark.skipif(not eb._odafc_available(), reason="needs the ODA File Converter")
def test_the_real_converter_waits_for_a_slow_xvfb(tmp_path, monkeypatch):
    """ezdxf starts Xvfb and sleeps 0.1 s; an Xvfb slower than that (a loaded CI
    runner) made the converter abort with "could not connect to display"."""
    import shutil

    real = shutil.which("Xvfb")
    if real is None:
        pytest.skip("needs Xvfb")
    slow = tmp_path / "bin" / "Xvfb"
    slow.parent.mkdir()
    slow.write_text(f'#!/bin/sh\nsleep 1\nexec "{real}" "$@"\n')
    slow.chmod(0o755)
    monkeypatch.setenv("PATH", f"{slow.parent}:{__import__('os').environ['PATH']}")

    doc = ezdxf.new()
    doc.modelspace().add_line((0, 0), (10, 5))
    target = tmp_path / "slow.dwg"
    dwg_read.odafc_module().export_dwg(doc, str(target), replace=True)
    back, report = dwg_read.read_dwg(str(target), odafc=True)
    assert report["route"] == "odafc"
    assert [e.dxftype() for e in back.modelspace()] == ["LINE"]
