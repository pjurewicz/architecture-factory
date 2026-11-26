"""Named views and UCS.

Headless, a named view is a VIEW table entry and "restore" writes the
``*Active`` VPORT (the view the file opens on) — ezdxf's header has no
``$VIEWCTR``; measured. A UCS is a UCS table entry plus the ``$UCS*`` header
variables. Tool coordinates stay WCS: a UCS is stored and
made current for the operator, and no tool interprets its inputs in it.
"""

from __future__ import annotations

import pytest
from fastmcp import Client

import server
from engineering.environment.ucs import is_world_axes, resolve_ucs_axes
from engineering.environment.views import resolve_view_args

pytestmark = pytest.mark.asyncio


# ── pure helpers ────────────────────────────────────────────────────────────


def test_view_args_validate_before_any_write():
    assert resolve_view_args([10, 20], 50, None) == {
        "center": (10.0, 20.0),
        "height": 50.0,
        "width": None,
    }
    assert resolve_view_args(None, None, None) == {"center": None, "height": None, "width": None}
    with pytest.raises(ValueError, match="height must be > 0"):
        resolve_view_args(None, 0, None)
    with pytest.raises(ValueError, match="width must be > 0"):
        resolve_view_args(None, 10, -1)
    with pytest.raises(ValueError, match="center"):
        resolve_view_args([1], None, None)
    with pytest.raises(ValueError, match="finite"):
        resolve_view_args([float("nan"), 0], None, None)


def test_ucs_axes_are_normalised_and_orthogonality_is_measured():
    axes = resolve_ucs_axes([10, 10, 0], [0, 2, 0], [-3, 0, 0])
    assert axes["origin"] == (10.0, 10.0, 0.0)
    assert axes["x_axis"] == (0.0, 1.0, 0.0) and axes["y_axis"] == (-1.0, 0.0, 0.0)
    assert axes["z_axis"] == (0.0, 0.0, 1.0)
    assert axes["angle_deg"] == pytest.approx(90.0)
    with pytest.raises(ValueError, match=r"measured 45\.00"):
        resolve_ucs_axes([0, 0, 0], [1, 0, 0], [1, 1, 0])
    with pytest.raises(ValueError, match="x_axis.*zero"):
        resolve_ucs_axes([0, 0, 0], [0, 0, 0], [0, 1, 0])
    with pytest.raises(ValueError, match="origin"):
        resolve_ucs_axes([0, 0], [1, 0, 0], [0, 1, 0])


def test_world_is_decided_by_the_frame_not_the_name():
    assert is_world_axes((0, 0, 0), (1, 0, 0), (0, 1, 0))
    assert is_world_axes([0.0, 0.0, 1e-12], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert not is_world_axes((0, 20, 0), (1, 0, 0), (0, 1, 0))  # UCS Origin
    assert not is_world_axes((0, 0, 0), (0, 1, 0), (-1, 0, 0))  # rotated 90°
    assert not is_world_axes((0, 0), (1, 0, 0), (0, 1, 0))


# ── headless: named views ───────────────────────────────────────────────────


async def test_view_defaults_to_the_drawing_extents(backend):
    await backend.entity_create_line(0, 0, 100, 50)
    result = await backend.view_named_save("OVERALL")
    assert result["ok"] and result["replaced"] is False
    assert result["center"] == [50.0, 25.0]
    # extents 100 x 50 at the *Active VPORT's default aspect 1.34: the width
    # is the binding side, so height = 100 / 1.34
    assert result["height"] == pytest.approx(100.0 / 1.34)
    assert result["width"] == pytest.approx(100.0)
    view = backend._doc.views.get("OVERALL")
    assert tuple(view.dxf.center)[:2] == (50.0, 25.0)


async def test_view_on_an_empty_drawing_needs_an_explicit_window(backend):
    with pytest.raises(ValueError, match="no extents"):
        await backend.view_named_save("EMPTY")
    assert "EMPTY" not in backend._doc.views
    result = await backend.view_named_save("EMPTY", center=[0, 0], height=10)
    assert result["center"] == [0.0, 0.0] and result["height"] == 10.0
    assert result["width"] == pytest.approx(13.4)


async def test_view_replace_restore_and_list(backend):
    await backend.view_named_save("V", center=[1, 2], height=10, width=15)
    again = await backend.view_named_save("v", center=[5, 5], height=20)
    assert again["replaced"] is True and again["name"] == "V"
    assert len(list(backend._doc.views)) == 1
    restored = await backend.view_named_restore("v")
    assert restored == {
        "ok": True,
        "name": "V",
        "center": [5.0, 5.0],
        "height": 20.0,
        "width": pytest.approx(26.8),
        "applied": "header_only",
        "store": "vport_active",
        "vport_height": 20.0,
        "aspect_ratio": pytest.approx(1.34),
        "backend": "ezdxf",
    }
    (vport,) = backend._doc.viewports.get("*Active")
    assert tuple(vport.dxf.center)[:2] == (5.0, 5.0) and vport.dxf.height == 20.0
    assert await backend.view_named_list() == [
        {"name": "V", "center": [5.0, 5.0], "height": 20.0, "width": pytest.approx(26.8)}
    ]
    with pytest.raises(ValueError, match="no named view 'NOPE'"):
        await backend.view_named_restore("NOPE")
    with pytest.raises(ValueError, match="height must be > 0"):
        await backend.view_named_save("BAD", center=[0, 0], height=0)
    assert "BAD" not in backend._doc.views


async def test_view_restore_keeps_the_vport_aspect_and_fits_the_window(backend):
    """Measured (AutoCAD 2026, display aspect 2.013): the *Active VPORT's
    aspect ratio is reconciled against the real display by the viewport's
    lower-left corner — a 10x20 view at (5, 6) written with aspect 0.5
    opened at VIEWCTR (20.134, 6), and the default 1.34 at (11.734, 6). The
    aspect is the display's (AutoCAD writes it at save), so restore leaves it
    alone and fits the window into it, like `-VIEW _R` does."""
    (vport,) = backend._doc.viewports.get("*Active") or [backend._doc.viewports.new("*Active")]
    vport.dxf.aspect_ratio = 2.0125  # what AutoCAD wrote for a 2262 x 1124 display
    await backend.view_named_save("TALL", center=[5, 6], height=20, width=10)
    await backend.view_named_save("WIDE", center=[5, 6], height=20, width=100)
    tall = await backend.view_named_restore("TALL")
    assert vport.dxf.aspect_ratio == pytest.approx(2.0125), "never rewritten from w / h"
    assert tuple(vport.dxf.center)[:2] == (5.0, 6.0)
    assert vport.dxf.height == 20.0 and tall["vport_height"] == 20.0
    assert tall["aspect_ratio"] == pytest.approx(2.0125)
    wide = await backend.view_named_restore("WIDE")
    assert vport.dxf.aspect_ratio == pytest.approx(2.0125)
    assert vport.dxf.height == pytest.approx(100.0 / 2.0125)
    assert wide["vport_height"] == pytest.approx(100.0 / 2.0125) and wide["height"] == 20.0


async def test_view_is_written_as_a_plan_view_on_disk(backend, tmp_path):
    """ezdxf's VIEW default is ``direction=(1, 1, 1)`` — an isometric view. The
    tool stores the plan window of ``center``/``height``, so the entry must
    say so explicitly (DXF 11/21/31, 12/22/32 and 50) or AutoCAD restores it
    oblique and reads ``center`` in that DCS."""
    await backend.entity_create_line(0, 0, 100, 50)
    await backend.view_named_save("OVERALL")
    view = backend._doc.views.get("OVERALL")
    assert tuple(view.dxf.direction) == (0.0, 0.0, 1.0)
    assert tuple(view.dxf.target) == (0.0, 0.0, 0.0)
    assert view.dxf.view_twist == 0.0

    import ezdxf

    path = tmp_path / "views.dxf"
    await backend.drawing_save_as(str(path))
    reopened = ezdxf.readfile(str(path)).views.get("OVERALL")
    assert tuple(reopened.dxf.direction) == (0.0, 0.0, 1.0)
    assert tuple(reopened.dxf.target) == (0.0, 0.0, 0.0)
    assert tuple(reopened.dxf.center)[:2] == (50.0, 25.0)


async def test_view_restore_carries_the_orientation_into_the_vport(backend):
    """A foreign drawing's VIEW may be oblique or twisted; restoring it must
    not silently flatten it to a plan view of the same center."""
    backend._doc.views.add(
        "ISO",
        dxfattribs={
            "center": (5.0, 6.0, 0.0),
            "height": 20.0,
            "width": 30.0,
            "direction": (1.0, 1.0, 1.0),
            "target": (1.0, 2.0, 3.0),
            "view_twist": 15.0,
        },
    )
    result = await backend.view_named_restore("iso")
    assert result["name"] == "ISO" and result["center"] == [5.0, 6.0]
    (vport,) = backend._doc.viewports.get("*Active")
    assert tuple(vport.dxf.direction) == (1.0, 1.0, 1.0)
    assert tuple(vport.dxf.target) == (1.0, 2.0, 3.0)
    assert vport.dxf.view_twist == 15.0
    # ...and restoring a view the tool saved puts the VPORT back on plan.
    await backend.view_named_save("PLAN", center=[0, 0], height=10)
    await backend.view_named_restore("PLAN")
    (vport,) = backend._doc.viewports.get("*Active")
    assert tuple(vport.dxf.direction) == (0.0, 0.0, 1.0)
    assert tuple(vport.dxf.target) == (0.0, 0.0, 0.0)
    assert vport.dxf.view_twist == 0.0


# ── headless: UCS ───────────────────────────────────────────────────────────


async def test_ucs_set_stores_the_entry_and_makes_it_current(backend):
    result = await backend.ucs_set("FRONT", [10, 10, 0], [0, 1, 0], [-1, 0, 0])
    assert result == {
        "ok": True,
        "name": "FRONT",
        "origin": [10.0, 10.0, 0.0],
        "x_axis": [0.0, 1.0, 0.0],
        "y_axis": [-1.0, 0.0, 0.0],
        "replaced": False,
        "current": True,
        "backend": "ezdxf",
    }
    header = backend._doc.header
    assert header["$UCSNAME"] == "FRONT" and tuple(header["$UCSORG"]) == (10.0, 10.0, 0.0)
    assert tuple(header["$UCSXDIR"]) == (0.0, 1.0, 0.0)
    rows = await backend.ucs_list()
    assert rows[0] == {
        "name": "world",
        "origin": [0.0, 0.0, 0.0],
        "x_axis": [1.0, 0.0, 0.0],
        "y_axis": [0.0, 1.0, 0.0],
        "current": False,
    }
    assert rows[1]["name"] == "FRONT" and rows[1]["current"] is True


async def test_ucs_non_orthogonal_is_refused_before_any_write(backend):
    with pytest.raises(ValueError, match="measured 60"):
        await backend.ucs_set("SKEW", [0, 0, 0], [1, 0, 0], [0.5, 0.8660254, 0])
    assert "SKEW" not in backend._doc.ucs and backend._doc.header["$UCSNAME"] == ""


async def test_ucs_restore_world_and_named(backend):
    await backend.ucs_set("FRONT", [10, 10, 0], [0, 1, 0], [-1, 0, 0])
    world = await backend.ucs_restore("world")
    assert world["name"] == "world" and backend._doc.header["$UCSNAME"] == ""
    assert tuple(backend._doc.header["$UCSXDIR"]) == (1.0, 0.0, 0.0)
    assert (await backend.ucs_list())[0]["current"] is True
    named = await backend.ucs_restore("front")
    assert named["name"] == "FRONT" and backend._doc.header["$UCSNAME"] == "FRONT"
    with pytest.raises(ValueError, match="no UCS named 'NOPE'"):
        await backend.ucs_restore("NOPE")
    with pytest.raises(ValueError, match="reserved"):
        await backend.ucs_set("world", [0, 0, 0], [1, 0, 0], [0, 1, 0])


async def test_ucs_replace_keeps_one_entry(backend):
    await backend.ucs_set("A", [0, 0, 0], [1, 0, 0], [0, 1, 0])
    again = await backend.ucs_set("a", [1, 1, 0], [1, 0, 0], [0, 1, 0])
    assert again["replaced"] is True and again["name"] == "A"
    assert len(list(backend._doc.ucs)) == 1


async def test_ucs_list_does_not_call_an_unnamed_ucs_world(backend):
    """``UCS Origin`` / ``UCS 3P`` without saving — the common kind — leaves
    ``$UCSNAME`` empty exactly like WCS does (AutoCAD 2026 writes
    ``$UCSNAME=''``, ``$UCSORG=(0,20,0)`` after ``_.UCS _O``). The frame, not
    the name, says whether the drawing is in WCS."""
    await backend.ucs_set("FRONT", [10, 10, 0], [0, 1, 0], [-1, 0, 0])
    header = backend._doc.header
    header["$UCSNAME"] = ""
    header["$UCSORG"] = (50.0, 20.0, 0.0)
    header["$UCSXDIR"] = (0.0, 1.0, 0.0)
    header["$UCSYDIR"] = (-1.0, 0.0, 0.0)
    rows = await backend.ucs_list()
    assert [r["name"] for r in rows] == ["world", "FRONT", None]
    assert [r["current"] for r in rows] == [False, False, True]
    assert rows[2] == {
        "name": None,
        "origin": [50.0, 20.0, 0.0],
        "x_axis": [0.0, 1.0, 0.0],
        "y_axis": [-1.0, 0.0, 0.0],
        "current": True,
    }
    # ...and the same empty name with the WCS frame really is world.
    await backend.ucs_restore("world")
    rows = await backend.ucs_list()
    assert [r["name"] for r in rows] == ["world", "FRONT"]
    assert rows[0]["current"] is True and rows[1]["current"] is False


async def test_ucs_list_a_stale_name_falls_back_to_the_frame(backend):
    """``$UCSNAME`` naming an entry the table no longer holds is not a
    current row; the frame decides between world and unnamed."""
    header = backend._doc.header
    header["$UCSNAME"] = "GONE"
    rows = await backend.ucs_list()
    assert [(r["name"], r["current"]) for r in rows] == [("world", True)]
    header["$UCSORG"] = (0.0, 20.0, 0.0)
    rows = await backend.ucs_list()
    assert [(r["name"], r["current"]) for r in rows] == [("world", False), (None, True)]
    assert rows[1]["origin"] == [0.0, 20.0, 0.0]


async def test_ucs_list_tool_reports_an_unnamed_ucs_over_the_wire(monkeypatch, tmp_path):
    """The exact file AutoCAD writes after ``_.UCS _O 0,20,0`` without saving:
    ``$UCSNAME=''`` and ``$UCSORG=(0,20,0)``. The tool used to answer
    ``current: "world"``."""
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.ucs.add(
        "FRONT", dxfattribs={"origin": (10, 10, 0), "xaxis": (0, 1, 0), "yaxis": (-1, 0, 0)}
    )
    doc.header["$UCSNAME"] = ""
    doc.header["$UCSORG"] = (0.0, 20.0, 0.0)
    path = tmp_path / "unnamed_ucs.dxf"
    doc.saveas(str(path))

    monkeypatch.setenv("AUTOCAD_MCP_BACKEND", "ezdxf")
    async with Client(server.mcp) as client:
        await client.call_tool("drawing_open", {"path": str(path)})
        payload = (await client.call_tool("ucs_list", {})).structured_content
    assert payload["current"] is None and payload["current_unnamed"] is True
    assert payload["count"] == 3
    assert [r["name"] for r in payload["ucs"]] == ["world", "FRONT", None]
    assert payload["ucs"][0]["current"] is False
    assert payload["ucs"][2]["current"] is True
    assert payload["ucs"][2]["origin"] == [0.0, 20.0, 0.0]
