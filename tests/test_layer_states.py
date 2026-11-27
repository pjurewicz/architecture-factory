"""Portable layer states: save → change → restore equals the snapshot.

These are the server's own layer states — JSON chunks in an XRECORD under
``ACADMCP_LAYERSTATES`` in the drawing's named object dictionary. They live
in the file and travel with it; they do NOT appear in AutoCAD's Layer States
Manager. The ezdxf tests prove the format.
"""

from __future__ import annotations

import json

import pytest

from backends.base import LayerInfo
from engineering.environment.layer_states import (
    DICT_NAME,
    PROPERTIES,
    decode_state,
    diff_snapshot,
    encode_state,
    snapshot_from_layers,
    validate_properties,
)


def _layer(name, **overrides) -> LayerInfo:
    base = dict(
        name=name,
        color=7,
        linetype="Continuous",
        lineweight=-3,
        is_on=True,
        is_frozen=False,
        is_locked=False,
        is_current=False,
    )
    base.update(overrides)
    return LayerInfo(**base)


# ── codec ───────────────────────────────────────────────────────────────────


def test_snapshot_records_every_property_and_the_current_layer():
    state = snapshot_from_layers(
        [_layer("0"), _layer("HIDDEN", color=8, linetype="HIDDEN", lineweight=35, is_frozen=True)],
        "HIDDEN",
        plot={"HIDDEN": False},
        description="before plotting",
    )
    assert state == {
        "version": 1,
        "description": "before plotting",
        "current_layer": "HIDDEN",
        "layers": {
            "0": {
                "on": True,
                "frozen": False,
                "locked": False,
                "color": 7,
                "linetype": "Continuous",
                "lineweight": -3,
                "plot": True,
            },
            "HIDDEN": {
                "on": True,
                "frozen": True,
                "locked": False,
                "color": 8,
                "linetype": "HIDDEN",
                "lineweight": 35,
                "plot": False,
            },
        },
    }


def test_encode_chunks_are_at_most_255_ascii_characters_and_round_trip():
    layers = [_layer(f"LAYER-{i:03d}-{'x' * 40}") for i in range(60)]
    state = snapshot_from_layers(layers, "0")
    chunks = encode_state(state)
    assert len(chunks) > 5
    assert all(len(c) <= 255 and c.isascii() for c in chunks)
    assert decode_state(chunks) == state


def test_unicode_layer_names_survive_the_ascii_chunking():
    state = snapshot_from_layers([_layer("KÖRPER-Ø20")], "KÖRPER-Ø20")
    chunks = encode_state(state)
    assert all(c.isascii() for c in chunks), "escaped so a chunk boundary never splits a code point"
    assert decode_state(chunks)["current_layer"] == "KÖRPER-Ø20"


@pytest.mark.parametrize(
    "chunks, fragment",
    [
        (["not json"], "not valid JSON"),
        ([json.dumps({"version": 2, "layers": {}})], "version"),
        ([json.dumps({"version": 1, "layers": {"A": {"on": True}}})], "lacks"),
        ([json.dumps([1, 2])], "version"),
    ],
)
def test_decode_refuses_anything_that_is_not_a_state(chunks, fragment):
    with pytest.raises(ValueError, match=fragment):
        decode_state(chunks)


def test_diff_reports_missing_and_new_layers_sorted():
    state = snapshot_from_layers([_layer("A"), _layer("B"), _layer("C")], "0")
    missing, new = diff_snapshot(state, [_layer("C"), _layer("Z"), _layer("A"), _layer("Y")])
    assert missing == ["B"] and new == ["Y", "Z"]


def test_validate_properties_defaults_to_all_and_names_the_bad_one():
    assert validate_properties(None) == PROPERTIES
    assert validate_properties(["color", "on", "color"]) == ("color", "on")
    with pytest.raises(ValueError, match=r"properties\[1\]: 'colour'"):
        validate_properties(["on", "colour"])
    with pytest.raises(TypeError, match="list"):
        validate_properties("on")
    with pytest.raises(ValueError, match="at least one"):
        validate_properties([])


# ── headless engine ─────────────────────────────────────────────────────────


async def _seed(backend):
    await backend.layer_create("GEOMETRY", color=7, linetype="Continuous", lineweight=50)
    await backend.layer_create("HIDDEN", color=8, linetype="HIDDEN", lineweight=35)
    await backend.layer_create("CENTER", color=1, linetype="CENTER", lineweight=25)
    await backend.layer_set_current("GEOMETRY")
    await backend.layer_freeze("CENTER")
    await backend.layer_hide("HIDDEN")


async def _table(backend) -> dict:
    return {
        lyr.name: (
            lyr.color,
            lyr.linetype,
            lyr.lineweight,
            lyr.is_on,
            lyr.is_frozen,
            lyr.is_locked,
            lyr.is_current,
        )
        for lyr in await backend.layer_list()
    }


@pytest.mark.asyncio
async def test_save_change_restore_equals_the_snapshot(backend):
    await _seed(backend)
    before = await _table(backend)
    saved = await backend.layer_state_save("PLOT", description="plot set")
    # measured: ezdxf.new() already has "0" and "Defpoints", so the seeded
    # table is five layers and serialises to three <= 255-character chunks
    assert saved == {
        "ok": True,
        "name": "PLOT",
        "layer_count": 5,
        "replaced": False,
        "chunks": 3,
        "backend": "ezdxf",
    }

    await backend.layer_thaw("CENTER")
    await backend.layer_show("HIDDEN")
    await backend.layer_lock("GEOMETRY")
    await backend.layer_modify("HIDDEN", color=3, linetype="Continuous", lineweight=13)
    await backend.layer_set_current("0")
    assert await _table(backend) != before

    restored = await backend.layer_state_restore("PLOT")
    assert restored["applied"] == {
        "layers": 5,
        "properties": list(PROPERTIES),
        "current_layer": "GEOMETRY",
    }
    assert restored["missing_layers"] == [] and restored["new_layers"] == []
    assert await _table(backend) == before


@pytest.mark.asyncio
async def test_restore_applies_only_the_requested_properties(backend):
    await _seed(backend)
    await backend.layer_state_save("S")
    await backend.layer_thaw("CENTER")
    await backend.layer_modify("CENTER", color=5)
    result = await backend.layer_state_restore("S", properties=["color"])
    assert result["applied"]["properties"] == ["color"]
    center = next(lyr for lyr in await backend.layer_list() if lyr.name == "CENTER")
    assert center.color == 1 and center.is_frozen is False, "frozen was not requested"


@pytest.mark.asyncio
async def test_restore_reports_missing_and_new_layers_and_never_creates(backend):
    await _seed(backend)
    await backend.layer_state_save("S")
    await backend.layer_delete("CENTER")
    await backend.layer_create("NEW-ONE")
    result = await backend.layer_state_restore("S")
    assert result["missing_layers"] == ["CENTER"] and result["new_layers"] == ["NEW-ONE"]
    assert "CENTER" not in [lyr.name for lyr in await backend.layer_list()]


@pytest.mark.asyncio
async def test_plot_flag_and_current_layer_round_trip(backend):
    await _seed(backend)
    backend._doc.layers.get("HIDDEN").dxf.plot = 0
    await backend.layer_state_save("S")
    backend._doc.layers.get("HIDDEN").dxf.plot = 1
    await backend.layer_set_current("0")
    await backend.layer_state_restore("S", properties=["plot", "current"])
    assert backend._doc.layers.get("HIDDEN").dxf.plot == 0
    assert backend._current_layer == "GEOMETRY"
    assert backend._doc.header["$CLAYER"] == "GEOMETRY"


@pytest.mark.asyncio
async def test_state_survives_save_and_reopen(backend, tmp_path):
    await _seed(backend)
    await backend.layer_state_save("PLOT", description="d")
    path = tmp_path / "states.dxf"
    await backend.drawing_save(str(path))
    await backend.document_close(None)
    await backend.drawing_open(str(path))
    assert await backend.layer_state_list() == [
        {"name": "PLOT", "description": "d", "layer_count": 5}
    ]
    await backend.layer_thaw("CENTER")
    await backend.layer_state_restore("PLOT")
    center = next(lyr for lyr in await backend.layer_list() if lyr.name == "CENTER")
    assert center.is_frozen is True


@pytest.mark.asyncio
async def test_xrecord_chunks_stay_under_256_characters(backend):
    for i in range(80):
        await backend.layer_create(f"LONG-LAYER-NAME-{i:03d}-{'x' * 30}")
    saved = await backend.layer_state_save("BIG")
    assert saved["chunks"] > 10
    xrecord = backend._doc.rootdict.get(DICT_NAME).get("BIG")
    tags = [t for t in xrecord.tags if t.code == 1000]
    assert len(tags) == saved["chunks"]
    assert all(len(t.value) <= 255 for t in tags)


@pytest.mark.asyncio
async def test_replace_delete_and_missing_names(backend):
    await backend.layer_state_save("S")
    again = await backend.layer_state_save("s")
    assert again["replaced"] is True, "AutoCAD dictionaries are case-insensitive"
    assert [row["name"] for row in await backend.layer_state_list()] == ["s"]
    assert await backend.layer_state_delete("S") == {"ok": True, "deleted": "s", "backend": "ezdxf"}
    assert await backend.layer_state_list() == []
    with pytest.raises(ValueError, match="no layer state named 'S'"):
        await backend.layer_state_restore("S")
    with pytest.raises(ValueError, match="no layer state named 'S'"):
        await backend.layer_state_delete("S")


@pytest.mark.asyncio
async def test_refusals_happen_before_any_write(backend):
    with pytest.raises(ValueError, match="layer state name"):
        await backend.layer_state_save("   ")
    with pytest.raises(TypeError, match="description"):
        await backend.layer_state_save("S", description=3)
    assert DICT_NAME not in backend._doc.rootdict, "a refused save creates no dictionary"
    await backend.layer_state_save("S")
    with pytest.raises(ValueError, match=r"properties\[0\]: 'plottable'"):
        await backend.layer_state_restore("S", properties=["plottable"])
    assert len(list(backend._doc.rootdict.get(DICT_NAME).keys())) == 1
