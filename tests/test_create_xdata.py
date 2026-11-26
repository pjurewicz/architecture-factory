"""Create calls that carry their own XDATA (B17): one call, one lock, one undo step.

The mechanical generators tag every entity they draw; doing it with a second
entity_set_xdata per entity would double the headless undo snapshots. The tag
rides in the create call instead.
"""

from __future__ import annotations

import pytest

from backends.xdata_specs import prepare_xdata

TAG = {"ACADMCP_MECH": ['{"kind":"feature","v":1}']}


def test_prepare_xdata_validates_before_anything_is_written():
    assert prepare_xdata(None) == []
    assert prepare_xdata(TAG) == [("ACADMCP_MECH", [(1000, '{"kind":"feature","v":1}')])]
    with pytest.raises(TypeError, match="app_name"):
        prepare_xdata(["ACADMCP_MECH"])
    with pytest.raises(ValueError, match="app name"):
        prepare_xdata({"bad name!": ["x"]})
    with pytest.raises(ValueError, match="no values"):
        prepare_xdata({"ACADMCP_MECH": []})
    with pytest.raises(ValueError, match="16 KB"):
        prepare_xdata({"ACADMCP_MECH": ["x" * 255] * 70})  # 17,850 characters


@pytest.mark.asyncio
async def test_every_tagged_create_call_writes_its_xdata_in_one_step(backend, monkeypatch):
    marks = []
    original = backend._mark_dirty

    def counted():
        marks.append(1)
        original()

    monkeypatch.setattr(backend, "_mark_dirty", counted)
    made = [
        await backend.entity_create_line(0, 0, 1, 0, xdata=TAG),
        await backend.entity_create_circle(0, 0, 1, xdata=TAG),
        await backend.entity_create_arc(0, 0, 1, 0, 90, xdata=TAG),
        await backend.entity_create_polyline([[0, 0], [1, 0], [1, 1]], xdata=TAG),
        await backend.entity_create_text("A", 0, 0, xdata=TAG),
        await backend.entity_create_hatch("ANSI31", [[0, 0], [1, 0], [1, 1]], xdata=TAG),
    ]
    assert len(marks) == len(made)  # one undo step per create, the tag included
    for info in made:
        got = await backend.entity_get_xdata(info.handle, "ACADMCP_MECH")
        assert got["xdata"]["ACADMCP_MECH"] == ['{"kind":"feature","v":1}']


@pytest.mark.asyncio
async def test_a_refused_tag_creates_nothing(backend):
    before = len(await backend.entity_list(limit=1000))
    with pytest.raises(ValueError, match="16 KB"):
        await backend.entity_create_line(0, 0, 1, 0, xdata={"ACADMCP_MECH": ["x" * 255] * 70})
    assert len(await backend.entity_list(limit=1000)) == before


@pytest.mark.asyncio
async def test_an_untagged_create_call_is_unchanged(backend):
    info = await backend.entity_create_line(0, 0, 1, 0)
    assert (await backend.entity_get_xdata(info.handle))["xdata"] == {}
