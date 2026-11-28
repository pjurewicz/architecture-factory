"""The furniture and sanitary catalogue: authored outlines, search, blocks.

Pure half first (every item's primitives against its declared nominal size,
the search in both languages, the refusal), then the insert through the
headless backend - the `backend` fixture in tests/conftest.py.
"""

from __future__ import annotations

import pytest

from backends.block_specs import validate_attdef_specs, validate_entity_specs
from engineering.arch.catalogue import (
    CATALOGUE_NAMES,
    FAMILIES,
    ITEMS,
    SIZE_BASIS,
    block_spec,
    catalogue,
    insert_item,
)
from engineering.arch.layers import ARCH_LAYERS, ARCH_ROLE_LAYER
from engineering.mech.primitives import Arc, points_of


def _spec_points(spec: dict) -> list[tuple[float, float]]:
    """Every point that bounds one block_define spec - an arc by its real sweep."""
    kind = spec["type"]
    if kind == "line":
        return [(spec["x1"], spec["y1"]), (spec["x2"], spec["y2"])]
    if kind == "circle":
        cx, cy, r = spec["cx"], spec["cy"], spec["radius"]
        return [(cx - r, cy), (cx + r, cy), (cx, cy - r), (cx, cy + r)]
    if kind == "arc":
        arc = Arc((spec["cx"], spec["cy"]), spec["radius"], spec["start_angle"], spec["end_angle"])
        return list(points_of(arc))
    if kind in ("polyline", "wipeout"):
        return [tuple(p) for p in spec["points"]]
    raise AssertionError(f"unexpected spec type {kind!r}")


#: Items whose drawing reaches past the w x d footprint: door swings in front
#: of a wardrobe, a niche's mask over the wall face line.
BEYOND = {"wardrobe", "wall_niche"}


def test_the_catalogue_holds_forty_eight_items_in_two_families():
    assert len(ITEMS) == 48
    assert len(set(CATALOGUE_NAMES)) == 48
    assert FAMILIES == ("furniture", "sanitary")
    assert sum(item.family == "furniture" for item in ITEMS) == 31
    assert sum(item.family == "sanitary" for item in ITEMS) == 17
    assert {
        "single_bed",
        "double_bed",
        "wardrobe",
        "sofa_3_seat",
        "armchair",
        "dining_table_4",
        "desk",
        "kitchen_counter",
        "fridge",
        "bookshelf",
        "wc",
        "wall_basin",
        "shower_tray",
        "bathtub",
        "kitchen_sink",
        "washing_machine",
        "corner_sofa",
        "dining_table_6",
        "dining_table_8",
        "induction_hob",
        "oven_tall_unit",
        "dishwasher",
        "walk_in_shower",
        "freestanding_bathtub",
        "countertop_basin",
    } <= set(CATALOGUE_NAMES)
    for name in CATALOGUE_NAMES:
        assert name == name.lower() and " " not in name


@pytest.mark.parametrize("name", CATALOGUE_NAMES)
def test_every_item_stays_inside_its_declared_nominal_size(name):
    spec = block_spec(name)
    w, d = spec["size"]
    assert w > 0 and d > 0
    x0, y0, x1, y1 = spec["extent"]
    if name not in BEYOND:
        assert [x0, y0] == pytest.approx([0, 0]) and x1 <= w + 1e-9 and y1 <= d + 1e-9
    for entity in spec["entities"]:
        for x, y in _spec_points(entity):
            assert x0 - 1e-6 <= x <= x1 + 1e-6, (name, entity)
            assert y0 - 1e-6 <= y <= y1 + 1e-6, (name, entity)
    for att in spec["attdefs"]:
        assert 0.0 <= att["x"] <= w and 0.0 <= att["y"] <= d


@pytest.mark.parametrize("name", CATALOGUE_NAMES)
def test_every_block_spec_validates_against_block_define(name):
    spec = block_spec(name)
    normalised = validate_entity_specs(spec["entities"])
    assert {entity["layer"] for entity in normalised} == {"0"}
    item_att, size_att = validate_attdef_specs(spec["attdefs"])
    assert item_att["tag"] == "ITEM" and item_att["default"] == name
    assert item_att["invisible"] is True and size_att["tag"] == "SIZE"
    assert spec["name"] == "ARCH_" + name.upper()
    assert spec["size_basis"] == SIZE_BASIS


def _inside(point, polygon) -> bool:
    x, y = point
    hit = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


@pytest.mark.parametrize("name", CATALOGUE_NAMES)
def test_every_block_starts_with_its_wipeouts(name):
    """Inside a block a WIPEOUT hides only what is drawn before it."""
    kinds = [entity["type"] for entity in block_spec(name)["entities"]]
    masks = kinds.count("wipeout")
    # the concealed WC frame is drawn dashed inside the pre-wall: it masks nothing
    assert (masks >= 1) != (name == "concealed_wc_frame"), name
    assert kinds[:masks] == ["wipeout"] * masks, name


def test_a_shaped_item_masks_its_outline_not_its_bounding_box():
    """The floor shows beside a WC bowl, in the L of a corner sofa, between chairs."""

    def masks(name):
        spec = block_spec(name)
        return [e["points"] for e in spec["entities"] if e["type"] == "wipeout"], spec["size"]

    for name, covered, open_ in (
        ("wc", (190, 400), (10, 690)),  # inside the bowl / the corner beside it
        ("corner_sofa", (1300, 450), (500, 1300)),  # the seat / inside the L
        ("freestanding_bathtub", (850, 400), (5, 5)),  # the tub / its rounded corner
        ("dining_table_4", (700, 850), (20, 200)),  # the table / beside the first chair
    ):
        polys, _ = masks(name)
        assert any(_inside(covered, poly) for poly in polys), name
        assert not any(_inside(open_, poly) for poly in polys), name
    polys, (w, d) = masks("double_bed")
    assert polys == [[[0.0, 0.0], [w, 0.0], [w, d], [0.0, d]]]


def test_every_outline_reaches_its_declared_size():
    """The declared size is the item's footprint, not a loose bound."""
    for name in CATALOGUE_NAMES:
        spec = block_spec(name)
        points = [p for entity in spec["entities"] for p in _spec_points(entity)]
        w, d = spec["size"]
        assert min(x for x, _ in points) == pytest.approx(0.0, abs=1e-9), name
        assert max(x for x, _ in points) == pytest.approx(w, abs=1e-9), name
        assert min(y for _, y in points) == pytest.approx(0.0, abs=1e-9), name
        # the wc's bowl arc stops at y = 680 inside its 700 mm nominal depth
        if name not in BEYOND:
            assert max(y for _, y in points) <= d + 1e-9, name


def test_the_wc_bowl_is_a_semicircle_joining_its_two_sides():
    spec = block_spec("wc")
    arc = next(e for e in spec["entities"] if e["type"] == "arc")
    assert (arc["cx"], arc["cy"], arc["radius"], arc["start_angle"], arc["end_angle"]) == (
        190.0,
        530.0,
        150.0,
        0.0,
        180.0,
    )
    sides = [e for e in spec["entities"] if e["type"] == "line"]
    assert [(s["x1"], s["y2"]) for s in sides] == [(40.0, 530.0), (340.0, 530.0)]


def test_the_sofa_cushions_split_the_seat_into_three_equal_widths():
    spec = block_spec("sofa_3_seat")
    verticals = sorted(
        e["x1"] for e in spec["entities"] if e["type"] == "line" and e["x1"] == e["x2"]
    )
    assert verticals == [200.0, 800.0, 1400.0, 2000.0]


def test_search_by_english_word():
    names = [row["name"] for row in catalogue("bed")]
    assert names == ["single_bed", "double_bed"]
    assert [row["name"] for row in catalogue("washing machine")] == ["washing_machine"]


def test_search_by_turkish_word_with_and_without_diacritics():
    assert [row["name"] for row in catalogue("yatak")] == ["single_bed", "double_bed"]
    assert [row["name"] for row in catalogue("buzdolabı")] == ["fridge"]
    assert [row["name"] for row in catalogue("buzdolabi")] == ["fridge"]
    assert [row["name"] for row in catalogue("KLOZET")] == ["wc"]
    assert [row["name"] for row in catalogue("çamaşır")] == [
        "washing_machine",
        "washer_dryer_column",
    ]
    assert [row["name"] for row in catalogue("camasir")] == [
        "washing_machine",
        "washer_dryer_column",
    ]


def test_search_every_word_must_match():
    assert [row["name"] for row in catalogue("masa")] == [
        "dining_table_4",
        "desk",
        "dining_table_6",
        "dining_table_8",
        "round_table",  # yuvarlak masa
    ]
    assert [row["name"] for row in catalogue("dört yemek masa")] == ["dining_table_4"]
    assert catalogue("bed sink") == ()


def test_family_filter_and_the_label_language():
    sanitary = catalogue(family="sanitary")
    assert [row["name"] for row in sanitary] == [
        "wc",
        "bidet",
        "wall_basin",
        "shower_tray",
        "bathtub",
        "kitchen_sink",
        "washing_machine",
        "walk_in_shower",
        "freestanding_bathtub",
        "sink_bowl",
        "countertop_basin",
        "radiator",
        "concealed_wc_frame",
        "prewall",
        "boiler",
        "dhw_tank",
        "washer_dryer_column",
    ]
    assert all(row["layer"] == ARCH_ROLE_LAYER["sanitary"] for row in sanitary)
    (row,) = catalogue("fridge", lang="tr")
    assert row["label"] == "buzdolabı" and row["label_en"] == "fridge"
    assert row["size"] == [600.0, 650.0]
    assert row["size_basis"] == SIZE_BASIS
    assert row["block"] == "ARCH_FRIDGE"
    assert len(catalogue()) == 48


def test_polish_labels_cover_every_item_and_are_searchable_without_diacritics():
    from engineering.arch.catalogue import ITEMS, LABEL_PL

    assert set(LABEL_PL) == {item.name for item in ITEMS}
    (row,) = catalogue("lodowka", lang="pl")
    assert (row["name"], row["label"], row["label_pl"]) == ("fridge", "lodówka", "lodówka")
    assert [r["name"] for r in catalogue("ŁÓŻKO")] == ["single_bed", "double_bed"]
    assert catalogue("fridge", lang="en")[0]["label"] == "fridge"


def test_unknown_family_and_language_are_refused_with_the_list():
    with pytest.raises(ValueError, match="furniture, sanitary"):
        catalogue(family="lighting")
    with pytest.raises(ValueError, match="en, tr, pl"):
        catalogue(lang="de")


def test_an_unknown_name_is_refused_with_the_nearest_names():
    with pytest.raises(ValueError, match=r"no item 'singlebed'; nearest: single_bed"):
        block_spec("singlebed")
    with pytest.raises(ValueError, match="the catalogue holds single_bed"):
        block_spec("zzz")


def test_a_label_spelling_resolves_to_the_item():
    assert block_spec("Single Bed")["item"] == "single_bed"
    assert block_spec("washing-machine")["item"] == "washing_machine"


@pytest.mark.asyncio
async def test_insert_defines_the_block_once_across_two_inserts(backend):
    first = await insert_item(backend, "double_bed", at=(1000.0, 2000.0))
    assert first["ok"] is True
    assert first["block_name"] == "ARCH_DOUBLE_BED"
    assert first["defined"] is True
    assert first["layer"] == ARCH_ROLE_LAYER["furniture"]
    assert first["layer_created"] is True
    assert first["size"] == [1600.0, 2000.0]
    assert first["size_basis"] == SIZE_BASIS

    second = await insert_item(backend, "double_bed", at=(4000.0, 2000.0), rotation=90.0)
    assert second["defined"] is False
    assert second["layer_created"] is False
    assert second["rotation"] == 90.0

    blocks = [b.name for b in await backend.block_list() if b.name == "ARCH_DOUBLE_BED"]
    assert blocks == ["ARCH_DOUBLE_BED"]
    inserts = await backend.entity_list(type_filter="INSERT", limit=100)
    assert sorted(e.handle for e in inserts) == sorted([first["handle"], second["handle"]])
    assert {e.layer for e in inserts} == {ARCH_ROLE_LAYER["furniture"]}
    assert await backend.block_get_attributes(first["handle"]) == {
        "ITEM": "double_bed",
        "SIZE": "1600x2000x450",
    }


@pytest.mark.asyncio
async def test_insert_creates_the_layer_from_its_arch_layers_row(backend):
    result = await insert_item(backend, "wc", at=(0.0, 0.0))
    layer = ARCH_ROLE_LAYER["sanitary"]
    assert result["layer"] == layer
    row = next(row for row in ARCH_LAYERS if row[0] == layer)
    info = next(info for info in await backend.layer_list() if info.name == layer)
    assert info.color == row[1]


@pytest.mark.asyncio
async def test_insert_refuses_before_writing_anything(backend):
    before = len(await backend.entity_list())
    with pytest.raises(ValueError, match="nearest: bathtub"):
        await insert_item(backend, "bathtup", at=(0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        await insert_item(backend, "bathtub", at=(float("nan"), 0.0))
    with pytest.raises(ValueError, match="rotation must be finite"):
        await insert_item(backend, "bathtub", at=(0.0, 0.0), rotation=float("inf"))
    assert len(await backend.entity_list()) == before
    assert "ARCH_BATHTUB" not in {b.name for b in await backend.block_list()}


@pytest.mark.asyncio
async def test_an_anchor_lands_on_the_point_and_the_result_reports_the_footprint(backend):
    # wc is 380 x 700; back_center turned 90 deg CCW onto (1000, 500): the back
    # runs up +Y, so the block base sits 190 below and the front faces -X.
    result = await insert_item(
        backend, "wc", at=(1000.0, 500.0), rotation=90.0, anchor="back_center"
    )
    assert result["insert_point"] == pytest.approx([1000.0, 310.0])
    assert result["back_center"] == pytest.approx([1000.0, 500.0])
    assert result["front_dir"] == pytest.approx([-1.0, 0.0])
    assert result["footprint"] == [
        pytest.approx(p) for p in ([1000, 310], [1000, 690], [300, 690], [300, 310])
    ]
    insert = await backend.entity_get(result["handle"])
    assert insert.properties["insertion"] == pytest.approx([1000.0, 310.0])
    with pytest.raises(ValueError, match="anchor must be one of"):
        await insert_item(backend, "wc", at=(0.0, 0.0), anchor="middle")


def test_every_row_states_its_frame():
    for row in catalogue():
        assert row["origin"] == "back_left" and row["front_dir"] == [0.0, 1.0]
        assert [row["width"], row["depth"]] == row["size"]


# -- parametric sizes ----------------------------------------------------------


@pytest.mark.parametrize("name", CATALOGUE_NAMES)
def test_every_item_redraws_inside_its_footprint_at_the_ends_of_its_range(name):
    item = next(i for i in ITEMS if i.name == name)
    for w in item.w_range:
        for d in item.d_range:
            try:
                spec = block_spec(name, w, d)
            except ValueError as exc:  # a combination the symbol cannot hold, named
                assert "needs a depth of at least" in str(exc), (name, w, d)
                continue
            validate_entity_specs(spec["entities"])
            points = [p for entity in spec["entities"] for p in _spec_points(entity)]
            assert max(x for x, _ in points) == pytest.approx(w, abs=1e-6), (name, w, d)
            top = spec["extent"][3] if name in BEYOND else d
            assert all(-1e-6 <= x <= w + 1e-6 and -1e-6 <= y <= top + 1e-6 for x, y in points)


def test_a_resized_item_is_redrawn_not_scaled():
    """A wider sofa keeps its 200 mm arms; three equal seats share the rest."""
    spec = block_spec("sofa_3_seat", width=2800)
    verticals = sorted(
        e["x1"] for e in spec["entities"] if e["type"] == "line" and e["x1"] == e["x2"]
    )
    assert verticals == [200.0, 1000.0, 1800.0, 2600.0]
    assert spec["name"] == "ARCH_SOFA_3_SEAT_2800x900"
    assert spec["attdefs"][1]["default"] == "2800x900x850"


def test_a_longer_table_seats_its_chairs_evenly_and_a_short_one_is_refused():
    spec = block_spec("dining_table_8", width=2800)
    seats = sorted(
        e["points"][0][0]
        for e in spec["entities"]
        if e["type"] == "polyline" and e["points"][0][1] == 0.0
    )
    assert seats == [125.0, 825.0, 1525.0, 2225.0]  # 2800 / 8 - 225 + k * 700
    with pytest.raises(ValueError, match=r"width 2100 mm is outside 2200-3000"):
        block_spec("dining_table_8", width=2100)


def test_out_of_range_and_non_numeric_sizes_are_refused_by_name():
    with pytest.raises(ValueError, match="wc depth 900 mm is outside 480-750"):
        block_spec("wc", depth=900)
    with pytest.raises(ValueError, match="height must be a number"):
        block_spec("wardrobe", height="tall")
    with pytest.raises(ValueError, match="needs a depth of at least 400"):
        block_spec("wall_basin", width=700, depth=350)


@pytest.mark.asyncio
async def test_each_footprint_is_its_own_block_and_the_nominal_one_keeps_its_name(backend):
    nominal = await insert_item(backend, "wardrobe", at=(0.0, 0.0))
    wide = await insert_item(backend, "wardrobe", at=(2000.0, 0.0), width=2400, height=2600)
    again = await insert_item(backend, "wardrobe", at=(5000.0, 0.0), width=2400)
    assert nominal["block_name"] == "ARCH_WARDROBE" and nominal["nominal"] is True
    assert wide["block_name"] == again["block_name"] == "ARCH_WARDROBE_2400x600"
    assert (wide["defined"], again["defined"]) == (True, False)
    assert wide["size"] == [2400.0, 600.0] and wide["height"] == 2600.0
    assert (await backend.block_get_attributes(wide["handle"]))["SIZE"] == "2400x600x2600"
    assert (await backend.block_get_attributes(again["handle"]))["SIZE"] == "2400x600x2200"


def test_round_3s_refused_sizes_now_fit_or_name_the_item_that_does():
    # round 3: a 420 mm sink, an 890 mm walk-in shower, a nightstand deeper than
    # 600 and a 365 mm vanity top were refused as out of range
    assert block_spec("walk_in_shower", 890, 900)["size"] == [890.0, 900.0]
    assert block_spec("nightstand", 500, 650)["size"] == [500.0, 650.0]
    assert block_spec("countertop_basin", 600, 365)["size"] == [600.0, 365.0]
    assert block_spec("sink_bowl", 420)["size"] == [420.0, 450.0]
    with pytest.raises(ValueError, match="take 420 x 500: sink_bowl\\.$"):
        block_spec("kitchen_sink", 420)  # a drainer needs the width; the single bowl does not


# -- round 5: many items, tiers, the new items ---------------------------------


@pytest.mark.asyncio
async def test_a_refused_item_is_reported_and_does_not_stop_the_others(backend):
    """Round 5 (A1): step 53 of 75 single inserts was refused and the batch
    rollback undid the 52 good ones."""
    from engineering.arch.catalogue import check_insert_args, check_items, insert_items

    items = [
        {"name": "double_bed", "x": 0, "y": 0},
        {"name": "round_table", "x": 3000, "y": 0, "width": 1480, "depth": 2600},
        {"name": "wc", "x": 6000, "y": 0, "anchor": "middle"},
        {"name": "wc", "x": 6000, "y": 0, "colour": 3},
        {"name": "plant", "x": 9000, "y": 0},
    ]
    problems = check_items(items)
    assert [p["i"] for p in problems] == [1, 2, 3]
    assert "items[1]: round_table depth 2600" in problems[0]["message"]
    assert "anchor must be one of" in problems[1]["message"]
    assert "unknown key(s) colour" in problems[2]["message"]
    check_insert_args({"items": items})  # some refused: the step still runs
    result = await insert_items(backend, items)
    assert [row["i"] for row in result["inserted"]] == [0, 4]
    assert set(result["inserted"][0]) == {"i", "handle", "block", "footprint"}
    assert result["refused"] == problems
    assert len(await backend.entity_list(type_filter="INSERT")) == 2

    from engineering.errors import InvalidArgsError

    before = len(await backend.entity_list())
    bad = [items[2], {"name": "bathtup", "x": 0, "y": float("nan")}]
    with pytest.raises(ValueError, match=r"items\[1\]: y must be finite"):
        check_insert_args({"items": bad})
    with pytest.raises(InvalidArgsError, match=r"every item was refused.*items\[0\].*items\[1\]"):
        await insert_items(backend, bad)
    with pytest.raises(InvalidArgsError, match="non-empty list"):
        await insert_items(backend, [])
    assert len(await backend.entity_list()) == before


def _order(backend) -> list[str]:
    return [e.dxf.handle for e in backend._doc.modelspace().entities_in_redraw_order()]


@pytest.mark.asyncio
async def test_items_are_layered_by_tier_whatever_order_they_come_in(backend):
    """Round 5 (A6): send_to_back pushed a table and a plant under the rug's
    mask, inserted earlier - 22 items vanished with no warning."""
    from engineering.arch.catalogue import insert_items
    from engineering.critique import run_critique

    await backend.layer_create("TEXT")
    note = await backend.entity_create_text("SALON", 900, 900, 100, layer="TEXT")
    first = await insert_items(
        backend,
        [
            {"name": "plant", "x": 1500, "y": 1500, "anchor": "center"},
            {
                "name": "round_table",
                "x": 900,
                "y": 900,
                "anchor": "center",
                "options": {"chairs": 4},
            },
        ],
    )
    rug = await insert_items(
        backend, [{"name": "round_rug", "x": 900, "y": 900, "anchor": "center", "width": 2600}]
    )
    plant, table = (row["handle"] for row in first["inserted"])
    order = _order(backend)
    rank = [order.index(h) for h in (rug["inserted"][0]["handle"], table, plant, note.handle)]
    assert rank == sorted(rank)  # rug < table < plant < what else the space draws
    assert rug["warnings"] == [] and first["warnings"] == []
    assert await run_critique(backend, focus=["wipeout_mask"]) == []


@pytest.mark.asyncio
async def test_an_item_masking_another_of_its_tier_is_a_warning(backend):
    from engineering.arch.catalogue import insert_items

    desk = await insert_items(backend, [{"name": "desk", "x": 0, "y": 0}])
    chair = await insert_items(backend, [{"name": "chair", "x": 500, "y": 100}])
    (warning,) = chair["warnings"]
    assert warning["i"] == 0 and warning["handle"] == chair["inserted"][0]["handle"]
    assert warning["masks"] == [desk["inserted"][0]["handle"]]


@pytest.mark.asyncio
async def test_a_niche_goes_on_top_and_cuts_the_wall_without_a_warning(backend):
    from engineering.arch.catalogue import insert_items

    await backend.layer_create("A-WALL-E-N")
    wall = await backend.entity_create_line(0, 0, 3000, 0, layer="A-WALL-E-N")
    niche = await insert_items(
        backend,
        [{"name": "wall_niche", "x": 1500, "y": 0, "rotation": 180, "anchor": "back_center"}],
    )
    await insert_items(backend, [{"name": "bookshelf", "x": 0, "y": 0}])
    order = _order(backend)
    # on top of the wall it cuts, and of a shelf inserted after it
    assert order[-1] == niche["inserted"][0]["handle"] and wall.handle in order[:-1]
    assert niche["warnings"] == []


def test_options_are_validated_by_name_and_name_the_block():
    spec = block_spec("round_table", 1800, 1800, options={"chairs": 8})
    assert spec["name"] == "ARCH_ROUND_TABLE_1800x1800_C8"
    seats = [e for e in spec["entities"] if e["type"] == "polyline" and e["closed"]]
    assert len(seats) == 8
    x0, y0, x1, y1 = spec["extent"]
    assert x0 < 0 and y0 < 0 and x1 > 1800 and y1 > 1800  # the chairs stand round the top
    assert block_spec("round_table")["name"] == "ARCH_ROUND_TABLE"  # 0 chairs, nominal
    with pytest.raises(ValueError, match="options.chairs 13 is outside 0-12"):
        block_spec("round_table", options={"chairs": 13})
    with pytest.raises(ValueError, match="options.legs: unknown; round_table takes chairs"):
        block_spec("round_table", options={"legs": 4})
    with pytest.raises(ValueError, match="takes no options"):
        block_spec("plant", options={"chairs": 1})
    with pytest.raises(ValueError, match="8 chairs need 4000 mm"):
        block_spec("round_table", 800, 800, options={"chairs": 8})


def test_a_wardrobe_draws_a_rail_hangers_and_one_dashed_swing_per_door():
    spec = block_spec("wardrobe", 2400, options={"doors": 3})
    assert spec["name"] == "ARCH_WARDROBE_2400x600_D3"
    arcs = [e for e in spec["entities"] if e["type"] == "arc"]
    # hinged left, right, left: the second and third leaves are a pair on x = 1600
    assert sorted({(a["cx"], a["cy"]) for a in arcs}) == [(0.0, 600.0), (1600.0, 600.0)]
    assert len(arcs) > 3 * 3  # dashed: several short arcs per swing
    assert spec["extent"][3] == pytest.approx(600 + 800)  # a leaf's swing in front
    assert block_spec("wardrobe")["name"] == "ARCH_WARDROBE"  # 2 doors by default
    with pytest.raises(ValueError, match="a leaf takes 300-1000 mm"):
        block_spec("wardrobe", 2400, options={"doors": 1})


@pytest.mark.asyncio
async def test_against_wall_turns_the_item_to_the_room_and_mask_false_drops_the_wipeouts(backend):
    """Round 7: agents computed rotation and anchor by hand; the source hatches
    through movable furniture. Ring faces at 125 / 4875 and 125 / 3875."""
    from engineering.arch.catalogue import insert_items
    from engineering.arch.spec import draw_plan_from_spec

    await draw_plan_from_spec(
        backend,
        {
            "sheet": {"intent": "flat", "sheet_size": "A3", "scale": 50},
            "walls": [
                {
                    "id": "EXT",
                    "axis": [[0, 0], [5000, 0], [5000, 4000], [0, 4000]],
                    "thickness": 250,
                    "closed": True,
                }
            ],
        },
    )
    out = await insert_items(
        backend,
        [
            {"name": "boiler", "x": 4700, "y": 2000, "against_wall": True},  # right wall
            {"name": "double_bed", "x": 2500, "y": 3500, "against_wall": True, "mask": False},
            {"name": "sofa_3_seat", "x": 2500, "y": 2000, "against_wall": True},  # 1.9 m off
        ],
    )
    boiler, bed = out["inserted"]
    assert boiler["rotation"] == pytest.approx(90.0)  # back on x = 4875, front towards -X
    # back_center on the face: footprint back corners on x = 4875, around y = 2000
    (bx0, by0), (bx1, by1) = boiler["footprint"][:2]
    assert bx0 == pytest.approx(4875.0) and bx1 == pytest.approx(4875.0)
    assert (by0 + by1) / 2 == pytest.approx(2000.0)
    assert bed["rotation"] == pytest.approx(180.0) and bed["block"] == "ARCH_DOUBLE_BED_NOMASK"
    block = backend._doc.blocks.get("ARCH_DOUBLE_BED_NOMASK")
    assert not [e for e in block if e.dxftype() == "WIPEOUT"]
    (refused,) = out["refused"]
    assert refused["i"] == 2 and "no wall face within 1000 mm" in refused["message"]
