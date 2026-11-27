"""External references, raster images and DWG output (v1.6 tracks B+G, group G).

These are the only four operations of the sheet track that genuinely differ per
engine, so they are the only four that become contract members. Everything else
in `engineering/sheet/` composes the generic members and is written once.

``AutoCADBackend`` composes every contract in this package, so importing
``backends.base.AutoCADBackend`` is unchanged for callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

#: An xref is attached (its own xrefs come with it) or overlaid (they do not).
XREF_KINDS = ("attach", "overlay")

#: `list`, `detach` and `path` are headless; `reload` and `bind` need AutoCAD's
#: xref manager and are refused with the `xref_live` capability.
XREF_ACTIONS = ("list", "reload", "bind", "detach", "path")

#: DWG formats `drawing_export_dwg` writes, mapped through
#: `ezdxf.addons.odafc.VERSION_MAP`, which covers exactly these six and more.
DWG_VERSIONS = ("R2000", "R2004", "R2007", "R2010", "R2013", "R2018")

#: `image_attach`'s layers: a working underlay goes on a layer that does not
#: plot, an image meant for the sheet (a logo, a site map) on one that does.
IMAGE_UNDERLAY_LAYER = "UNDERLAY"
IMAGE_PLOT_LAYER = "IMAGE"
#: AutoCAD's fade (0-90 %) given to an underlay, so CAD lines read over a scan.
IMAGE_UNDERLAY_FADE = 50


class RefsContract(ABC):
    @abstractmethod
    async def xref_attach(
        self,
        path: str,
        at,
        scale: float,
        rotation: float,
        kind: str,
    ) -> dict:
        """Attach an external reference and insert it once.

        The block name is the referenced file's stem, which is what both
        engines do. Returns ``{ok, name, path, kind, handle, at, scale,
        rotation, backend}``.

        Refuses, before anything is written: a `kind` outside
        :data:`XREF_KINDS`, a file that does not exist, a non-finite
        coordinate/scale/rotation, a non-positive scale, and a block name the
        drawing already holds (re-attaching would either silently replace the
        definition or raise mid-write).
        """
        ...

    @abstractmethod
    async def xref_manage(self, name: str, action: str, new_path: str | None) -> dict:
        """One of :data:`XREF_ACTIONS` on one external reference.

        ``list`` ignores `name` and returns
        ``{ok, xrefs: [{name, path, kind, inserts}], backend}``. ``path``
        rewrites the saved path to `new_path` and returns the new one.
        ``detach`` removes every insert of the xref -- in *every* entity space,
        model space and each paper-space layout, because an insert left behind
        with no block definition is a corrupt drawing -- then the definition,
        and reports ``inserts_removed``. ``inserts`` counts the same set.
        ``reload`` and ``bind`` need AutoCAD's xref manager: the headless
        engine raises ``UnsupportedCapabilityError("xref_live", ...)`` rather than pretending,
        which is why the key exists; ``list``/``detach``/``path`` are *not*
        behind it.

        Refuses an action outside the vocabulary, an unknown xref name, and a
        ``path`` action with no `new_path`.
        """
        ...

    @abstractmethod
    async def image_attach(
        self,
        path: str,
        at,
        scale: float,
        rotation: float,
        *,
        plot: bool = False,
        fade: float | None = None,
        layer: str | None = None,
    ) -> dict:
        """Attach a raster image as an underlay.

        `scale` is drawing units per pixel, so the placed size is the image's
        pixel size times `scale` and the aspect ratio is the file's. `plot`
        sets the layer's plot flag, which decides whether a PDF shows the
        image (a screenshot always does): `layer` defaults to
        :data:`IMAGE_UNDERLAY_LAYER` without plot and :data:`IMAGE_PLOT_LAYER`
        with it, created before the image. `fade` (0-90 %) defaults to
        :data:`IMAGE_UNDERLAY_FADE` without plot and 0 with it. Returns
        ``{ok, path, handle, at, scale, rotation, pixels: [w, h],
        size_mm: [w, h], layer, plot, fade, frame, backend}`` (``frame`` is
        the drawing's IMAGEFRAME), plus ``layer_plot_changed`` when an
        existing layer's flag moved.

        Refuses a file that does not exist, a file no image reader can open
        (naming it), a non-positive scale, a non-finite number, a fade outside
        0-90 and an invalid layer name -- all before anything is written.
        """
        ...

    @abstractmethod
    async def drawing_export_dwg(self, path: str, version: str) -> dict:
        """Write the current drawing as a real DWG.

        Returns ``{ok, path, version, bytes, backend}``. Refuses a `version`
        outside :data:`DWG_VERSIONS` by name. Headlessly this needs the ODA
        File Converter and otherwise raises
        ``UnsupportedCapabilityError("dwg_write", ...)`` naming the install
        route.
        """
        ...
