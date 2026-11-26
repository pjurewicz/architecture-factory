"""Reading a DWG on the headless engine (K1, 1.8.0): convert it to DXF first.

ezdxf reads DXF only. ``drawing_open`` on a ``.dwg`` path (the extension is
authoritative, N2) therefore converts it through the ODA File Converter
(``ezdxf.addons.odafc``, the route ``drawing_export_dwg`` already writes with)
when it is installed, and is refused with capability ``dwg_read`` otherwise.

A conversion that fails is refused ``dwg_conversion_failed`` with the attempt
(route, reason, seconds). Nothing is opened and the open document is unchanged
either way. (Upstream's second route, AutoCAD's ``accoreconsole.exe``, needs
Windows and an AutoCAD install; this fork removed it with the live engine.)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .base import FeatureCapability, UnsupportedCapabilityError

__all__ = [
    "DwgConversionError",
    "capability",
    "read_dwg",
    "refusal",
]

_ODA_URL = "https://www.opendesign.com/guestfiles/oda_file_converter"


class DwgConversionError(RuntimeError):
    """The DWG -> DXF conversion failed; nothing was opened.

    ``attempts`` lists ``{route, reason, seconds}`` for the route tried.
    """

    kind = "dwg_conversion_failed"

    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = [dict(row) for row in attempts]
        self.details = {"attempts": self.attempts}


def capability(odafc: bool) -> FeatureCapability:
    """The ezdxf map's ``dwg_read`` row, re-evaluated per call."""
    if odafc:
        return FeatureCapability(True, "odafc", reason="oda_file_converter_installed")
    return FeatureCapability(False, reason="requires_oda_file_converter")


def refusal(path: str, backend: str = "ezdxf") -> UnsupportedCapabilityError:
    """``dwg_read``: no converter on this machine."""
    return UnsupportedCapabilityError(
        "dwg_read",
        f"drawing_open: the headless {backend} backend cannot read '{path}': a DWG is read by "
        "converting it to DXF and the ODA File Converter is not installed here. Nothing was "
        f"opened and the open document is unchanged. Install the ODA File Converter "
        f"({_ODA_URL}), or open the .dxf instead.",
    )


def _read_with_odafc(source: str):
    from ezdxf.addons import odafc

    return odafc.readfile(source)


def read_dwg(path: str, *, odafc: bool) -> tuple[object, dict]:
    """``(document, dwg_read report)`` for the DWG at ``path``.

    ``odafc`` says whether the ODA File Converter is installed (the engine's
    own probe). Refused ``dwg_read`` without it, raises
    :class:`DwgConversionError` when the conversion failed.
    """
    if not odafc:
        raise refusal(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"drawing_open: {path} does not exist")
    started = time.monotonic()
    try:
        doc = _read_with_odafc(path)
    except Exception as exc:  # noqa: BLE001 - reported as the failed attempt
        reason = f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:400]
        attempt = {
            "route": "odafc",
            "reason": reason,
            "seconds": round(time.monotonic() - started, 3),
        }
        raise DwgConversionError(
            f"drawing_open: {Path(path).name} could not be converted to DXF (odafc: {reason}). "
            "Nothing was opened and the open document is unchanged.",
            [attempt],
        ) from None
    report = {"route": "odafc", "seconds": round(time.monotonic() - started, 3)}
    report["dxf_version"] = getattr(doc, "dxfversion", None)
    return doc, report
