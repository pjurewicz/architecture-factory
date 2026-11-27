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
import select
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from .base import FeatureCapability, UnsupportedCapabilityError

__all__ = [
    "DwgConversionError",
    "capability",
    "odafc_module",
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


@contextmanager
def _ready_xvfb_display():
    """A private Xvfb display, yielded once it accepts connections.

    Replaces ezdxf's ``odafc._linux_dummy_display``, which starts Xvfb on
    ``:<pid>`` and sleeps 0.1 s: on a loaded machine Xvfb is not up yet and the
    converter aborts with "could not connect to display"; two conversions in one
    process also claim the same display. ``-displayfd`` makes Xvfb pick a free
    display and write its number only when it is ready.

    The wait is bounded by ``XVFB_START_TIMEOUT`` (seconds, default 30) and
    watches the process too: an Xvfb that exits is reported at once, with the
    end of its stderr, even when a child it started still holds the pipe.
    """
    if not shutil.which("Xvfb"):
        with _EZDXF_DUMMY_DISPLAY() as display:
            yield display
        return
    timeout = float(os.environ.get("XVFB_START_TIMEOUT", "").strip() or _XVFB_START_TIMEOUT_S)
    read_fd, write_fd = os.pipe()
    with tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(
            ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "800x600x24"],
            pass_fds=(write_fd,),
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )
        os.close(write_fd)

        def failed(what: str) -> RuntimeError:
            stderr.seek(0)
            tail = " ".join(stderr.read().decode(errors="replace").split())[-300:]
            return RuntimeError(f"Xvfb {what}" + (f"; stderr: {tail}" if tail else ""))

        try:
            with os.fdopen(read_fd) as ready:
                deadline = time.monotonic() + timeout
                while not select.select([ready], [], [], 0.1)[0]:
                    if proc.poll() is not None:
                        raise failed(f"exited before it was ready (exit code {proc.returncode})")
                    if time.monotonic() >= deadline:
                        raise failed(f"did not start within {timeout:g} s (XVFB_START_TIMEOUT)")
                number = ready.readline().strip()
            if not number:
                raise failed(f"exited before it was ready (exit code {proc.wait()})")
            yield f":{number}"
        finally:
            proc.terminate()
            proc.wait()


#: Measured: 0.04 s warm, 0.2 s beside a parallel test run, 2.3 s from a cold
#: cache. The 1.8.0 value, 10 s, was exceeded once on a loaded CI runner.
_XVFB_START_TIMEOUT_S = 30.0
_EZDXF_DUMMY_DISPLAY = None


def odafc_module():
    """``ezdxf.addons.odafc``, running the converter on a display that is ready."""
    global _EZDXF_DUMMY_DISPLAY
    from ezdxf.addons import odafc

    if _EZDXF_DUMMY_DISPLAY is None:
        _EZDXF_DUMMY_DISPLAY = odafc._linux_dummy_display
        odafc._linux_dummy_display = _ready_xvfb_display
    return odafc


def _read_with_odafc(source: str):
    return odafc_module().readfile(source)


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
