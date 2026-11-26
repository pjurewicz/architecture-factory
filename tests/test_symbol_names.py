"""A linetype name is checked by the DXF symbol-name rule before it is used.

A newline and ``;`` end an AutoCAD macro segment, so a name carrying either
could inject further commands wherever a name reaches a command line. The
guard here is the DXF symbol-name rule rather than a bespoke blocklist:
AutoCAD already forbids these characters in a table name, so anything this
rejects was never a loadable linetype, and the injection characters fall out of
the rule rather than having to be enumerated.
"""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

from security import sanitize_symbol_name


@pytest.mark.parametrize(
    "name",
    [
        "CENTER",
        "HIDDEN2",
        "DASHDOT",
        "ACAD_ISO02W100",
        "My-Linetype_01",
    ],
)
def test_ordinary_linetype_names_pass(name):
    assert sanitize_symbol_name(name, kind="linetype") == name


@pytest.mark.parametrize(
    "payload",
    [
        "CENTER\n_ERASE ALL\n",  # newline ends the macro segment
        "CENTER;_ERASE ALL",  # so does a semicolon
        "CENTER\r_QUIT",
        'CENTER"',
        "CENTER|X",
        "CENTER<X",
        "CENTER>X",
        "CENTER?X",
        "CENTER*X",
        "CENTER,X",
        "CENTER=X",
        "CENTER:X",
        "CENTER/X",
        "CENTER" + chr(92) + "X",
    ],
)
def test_a_name_that_could_end_the_macro_segment_is_refused(payload):
    with pytest.raises(ToolError) as excinfo:
        sanitize_symbol_name(payload, kind="linetype")

    assert "linetype" in str(excinfo.value).lower()


def test_an_empty_name_is_refused():
    with pytest.raises(ToolError):
        sanitize_symbol_name("   ", kind="linetype")


def test_the_refusal_does_not_echo_the_payload_back_verbatim():
    """An error message is a channel too."""
    with pytest.raises(ToolError) as excinfo:
        sanitize_symbol_name("CENTER\n_ERASE ALL\n", kind="linetype")

    assert "\n_ERASE" not in str(excinfo.value)
