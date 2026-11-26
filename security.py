"""Security utilities for AutoCAD MCP Pro.

Path validation and the symbol-name / macro-argument guards.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from fastmcp.exceptions import ToolError

import config

log = logging.getLogger("autocad_mcp.security")


class PathRefusedError(ToolError):
    """``validate_path`` refused a path: a traversal pattern, a non-local
    anchor, a system directory, or a path outside ``ALLOWED_PATHS``.

    A ``ToolError`` as before (same message, same handling everywhere); the
    type is what lets ``engineering.errors.classify`` say ``path_refused``.
    """

    kind = "path_refused"

    def __init__(self, message: str, **details):
        super().__init__(message)
        #: The refusal's own envelope fields (1.8-17: ``reason``, ``extension``,
        #: ``allowed``); empty for the 1.7 traversal / directory refusals.
        self.details = {key: value for key, value in details.items() if value is not None}


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

_BLOCKED_PATH_PATTERNS = re.compile(
    r"(\.\.[\\/])"
    r"|(%2e%2e)"
    r"|(\\\\[?.]\\)"
    r"|(^\\\\)"
    r"|(\x00)",
    re.IGNORECASE,
)


_DRIVE_ANCHOR = re.compile(r"^[A-Za-z]:$")


def _reject_non_local_anchor(original: str, resolved: Path) -> None:
    r"""Every guard below compares against plain ``X:\`` paths.

    Windows accepts three other anchors that name the same files but compare
    unequal: the extended-length prefix (``\\?\C:\...``), UNC (``\\host\share``)
    and the admin shares (``\\localhost\C$``). The backslash spellings are
    already rejected by ``_BLOCKED_PATH_PATTERNS``, but ``pathlib`` normalises
    the *forward-slash* spellings into them only after ``.resolve()`` — so they
    slipped past the pattern check and then failed to match ``C:/Windows`` in
    the system-directory loop. Proven: ``//?/C:/Windows/win.ini`` handed back a
    working handle to ``C:\Windows\win.ini``.

    A hard input rejection, so a plain ``ToolError`` like its siblings — not a
    capability refusal, which means "this backend cannot" and would wrongly
    suggest another engine would accept it.
    """
    drive = resolved.drive
    if _DRIVE_ANCHOR.match(drive):
        return
    log.warning("Blocked non-local path anchor: %s -> %s", original, resolved)
    raise PathRefusedError(
        f"Path rejected: '{original}' resolves to '{resolved}', which is not a "
        f"plain local drive path (anchor '{drive}'). Extended-length (\\\\?\\), "
        "device and UNC (\\\\host\\share, \\\\localhost\\C$) paths bypass the "
        "system-directory and allowed-root checks and are not accepted. Use a "
        "normal 'X:\\folder\\file.dxf' path."
    )


#: ``kinds=DIRECTORY``: the write target is a folder the tool writes fixed,
#: named files into (``drawing_deliver``, ``batch_plot``), each of which is
#: validated with its own extension set when the tool names it.
DIRECTORY = "directory"

#: File names AutoCAD (or Windows) loads or runs: never written by any tool,
#: whatever ``kinds`` or ``ALLOWED_PATHS`` say (1.8-17). ``acad*.lsp`` and
#: ``acaddoc*.lsp`` are the autoload names; every ``.lsp`` is refused, as is
#: compiled LISP, menu LISP, scripts, ObjectARX / DBX modules and the
#: Windows executables, and the command-alias file ``acad.pgp``.
WRITE_REFUSED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".lsp",
        ".fas",
        ".vlx",
        ".mnl",
        ".scr",
        ".arx",
        ".dbx",
        ".dll",
        ".exe",
        ".bat",
        ".cmd",
        ".ps1",
        ".vbs",
        ".js",
    }
)
WRITE_REFUSED_NAMES: frozenset[str] = frozenset({"acad.pgp"})

#: The extension sets the writers pass as ``kinds`` (1.8-17).
KINDS_DRAWING: tuple[str, ...] = (".dxf", ".dwg", ".dwt")
KINDS_DXF: tuple[str, ...] = (".dxf",)
KINDS_DWG: tuple[str, ...] = (".dwg",)
KINDS_TEMPLATE: tuple[str, ...] = (".dxf", ".dwt")
KINDS_PDF: tuple[str, ...] = (".pdf",)
KINDS_TABLE: tuple[str, ...] = (".csv", ".xlsx")
KINDS_CSV: tuple[str, ...] = (".csv",)
KINDS_IMAGE: tuple[str, ...] = (".png", ".jpg")
KINDS_REPORT: tuple[str, ...] = (".json", ".md")


def _write_name(resolved: Path) -> str:
    """The name Windows will actually create: trailing dots and spaces dropped."""
    return resolved.name.rstrip(" .").lower()


def _check_write_target(original: str, resolved: Path, kinds) -> None:
    """The extension rules of a write (1.8-17); raises ``PathRefusedError``."""
    name = _write_name(resolved)
    if ":" in resolved.name:
        # An NTFS alternate data stream (`report.csv:acaddoc.lsp`) is a second
        # file the extension check below would never see.
        raise PathRefusedError(
            f"Path rejected: '{original}' names an alternate data stream (':' in the file "
            "name); write a plain file.",
            reason="alternate_data_stream",
        )
    suffix = Path(name).suffix
    if suffix in WRITE_REFUSED_SUFFIXES or name in WRITE_REFUSED_NAMES:
        log.warning("Blocked write of a loadable/executable file: %s", resolved)
        raise PathRefusedError(
            f"Path rejected: '{resolved.name}' is a file AutoCAD or Windows loads or runs "
            f"({suffix or name}); no tool writes one, inside ALLOWED_PATHS or not.",
            reason="executable_name",
            extension=suffix or name,
        )
    if kinds is None:
        return
    if kinds == DIRECTORY:
        if resolved.exists() and not resolved.is_dir():
            raise PathRefusedError(
                f"Path rejected: '{resolved}' is a file; name the folder to write into.",
                reason="not_a_directory",
            )
        return
    allowed = tuple(str(kind).lower() for kind in kinds)
    names = [kind or "(no extension)" for kind in allowed]
    shown = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
    if resolved.is_dir():
        raise PathRefusedError(
            f"Path rejected: '{resolved}' is a directory; name the {shown} file to write.",
            reason="is_a_directory",
            allowed=list(allowed),
        )
    if suffix not in allowed:
        raise PathRefusedError(
            f"Path rejected: '{resolved.name}' must end in {shown} - this tool "
            f"writes only those (got {suffix or 'no extension'!r}).",
            reason="extension_not_allowed",
            extension=suffix,
            allowed=list(allowed),
        )


def validate_path(path: str, *, allow_write: bool = False, kinds=None) -> Path:
    """Resolve and validate a file path against security rules.

    Raises ToolError if:
      - Path contains traversal sequences (..)
      - Path is outside ALLOWED_PATHS (when configured)
      - Path targets a system directory

    With ``allow_write=True`` (1.8-17) also:
      - the server's tool policy forbids writes (``READ_ONLY``, a ceiling below
        ``mutate``, a read-only token) - ``PolicyRefusedError``, kind
        ``policy_refused``, whatever the calling tool's label;
      - the name is one AutoCAD or Windows loads or runs (``*.lsp``,
        ``acad.pgp``, ...: :data:`WRITE_REFUSED_SUFFIXES`) - always, even
        inside ``ALLOWED_PATHS``; kind ``path_refused``;
      - ``kinds`` (the writer's extension set, e.g. ``(".pdf",)``, or
        :data:`DIRECTORY`) does not admit the name - kind ``path_refused``.
        Every writer passes its ``kinds`` (``tests/test_write_path_policy.py``).
    """
    if not path or not path.strip():
        raise ToolError("File path cannot be empty.")

    if _BLOCKED_PATH_PATTERNS.search(path):
        log.warning("Blocked path traversal attempt: %s", path)
        raise PathRefusedError(
            f"Path rejected: suspicious traversal pattern detected in '{path}'. "
            "Use absolute paths without '..' components."
        )

    resolved = Path(path).resolve()

    # After .resolve() (drive is only meaningful once resolved) and before the
    # system-directory loop, whose comparisons this is protecting. POSIX has no
    # drive component at all, so the check is Windows-only by construction.
    if os.name == "nt":
        _reject_non_local_anchor(path, resolved)

    _SYSTEM_DIRS = [
        Path("C:/Windows").resolve(),
        Path("C:/Program Files").resolve(),
        Path("C:/Program Files (x86)").resolve(),
        Path("/etc").resolve(),
        Path("/usr").resolve(),
        Path("/bin").resolve(),
        Path("/sbin").resolve(),
        Path("/boot").resolve(),
        Path("/sys").resolve(),
        Path("/proc").resolve(),
    ]
    for sys_dir in _SYSTEM_DIRS:
        try:
            if resolved.is_relative_to(sys_dir):
                log.warning("Blocked system directory access: %s", resolved)
                raise PathRefusedError(
                    f"Path rejected: '{resolved}' is inside a system directory. "
                    "File operations in system directories are not allowed."
                )
        except (ValueError, TypeError):
            pass

    if config.settings.allowed_paths:
        allowed = False
        for allowed_dir in config.settings.allowed_paths:
            try:
                if resolved.is_relative_to(allowed_dir):
                    allowed = True
                    break
            except (ValueError, TypeError):
                pass
        if not allowed:
            allowed_str = ", ".join(str(p) for p in config.settings.allowed_paths)
            log.warning("Path outside allowed directories: %s", resolved)
            raise PathRefusedError(
                f"Path rejected: '{resolved}' is not inside any allowed directory. "
                f"Allowed directories: {allowed_str}. "
                "Set ALLOWED_PATHS environment variable to configure."
            )

    if allow_write:
        from policy import check_write

        check_write(resolved)
        _check_write_target(path, resolved, kinds)
        parent = resolved.parent
        if not parent.exists():
            raise ToolError(
                f"Parent directory does not exist: '{parent}'. "
                "Create the directory first or use a valid path."
            )

    return resolved


# ---------------------------------------------------------------------------
# Symbol-table names interpolated into a SendCommand macro
# ---------------------------------------------------------------------------
#
# `_ensure_linetype_loaded` built `_-LINETYPE _LOAD {name} {file}` from a
# caller's `linetype=` argument, and `ComBackend.linetype_load` added a
# caller-supplied path. SendCommand takes a whole macro and treats a newline
# and `;` as segment separators, so either string could append commands that
# no command check ever saw. (v1.6 moved both loaders to the ActiveX
# `Linetypes.Load` member, which takes the name as data; the guard stays,
# because a name this rule refuses was never a loadable linetype either.)
#
# The guard is the DXF symbol-name rule rather than a second blocklist. AutoCAD
# already forbids these characters in a table name, so nothing this rejects was
# ever a loadable linetype, and the injection characters fall out of the rule
# instead of having to be enumerated and kept up to date.

#: Characters DXF forbids in a symbol-table name, plus the whitespace that ends
#: a SendCommand segment.
_FORBIDDEN_SYMBOL_CHARS = set('<>/\\":;?*|,=`') | set("\n\r\t")

#: Long enough for any real linetype, short enough not to be a payload.
_MAX_SYMBOL_NAME = 255


def illegal_symbol_name_chars(name: str) -> list[str]:
    """The characters in `name` that DXF forbids in a symbol-table name, sorted.

    The one rule behind every symbol-name gate in this repo: `sanitize_symbol_name`
    refuses on it for macro interpolation, and pre-write validators (block
    primitive specs, for one) refuse on it so ezdxf's own `DXFValueError` never
    fires mid-write after earlier entities have already landed.
    """
    return sorted({c for c in name if c in _FORBIDDEN_SYMBOL_CHARS or ord(c) < 32})


def sanitize_symbol_name(name: str, *, kind: str = "symbol") -> str:
    """Return `name` if it is a legal DXF symbol-table name, else refuse.

    Used at every point where a caller-controlled name is interpolated into an
    AutoCAD command macro. The refusal deliberately does not echo the rejected
    string back: an error message that a client logs or renders is a channel
    too, and a payload that reaches a log is a payload that reached something.
    """
    candidate = (name or "").strip()
    if not candidate:
        raise ToolError(f"{kind} name rejected: empty.")
    if len(candidate) > _MAX_SYMBOL_NAME:
        raise ToolError(f"{kind} name rejected: longer than {_MAX_SYMBOL_NAME} characters.")
    offenders = illegal_symbol_name_chars(candidate)
    if offenders:
        shown = ", ".join(repr(c) for c in offenders)
        raise ToolError(
            f"{kind} name rejected: DXF symbol-table names cannot contain {shown}. "
            "This name would also end the AutoCAD command macro it is placed in."
        )
    return candidate


#: What ends a segment inside a SendCommand macro. A path legitimately contains
#: ':' and '\', so a file argument cannot be held to the symbol-name rule -- it
#: gets this narrower one plus `validate_path`.
_MACRO_SEPARATORS = set("\n\r;") | {chr(c) for c in range(32)}


def sanitize_macro_argument(value: str, *, kind: str = "argument") -> str:
    """Return `value` if it cannot end the AutoCAD command macro it sits in.

    For arguments that are legitimately allowed the characters DXF forbids in a
    symbol name -- a file path, mostly. Narrower than `sanitize_symbol_name` on
    purpose: the goal is only that the string cannot start a second command.
    """
    candidate = (value or "").strip()
    if not candidate:
        raise ToolError(f"{kind} rejected: empty.")
    if any(c in _MACRO_SEPARATORS for c in candidate):
        raise ToolError(
            f"{kind} rejected: contains a newline, carriage return, semicolon or "
            "control character, which would end the AutoCAD command macro it is "
            "placed in and start another."
        )
    return candidate
