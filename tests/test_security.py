"""Security tests for path validation and command sanitization."""

from __future__ import annotations

import os

import pytest
from fastmcp.exceptions import ToolError

from security import validate_path

# ---------------------------------------------------------------------------
# Path validation tests
# ---------------------------------------------------------------------------


class TestValidatePath:
    """Tests for validate_path function."""

    def test_valid_absolute_path(self, tmp_path):
        """Valid absolute path should resolve correctly."""
        test_file = tmp_path / "drawing.dxf"
        test_file.touch()
        result = validate_path(str(test_file), allow_write=False)
        assert result.exists()

    def test_empty_path_rejected(self):
        """Empty path should raise ToolError."""
        with pytest.raises(ToolError, match="cannot be empty"):
            validate_path("")

    def test_whitespace_path_rejected(self):
        """Whitespace-only path should raise ToolError."""
        with pytest.raises(ToolError, match="cannot be empty"):
            validate_path("   ")

    def test_traversal_dot_dot_slash(self):
        """Path with ../ should be rejected."""
        with pytest.raises(ToolError, match="traversal"):
            validate_path("../../../etc/passwd")

    def test_traversal_dot_dot_backslash(self):
        """Path with ..\\ should be rejected."""
        with pytest.raises(ToolError, match="traversal"):
            validate_path("..\\..\\Windows\\System32\\cmd.exe")

    def test_traversal_encoded(self):
        """URL-encoded traversal should be rejected."""
        with pytest.raises(ToolError, match="traversal"):
            validate_path("%2e%2e/secret")

    def test_null_byte(self):
        """Null byte in path should be rejected."""
        with pytest.raises(ToolError, match="traversal"):
            validate_path("file\x00.dxf")

    def test_write_requires_existing_parent(self, tmp_path):
        """Write mode should check parent directory exists."""
        nonexistent = tmp_path / "nonexistent_dir" / "file.dxf"
        with pytest.raises(ToolError, match="Parent directory"):
            validate_path(str(nonexistent), allow_write=True)

    def test_write_with_valid_parent(self, tmp_path):
        """Write mode should pass when parent exists."""
        target = tmp_path / "output.dxf"
        result = validate_path(str(target), allow_write=True)
        assert result.parent.exists()

    def test_allowed_paths_enforcement(self, tmp_path, monkeypatch):
        """When ALLOWED_PATHS is set, paths outside should be rejected."""
        from config import Settings

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        monkeypatch.setenv("ALLOWED_PATHS", str(allowed_dir))

        import config

        config.settings = Settings()

        blocked = tmp_path / "blocked" / "file.dxf"
        blocked.parent.mkdir(exist_ok=True)
        blocked.touch()
        with pytest.raises(ToolError, match="not inside any allowed"):
            validate_path(str(blocked))

        config.settings = Settings.__new__(Settings)
        config.settings.allowed_paths = []
        monkeypatch.delenv("ALLOWED_PATHS", raising=False)
        config.settings = Settings()

    def test_allowed_paths_permits_inside(self, tmp_path, monkeypatch):
        """When ALLOWED_PATHS is set, paths inside should be allowed."""
        from config import Settings

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        test_file = allowed_dir / "drawing.dxf"
        test_file.touch()
        monkeypatch.setenv("ALLOWED_PATHS", str(allowed_dir))

        import config

        config.settings = Settings()

        result = validate_path(str(test_file))
        assert result.exists()

        monkeypatch.delenv("ALLOWED_PATHS", raising=False)
        config.settings = Settings()


# ---------------------------------------------------------------------------
# Command sanitization tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LISP sanitization tests
# ---------------------------------------------------------------------------


# ── T0.8-A — what the guards actually guard ────────────────────────────────


class TestNonLocalPathAnchors:
    r"""Windows accepts several spellings of the same file that compare unequal.

    ``_BLOCKED_PATH_PATTERNS`` rejects the backslash forms, but ``Path.resolve()``
    normalises the *forward-slash* spellings into them afterwards — so
    ``//?/C:/Windows/win.ini`` slipped past the pattern check and then compared
    unequal to ``C:/Windows`` in the system-directory loop. Proven: it returned a
    handle that ``os.path.samefile``'d ``C:\Windows\win.ini``.
    """

    @pytest.mark.skipif(os.name != "nt", reason="drive anchors are a Windows concept")
    def test_extended_length_prefix_is_rejected(self):
        with pytest.raises(ToolError, match="not a plain local drive path"):
            validate_path("//?/C:/Windows/win.ini")

    @pytest.mark.skipif(os.name != "nt", reason="drive anchors are a Windows concept")
    def test_unc_forward_slash_form_is_rejected(self):
        with pytest.raises(ToolError, match="not a plain local drive path"):
            validate_path("//localhost/C$/Windows/win.ini")

    @pytest.mark.skipif(os.name != "nt", reason="drive anchors are a Windows concept")
    def test_device_form_still_reaches_the_system_directory_check(self):
        """``//./C:/...`` normalises to a plain drive path — it must be caught
        by the system-directory rule, not by the anchor rule, or the anchor
        check would be masking rather than adding a guard."""
        with pytest.raises(ToolError, match="system directory"):
            validate_path("//./C:/Windows/win.ini")

    @pytest.mark.skipif(os.name != "nt", reason="drive anchors are a Windows concept")
    def test_an_ordinary_path_is_unaffected(self, tmp_path):
        assert validate_path(str(tmp_path / "part.dxf"))
