"""Drawing audit (the module kept its pre-1.5 name; the solid tools are gone).

Split out of the single 1854-line ``AutoCADBackend`` ABC in v1.5.0 (M7).
``AutoCADBackend`` composes every contract in this package, so importing
``backends.base.AutoCADBackend`` is unchanged for callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class SolidContract(ABC):
    @abstractmethod
    async def drawing_audit(self) -> dict:
        """Audit the drawing, applying every repair the backend can make.

        This MUTATES the drawing. Returns ``{"ok", "repaired", "fixes",
        "fix_count", "errors", "error_count"}``, where ``fixes`` are problems
        already repaired and ``errors`` are problems that could not be; both are
        lists of ``{"code": int, "name": str, "message": str, "handle": str |
        None}``.

        A backend that repairs but cannot observe its own result reports
        ``repaired``/``fix_count``/``error_count`` as ``None`` — never ``0``,
        which would claim nothing was wrong — plus ``"detail": "unavailable"``
        and ``"capability": "audit_detail"`` so a client can pre-check the
        boundary. Any implementation that repairs MUST mark the document dirty,
        or the repair is discarded the next time the document is closed.
        """
        ...

    @abstractmethod
    async def drawing_close(self, save: bool = True) -> dict: ...

    @abstractmethod
    async def drawing_undo(self) -> dict: ...

    @abstractmethod
    async def drawing_redo(self) -> dict: ...
