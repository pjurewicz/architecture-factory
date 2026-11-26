"""The durability docs name each engine's checkpoint points (1.8-08).

Headless, a save, an open, a new drawing and a batch boundary keep a durable
checkpoint (``backends/ezdxf_checkpoints.py``). Live, only
``checkpoint_create``, ``transaction_begin`` and the destructive tools do:
no live save, open or new drawing takes one.
A sentence in the skill or the README that lists save / open / new drawing as
checkpoint points must therefore say *headless*, and the same document must say
the live engine does not, or a model on a live seat expects a restore point at
its last save that does not exist.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REFERENCE = Path("references") / "recovery-and-transactions.md"
DOCS = (
    ROOT / ".claude" / "skills" / "autocad-mcp-premium" / REFERENCE,
    ROOT / "plugin" / "skills" / "premium" / REFERENCE,
    ROOT / "README.md",
)

SAVE_OPEN_NEW = re.compile(r"save, (?:an )?open, (?:a )?new drawing", re.IGNORECASE)
LIVE_NEGATION = "not at a save, an open or a new drawing"


def test_the_docs_scope_save_open_new_checkpoints_to_the_headless_engine():
    for doc in DOCS:
        text = " ".join(doc.read_text(encoding="utf-8").split())
        matches = list(SAVE_OPEN_NEW.finditer(text))
        assert matches, f"{doc.name}: the checkpoint points are no longer listed"
        for match in matches:
            before = text[max(0, match.start() - 60) : match.start()].lower()
            assert "headless" in before, (
                f"{doc.relative_to(ROOT)}: '{text[match.start() - 60 : match.end()]}' "
                "lists save / open / new drawing as checkpoint points without saying headless"
            )
        assert LIVE_NEGATION in text, f"{doc.relative_to(ROOT)}: the live points are not stated"
        for point in ("checkpoint_create", "transaction_begin"):
            assert f"`{point}" in text
