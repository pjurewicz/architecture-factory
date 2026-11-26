"""The durability docs name the checkpoint points (1.8-08).

A save, an open, a new drawing and a batch boundary keep a durable checkpoint
(``backends/ezdxf_checkpoints.py``), as do ``checkpoint_create`` and
``transaction_begin``; the skill and the README must list them.
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


def test_the_docs_list_the_checkpoint_points():
    for doc in DOCS:
        text = " ".join(doc.read_text(encoding="utf-8").split())
        assert SAVE_OPEN_NEW.search(text), f"{doc.name}: the checkpoint points are no longer listed"
        for point in ("checkpoint_create", "transaction_begin"):
            assert f"`{point}" in text, f"{doc.relative_to(ROOT)}: {point} is not named"
