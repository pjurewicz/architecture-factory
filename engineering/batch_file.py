"""A batch read from a JSON file instead of the call's arguments (B6).

``cad_batch(steps_path=)`` and ``entity_batch_create(entities_path=)`` take
the path of a ``.json`` file (an upload's, ``POST /files?name=steps.json``)
holding exactly the list the inline argument would: the model sends a path
instead of retyping the list. The list then goes through the same checks as
the inline one, and a refusal names ``<arg> '<file name>'[index]``.

Bounded: :data:`MAX_BYTES` before the file is read, nesting deeper than
:data:`MAX_DEPTH` refused (Python's parser would otherwise end in a
``RecursionError``), and the list at most the caller's own item ceiling.
JSON has no entity expansion, so a "JSON bomb" is only its size and depth.
"""

from __future__ import annotations

import json
from pathlib import Path

MAX_BYTES = 16 * 1024 * 1024
MAX_DEPTH = 64


def _depth(value, limit: int) -> int:
    """The nesting depth of ``value``, stopping once it passes ``limit``."""
    deepest = 0
    stack = [(value, 1)]
    while stack:
        item, level = stack.pop()
        if isinstance(item, dict | list):
            deepest = max(deepest, level)
            if deepest > limit:
                return deepest
            children = item.values() if isinstance(item, dict) else item
            stack.extend((child, level + 1) for child in children)
    return deepest


def parse(data: bytes, where: str):
    """``data`` as JSON, or ``ValueError`` naming ``where``."""
    if len(data) > MAX_BYTES:
        raise ValueError(f"{where}: {len(data)} bytes is over MAX_BYTES={MAX_BYTES}.")
    try:
        value = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{where}: not UTF-8 JSON ({str(exc)[:120]}).") from None
    if _depth(value, MAX_DEPTH) > MAX_DEPTH:
        raise ValueError(f"{where}: nested deeper than MAX_DEPTH={MAX_DEPTH}.")
    return value


def load_list(path, arg: str) -> tuple[list, str]:
    """The JSON list in ``path`` (already validated) and its label ``<arg> '<name>'``."""
    target = Path(path)
    label = f"{arg} '{target.name}'"
    if target.suffix.lower() != ".json":
        raise ValueError(f"{label}: a .json file, got {target.suffix or 'no extension'!r}.")
    size = target.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"{label}: {size} bytes is over MAX_BYTES={MAX_BYTES}.")
    value = parse(target.read_bytes(), label)
    if not isinstance(value, list):
        raise ValueError(f"{label}: the file must hold a JSON list, got {type(value).__name__}.")
    return value, label
