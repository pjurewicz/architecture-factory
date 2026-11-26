"""One complete read of the drawing per QA run: every space, every entity, no cap.

``drawing_critique`` and the validator used to read ``entity_list(limit=5000)``
(10,000 for the validator) of the *current* space only, so a construction line
at position 5,001, a duplicate pair after the 5,000th line or anything on a
paper layout was never seen - and the result still read "zero issues". This
module is the one reader they share now:

* :func:`qa_snapshot` takes one all-spaces :class:`Snapshot` per run and caches
  it, with its :class:`ScanCoverage`, in ``shared["qa_snapshot"]``. A finalize
  passes one dict to the validator, the critique and the topology focuses, so
  it costs one read, not three.
  A dict is valid only while the drawing does not change: a caller that edits
  between two runs (``drawing_refine``) passes a fresh one.
* Before reading, the entities of every space are counted (``len`` of each
  ezdxf layout). Above
  ``MAX_QA_ENTITIES`` (default 2,000,000; 0 disables) the read is refused and
  the coverage says why - a partial scan is never passed off as a whole one.
* The read runs under the backend's lock as a **read-only** call:
  :func:`_async_readonly` passes ``readonly=True`` *and* marks the function
  (``quarantine.mark_readonly``), so a wrapper around ``_async`` that drops
  the keyword - a monkeypatch, a tracer - still hands ``_async`` a read. An
  overrunning QA read is detached, never quarantined.

The QA read computes no INSERT box (``records_from_doc(...,
insert_boxes=False)``): a box walks the block's geometry, about 1.8 ms an
INSERT, so 5,000 INSERTs of a 20-line block took 4.7-9.3 s under the lock (W1
review, measured) for checks that never read them. The one QA consumer that
does - the topology lane, whose dangling-end test asks whether an end touches
equipment - fills the boxes it needs with :func:`with_insert_boxes`.

A read that fails for any reason other than a quarantine is cached as an
incomplete coverage carrying the reason, so every check of the run reports
``scan_incomplete`` once instead of retrying the read. A quarantine refusal
propagates: a quarantined document must never read as clean.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import config
from backends.quarantine import DOCUMENT_REFUSALS, mark_readonly
from engineering.understand.snapshot import (
    Snapshot,
    _box,
    records_from_doc,
)

__all__ = [
    "QA_SNAPSHOT_KEY",
    "ScanCoverage",
    "entity_infos_all_spaces",
    "qa_snapshot",
    "shared_index",
    "with_insert_boxes",
]

#: The key :func:`qa_snapshot` caches its result under in a run's shared dict.
QA_SNAPSHOT_KEY = "qa_snapshot"


@dataclass(frozen=True)
class ScanCoverage:
    """What a QA read covered: ``complete`` only when every entity of every
    space was read."""

    complete: bool
    records: int
    spaces: tuple[str, ...]
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "complete": self.complete,
            "records": self.records,
            "spaces": list(self.spaces),
            "reason": self.reason,
        }


async def _async_readonly(backend, func):
    """``backend._async(func, readonly=True)``, with ``func`` marked as a read
    too, so the flag survives a wrapper that drops the keyword."""
    return await backend._async(mark_readonly(func), readonly=True)


def _limit() -> int:
    return int(config.settings.max_qa_entities)


def _over_limit(counts: dict[str, int], limit: int) -> str | None:
    total = sum(counts.values())
    if limit > 0 and total > limit:
        spaces = ", ".join(f"{name} {count:,}" for name, count in counts.items())
        return (
            f"{total:,} entities ({spaces}) exceed MAX_QA_ENTITIES={limit:,}; the drawing "
            "was not read. Raise MAX_QA_ENTITIES (0 disables the check) to scan it."
        )
    return None


def _ezdxf_counts(doc) -> dict[str, int]:
    return {name: len(doc.layouts.get(name)) for name in doc.layouts.names_in_taborder()}


async def _read_ezdxf(backend) -> tuple[Snapshot | None, ScanCoverage]:
    limit = _limit()

    def _sync():
        doc = backend._require_doc()
        counts = _ezdxf_counts(doc)
        refusal = _over_limit(counts, limit)
        if refusal is not None:
            return None, counts, refusal
        source = backend._doc_path or "memory:untitled"
        return records_from_doc(doc, source=source, insert_boxes=False), counts, None

    snap, counts, refusal = await _async_readonly(backend, _sync)
    if snap is None:
        return None, ScanCoverage(False, 0, tuple(counts), refusal)
    return snap, ScanCoverage(True, len(snap.records), snap.layouts)


async def _read_all_spaces(backend) -> tuple[Snapshot | None, ScanCoverage]:
    return await _read_ezdxf(backend)


async def qa_snapshot(backend, shared: dict) -> tuple[Snapshot | None, ScanCoverage]:
    """The run's one all-spaces read, cached in ``shared["qa_snapshot"]``.

    Returns ``(snapshot, coverage)``. ``snapshot`` is None when nothing was
    read (over ``MAX_QA_ENTITIES``, or the read failed); a live export whose
    counts disagree with AutoCAD's returns what it read with
    ``coverage.complete`` False. A ``DocumentQuarantineError`` or a live
    ``ActiveDocumentChangedError`` (``DOCUMENT_REFUSALS``) propagates, typed,
    and is not cached.
    """
    cached = shared.get(QA_SNAPSHOT_KEY)
    if cached is not None:
        return cached
    try:
        result = await _read_all_spaces(backend)
    except DOCUMENT_REFUSALS:
        raise
    except Exception as exc:  # noqa: BLE001 - reported as an incomplete scan, never as clean
        reason = f"the drawing could not be read: {type(exc).__name__}: {exc}"
        result = (None, ScanCoverage(False, 0, (), reason))
    shared[QA_SNAPSHOT_KEY] = result
    return result


async def with_insert_boxes(backend, snap: Snapshot, spaces=None) -> Snapshot:
    """``snap`` with the box of every INSERT in ``spaces`` (every space when
    None) filled in - for a QA consumer that reads INSERT boxes.

    Headless, the boxes are read by handle from the document in one read-only
    call; live, the drawing is exported again and read with its boxes (the QA
    export is deleted once read). A handle the drawing no longer holds raises:
    the drawing changed during the run, and a box missing from the lane's
    picture would turn into a false finding. A snapshot that already carries
    its boxes is returned as it is.
    """
    if snap.insert_boxes:
        return snap
    wanted = {
        r.handle
        for r in snap.records
        if r.type == "INSERT" and (spaces is None or r.space in spaces)
    }
    if not wanted:
        return snap

    def _sync():
        doc = backend._require_doc()
        found = {}
        for handle in wanted:
            entity = doc.entitydb.get(handle)
            if entity is not None and entity.is_alive:
                found[handle] = _box(entity)
        return found

    boxes = await _async_readonly(backend, _sync)
    missing = sorted(wanted - set(boxes))
    if missing:
        raise RuntimeError(
            f"{len(missing)} INSERT(s) of this run's QA read are no longer in the drawing "
            f"({', '.join(missing[:5])}); the drawing changed during the run"
        )
    records = tuple(
        replace(r, bbox=boxes[r.handle]) if r.handle in wanted else r for r in snap.records
    )
    return replace(snap, records=records, insert_boxes=spaces is None)


def _ezdxf_infos(doc):
    from backends.ezdxf_backend import _entity_info_dxf

    return [
        (_entity_info_dxf(entity), name)
        for name in doc.layouts.names_in_taborder()
        for entity in doc.layouts.get(name)
    ]


async def entity_infos_all_spaces(backend) -> list:
    """``(EntityInfo, space)`` for every entity of every layout, in tab order,
    read in one locked call and without switching the current layout.

    ``pid_graph(scope="all")`` walked the layouts with ``layout_set_current``,
    which rebinds the active paper layout and marks the document dirty - a
    read that changed the drawing (backlog W3-T14-09). An entity the live
    engine cannot read refuses the whole read instead of being skipped.
    """
    return await _async_readonly(backend, lambda: _ezdxf_infos(backend._require_doc()))


class _FailedBuild:
    """A domain index whose build raised, cached so the run reads once."""

    def __init__(self, error: Exception):
        self.error = error


async def shared_index(shared: dict, key: str, build, backend):
    """``shared[key]``, built once per run by ``await build(backend)``.

    A build that raises is cached and re-raised for every later focus of the
    run, so each focus reports ``check_failed`` (``run_critique`` turns the
    exception into that finding) and the drawing is read once. A quarantine
    refusal propagates and is not cached.
    """
    cached = shared.get(key)
    if isinstance(cached, _FailedBuild):
        raise cached.error
    if cached is not None:
        return cached
    try:
        value = await build(backend)
    except DOCUMENT_REFUSALS:
        raise
    except Exception as exc:
        shared[key] = _FailedBuild(exc)
        raise
    shared[key] = value
    return value
