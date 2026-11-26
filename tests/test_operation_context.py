"""B12 - the operation id reaches the worker threads and the call_outcomes rows.

ezdxf's ``_async`` runs its worker in ``contextvars.copy_context()`` (W1).
Every cancellation row names the operation it belongs to - that is what lets
the middleware call a cancelled write ``unknown_commit`` and a cancelled queue
entry ``cancelled``.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from backends.operations import CURRENT_OPERATION

pytestmark = pytest.mark.asyncio


def _wait_on(entered: threading.Event, release: threading.Event) -> str:
    entered.set()
    assert release.wait(10), "the test never released the worker"
    return "late"


async def _until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout:.1f}s")
        await asyncio.sleep(0.01)


async def test_a_cancelled_ezdxf_call_records_its_operation_id(backend):
    entered, release = threading.Event(), threading.Event()
    token = CURRENT_OPERATION.set("op-ezdxf-1")
    try:
        task = asyncio.ensure_future(backend._async(_wait_on, entered, release))
    finally:
        CURRENT_OPERATION.reset(token)
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        row = backend.call_outcomes[-1]
        assert (row["call"], row["outcome"]) == ("_wait_on", "cancelled_running")
        assert row["operation_id"] == "op-ezdxf-1"
    finally:
        release.set()
    await _until(lambda: backend.call_outcomes[-1]["ended"] is not None)


async def test_the_ezdxf_worker_sees_the_callers_operation_id(backend):
    token = CURRENT_OPERATION.set("op-ezdxf-2")
    try:
        assert await backend._async(CURRENT_OPERATION.get) == "op-ezdxf-2"
    finally:
        CURRENT_OPERATION.reset(token)
