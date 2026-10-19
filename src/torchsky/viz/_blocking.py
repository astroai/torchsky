"""Run blocking torch/IO work off the FastAPI event loop."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

from starlette.concurrency import run_in_threadpool

_T = TypeVar("_T")


async def run_in_thread(fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Await ``fn(*args, **kwargs)`` in Starlette's thread pool."""
    return await run_in_threadpool(partial(fn, *args, **kwargs))
