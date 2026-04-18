"""Run CPU-heavy viz work off the asyncio event loop."""

from __future__ import annotations

import asyncio
from typing import Callable, TypeVar

T = TypeVar("T")


async def run_in_thread(fn: Callable[..., T], *args, **kwargs) -> T:
    """Execute ``fn`` in a worker thread (``asyncio.to_thread``)."""
    return await asyncio.to_thread(fn, *args, **kwargs)
