from __future__ import annotations
import asyncio
import random
import time
from typing import Callable, List, Optional

Subscriber = Callable[[str, float, float], None]  # (source_id, value, timestamp)


class Generator:
    """
    Produces a stream of random numbers every `interval` seconds.
    Subscribers are called with (generator_id, value, timestamp).
    """

    def __init__(self, generator_id: str, interval: float = 5.0) -> None:
        self.id = generator_id
        self.interval = interval
        self._subscribers: List[Subscriber] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        """Subscribe a callable. Returns an unsubscribe function."""
        self._subscribers.append(fn)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(fn)
            except ValueError:
                pass

        return unsubscribe

    async def _run(self) -> None:
        self._running = True
        try:
            while self._running:
                ts = time.time()
                value = random.random()
                # notify subscribers (do not await them synchronously to avoid blocking)
                for sub in list(self._subscribers):
                    # schedule callback; allow subscriber to be sync or async
                    try:
                        maybe_coro = sub(self.id, value, ts)
                        if asyncio.iscoroutine(maybe_coro):
                            # fire-and-forget
                            asyncio.create_task(maybe_coro)
                    except Exception:
                        # Protect generator loop from subscriber errors
                        pass
                await asyncio.sleep(self.interval)
        finally:
            self._running = False

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        if self._task is not None and not self._task.done():
            return
        # Describe what this Generator will do when starting
        print(f"Generator {self.id} starting: will emit a random value every {self.interval} seconds.")
        loop = loop or asyncio.get_event_loop()
        self._task = loop.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None