from __future__ import annotations
import asyncio
import random
import time
import math
from typing import Callable, List, Optional

Subscriber = Callable[[str, float, float], None]  # (source_id, value, timestamp)


class Generator:
    """
    Produces a stream of FX-like prices every `interval` seconds.
    Subscribers are called with (generator_id, value, timestamp).

    Price evolution uses geometric Brownian motion:
        S_next = S * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    where `sigma` is `volatility` (annualized) and `dt` is fraction of year
    corresponding to `interval`. `mu` is the drift (default 0).
    """

    _SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0

    def __init__(
        self,
        generator_id: str,
        interval: float = 5.0,
        initial_price: float = 1.0,
        volatility: float = 0.10,
        mu: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.id = generator_id
        self.interval = interval
        self._price: float = float(initial_price)
        self._volatility: float = float(volatility)  # annualized volatility (sigma)
        self._mu: float = float(mu)  # annualized drift
        self._rng = rng or random.Random()
        self._subscribers: List[Subscriber] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def price(self) -> float:
        """Latest generated price."""
        return self._price

    @property
    def volatility(self) -> float:
        """Annualized volatility (sigma)."""
        return self._volatility

    @volatility.setter
    def volatility(self, v: float) -> None:
        self._volatility = float(v)

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

                # time step as fraction of year
                dt = max(self.interval, 0.0) / self._SECONDS_PER_YEAR

                # If volatility or dt is zero, produce deterministic evolution
                if self._volatility <= 0.0 or dt <= 0.0:
                    next_price = self._price * math.exp(self._mu * dt)
                else:
                    z = self._rng.gauss(0.0, 1.0)
                    drift = (self._mu - 0.5 * (self._volatility ** 2)) * dt
                    diffusion = self._volatility * math.sqrt(dt) * z
                    next_price = self._price * math.exp(drift + diffusion)

                # protect against non-positive prices (shouldn't happen with GBM, but be safe)
                if next_price <= 0.0:
                    next_price = max(1e-12, abs(next_price))

                self._price = next_price
                value = float(self._price)

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
        print(
            f"Generator {self.id} starting: will emit a price every {self.interval} seconds. "
            f"initial_price={self._price:.6f}, volatility={self._volatility:.4f}, mu={self._mu:.4f}"
        )
        loop = loop or asyncio.get_event_loop()
        self._task = loop.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None