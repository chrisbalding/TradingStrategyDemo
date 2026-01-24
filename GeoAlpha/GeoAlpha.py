"""
GeoAlpha trading lifecycle model.

- Generator: produces a new random number every 5s and notifies subscribers.
- Strategy: subscribes to one or more Generators, computes a numeric decision
  from the latest values and notifies its subscribers when a new decision is made.
- Trader: subscribes to Strategies, keeps the latest decision from each, computes
  the median across current strategy decisions and prints the result.

Run this module to see a short demo where generators/strategies are created and
destroyed while the trader continues to operate.
"""
from __future__ import annotations
import asyncio
import random
import statistics
import time
import math
from typing import Callable, Dict, List, Optional, Tuple

Subscriber = Callable[[str, float, float], None]  # (source_id, value, timestamp)
DecisionFn = Callable[[List[float]], float]


def geometric_mean(values: List[float]) -> float:
    """Geometric mean. Protect against zeros by using small epsilon if needed."""
    if not values:
        raise ValueError("No values provided")
    # product can underflow for many small numbers; use log-sum approach
    log_sum = 0.0
    for v in values:
        if v <= 0:
            # geometric mean undefined for non-positive values; fallback to arithmetic mean
            return sum(values) / len(values)
        log_sum += math.log(v)
    return math.exp(log_sum / len(values))


def trimmed_mean(values: List[float], trim_fraction: float = 0.1) -> float:
    """Trim the lowest and highest `trim_fraction` of values, then take the mean."""
    if not values:
        raise ValueError("No values provided")
    n = len(values)
    k = int(n * trim_fraction)
    if 2 * k >= n:
        # trimming would remove all values; fallback to simple mean
        return sum(values) / n
    sorted_vals = sorted(values)
    trimmed = sorted_vals[k : n - k]
    return sum(trimmed) / len(trimmed)


def pick_decision_fn(method: Optional[str]) -> DecisionFn:
    """Map a method name to a decision function. Defaults to arithmetic mean."""
    if method is None or method == "mean":
        return lambda vals: sum(vals) / len(vals)
    if method == "median":
        return lambda vals: statistics.median(vals)
    if method == "geometric_mean":
        return geometric_mean
    if method == "trimmed_mean":
        # default trim fraction 10%
        return lambda vals: trimmed_mean(vals, 0.1)
    # unknown method -> default mean
    return lambda vals: sum(vals) / len(vals)


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


class Strategy:
    """
    Subscribes to multiple Generators. Maintains latest value per generator,
    computes a decision using a configurable statistical method whenever any
    subscribed generator produces a new value, and notifies subscribers with
    (strategy_id, decision, timestamp).

    You can pass either a `method` string (e.g. "median", "geometric_mean",
    "trimmed_mean") or a custom `decision_fn` callable that takes a List[float]
    and returns a float.
    """

    def __init__(
        self,
        strategy_id: str,
        method: Optional[str] = None,
        decision_fn: Optional[DecisionFn] = None,
    ) -> None:
        self.id = strategy_id
        self._generator_unsub: Dict[str, Callable[[], None]] = {}
        self._latest_values: Dict[str, float] = {}
        self._subscribers: List[Subscriber] = []
        if decision_fn is not None:
            self._decision_fn: DecisionFn = decision_fn
            self._method_name: str = getattr(decision_fn, "__name__", "custom")
        else:
            self._decision_fn = pick_decision_fn(method)
            # remember method name for logging (fallback to "mean")
            self._method_name: str = method or "mean"

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        self._subscribers.append(fn)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(fn)
            except ValueError:
                pass

        return unsubscribe

    def attach_generator(self, gen: Generator) -> None:
        if gen.id in self._generator_unsub:
            return  # already attached

        # Describe what the strategy will do when attaching to a generator
        print(
            f"Strategy {self.id} attaching to Generator {gen.id}: "
            f"will update latest value from this generator and include it in decisions "
            f"using method '{self._method_name}'."
        )

        def _on_value(gen_id: str, value: float, ts: float):
            # update latest value and produce a decision
            self._latest_values[gen_id] = value
            self._produce_decision(ts)

        unsub = gen.subscribe(_on_value)
        self._generator_unsub[gen.id] = unsub

    def detach_generator(self, gen: Generator) -> None:
        unsub = self._generator_unsub.pop(gen.id, None)
        if unsub:
            unsub()
        self._latest_values.pop(gen.id, None)

    def _produce_decision(self, ts: float) -> None:
        if not self._latest_values:
            return
        values = list(self._latest_values.values())
        try:
            decision = self._decision_fn(values)
        except Exception:
            # fallback to arithmetic mean if custom fn fails
            decision = sum(values) / len(values)
        for sub in list(self._subscribers):
            try:
                maybe_coro = sub(self.id, decision, ts)
                if asyncio.iscoroutine(maybe_coro):
                    asyncio.create_task(maybe_coro)
            except Exception:
                pass

    def shutdown(self) -> None:
        # detach from all generators
        for unsub in list(self._generator_unsub.values()):
            try:
                unsub()
            except Exception:
                pass
        self._generator_unsub.clear()
        self._latest_values.clear()
        self._subscribers.clear()


class Trader:
    """
    Subscribes to strategies' decision outputs. Keeps the latest decision per
    strategy, computes median across all available decisions and prints the
    resulting value whenever any strategy produces a new decision.
    """

    def __init__(self) -> None:
        self._latest_decisions: Dict[str, float] = {}
        self._strategy_unsub: Dict[str, Callable[[], None]] = {}

    def attach_strategy(self, strategy: Strategy) -> None:
        if strategy.id in self._strategy_unsub:
            return

        def _on_decision(strategy_id: str, decision: float, ts: float):
            # update stored decision and compute median
            self._latest_decisions[strategy_id] = decision
            self._emit_median(ts)

        unsub = strategy.subscribe(_on_decision)
        self._strategy_unsub[strategy.id] = unsub

    def detach_strategy(self, strategy: Strategy) -> None:
        unsub = self._strategy_unsub.pop(strategy.id, None)
        if unsub:
            unsub()
        self._latest_decisions.pop(strategy.id, None)

    def _emit_median(self, ts: float) -> None:
        if not self._latest_decisions:
            return
        values = list(self._latest_decisions.values())
        med = statistics.median(values)
        print(f"[{time.strftime('%X', time.localtime(ts))}] Trader median across {len(values)} strategies: {med:.6f}")

    def shutdown(self) -> None:
        for unsub in list(self._strategy_unsub.values()):
            try:
                unsub()
            except Exception:
                pass
        self._strategy_unsub.clear()
        self._latest_decisions.clear()


# Demo / integration example
async def main_demo() -> None:

    print("Demo started. Generators emit every 5 seconds. Watch trader output.")

    # Create generators
    g1 = Generator("G1", interval=5.0)
    g2 = Generator("G2", interval=5.0)
    g3 = Generator("G3", interval=5.0)

    # Start them
    g1.start()
    g2.start()
    g3.start()

    # Create strategies with different decision methods
    sA = Strategy("S-A", method="mean")                 # arithmetic mean (default)
    sB = Strategy("S-B", method="median")               # median
    sC = Strategy("S-C", method="geometric_mean")       # geometric mean
    # Custom trimmed mean with 20% trim using a decision_fn
    sD = Strategy("S-D", method="trimmed_mean_20", decision_fn=lambda vals: trimmed_mean(vals, 0.2))

    # Attach generators to strategies
    sA.attach_generator(g1)
    sA.attach_generator(g2)

    sB.attach_generator(g2)
    sB.attach_generator(g3)

    sC.attach_generator(g1)
    sC.attach_generator(g3)

    sD.attach_generator(g1)
    sD.attach_generator(g2)
    sD.attach_generator(g3)

    # Create trader and attach strategies
    trader = Trader()
    trader.attach_strategy(sA)
    trader.attach_strategy(sB)
    trader.attach_strategy(sC)
    trader.attach_strategy(sD)

    # Let system run for 25 seconds
    await asyncio.sleep(25)

    # Dynamically add a new strategy that listens to all generators (example using median)
    sE = Strategy("S-E", method="median")
    sE.attach_generator(g1)
    sE.attach_generator(g2)
    sE.attach_generator(g3)
    trader.attach_strategy(sE)
    print("Added strategy S-E (subscribes to G1,G2,G3)")

    await asyncio.sleep(20)

    # Stop one generator and remove one strategy to demonstrate resilience
    print("Shutting down generator G2 and removing strategy S-B")
    await g2.stop()
    sB.shutdown()
    trader.detach_strategy(sB)

    await asyncio.sleep(20)

    # Clean shutdown
    print("Shutting down remaining components...")
    sA.shutdown()
    sC.shutdown()
    sD.shutdown()
    sE.shutdown()
    trader.shutdown()
    await g1.stop()
    await g3.stop()
    print("Demo finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main_demo())
    except KeyboardInterrupt:
        pass
