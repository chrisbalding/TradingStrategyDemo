from __future__ import annotations
import asyncio
from typing import Callable, Dict, List

# Try relative import for package usage, fall back to local import so the top-level
# script can still be executed directly (python GeoAlpha.py).
try:
    from .generator import Generator
except Exception:
    from generator import Generator  # type: ignore

Subscriber = Callable[[str, float, float], None]  # (source_id, value, timestamp)
DecisionFn = Callable[[List[float]], float]


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
        method: str,
        decision_fn: DecisionFn,
    ) -> None:
        self.id = strategy_id
        self._generator_unsub: Dict[str, Callable[[], None]] = {}
        self._latest_values: Dict[str, float] = {}
        self._subscribers: List[Subscriber] = []
        self._decision_fn: DecisionFn = decision_fn
        self._method_name: str = method

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