from __future__ import annotations
import statistics
import time
from typing import Callable, Dict

# Try relative import for package usage, fall back to local import so the top-level
# script can still be executed directly (python GeoAlpha.py).
try:
    from .strategy import Strategy
except Exception:
    from strategy import Strategy  # type: ignore

Subscriber = Callable[[str, float, float], None]  # (source_id, value, timestamp)


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