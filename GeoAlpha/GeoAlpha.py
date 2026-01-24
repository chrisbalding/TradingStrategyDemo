"""
GeoAlpha trading lifecycle model.

- Generator: produces a new random number on an interval and notifies subscribers.
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

# Try relative imports first (package), then fall back to local imports so the
# module can be executed directly as a script.
try:
    from .generator import Generator
    from .strategy import Strategy
    from .trader import Trader
except Exception:
    from generator import Generator  # type: ignore
    from strategy import Strategy  # type: ignore
    from trader import Trader  # type: ignore

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


# Demo / integration example
async def main_demo() -> None:

    print("*** Demo started ***")

    print("*** Create Generators ***")
    interval = 1.0 # would normally be higher (e.g., 5.0 seconds)
    system_cycle = interval * 5
    g1 = Generator("USDJPY", interval=interval, initial_price=155.69)
    g2 = Generator("GBPUSD", interval=interval, initial_price=1.36)
    g3 = Generator("EURUSD", interval=interval, initial_price=1.18)

    print("*** Start Generators ***")
    g1.start()
    g2.start()
    g3.start()

    print ("*** Create strategies with different decision methods ***")
    sA = Strategy("S-A", method="mean", decision_fn=lambda vals: sum(vals) / len(vals))
    sB = Strategy("S-B", method="median", decision_fn=statistics.median)
    sC = Strategy("S-C", method="geometric_mean", decision_fn=geometric_mean)
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

    # Let system run for a few cycles
    await asyncio.sleep(system_cycle)

    # Dynamically add a new strategy that listens to all generators (example using median)
    sE = Strategy("S-E", method="median", decision_fn=statistics.median)
    sE.attach_generator(g1)
    sE.attach_generator(g2)
    sE.attach_generator(g3)
    trader.attach_strategy(sE)
    print("Added strategy S-E (subscribes to G1,G2,G3)")

    await asyncio.sleep(system_cycle)

    # Stop one generator and remove one strategy to demonstrate resilience
    print("Shutting down generator G2 and removing strategy S-B")
    await g2.stop()
    sB.shutdown()
    trader.detach_strategy(sB)

    await asyncio.sleep(system_cycle)

    # Clean shutdown
    print("Shutting down remaining components...")
    sA.shutdown()
    sC.shutdown()
    sD.shutdown()
    sE.shutdown()
    trader.shutdown()
    await g1.stop()
    await g3.stop()
    print("*** Demo finished ***")


if __name__ == "__main__":
    try:
        asyncio.run(main_demo())
    except KeyboardInterrupt:
        pass
