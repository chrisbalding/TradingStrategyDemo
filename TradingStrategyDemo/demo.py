"""
TradingStrategyDemo trading lifecycle model.

- Generator: produces FX like prices on an interval and notifies subscribers.
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


def max_seen() -> DecisionFn:
    """
    Factory that returns a decision function which tracks the largest value seen
    across invocations. The returned callable accepts the current list of latest
    values and returns the maximum observed so far.
    """
    max_value = float("-inf")

    def _fn(values: List[float]) -> float:
        nonlocal max_value
        if not values:
            return max_value
        current_max = max(values)
        if current_max > max_value:
            max_value = current_max
            print(f"New max seen: {max_value}")
        return max_value

    return _fn


def trailing_stop_sell(stop_distance: float = 0.001) -> DecisionFn:
    """
    Factory that returns a decision function which maintains a trailing stop level 
    at a stop_distance below the maximum observed value. If the current value
    drops below the trailing stop, it signals a sell by returning the null value
    """
    trailing_stop = float("-inf")

    def _fn(values: List[float]) -> float:
        nonlocal trailing_stop
        current_val = max(values)
        if trailing_stop == float("-inf"):
            # first call, initialize trailing stop
            trailing_stop = current_val - stop_distance
        if current_val > trailing_stop + stop_distance:
            # update trailing stop upwards
            trailing_stop = current_val - stop_distance
        if current_val < trailing_stop:
            # signal sell
            print(f"Trailing stop hit: {trailing_stop}")
            trailing_stop = float("-inf")
        return trailing_stop

    return _fn


# Demo / integration example
async def main_demo() -> None:

    print("*** Demo started ***")

    print("*** Create Generators ***")
    interval = 5.0 # set to something smaller when testing
    system_cycle = interval * 5
    gUsdJpy = Generator("USDJPY", interval=interval, initial_price=155.69)
    gGbpUsd = Generator("GBPUSD", interval=interval, initial_price=1.36)
    gEurUsd = Generator("EURUSD", interval=interval, initial_price=1.18)

    print("*** Start Generators ***")
    gUsdJpy.start()
    gGbpUsd.start()
    gEurUsd.start()

    print ("*** Create strategies with different decision methods ***")
    sMean = Strategy("sMean", method="mean", decision_fn=lambda vals: sum(vals) / len(vals))
    sMedian = Strategy("sMedian", method="median", decision_fn=statistics.median)
    sGeometric = Strategy("sGeometric", method="geometric_mean", decision_fn=geometric_mean)
    # Custom trimmed mean with 20% trim using a decision_fn
    sTrimmed = Strategy("sTrimmed", method="trimmed_mean_20", decision_fn=lambda vals: trimmed_mean(vals, 0.2))
    # max seen strategy that holds some state
    sMaxSeen = Strategy("sMaxSeen", method="max_seen", decision_fn=max_seen())
    # trailing stop strategy
    sTrailingStop = Strategy("sTrailingStop", method="trailing_stop_sell", decision_fn=trailing_stop_sell())

    # Attach generators to strategies
    sMean.attach_generator(gUsdJpy)
    sMean.attach_generator(gGbpUsd)

    sMedian.attach_generator(gGbpUsd)
    sMedian.attach_generator(gEurUsd)

    sGeometric.attach_generator(gUsdJpy)
    sGeometric.attach_generator(gEurUsd)

    sTrimmed.attach_generator(gUsdJpy)
    sTrimmed.attach_generator(gGbpUsd)
    sTrimmed.attach_generator(gEurUsd)

    sMaxSeen.attach_generator(gUsdJpy)
    sTrailingStop.attach_generator(gUsdJpy)

    # Create trader and attach strategies
    trader = Trader()
    trader.attach_strategy(sMean)
    trader.attach_strategy(sMedian)
    trader.attach_strategy(sGeometric)
    trader.attach_strategy(sTrimmed)
    trader.attach_strategy(sMaxSeen)
    trader.attach_strategy(sTrailingStop)

    # Let system run for a few cycles
    await asyncio.sleep(system_cycle)

    # Dynamically add a new strategy that listens to all generators (example using median)
    sMedianAll = Strategy("S-E", method="median", decision_fn=statistics.median)
    sMedianAll.attach_generator(gUsdJpy)
    sMedianAll.attach_generator(gGbpUsd)
    sMedianAll.attach_generator(gEurUsd)
    trader.attach_strategy(sMedianAll)
    print("Added strategy S-E (subscribes to G1,G2,G3)")

    await asyncio.sleep(system_cycle)

    # Stop one generator and remove one strategy to demonstrate resilience
    print("Shutting down generator G2 and removing strategy S-B")
    await gGbpUsd.stop()
    sMedian.shutdown()
    trader.detach_strategy(sMedian)

    await asyncio.sleep(system_cycle)

    # Clean shutdown
    print("Shutting down remaining components...")
    sMean.shutdown()
    sGeometric.shutdown()
    sTrimmed.shutdown()
    sMedianAll.shutdown()
    sMaxSeen.shutdown()
    trader.shutdown()
    await gUsdJpy.stop()
    await gEurUsd.stop()

    print("*** Demo finished ***")


if __name__ == "__main__":
    try:
        asyncio.run(main_demo())
    except KeyboardInterrupt:
        pass
