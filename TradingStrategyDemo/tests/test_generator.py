import math
import random
import time
import asyncio
import pytest

from generator import Generator


def _run_async(coro):
    """Helper so tests don't require pytest-asyncio; run an async coro synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def test_properties_and_volatility_setter():
    g = Generator("T0", interval=0.1, initial_price=1.5, volatility=0.25, mu=0.01)
    assert isinstance(g.price, float)
    assert g.price == pytest.approx(1.5)
    assert g.volatility == pytest.approx(0.25)

    g.volatility = 0.123
    assert g.volatility == pytest.approx(0.123)


def test_deterministic_evolution_when_zero_volatility():
    # With zero volatility the evolution should be deterministic: S_next = S * exp(mu * dt)
    seed = 42
    rng = random.Random(seed)
    initial = 1.2345
    mu = 0.05
    interval = 0.01  # seconds (small for test)
    g = Generator("T1", interval=interval, initial_price=initial, volatility=0.0, mu=mu, rng=rng)

    async def runner():
        g.start()
        # wait a bit more than one interval so the generator advances at least once
        await asyncio.sleep(interval * 1.2)
        await g.stop()

    _run_async(runner())

    dt = interval / g._SECONDS_PER_YEAR
    expected = initial * math.exp(mu * dt)
    assert g.price == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_price_update_with_seeded_rng_and_subscriber_receives_value():
    # Use two RNGs seeded the same: one to compute expected, one for the generator instance
    seed = 12345
    rng_for_gen = random.Random(seed)
    rng_for_expected = random.Random(seed)

    initial = 1.0
    volatility = 0.2
    mu = 0.0
    interval = 0.01

    g = Generator("T2", interval=interval, initial_price=initial, volatility=volatility, mu=mu, rng=rng_for_gen)

    # compute expected price after one step using same first Gaussian draw
    z = rng_for_expected.gauss(0.0, 1.0)
    dt = interval / g._SECONDS_PER_YEAR
    drift = (mu - 0.5 * (volatility ** 2)) * dt
    diffusion = volatility * math.sqrt(dt) * z
    expected_price = initial * math.exp(drift + diffusion)

    received = []

    def subscriber(src_id: str, value: float, ts: float):
        # record what subscriber receives
        received.append((src_id, value, ts))

    g.subscribe(subscriber)

    async def runner():
        g.start()
        # wait slightly longer than one interval to allow one emission
        await asyncio.sleep(interval * 1.2)
        await g.stop()

    _run_async(runner())

    # generator should have advanced and match expected price (within tolerance)
    assert g.price == pytest.approx(expected_price, rel=1e-9)

    # subscriber should have been called at least once with the same value
    assert len(received) >= 1
    # last received value should approximately equal final price
    last_src, last_value, last_ts = received[-1]
    assert last_src == g.id
    assert last_value == pytest.approx(g.price, rel=1e-9)
    assert isinstance(last_ts, float)