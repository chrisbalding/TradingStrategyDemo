import time
import importlib
import pytest

from GeoAlpha import geometric_mean, trimmed_mean, max_seen, trailing_stop_sell


def test_max_seen_tracks_largest_value_across_calls():

    max_seen_fn = max_seen()
    
    # first call sets baseline
    assert max_seen_fn([1.0]) == pytest.approx(1.0)
    # lower input shouldn't decrease max
    assert max_seen_fn([0.5]) == pytest.approx(1.0)
    # higher input updates max
    assert max_seen_fn([2.0]) == pytest.approx(2.0)
    # empty input should return last-seen max
    assert max_seen_fn([]) == pytest.approx(2.0)


def test_trailing_stop_sell():

    trailing_stop_sell_fn = trailing_stop_sell(stop_distance=1.0)

    # test trailing stop updates
    assert trailing_stop_sell_fn([10.0]) == pytest.approx(9.0)
    assert trailing_stop_sell_fn([11.0]) == pytest.approx(10.0)
    assert trailing_stop_sell_fn([11.5]) == pytest.approx(10.5)
    assert trailing_stop_sell_fn([11.0]) == pytest.approx(10.5)
    # test drop below trailing stop
    assert trailing_stop_sell_fn([10.0]) == pytest.approx(float("-inf"))


def test_geometric_mean_basic():
    # geometric mean of [1, 4] == 2
    assert geometric_mean([1.0, 4.0]) == pytest.approx(2.0)


def test_geometric_mean_nonpositive_falls_back_to_arithmetic_mean():
    # implementation falls back to arithmetic mean for non-positive values
    vals = [0.0, 2.0]
    assert geometric_mean(vals) == pytest.approx(sum(vals) / len(vals))


def test_trimmed_mean_basic_trim25():
    vals = [1.0, 2.0, 3.0, 100.0]
    # with trim_fraction=0.25 k=1 -> trimmed values [2.0,3.0] -> mean 2.5
    assert trimmed_mean(vals, trim_fraction=0.25) == pytest.approx(2.5)


def test_trimmed_mean_fallback_when_trim_too_large():
    vals = [1.0, 10.0]
    # trim_fraction=0.5 -> k=1 -> 2*k == n -> fallback to arithmetic mean
    assert trimmed_mean(vals, trim_fraction=0.5) == pytest.approx(sum(vals) / len(vals))
