import time
import pytest

from strategy import Strategy


class MockGenerator:
    """Minimal generator shim implementing subscribe/emit used by Strategy tests."""
    def __init__(self, id: str):
        self.id = id
        self._subs = []

    def subscribe(self, fn):
        self._subs.append(fn)

        def unsubscribe():
            try:
                self._subs.remove(fn)
            except ValueError:
                pass

        return unsubscribe

    def emit(self, value: float, ts: float | None = None):
        ts = ts or time.time()
        for s in list(self._subs):
            s(self.id, float(value), ts)


def test_strategy_emits_decision_on_single_generator():
    decisions = []

    def mean_fn(vals):
        return sum(vals) / len(vals)

    s = Strategy("S-single", method="mean", decision_fn=mean_fn)
    s.subscribe(lambda sid, dec, ts: decisions.append((sid, dec, ts)))

    g = MockGenerator("G1")
    s.attach_generator(g)

    g.emit(1.234)
    assert len(decisions) >= 1
    sid, dec, ts = decisions[-1]
    assert sid == s.id
    assert dec == pytest.approx(1.234)
    assert isinstance(ts, float)


def test_strategy_aggregates_multiple_generators_and_updates_latest_values():
    decisions = []

    def mean_fn(vals):
        return sum(vals) / len(vals)

    s = Strategy("S-multi", method="mean", decision_fn=mean_fn)
    s.subscribe(lambda sid, dec, ts: decisions.append((sid, dec, ts)))

    g1 = MockGenerator("G1")
    g2 = MockGenerator("G2")
    s.attach_generator(g1)
    s.attach_generator(g2)

    # first emission from g1 -> decision should be g1 value only
    g1.emit(2.0)
    assert decisions[-1][1] == pytest.approx(2.0)

    # then emit from g2 -> decision should be mean(2.0, 4.0) == 3.0
    g2.emit(4.0)
    assert decisions[-1][1] == pytest.approx(3.0)


def test_detach_generator_and_shutdown_behaviour():
    decisions = []

    def mean_fn(vals):
        return sum(vals) / len(vals)

    s = Strategy("S-detach", method="mean", decision_fn=mean_fn)
    s.subscribe(lambda sid, dec, ts: decisions.append((sid, dec, ts)))

    g = MockGenerator("G1")
    s.attach_generator(g)

    g.emit(1.0)
    assert decisions[-1][1] == pytest.approx(1.0)

    s.detach_generator(g)
    # after detach, emits should not change decisions
    g.emit(5.0)
    assert decisions[-1][1] == pytest.approx(1.0)

    # re-attach, emit, then shutdown -> further emits ignored
    s.attach_generator(g)
    g.emit(2.0)
    assert decisions[-1][1] == pytest.approx(2.0)

    s.shutdown()
    g.emit(3.0)
    assert decisions[-1][1] == pytest.approx(2.0)