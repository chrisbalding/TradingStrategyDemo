import statistics
import time
import pytest

from trader import Trader


class MockStrategy:
    """Minimal strategy shim implementing subscribe/emit used by Trader tests."""
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

    def emit(self, decision: float, ts: float | None = None):
        ts = ts or time.time()
        for s in list(self._subs):
            s(self.id, float(decision), ts)


class RecordingTrader(Trader):
    """Subclass Trader to record emitted medians for assertions."""
    def __init__(self):
        super().__init__()
        self.emitted: list[tuple[float, float]] = []

    def _emit_median(self, ts: float) -> None:
        if not self._latest_decisions:
            return
        values = list(self._latest_decisions.values())
        med = statistics.median(values)
        self.emitted.append((med, ts))


def test_trader_emits_median_single_strategy():
    rt = RecordingTrader()
    s = MockStrategy("S1")
    rt.attach_strategy(s)

    ts = time.time()
    s.emit(2.0, ts)

    assert len(rt.emitted) == 1
    med, recorded_ts = rt.emitted[-1]
    assert med == pytest.approx(2.0)
    assert recorded_ts == pytest.approx(ts)


def test_trader_median_multiple_strategies():
    rt = RecordingTrader()
    s1 = MockStrategy("S1")
    s2 = MockStrategy("S2")

    rt.attach_strategy(s1)
    rt.attach_strategy(s2)

    s1.emit(1.0)
    assert rt.emitted[-1][0] == pytest.approx(1.0)

    s2.emit(3.0)
    # median of [1.0, 3.0] == 2.0
    assert rt.emitted[-1][0] == pytest.approx(2.0)


def test_detach_strategy_and_shutdown_behavior():
    rt = RecordingTrader()
    s = MockStrategy("S1")
    rt.attach_strategy(s)

    s.emit(1.0)
    assert rt.emitted[-1][0] == pytest.approx(1.0)

    rt.detach_strategy(s)
    # after detach, emits should not change recorded medians
    s.emit(5.0)
    assert rt.emitted[-1][0] == pytest.approx(1.0)

    # re-attach, emit, then shutdown -> further emits ignored
    rt.attach_strategy(s)
    s.emit(2.0)
    assert rt.emitted[-1][0] == pytest.approx(2.0)

    rt.shutdown()
    s.emit(3.0)
    assert rt.emitted[-1][0] == pytest.approx(2.0)