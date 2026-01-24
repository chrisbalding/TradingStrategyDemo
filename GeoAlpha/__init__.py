"""
GeoAlpha package initializer.

Expose the primary public API so callers can do:
    from GeoAlpha import Generator, Strategy, Trader
"""
from .generator import Generator
from .strategy import Strategy
from .trader import Trader

__all__ = ["Generator", "Strategy", "Trader"]