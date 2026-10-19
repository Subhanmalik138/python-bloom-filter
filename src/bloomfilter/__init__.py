"""Bloom filter: a space-efficient probabilistic membership test.

Public API:
    BloomFilter  -- the filter itself
"""

from .bloom import BloomFilter

__all__ = ["BloomFilter"]
__version__ = "1.0.0"
