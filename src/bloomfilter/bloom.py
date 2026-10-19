"""A classic Bloom filter implementation.

A Bloom filter answers "is this item in the set?" using a fixed-size bit
array and k independent hash functions. Membership queries never produce
false negatives, but may produce false positives with a tunable probability.

Bit array size (m) and hash count (k) are derived from the expected number
of items (n) and the target false-positive rate (p) with the standard
optimal formulas:

    m = ceil(-(n * ln(p)) / (ln(2)^2))
    k = round((m / n) * ln(2))

Hashing uses double hashing (Kirsch & Mitzenmacher): two 64-bit hashes are
taken from a single SHA-256 digest and combined as h_i = h1 + i*h2,
which gives k effective hash functions for the cost of one digest.
"""

from __future__ import annotations

import hashlib
import math
import struct

_HASH_DIGEST_BYTES = 32  # SHA-256


class BloomFilter:
    """Probabilistic set-membership structure with a bounded error rate."""

    def __init__(self, capacity: int, error_rate: float = 0.01) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not 0 < error_rate < 1:
            raise ValueError("error_rate must be in the open interval (0, 1)")

        self.capacity = capacity
        self.error_rate = error_rate
        self.num_bits = _optimal_num_bits(capacity, error_rate)
        self.num_hashes = _optimal_num_hashes(self.num_bits, capacity)
        self._bits = bytearray((self.num_bits + 7) // 8)
        self._inserted = 0

    # ------------------------------------------------------------------
    # core operations
    # ------------------------------------------------------------------
    def add(self, item: str | bytes) -> None:
        """Insert an item into the filter."""
        for index in self._hash_indices(item):
            self._bits[index >> 3] |= 1 << (index & 7)
        self._inserted += 1

    def __contains__(self, item: str | bytes) -> bool:
        """Return True if the item is *possibly* in the set.

        False is a definite "not present"; True means "present or a false
        positive".
        """
        return all(self._bits[index >> 3] & (1 << (index & 7))
                   for index in self._hash_indices(item))

    def __len__(self) -> int:
        """Number of items inserted (counting duplicates)."""
        return self._inserted

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    def false_positive_rate(self) -> float:
        """Theoretical false-positive rate for the current fill level.

        Uses (1 - e^(-k*n/m))^k, the standard closed-form approximation.
        """
        if self._inserted == 0:
            return 0.0
        exponent = -self.num_hashes * self._inserted / self.num_bits
        return (1.0 - math.exp(exponent)) ** self.num_hashes

    def estimated_count(self) -> float:
        """Estimate the number of distinct inserted items from the bit density.

        Inverts the fill model: n* = -(m/k) * ln(1 - X/m), where X is the
        number of set bits.
        """
        set_bits = sum(bin(byte).count("1") for byte in self._bits)
        if set_bits == 0:
            return 0.0
        return -(self.num_bits / self.num_hashes) * math.log(
            1.0 - set_bits / self.num_bits
        )

    # ------------------------------------------------------------------
    # set operations
    # ------------------------------------------------------------------
    def _check_compatible(self, other: "BloomFilter") -> None:
        if not isinstance(other, BloomFilter):
            raise TypeError("expected a BloomFilter")
        if (self.num_bits != other.num_bits
                or self.num_hashes != other.num_hashes):
            raise ValueError(
                "filters must share num_bits and num_hashes to combine"
            )

    def union(self, other: "BloomFilter") -> "BloomFilter":
        """Return a new filter representing the union of both sets (bitwise OR)."""
        self._check_compatible(other)
        merged = BloomFilter(self.capacity, self.error_rate)
        merged._bits = bytearray(
            a | b for a, b in zip(self._bits, other._bits)
        )
        merged._inserted = max(self._inserted, other._inserted)
        return merged

    def intersection(self, other: "BloomFilter") -> "BloomFilter":
        """Return a new filter approximating the intersection (bitwise AND)."""
        self._check_compatible(other)
        merged = BloomFilter(self.capacity, self.error_rate)
        merged._bits = bytearray(
            a & b for a, b in zip(self._bits, other._bits)
        )
        merged._inserted = min(self._inserted, other._inserted)
        return merged

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        """Serialize the filter (header + bit array) to bytes."""
        header = struct.pack(
            ">IIIdI",
            self.capacity,
            self.num_bits,
            self.num_hashes,
            self.error_rate,
            self._inserted,
        )
        return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        """Rebuild a filter previously serialized with :meth:`to_bytes`."""
        header_size = struct.calcsize(">IIIdI")
        if len(data) < header_size:
            raise ValueError("data too short to be a serialized BloomFilter")
        capacity, num_bits, num_hashes, error_rate, inserted = struct.unpack(
            ">IIIdI", data[:header_size]
        )
        bits = data[header_size:]
        expected = (num_bits + 7) // 8
        if len(bits) != expected:
            raise ValueError(
                f"bit array length mismatch: expected {expected}, got {len(bits)}"
            )
        instance = cls(capacity, error_rate)
        # parameters are fixed by (capacity, error_rate); sanity check them
        if instance.num_bits != num_bits or instance.num_hashes != num_hashes:
            raise ValueError("header parameters inconsistent with payload")
        instance._bits = bytearray(bits)
        instance._inserted = inserted
        return instance

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _to_bytes(item: str | bytes) -> bytes:
        if isinstance(item, str):
            return item.encode("utf-8")
        if isinstance(item, bytes):
            return item
        raise TypeError("item must be str or bytes")

    def _hash_indices(self, item: str | bytes) -> list[int]:
        """Yield k bit indices via double hashing over one SHA-256 digest."""
        digest = hashlib.sha256(self._to_bytes(item)).digest()
        h1, h2 = struct.unpack(">QQ", digest[:16])
        if h2 == 0:  # degenerate guard; keep all k indices distinct
            h2 = 0x9E3779B97F4A7C15
        m = self.num_bits
        return [(h1 + i * h2) % m for i in range(self.num_hashes)]


def _optimal_num_bits(n: int, p: float) -> int:
    return math.ceil(-(n * math.log(p)) / (math.log(2) ** 2))


def _optimal_num_hashes(m: int, n: int) -> int:
    return max(1, round((m / n) * math.log(2)))
