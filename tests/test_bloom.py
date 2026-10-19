"""Tests for the Bloom filter."""

import os
import random
import string
import subprocess
import sys
from pathlib import Path

import pytest

from bloomfilter import BloomFilter


def random_words(count: int, seed: int = 1234) -> list[str]:
    rng = random.Random(seed)
    alphabet = string.ascii_lowercase + string.digits
    return [
        "".join(rng.choice(alphabet) for _ in range(10)) for _ in range(count)
    ]


def test_no_false_negatives() -> None:
    bloom = BloomFilter(capacity=1_000, error_rate=0.01)
    words = random_words(1_000)
    for word in words:
        bloom.add(word)
    assert all(word in bloom for word in words)


def test_false_positive_rate_within_tolerance() -> None:
    bloom = BloomFilter(capacity=5_000, error_rate=0.01)
    members = set(random_words(5_000, seed=1))
    for word in members:
        bloom.add(word)

    probes = [w for w in random_words(20_000, seed=2) if w not in members]
    observed = sum(1 for w in probes if w in bloom) / len(probes)
    theoretical = bloom.false_positive_rate()

    # observed rate should land near theory, well under 5x the target
    assert observed < 0.05
    assert abs(observed - theoretical) < 0.01


def test_accepts_bytes_and_str() -> None:
    bloom = BloomFilter(capacity=100, error_rate=0.01)
    bloom.add("hello")
    bloom.add(b"world")
    assert "hello" in bloom
    assert b"world" in bloom


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        BloomFilter(capacity=0)
    with pytest.raises(ValueError):
        BloomFilter(capacity=100, error_rate=0.0)
    with pytest.raises(ValueError):
        BloomFilter(capacity=100, error_rate=1.5)
    with pytest.raises(TypeError):
        BloomFilter(capacity=100).add(123)  # type: ignore[arg-type]


def test_optimal_sizing_matches_textbook_formulas() -> None:
    import math

    bloom = BloomFilter(capacity=10_000, error_rate=0.001)
    expected_bits = math.ceil(
        -(10_000 * math.log(0.001)) / (math.log(2) ** 2)
    )
    expected_hashes = max(1, round((expected_bits / 10_000) * math.log(2)))
    assert bloom.num_bits == expected_bits
    assert bloom.num_hashes == expected_hashes


def test_serialization_roundtrip(tmp_path: Path) -> None:
    bloom = BloomFilter(capacity=1_000, error_rate=0.01)
    words = random_words(1_000)
    for word in words:
        bloom.add(word)

    path = tmp_path / "filter.bloom"
    path.write_bytes(bloom.to_bytes())
    restored = BloomFilter.from_bytes(path.read_bytes())

    assert restored.num_bits == bloom.num_bits
    assert restored.num_hashes == bloom.num_hashes
    assert len(restored) == len(bloom)
    assert all(word in restored for word in words)


def test_serialization_rejects_corrupt_data() -> None:
    with pytest.raises(ValueError):
        BloomFilter.from_bytes(b"too short")
    bloom = BloomFilter(capacity=100, error_rate=0.01)
    data = bloom.to_bytes()
    with pytest.raises(ValueError):
        BloomFilter.from_bytes(data[:-5])  # truncated bit array


def test_union_contains_both_sides() -> None:
    a = BloomFilter(capacity=500, error_rate=0.01)
    b = BloomFilter(capacity=500, error_rate=0.01)
    words_a = random_words(500, seed=11)
    words_b = random_words(500, seed=22)
    for word in words_a:
        a.add(word)
    for word in words_b:
        b.add(word)

    merged = a.union(b)
    assert all(w in merged for w in words_a)
    assert all(w in merged for w in words_b)


def test_intersection_is_subset_of_union() -> None:
    a = BloomFilter(capacity=500, error_rate=0.01)
    b = BloomFilter(capacity=500, error_rate=0.01)
    for word in random_words(300, seed=33):
        a.add(word)
    for word in random_words(300, seed=44):
        b.add(word)

    inter = a.intersection(b)
    union = a.union(b)
    # every bit set in the intersection must also be set in the union
    inter_bits = inter.to_bytes()
    union_bits = union.to_bytes()
    assert all((i & u) == i for i, u in zip(inter_bits, union_bits))


def test_combining_incompatible_filters_raises() -> None:
    a = BloomFilter(capacity=500, error_rate=0.01)
    b = BloomFilter(capacity=1_000, error_rate=0.01)  # different bit size
    with pytest.raises(ValueError):
        a.union(b)
    with pytest.raises(ValueError):
        a.intersection(b)


def test_estimated_count_tracks_reality() -> None:
    bloom = BloomFilter(capacity=10_000, error_rate=0.01)
    words = random_words(5_000, seed=55)
    for word in words:
        bloom.add(word)
    estimate = bloom.estimated_count()
    assert 4_000 < estimate < 6_000


def test_cli_build_query_stats_roundtrip(tmp_path: Path) -> None:
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    query_file = tmp_path / "query.txt"
    query_file.write_text("alpha\ndelta\n", encoding="utf-8")
    filter_file = tmp_path / "words.bloom"

    env = dict(os.environ, PYTHONPATH="src")
    assert subprocess.run(
        [sys.executable, "-m", "bloomfilter.cli", "build",
         str(wordlist), str(filter_file), "--capacity", "100"],
        capture_output=True, text=True, env=env,
    ).returncode == 0

    query = subprocess.run(
        [sys.executable, "-m", "bloomfilter.cli", "query",
         str(filter_file), str(query_file)],
        capture_output=True, text=True, env=env,
    )
    assert query.returncode == 0
    assert "alpha: maybe" in query.stdout
    assert "delta: no" in query.stdout

    stats = subprocess.run(
        [sys.executable, "-m", "bloomfilter.cli", "stats", str(filter_file)],
        capture_output=True, text=True, env=env,
    )
    assert stats.returncode == 0
    assert "capacity:" in stats.stdout
