# Bloom Filter

A classic Bloom filter in Python: a space-efficient probabilistic set-membership
structure. It answers "is this item in the set?" in O(k) time using a fixed-size
bit array — roughly **10 bits per item** at a 1% false-positive rate, versus
hundreds of bytes per entry for a Python `set` of strings.

Use cases: cache-prefetch guards, distributed join pruning, spell-checker
dictionaries, web crawlers' seen-URL sets — anywhere you need a cheap
"definitely not present" test before an expensive lookup.

## Key design decisions

1. **Optimal sizing from (capacity, error rate).** The bit array size and hash
   count are derived with the textbook formulas `m = -(n·ln p) / ln²2` and
   `k = (m/n)·ln2`, so the caller only declares intent and the structure is
   sized correctly by construction.
2. **Double hashing (Kirsch & Mitzenmacher).** Instead of computing k separate
   hashes, two 64-bit values are taken from a single SHA-256 digest and combined
   as `h_i = h1 + i·h2`. This gives k effectively independent hash functions
   for the cost of one digest, with the standard modulo-m reduction.
3. **Self-describing serialization.** `to_bytes()`/`from_bytes()` pack a small
   header (capacity, bit size, hash count, error rate, insert count) plus the
   raw bit array, so a filter built on disk can be loaded by another process
   without passing parameters around. Combining filters (union/intersection)
   requires matching parameters and raises otherwise.

## How to run

```bash
# build a filter from a wordlist (one item per line)
PYTHONPATH=src python -m bloomfilter.cli build words.txt words.bloom \
    --capacity 100000 --error-rate 0.01

# query words against it ("maybe" = present or false positive; "no" = certain)
PYTHONPATH=src python -m bloomfilter.cli query words.bloom query.txt

# inspect parameters and statistics
PYTHONPATH=src python -m bloomfilter.cli stats words.bloom

# benchmark: observed false-positive rate vs theory + query throughput
PYTHONPATH=src python -m bloomfilter.cli benchmark --capacity 100000
```

## How to run tests

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -v
```

Tests cover: no false negatives, observed vs theoretical false-positive rate,
serialization round-trip (and corrupt-input rejection), union/intersection
correctness, optimal-sizing formulas, count estimation, and a CLI end-to-end
round trip.

## Tech stack

Python 3.10+ standard library only (`hashlib`, `struct`, `math`, `argparse`);
pytest for tests. No runtime dependencies.

## License

MIT — see [LICENSE](LICENSE).
