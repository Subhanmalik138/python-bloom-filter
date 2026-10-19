"""Command-line interface for the Bloom filter.

Commands:
    build     build a filter from a wordlist file and save it to disk
    query     check words against a saved filter
    stats     show parameters and statistics of a saved filter
    benchmark measure the observed false-positive rate on random data
"""

from __future__ import annotations

import argparse
import random
import string
import sys
import time

from .bloom import BloomFilter

FILTER_PATH_SUFFIX = ".bloom"


def _load(path: str) -> BloomFilter:
    with open(path, "rb") as fh:
        return BloomFilter.from_bytes(fh.read())


def _save(path: str, bloom: BloomFilter) -> None:
    with open(path, "wb") as fh:
        fh.write(bloom.to_bytes())


def cmd_build(args: argparse.Namespace) -> int:
    bloom = BloomFilter(args.capacity, args.error_rate)
    with open(args.wordlist, encoding="utf-8") as fh:
        for line in fh:
            word = line.strip()
            if word:
                bloom.add(word)
    _save(args.output, bloom)
    print(
        f"built {args.output}: capacity={bloom.capacity} "
        f"error_rate={bloom.error_rate} num_bits={bloom.num_bits} "
        f"num_hashes={bloom.num_hashes} inserted={len(bloom)}"
    )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    bloom = _load(args.filter)
    hits = 0
    with open(args.words, encoding="utf-8") as fh:
        for line in fh:
            word = line.strip()
            if not word:
                continue
            present = word in bloom
            hits += present
            print(f"{word}: {'maybe' if present else 'no'}")
    print(f"--- {hits} possible hits ---")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    bloom = _load(args.filter)
    print(f"capacity:            {bloom.capacity}")
    print(f"target error rate:   {bloom.error_rate}")
    print(f"bit array size:      {bloom.num_bits} bits "
          f"({bloom.num_bits / 8 / 1024:.1f} KiB)")
    print(f"hash functions:      {bloom.num_hashes}")
    print(f"inserted items:      {len(bloom)}")
    print(f"estimated distinct:  {bloom.estimated_count():.0f}")
    print(f"current fp rate:     {bloom.false_positive_rate():.6f}")
    return 0


def _random_word(length: int = 12) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def cmd_benchmark(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    bloom = BloomFilter(args.capacity, args.error_rate)

    members = {_random_word() for _ in range(args.capacity)}
    start = time.perf_counter()
    for word in members:
        bloom.add(word)
    build_time = time.perf_counter() - start

    misses: list[str] = []
    while len(misses) < args.probes:  # guaranteed probes are non-members
        candidate = _random_word()
        if candidate not in members:
            misses.append(candidate)
    start = time.perf_counter()
    false_positives = sum(1 for w in misses if w in bloom)
    query_time = time.perf_counter() - start

    observed = false_positives / args.probes
    print(f"capacity:        {args.capacity}")
    print(f"target fp rate:  {args.error_rate}")
    print(f"observed fp:     {observed:.6f} ({false_positives}/{args.probes})")
    print(f"theoretical fp:  {bloom.false_positive_rate():.6f}")
    print(f"build time:      {build_time:.3f}s")
    print(f"query throughput:{args.probes / query_time:,.0f} lookups/s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bloom",
        description="Bloom filter toolkit: build, query, inspect, benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build a filter from a wordlist")
    p_build.add_argument("wordlist", help="text file, one item per line")
    p_build.add_argument("output", help=f"output file (e.g. words{FILTER_PATH_SUFFIX})")
    p_build.add_argument("--capacity", type=int, default=100_000,
                         help="expected item count (default: 100000)")
    p_build.add_argument("--error-rate", type=float, default=0.01,
                         help="target false-positive rate (default: 0.01)")
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="check words against a filter")
    p_query.add_argument("filter", help="filter file created by 'build'")
    p_query.add_argument("words", help="text file of words to check")
    p_query.set_defaults(func=cmd_query)

    p_stats = sub.add_parser("stats", help="show filter parameters and stats")
    p_stats.add_argument("filter", help="filter file created by 'build'")
    p_stats.set_defaults(func=cmd_stats)

    p_bench = sub.add_parser("benchmark", help="measure observed false-positive rate")
    p_bench.add_argument("--capacity", type=int, default=100_000)
    p_bench.add_argument("--error-rate", type=float, default=0.01)
    p_bench.add_argument("--probes", type=int, default=20_000,
                         help="non-member probes (default: 20000)")
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
