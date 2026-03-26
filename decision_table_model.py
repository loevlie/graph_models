#!/usr/bin/env python3
"""
Decision table model: a correction over a base model (PA/ER).

Idea (from advisor):
  1. For each node, build a binary feature vector from neighbor degree histogram
     (how many neighbors with degree > 2^k for k = 0,1,2,...)
  2. For each potential edge (u,v), combine features → key
  3. For FREQUENT keys: learn empirical edge probability → correction over base
  4. For rare keys: fall back to base model (PA/ER)
  5. Total cost = base_params + decision_table + edges_given_corrected_probs

The decision table amortizes over many edges — should help on large graphs.

NOTE: This Python version uses sampling for scalability. A Rust implementation
would be needed for exact computation on large graphs. See the shuffle-coding
repo for the Rust infrastructure.

TODO: Batch version — process multiple graphs sharing the same decision table,
amortizing table cost across a collection.
"""

import os
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from analyze_graphs import (
    download_and_extract, parse_tu_dataset, er_bits_per_edge,
    pa_bits_per_edge, configuration_model_bits_per_edge,
    SZIP_DATASETS, TU_DATASETS, DATA_DIR
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def node_features(G, log_bins):
    """Compute binary feature vector for ALL nodes at once. Fast."""
    n = G.number_of_nodes()
    degrees = np.array([G.degree(i) for i in range(n)])
    feat = np.zeros((n, len(log_bins)), dtype=np.int8)

    for i in range(n):
        if degrees[i] == 0:
            continue
        nb_degs = [degrees[nb] for nb in G.neighbors(i)]
        for b, t in enumerate(log_bins):
            if any(d >= t for d in nb_degs):
                feat[i, b] = 1
    return feat, degrees


def decision_table_model(G, min_frequency=10, n_samples=500000, verbose=False):
    """
    Compute bits per edge using decision table correction over ER base.

    Uses sampling to estimate edge probabilities per feature-pair key.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if m == 0 or n < 4:
        return float('nan'), 0, 0

    density = m / (n * (n - 1) / 2)
    base_p = density

    # Log bins
    degrees = np.array([G.degree(i) for i in range(n)])
    max_deg = max(degrees)
    log_bins = []
    t = 1
    while t <= max_deg:
        log_bins.append(t)
        t *= 2
    n_feat_bits = len(log_bins)

    # Compute features
    feat, _ = node_features(G, log_bins)

    # Convert feature vectors to hashable tuples
    feat_tuples = [tuple(feat[i].tolist()) for i in range(n)]

    # Build adjacency set for fast lookup
    adj = set()
    for u, v in G.edges():
        adj.add((min(u,v), max(u,v)))

    # Sample pairs to build the decision table
    np.random.seed(42)
    pair_stats = defaultdict(lambda: [0, 0])  # key -> [edges, total]

    actual_samples = min(n_samples, n * (n - 1) // 2)
    sampled = 0
    attempts = 0
    seen_pairs = set()

    while sampled < actual_samples and attempts < actual_samples * 5:
        u = np.random.randint(0, n)
        v = np.random.randint(0, n)
        if u == v:
            attempts += 1
            continue
        if u > v:
            u, v = v, u
        if (u, v) in seen_pairs:
            attempts += 1
            continue
        seen_pairs.add((u, v))

        fu, fv = feat_tuples[u], feat_tuples[v]
        key = (min(fu, fv), max(fu, fv))
        pair_stats[key][1] += 1
        if (u, v) in adj:
            pair_stats[key][0] += 1
        sampled += 1
        attempts += 1

    # Build decision table for frequent keys
    decision_table = {}
    for key, (edges, total) in pair_stats.items():
        if total >= min_frequency:
            emp_p = max(1e-10, min(1 - 1e-10, edges / total))
            decision_table[key] = emp_p

    # Table description cost
    table_entry_bits = 2 * n_feat_bits + 10  # 2 feature vecs + probability
    table_bits = 32 + len(decision_table) * table_entry_bits

    # Compute encoding cost from sampled statistics
    total_enc_bits = 0.0
    total_sampled_pairs = 0

    for key, (edges, total) in pair_stats.items():
        if key in decision_table:
            p = decision_table[key]
        else:
            p = base_p

        non_edges = total - edges
        if edges > 0 and p > 0:
            total_enc_bits += edges * (-np.log2(p))
        if non_edges > 0 and (1 - p) > 0:
            total_enc_bits += non_edges * (-np.log2(1 - p))
        total_sampled_pairs += total

    # Extrapolate to full graph
    total_pairs = n * (n - 1) // 2
    if total_sampled_pairs > 0:
        avg_bits_per_pair = total_enc_bits / total_sampled_pairs
        full_enc_bits = avg_bits_per_pair * total_pairs
    else:
        full_enc_bits = float('nan')

    total_bits = 32 + table_bits + full_enc_bits  # base param + table + encoding
    bpe = total_bits / m

    if verbose:
        print(f"  Feature bits per node: {n_feat_bits}")
        print(f"  Unique keys sampled: {len(pair_stats)}")
        print(f"  Decision table entries: {len(decision_table)}")
        print(f"  Table cost: {table_bits:.0f} bits")
        print(f"  Encoding cost (extrapolated): {full_enc_bits:.0f} bits")
        print(f"  Total: {total_bits:.0f} bits → {bpe:.2f} bits/edge")

        # Show some table entries
        if decision_table:
            print(f"  Sample table entries (key → P(edge)):")
            sorted_entries = sorted(decision_table.items(), key=lambda x: -pair_stats[x[0]][1])
            for key, p in sorted_entries[:5]:
                edges, total = pair_stats[key]
                print(f"    {key[0]}+{key[1]} → P={p:.4f} (base={base_p:.4f}, "
                      f"{edges}/{total} edges, {'higher' if p > base_p else 'lower'} than base)")

    return bpe, len(decision_table), n_feat_bits


def main():
    print("=" * 80)
    print("DECISION TABLE MODEL — Correction over ER base")
    print("=" * 80)
    print()
    print("NOTE: Python prototype using sampling. Rust needed for exact/large-scale.")
    print("TODO: Batch mode — share one decision table across a graph collection.")
    print()

    # Test on a few representative datasets
    test_datasets = [
        ("USAir97", "SZIP", SZIP_DATASETS["USAir97"]),
        ("Erdos", "SZIP", SZIP_DATASETS["Erdos"]),
        ("as", "SZIP", SZIP_DATASETS["as"]),
    ]

    # Also test on single TU graphs
    tu_tests = [
        ("MUTAG", "TU", TU_DATASETS["MUTAG"]),
    ]

    print("=" * 80)
    print(f"{'Dataset':12s} {'n':>8s} {'m':>8s} {'ER':>8s} {'PA':>8s} {'Config':>8s} "
          f"{'DT(5)':>8s} {'DT(20)':>8s} {'Tbl':>5s}")
    print("-" * 80)

    for name, source, url in test_datasets:
        cache_dir = os.path.join(DATA_DIR, name)
        graphs = parse_tu_dataset(name, cache_dir)
        G = graphs[0]
        n, m_ = G.number_of_nodes(), G.number_of_edges()
        degrees = np.array([G.degree(i) for i in range(n)])
        density = m_ / (n * (n - 1) / 2)

        er = er_bits_per_edge(n, m_, density)
        pa = pa_bits_per_edge(degrees, n, m_)
        cfg = configuration_model_bits_per_edge(degrees, n, m_)

        dt5, ts5, _ = decision_table_model(G, min_frequency=5, verbose=False)
        dt20, ts20, _ = decision_table_model(G, min_frequency=20, verbose=False)

        print(f"{name:12s} {n:8d} {m_:8d} {er:8.2f} {pa:8.2f} {cfg:8.2f} "
              f"{dt5:8.2f} {dt20:8.2f} {ts20:5d}")

    # MUTAG: use a single representative graph
    for name, source, url in tu_tests:
        cache_dir = os.path.join(DATA_DIR, name)
        graphs = parse_tu_dataset(name, cache_dir)
        sizes = [g.number_of_edges() for g in graphs]
        med_idx = int(np.argmin(np.abs(np.array(sizes) - np.median(sizes))))
        G = graphs[med_idx]
        n, m_ = G.number_of_nodes(), G.number_of_edges()
        degrees = np.array([G.degree(i) for i in range(n)])
        density = m_ / (n * (n - 1) / 2) if n > 1 else 0

        er = er_bits_per_edge(n, m_, density)
        pa = pa_bits_per_edge(degrees, n, m_)
        cfg = configuration_model_bits_per_edge(degrees, n, m_)

        dt5, ts5, _ = decision_table_model(G, min_frequency=3, verbose=False)
        dt20, ts20, _ = decision_table_model(G, min_frequency=5, verbose=False)

        print(f"{name+'*':12s} {n:8d} {m_:8d} {er:8.2f} {pa:8.2f} {cfg:8.2f} "
              f"{dt5:8.2f} {dt20:8.2f} {ts20:5d}")

    print()
    print("* = single representative graph from collection")
    print()

    # Detailed output for USAir97
    print("\n" + "=" * 80)
    print("DETAILED: USAir97")
    print("=" * 80)
    cache_dir = os.path.join(DATA_DIR, "USAir97")
    G = parse_tu_dataset("USAir97", cache_dir)[0]
    decision_table_model(G, min_frequency=20, verbose=True)


if __name__ == "__main__":
    main()
