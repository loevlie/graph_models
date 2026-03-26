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
    SZIP_DATASETS, REC_DATASETS, TU_DATASETS, DATA_DIR
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


def config_edge_prob(du, dv, m):
    """Configuration model edge probability between nodes of degree du, dv.
    P(edge between u,v) ≈ du * dv / (2m) for sparse graphs."""
    p = du * dv / (2.0 * m)
    return max(1e-10, min(1 - 1e-10, p))


def pa_degree_prob(k, m0):
    """PA predicted probability of degree k."""
    if k < 1:
        return 1e-10
    return 2.0 * m0 * (m0 + 1) / (k * (k + 1.0) * (k + 2.0))


def decision_table_model(G, base='er', min_frequency=10, n_samples=500000, verbose=False):
    """
    Compute bits per edge using decision table correction over a base model.

    base: 'er' (flat density), 'pa' (config-like, degree-proportional),
          or 'config' (exact du*dv/(2m) per pair)

    Uses sampling to estimate edge probabilities per feature-pair key.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if m == 0 or n < 4:
        return float('nan'), 0, 0

    density = m / (n * (n - 1) / 2)
    degrees = np.array([G.degree(i) for i in range(n)])

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
    # For each sampled pair, we also stored which (u,v) it was so we can compute
    # per-pair base probabilities. But we only have key-level stats from sampling.
    # For config/PA base: compute the AVERAGE base_p per key from sampled pairs.
    # We need to track this during sampling — re-sample to get per-key base probs.

    # Quick re-pass: compute average base_p per key
    key_base_p = {}
    if base == 'er':
        for key in pair_stats:
            key_base_p[key] = density
    else:
        # Re-sample to estimate average base_p per key
        np.random.seed(42)
        key_base_accum = defaultdict(lambda: [0.0, 0])
        resampled = 0
        re_attempts = 0
        re_seen = set()
        while resampled < actual_samples and re_attempts < actual_samples * 5:
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u == v:
                re_attempts += 1
                continue
            if u > v:
                u, v = v, u
            if (u, v) in re_seen:
                re_attempts += 1
                continue
            re_seen.add((u, v))
            fu, fv = feat_tuples[u], feat_tuples[v]
            key = (min(fu, fv), max(fu, fv))
            if base == 'config':
                bp = config_edge_prob(degrees[u], degrees[v], m)
            elif base == 'pa':
                bp = config_edge_prob(degrees[u], degrees[v], m)  # PA uses same edge prob approx
            else:
                bp = density
            key_base_accum[key][0] += bp
            key_base_accum[key][1] += 1
            resampled += 1
            re_attempts += 1

        for key, (s, c) in key_base_accum.items():
            key_base_p[key] = max(1e-10, min(1 - 1e-10, s / c))

    # Encoding cost
    total_enc_bits = 0.0
    total_sampled_pairs = 0
    base_enc_bits = 0.0  # for comparison: what base alone would cost

    for key, (edges, total) in pair_stats.items():
        bp = key_base_p.get(key, density)

        if key in decision_table:
            p = decision_table[key]
        else:
            p = bp

        non_edges = total - edges
        if edges > 0:
            total_enc_bits += edges * (-np.log2(max(1e-10, p)))
            base_enc_bits += edges * (-np.log2(max(1e-10, bp)))
        if non_edges > 0:
            total_enc_bits += non_edges * (-np.log2(max(1e-10, 1 - p)))
            base_enc_bits += non_edges * (-np.log2(max(1e-10, 1 - bp)))
        total_sampled_pairs += total

    # Extrapolate to full graph
    total_pairs = n * (n - 1) // 2
    if total_sampled_pairs > 0:
        avg_bits_per_pair = total_enc_bits / total_sampled_pairs
        full_enc_bits = avg_bits_per_pair * total_pairs
        avg_base_bits = base_enc_bits / total_sampled_pairs
        full_base_bits = avg_base_bits * total_pairs
    else:
        full_enc_bits = float('nan')
        full_base_bits = float('nan')

    # Base model param cost
    if base == 'config':
        # Config: encode the degree sequence
        deg_counts = Counter(degrees)
        K = len(deg_counts)
        d_max = int(max(degrees))
        base_param_bits = K * (np.log2(d_max + 1) + np.log2(n + 1)) + np.log2(K + 1)
        deg_dist = {d: c / n for d, c in deg_counts.items()}
        base_param_bits += sum(-np.log2(deg_dist[d]) for d in degrees)
    elif base == 'pa':
        base_param_bits = 32
    else:
        base_param_bits = 32

    total_bits = base_param_bits + table_bits + full_enc_bits
    base_only_bits = base_param_bits + full_base_bits
    bpe = total_bits / m
    base_only_bpe = base_only_bits / m

    if verbose:
        print(f"  Base model: {base}")
        print(f"  Feature bits per node: {n_feat_bits}")
        print(f"  Unique keys sampled: {len(pair_stats)}")
        print(f"  Decision table entries: {len(decision_table)}")
        print(f"  Table cost: {table_bits:.0f} bits")
        print(f"  Base param cost: {base_param_bits:.0f} bits")
        print(f"  Base-only encoding: {full_base_bits:.0f} bits → {base_only_bpe:.2f} bpe")
        print(f"  DT encoding:       {full_enc_bits:.0f} bits")
        print(f"  DT total:          {total_bits:.0f} bits → {bpe:.2f} bpe")
        print(f"  Improvement over base alone: {base_only_bpe - bpe:.2f} bits/edge")

        if decision_table:
            print(f"  Sample corrections (key → DT_p vs base_p):")
            sorted_entries = sorted(decision_table.items(), key=lambda x: -pair_stats[x[0]][1])
            for key, p in sorted_entries[:5]:
                edges, total = pair_stats[key]
                bp = key_base_p.get(key, density)
                print(f"    {key[0]}+{key[1]}: DT={p:.4f} base={bp:.4f} "
                      f"({edges}/{total} edges)")

    return bpe, len(decision_table), n_feat_bits


def main():
    print("=" * 80)
    print("DECISION TABLE MODEL — Correction over ER / PA / Config base")
    print("=" * 80)
    print()
    print("NOTE: Python prototype using sampling. Rust needed for exact/large-scale.")
    print("TODO: Batch mode — share one decision table across a graph collection.")
    print()

    test_datasets = [
        ("USAir97", "SZIP", SZIP_DATASETS["USAir97"]),
        ("Erdos", "SZIP", SZIP_DATASETS["Erdos"]),
        ("as", "SZIP", SZIP_DATASETS["as"]),
        ("DBLP", "REC", REC_DATASETS["DBLP"]),
        ("Gowalla", "REC", REC_DATASETS["Gowalla"]),
    ]

    tu_tests = [
        ("MUTAG", "TU", TU_DATASETS["MUTAG"]),
    ]

    # Collect all graphs
    all_graphs = []
    for name, source, url in test_datasets:
        cache_dir = os.path.join(DATA_DIR, name)
        graphs = parse_tu_dataset(name, cache_dir)
        all_graphs.append((name, source, graphs[0]))

    for name, source, url in tu_tests:
        cache_dir = os.path.join(DATA_DIR, name)
        graphs = parse_tu_dataset(name, cache_dir)
        sizes = [g.number_of_edges() for g in graphs]
        med_idx = int(np.argmin(np.abs(np.array(sizes) - np.median(sizes))))
        all_graphs.append((name + "*", source, graphs[med_idx]))

    # Run with all base models
    print("=" * 100)
    print(f"{'Dataset':12s} {'n':>7s} {'m':>7s} {'ER':>7s} {'PA':>7s} {'Cfg':>7s} "
          f"{'DT+ER':>7s} {'DT+PA':>7s} {'DT+Cfg':>7s} {'Tbl':>5s}")
    print("-" * 100)

    for name, source, G in all_graphs:
        n, m_ = G.number_of_nodes(), G.number_of_edges()
        degs = np.array([G.degree(i) for i in range(n)])
        dens = m_ / (n * (n - 1) / 2) if n > 1 else 0

        er = er_bits_per_edge(n, m_, dens)
        pa = pa_bits_per_edge(degs, n, m_)
        cfg = configuration_model_bits_per_edge(degs, n, m_)

        mf = 10 if n > 100 else 3
        dt_er, ts1, _ = decision_table_model(G, base='er', min_frequency=mf)
        dt_pa, ts2, _ = decision_table_model(G, base='pa', min_frequency=mf)
        dt_cfg, ts3, _ = decision_table_model(G, base='config', min_frequency=mf)

        print(f"{name:12s} {n:7d} {m_:7d} {er:7.2f} {pa:7.2f} {cfg:7.2f} "
              f"{dt_er:7.2f} {dt_pa:7.2f} {dt_cfg:7.2f} {ts2:5d}")

    print()
    print("* = single representative graph from collection")

    # Detailed output for a small and large graph
    for dname in ['USAir97', 'DBLP']:
        for base_name in ['er', 'pa']:
            print(f"\n{'='*80}")
            print(f"DETAILED: {dname} with base={base_name}")
            print(f"{'='*80}")
            cache_dir = os.path.join(DATA_DIR, dname)
            G = parse_tu_dataset(dname, cache_dir)[0]
            ns = 2000000 if G.number_of_nodes() > 10000 else 500000
            decision_table_model(G, base=base_name, min_frequency=10,
                                 n_samples=ns, verbose=True)


if __name__ == "__main__":
    main()
