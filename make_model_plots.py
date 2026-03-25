#!/usr/bin/env python3
"""
Generate degree distribution plots with model predictions overlaid.
Left: empirical + model overlays. Right: model predictions alone for comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.special import gammaln
from collections import Counter
import networkx as nx

# Reuse the data loading from analyze_graphs
from analyze_graphs import (
    download_and_extract, parse_tu_dataset,
    SZIP_DATASETS, REC_DATASETS, TU_DATASETS, PLOTS_DIR, DATA_DIR
)

def simulate_py_degree_dist(alpha, theta, total_endpoints, n_nodes, n_sims=50):
    """Simulate PY CRP to get expected degree distribution.
    Assigns total_endpoints 'customers' to 'tables' (nodes) via CRP."""
    all_counts = Counter()
    for _ in range(n_sims):
        tables = []  # list of table sizes
        for t in range(total_endpoints):
            if len(tables) == 0:
                tables.append(1)
                continue
            # Probabilities
            denom = t + theta
            # Existing tables
            probs = [(s - alpha) / denom for s in tables]
            # New table
            new_prob = (theta + len(tables) * alpha) / denom
            probs.append(new_prob)

            r = np.random.random()
            cumsum = 0
            chosen = len(probs) - 1  # default: new table
            for i, p in enumerate(probs):
                cumsum += p
                if r < cumsum:
                    chosen = i
                    break

            if chosen == len(tables):
                tables.append(1)
            else:
                tables[chosen] += 1

            # Early stop if we have way more tables than expected
            if len(tables) > n_nodes * 3:
                break

        # Record degree distribution
        for s in tables:
            all_counts[s] += 1

    # Normalize
    total_tables = sum(all_counts.values())
    dist = {k: v / total_tables for k, v in all_counts.items()}
    return dist


def py_expected_degree_dist(alpha, theta, n_nodes, total_endpoints):
    """Compute expected degree distribution from PY analytically.
    For PY(alpha, theta) with N items and K groups, the expected number
    of groups of size j is:

    E[K_j] = (theta/alpha if alpha>0, else theta) * C(j) / normalization

    where C(j) = Gamma(j - alpha) / (Gamma(1 - alpha) * Gamma(j + 1))

    We normalize so that sum of E[K_j] = n_nodes and sum of j*E[K_j] = total_endpoints.
    """
    if alpha < 1e-6:
        # Dirichlet process: expected table sizes follow a different pattern
        # Use simulation for alpha ≈ 0
        return None

    max_k = min(total_endpoints, 10000)
    ks = np.arange(1, max_k + 1)

    # Unnormalized weights: Gamma(k - alpha) / (Gamma(1-alpha) * Gamma(k+1))
    log_weights = gammaln(ks - alpha) - gammaln(1 - alpha) - gammaln(ks + 1)
    weights = np.exp(log_weights - np.max(log_weights))  # prevent overflow

    # Normalize so sum = n_nodes
    weights = weights * (n_nodes / np.sum(weights))

    # Convert to probability: P(degree = k) = E[K_k] / n_nodes
    probs = weights / n_nodes

    return dict(zip(ks.tolist(), probs.tolist()))


def pa_degree_dist(mean_degree, max_k):
    """Preferential Attachment (Barabasi-Albert) predicted degree distribution.
    For BA model with parameter m0 (edges per new node), the stationary
    degree distribution is:
        P(k) = 2*m0*(m0+1) / (k*(k+1)*(k+2))  for k >= m0
    where m0 = mean_degree / 2 (since each new node adds m0 edges,
    and the mean degree converges to 2*m0).
    This is a power-law with exponent 3 in the tail: P(k) ~ k^(-3)."""
    m0 = max(1, int(round(mean_degree / 2)))
    ks = np.arange(max(1, m0), max_k + 1)
    probs = 2.0 * m0 * (m0 + 1) / (ks * (ks + 1.0) * (ks + 2.0))
    # Normalize (the formula is already normalized for k >= m0, but clip for safety)
    probs = probs / probs.sum()
    return ks, probs


def make_model_overlay_plot(G_or_degrees, name, source, er_density, hw_alpha, hw_theta,
                            n_nodes, n_edges, is_collection=False):
    """Create a two-panel plot: empirical + overlays on left, models compared on right."""
    if isinstance(G_or_degrees, nx.Graph):
        degrees = np.array([d for _, d in G_or_degrees.degree()])
    else:
        degrees = np.array(G_or_degrees)

    if len(degrees) == 0:
        return

    n = n_nodes
    m = n_edges
    total_endpoints = 2 * m
    mean_deg = np.mean(degrees)

    # Empirical degree distribution
    deg_counts = Counter(degrees)
    emp_ks = sorted(deg_counts.keys())
    emp_probs = [deg_counts[k] / len(degrees) for k in emp_ks]

    # ER model: Binomial(n-1, p) or Poisson approx
    p = er_density
    max_k_plot = max(emp_ks)
    er_ks = np.arange(0, min(max_k_plot + 1, 500))
    if is_collection or n > 10000:
        from scipy.stats import poisson
        lam = mean_deg
        er_probs = poisson.pmf(er_ks, lam)
    else:
        er_probs = binom.pmf(er_ks, n - 1, p)

    # PA model
    pa_ks, pa_probs = pa_degree_dist(mean_deg, max_k_plot)

    # Hollywood/PY model
    hw_dist = None
    if not np.isnan(hw_alpha) and not np.isnan(hw_theta):
        if hw_alpha > 0.01:
            hw_dist = py_expected_degree_dist(hw_alpha, hw_theta, n, total_endpoints)
        if hw_dist is None and n < 5000:
            print(f"    Simulating PY for {name} (alpha={hw_alpha:.3f}, theta={hw_theta:.1f})...")
            hw_dist = simulate_py_degree_dist(hw_alpha, hw_theta, total_endpoints, n, n_sims=20)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: empirical + all model overlays
    ax1.scatter(emp_ks, emp_probs, s=15, alpha=0.7, color='black', label='Empirical', zorder=5)

    # ER
    er_mask = er_probs > 1e-10
    ax1.plot(er_ks[er_mask], er_probs[er_mask], 'r-', alpha=0.7, linewidth=2,
             label=f'ER (p={p:.4f})')

    # PA
    pa_mask = pa_probs > 1e-10
    ax1.plot(pa_ks[pa_mask], pa_probs[pa_mask], '-', color='#e67e22', alpha=0.7, linewidth=2,
             label=f'PA (m₀={max(1, int(round(mean_deg/2)))}), tail~k⁻³')

    # Hollywood/PY
    if hw_dist:
        hw_ks_list = sorted(hw_dist.keys())
        hw_probs_list = [hw_dist[k] for k in hw_ks_list]
        valid = [(k, pr) for k, pr in zip(hw_ks_list, hw_probs_list) if pr > 1e-10 and k <= max_k_plot * 2]
        if valid:
            hk, hp = zip(*valid)
            tail_exp = f"1+α={1+hw_alpha:.2f}" if hw_alpha > 0.01 else "≈exp"
            ax1.plot(hk, hp, 'b-', alpha=0.7, linewidth=2,
                    label=f'Hollywood (α={hw_alpha:.2f}, θ={hw_theta:.0f}), tail~k⁻⁽{tail_exp}⁾')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Degree (k)', fontsize=11)
    ax1.set_ylabel('P(k)', fontsize=11)
    ax1.set_title(f'{name} — Empirical vs Models')
    ax1.legend(fontsize=7.5, loc='upper right')
    ax1.grid(True, alpha=0.2)
    min_prob = min(pr for pr in emp_probs if pr > 0) * 0.1
    ax1.set_ylim(bottom=max(min_prob, 1e-8))

    # Right panel: models side by side (what each predicts)
    # Config = empirical (green)
    ax2.scatter(emp_ks, emp_probs, s=10, alpha=0.4, color='#2ecc71',
                label='Config (= empirical)', zorder=3)

    # ER (red)
    ax2.plot(er_ks[er_mask], er_probs[er_mask], 'r-', alpha=0.8, linewidth=2.5,
             label='ER: Binomial/Poisson')

    # PA (orange)
    ax2.plot(pa_ks[pa_mask], pa_probs[pa_mask], '-', color='#e67e22', alpha=0.8, linewidth=2.5,
             label='PA: k⁻³ power law')

    # Hollywood (blue)
    if hw_dist:
        valid = [(k, pr) for k, pr in zip(hw_ks_list, hw_probs_list) if pr > 1e-10 and k <= max_k_plot * 2]
        if valid:
            hk, hp = zip(*valid)
            ax2.plot(hk, hp, 'b-', alpha=0.8, linewidth=2.5,
                    label=f'Hollywood: k⁻⁽¹⁺α⁾ (α={hw_alpha:.2f})')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Degree (k)', fontsize=11)
    ax2.set_ylabel('P(k)', fontsize=11)
    ax2.set_title(f'{name} — Model Predictions Compared')
    ax2.legend(fontsize=7.5, loc='upper right')
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(bottom=max(min_prob, 1e-8))

    plt.suptitle(f'{source} / {name}  (n={n:,}, m={m:,})', fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"model_overlay_{source}_{name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    # Load results
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_analysis_results.csv"))

    all_datasets = {}
    all_datasets.update({(k, "SZIP"): v for k, v in SZIP_DATASETS.items()})
    all_datasets.update({(k, "REC"): v for k, v in REC_DATASETS.items()})
    all_datasets.update({(k, "TU"): v for k, v in TU_DATASETS.items()})

    for (name, source), url in all_datasets.items():
        row = df[(df["dataset"] == name) & (df["source"] == source)]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        # Skip very large graphs for PY simulation
        if row["nodes"] > 500000:
            print(f"  Skipping {name} (too large for PY simulation, n={int(row['nodes']):,})")
            continue

        print(f"\nProcessing {source}/{name}...")
        cache_dir = os.path.join(DATA_DIR, name)
        graphs = parse_tu_dataset(name, cache_dir)

        if len(graphs) == 1:
            degrees = np.array([d for _, d in graphs[0].degree()])
        else:
            degrees = np.concatenate([[d for _, d in G.degree()] for G in graphs])

        hw_alpha = row.get("hollywood_alpha", float('nan'))
        hw_theta = row.get("hollywood_theta", float('nan'))

        make_model_overlay_plot(
            degrees, name, source,
            er_density=row["density"],
            hw_alpha=hw_alpha,
            hw_theta=hw_theta,
            n_nodes=int(row["nodes"]),
            n_edges=int(row["edges"]),
            is_collection=(len(graphs) > 1)
        )

    print("\nDone! Model overlay plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
