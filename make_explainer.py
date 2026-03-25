#!/usr/bin/env python3
"""
Generate explainer figures showing how each edge model works
and what degree distributions they predict.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from scipy.stats import binom, poisson
from scipy.special import gammaln
from collections import Counter
import networkx as nx

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def draw_small_graph(ax, G, pos, title, node_colors=None, highlight_edges=None):
    """Draw a small graph with nice styling."""
    if node_colors is None:
        node_colors = ['#3498db'] * len(G)
    edge_colors = []
    edge_widths = []
    for e in G.edges():
        if highlight_edges and e in highlight_edges:
            edge_colors.append('#e74c3c')
            edge_widths.append(2.5)
        else:
            edge_colors.append('#bdc3c7')
            edge_widths.append(1.5)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=300,
                          edgecolors='white', linewidths=1.5)
    degrees = dict(G.degree())
    labels = {n: str(degrees[n]) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9, font_weight='bold', font_color='white')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')


def make_mechanism_figure():
    """Create a figure showing HOW each model generates edges."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    np.random.seed(42)

    n = 8
    pos = nx.circular_layout(nx.complete_graph(n))

    # --- ER ---
    ax = axes[0]
    G = nx.erdos_renyi_graph(n, 0.3, seed=42)
    colors = ['#e74c3c'] * n  # all same = all equal probability
    draw_small_graph(ax, G, pos, 'Erdos-Renyi (ER)', node_colors=colors)
    ax.text(0, -1.45, 'Each possible edge flipped\nindependently with prob p',
            ha='center', va='top', fontsize=9, style='italic', color='#555')

    # --- PA ---
    ax = axes[1]
    G = nx.barabasi_albert_graph(n, 2, seed=42)
    pos_pa = nx.spring_layout(G, seed=42)
    degrees = dict(G.degree())
    max_d = max(degrees.values())
    # Size nodes by degree to show "rich get richer"
    sizes = [200 + 300 * (degrees[i] / max_d) for i in range(n)]
    colors = [plt.cm.Oranges(0.3 + 0.7 * degrees[i] / max_d) for i in range(n)]
    nx.draw_networkx_edges(G, pos_pa, ax=ax, edge_color='#bdc3c7', width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(G, pos_pa, ax=ax, node_color=colors, node_size=sizes,
                          edgecolors='white', linewidths=1.5)
    labels = {i: str(degrees[i]) for i in range(n)}
    nx.draw_networkx_labels(G, pos_pa, labels, ax=ax, font_size=9, font_weight='bold', font_color='white')
    ax.set_title('Preferential Attachment (PA)', fontsize=11, fontweight='bold', pad=10)
    ax.text(0, -1.45, 'New nodes attach to high-degree\nnodes more often ("rich get richer")',
            ha='center', va='top', fontsize=9, style='italic', color='#555')
    ax.axis('off')

    # --- Hollywood ---
    ax = axes[2]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # Simulate a PY-like process: some nodes get many connections
    edges = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(2,6),(3,7),(5,6)]
    G.add_edges_from(edges)
    pos_hw = nx.spring_layout(G, seed=42)
    degrees = dict(G.degree())
    max_d = max(degrees.values())
    colors = [plt.cm.Blues(0.3 + 0.7 * degrees[i] / max_d) for i in range(n)]
    draw_small_graph(ax, G, pos_hw, 'Hollywood (Pitman-Yor)', node_colors=colors)
    ax.text(0, -1.45, 'Edge endpoints assigned via CRP:\nP(existing node) ~ degree - α\nP(new node) ~ θ + Kα',
            ha='center', va='top', fontsize=9, style='italic', color='#555')

    # --- Configuration ---
    ax = axes[3]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # Use a specific degree sequence: [4, 3, 3, 2, 2, 2, 1, 1]
    target_degs = [4, 3, 3, 2, 2, 2, 1, 1]
    # Create a graph matching this sequence
    edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,5),(2,6),(3,7)]
    G.add_edges_from(edges)
    pos_cfg = nx.spring_layout(G, seed=42)
    degrees = dict(G.degree())
    # Color by degree
    unique_degs = sorted(set(degrees.values()))
    deg_colors = {d: plt.cm.Greens(0.3 + 0.7 * i / max(1, len(unique_degs)-1))
                  for i, d in enumerate(unique_degs)}
    colors = [deg_colors[degrees[i]] for i in range(n)]
    draw_small_graph(ax, G, pos_cfg, 'Configuration Model', node_colors=colors)
    ax.text(0, -1.45, 'Given exact degree sequence,\nrandomly wire stubs together.\nDegrees are INPUT, not predicted.',
            ha='center', va='top', fontsize=9, style='italic', color='#555')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'explainer_mechanisms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def make_degree_dist_explainer():
    """Create a figure comparing what each model predicts for the degree distribution,
    using a concrete example (n=1000, mean_degree=10)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    n = 1000
    mean_deg = 10
    m = n * mean_deg // 2
    p = mean_deg / (n - 1)

    k = np.arange(1, 200)

    # --- ER: Poisson ---
    er_probs = poisson.pmf(k, mean_deg)

    # --- PA: k^(-3) ---
    m0 = mean_deg // 2
    pa_probs = 2.0 * m0 * (m0 + 1) / (k * (k + 1.0) * (k + 2.0))
    pa_probs = pa_probs / pa_probs.sum()

    # --- Hollywood (PY) with alpha=0.5 ---
    # Asymptotic: P(k) ~ k^(-(1+alpha)) = k^(-1.5)
    alpha_hw = 0.5
    hw_probs = np.zeros_like(k, dtype=float)
    for i, ki in enumerate(k):
        hw_probs[i] = np.exp(gammaln(ki - alpha_hw) - gammaln(1 - alpha_hw) - gammaln(ki + 1))
    hw_probs = hw_probs / hw_probs.sum()

    # --- Hollywood (PY) with alpha=0.75 ---
    alpha_hw2 = 0.75
    hw2_probs = np.zeros_like(k, dtype=float)
    for i, ki in enumerate(k):
        hw2_probs[i] = np.exp(gammaln(ki - alpha_hw2) - gammaln(1 - alpha_hw2) - gammaln(ki + 1))
    hw2_probs = hw2_probs / hw2_probs.sum()

    # --- Configuration: generate a realistic degree sequence (e.g., from a power-law) ---
    np.random.seed(42)
    # Use a truncated power-law with exponent 2.5
    raw = np.random.pareto(1.5, n) + 1
    raw = np.clip(raw, 1, n-1).astype(int)
    # Force sum to be even
    if raw.sum() % 2 == 1:
        raw[0] += 1
    config_counts = Counter(raw)
    config_ks = sorted(config_counts.keys())
    config_probs = [config_counts[ki] / n for ki in config_ks]

    # LEFT: Linear scale (zoomed to k < 50)
    ax = ax1
    mask = k < 50
    ax.plot(k[mask], er_probs[mask], 'r-', linewidth=2.5, label='ER: Poisson(λ=10)', alpha=0.9)
    ax.plot(k[mask], pa_probs[mask], '-', color='#e67e22', linewidth=2.5, label='PA: k⁻³', alpha=0.9)
    ax.plot(k[mask], hw_probs[mask], 'b-', linewidth=2.5, label='Hollywood: k⁻¹·⁵ (α=0.5)', alpha=0.9)
    ax.plot(k[mask], hw2_probs[mask], 'b--', linewidth=2, label='Hollywood: k⁻¹·⁷⁵ (α=0.75)', alpha=0.7)
    cfg_mask = [ki for ki in config_ks if ki < 50]
    cfg_p = [config_probs[config_ks.index(ki)] for ki in cfg_mask]
    ax.scatter(cfg_mask, cfg_p, s=20, color='#2ecc71', alpha=0.6, label='Config: empirical', zorder=3)

    ax.set_xlabel('Degree (k)', fontsize=12)
    ax.set_ylabel('P(k)', fontsize=12)
    ax.set_title('Linear Scale (k < 50)', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 50)

    # RIGHT: Log-log scale (full range)
    ax = ax2
    mask2 = (k >= 1) & (er_probs > 0)
    ax.plot(k[mask2], er_probs[mask2], 'r-', linewidth=2.5, label='ER: Poisson(λ=10)', alpha=0.9)

    pa_valid = pa_probs > 1e-10
    ax.plot(k[pa_valid], pa_probs[pa_valid], '-', color='#e67e22', linewidth=2.5, label='PA: k⁻³', alpha=0.9)

    hw_valid = hw_probs > 1e-10
    ax.plot(k[hw_valid], hw_probs[hw_valid], 'b-', linewidth=2.5, label='Hollywood: k⁻¹·⁵ (α=0.5)', alpha=0.9)

    hw2_valid = hw2_probs > 1e-10
    ax.plot(k[hw2_valid], hw2_probs[hw2_valid], 'b--', linewidth=2, label='Hollywood: k⁻¹·⁷⁵ (α=0.75)', alpha=0.7)

    ax.scatter(config_ks, config_probs, s=20, color='#2ecc71', alpha=0.6, label='Config: empirical', zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Degree (k)', fontsize=12)
    ax.set_ylabel('P(k)', fontsize=12)
    ax.set_title('Log-Log Scale (power laws are straight lines)', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=1e-6)

    plt.suptitle('What Each Model Predicts for the Degree Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'explainer_degree_dists.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def make_expressiveness_figure():
    """A simple diagram showing the expressiveness hierarchy."""
    fig, ax = plt.subplots(figsize=(12, 3))

    models = [
        ("ER\n1 param: p\nPoisson degrees", "#e74c3c", 0),
        ("PA\n1 param: m₀\nk⁻³ (fixed slope)", "#e67e22", 1),
        ("Hollywood\n2 params: α, θ\nk⁻⁽¹⁺α⁾ (tunable slope)", "#3498db", 2),
        ("Configuration\nn params: d₁...dₙ\nExact degree sequence", "#2ecc71", 3),
    ]

    for label, color, i in models:
        x = i * 2.5
        rect = plt.Rectangle((x, 0.2), 2, 1.6, facecolor=color, alpha=0.15,
                             edgecolor=color, linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 1, 1.0, label, ha='center', va='center', fontsize=10,
                fontweight='bold', color=color, zorder=3)

    # Arrows
    for i in range(3):
        ax.annotate('', xy=((i+1)*2.5 - 0.1, 1.0), xytext=(i*2.5 + 2.1, 1.0),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    ax.text(5.0, -0.15, 'More expressive →', ha='center', fontsize=11, color='#555', style='italic')
    ax.text(5.0, 2.15, 'More parameters, better fit to data, but more bits to encode the model itself',
            ha='center', fontsize=9, color='#888')

    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.4, 2.4)
    ax.axis('off')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'explainer_expressiveness.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    make_mechanism_figure()
    make_degree_dist_explainer()
    make_expressiveness_figure()
    print("Done!")
