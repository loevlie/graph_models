#!/usr/bin/env python3
"""
Generate a presentation-ready PDF explaining each model with worked examples.
Uses a simple 4-node diamond graph throughout for consistency.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binom, poisson
from scipy.special import gammaln
from collections import Counter
import networkx as nx

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(OUTPUT_DIR, "edge_models_explained.pdf")

C_ER = '#e74c3c'
C_PA = '#e67e22'
C_HW = '#3498db'
C_CFG = '#2ecc71'


def text_page(pdf, lines, title=None, title_color='#1a1a2e'):
    """Render a page of monospace text with optional title. No overlap."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.92])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    y = 0.97
    if title:
        ax.text(0.01, y, title, fontsize=17, fontweight='bold', color=title_color,
                verticalalignment='top')
        y -= 0.05

    for line in lines:
        if y < 0.02:
            break
        ax.text(0.01, y, line, fontsize=9.8, verticalalignment='top',
                family='monospace', linespacing=1.3)
        y -= 0.027

    pdf.savefig(fig)
    plt.close()


def draw_diamond(ax, highlight_edges=None, label_degrees=True, title=None):
    """Draw the 4-node diamond graph used throughout."""
    G = nx.Graph()
    G.add_edges_from([(0,1), (0,2), (1,3), (2,3), (1,2)])
    pos = {0: (0, 1), 1: (-1, 0), 2: (1, 0), 3: (0, -1)}
    labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    edge_colors = []
    for e in G.edges():
        if highlight_edges and (e in highlight_edges or (e[1],e[0]) in highlight_edges):
            edge_colors.append(C_ER)
        else:
            edge_colors.append('#bdc3c7')

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=2.5, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#ecf0f1', node_size=500,
                          edgecolors='#2c3e50', linewidths=2)

    if label_degrees:
        degs = dict(G.degree())
        dlabels = {n: f"{labels[n]}\nd={degs[n]}" for n in G.nodes()}
    else:
        dlabels = labels
    nx.draw_networkx_labels(G, pos, dlabels, ax=ax, font_size=9, font_weight='bold')

    if title:
        ax.set_title(title, fontsize=10, pad=8)
    ax.axis('off')
    return G


def title_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.text(0.5, 0.68, 'Edge Models for\nGraph Compression', fontsize=32,
            ha='center', va='center', fontweight='bold', color='#1a1a2e')
    ax.text(0.5, 0.52, 'A step-by-step guide with worked examples',
            fontsize=14, ha='center', va='center', color='#555')
    ax.text(0.5, 0.46, 'Using a simple diamond graph throughout',
            fontsize=12, ha='center', va='center', color='#888')

    # Draw the diamond
    ax_g = fig.add_axes([0.3, 0.22, 0.4, 0.2])
    draw_diamond(ax_g, title='Our running example: 4 nodes, 5 edges')

    # Expressiveness arrow
    models = [("ER", C_ER, "1 param"), ("PA", C_PA, "1 param"),
              ("Hollywood", C_HW, "2 params"), ("Config", C_CFG, "n params")]
    for i, (name, color, params) in enumerate(models):
        x = 0.14 + i * 0.22
        ax.add_patch(plt.Rectangle((x - 0.07, 0.10), 0.14, 0.07,
                     facecolor=color, alpha=0.15, edgecolor=color, linewidth=2,
                     transform=ax.transAxes))
        ax.text(x, 0.14, f'{name}\n{params}', ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)
        if i < 3:
            ax.annotate('', xy=(x + 0.09, 0.135), xytext=(x + 0.13, 0.135),
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))

    ax.text(0.5, 0.065, 'fewer params, simpler  ←→  more params, better fit',
            ha='center', fontsize=9, color='#999', style='italic')

    pdf.savefig(fig)
    plt.close()


def setup_page(pdf):
    """Introduce the diamond graph and key concepts."""
    fig = plt.figure(figsize=(8.5, 11))

    # Title
    ax_t = fig.add_axes([0.06, 0.93, 0.88, 0.05])
    ax_t.axis('off')
    ax_t.text(0, 0.5, 'Setup: Our Example Graph', fontsize=17, fontweight='bold', color='#1a1a2e')

    # Draw diamond
    ax_g = fig.add_axes([0.08, 0.72, 0.35, 0.20])
    draw_diamond(ax_g, title='Diamond graph')

    # Graph stats
    ax_s = fig.add_axes([0.50, 0.72, 0.44, 0.20])
    ax_s.axis('off')
    stats = [
        "n = 4 nodes (A, B, C, D)",
        "m = 5 edges (A-B, A-C, B-C, B-D, C-D)",
        "",
        "Degree sequence: [2, 3, 3, 2]",
        "  A has 2 connections",
        "  B has 3 connections",
        "  C has 3 connections",
        "  D has 2 connections",
        "",
        "Total possible edges: n(n-1)/2 = 6",
        "Density: 5/6 = 0.833",
        "Mean degree: (2+3+3+2)/4 = 2.5",
    ]
    for i, line in enumerate(stats):
        ax_s.text(0, 0.95 - i*0.085, line, fontsize=9.5, family='monospace')

    # What is bits per edge
    ax_b = fig.add_axes([0.06, 0.38, 0.88, 0.32])
    ax_b.axis('off')
    bpe_text = [
        'What is "bits per edge"?',
        "",
        "A model assigns a probability P(G) to a graph.",
        "Bits needed to encode G = -log₂(P(G))",
        "Bits per edge = -log₂(P(G)) / m",
        "",
        "Higher P(G) → fewer bits → better model.",
        "",
        "Analogy: coin flips",
        "  Flip a coin 10 times, get 7 heads, 3 tails.",
        "  Fair coin model (P(H)=0.5):",
        "    P(sequence) = 0.5¹⁰ = 1/1024",
        "    bits = -log₂(1/1024) = 10.0 → 1.00 bits/flip",
        "",
        "  Better model (P(H)=0.7, matching the data):",
        "    P(sequence) = 0.7⁷ × 0.3³ = 0.00222",
        "    bits = -log₂(0.00222) = 8.82 → 0.88 bits/flip",
        "",
        "  The model that matches the data uses fewer bits.",
    ]
    for i, line in enumerate(bpe_text):
        ax_b.text(0.01, 0.98 - i*0.053, line, fontsize=9.5, family='monospace',
                  fontweight='bold' if i == 0 else 'normal')

    # What is a degree distribution
    ax_d = fig.add_axes([0.06, 0.04, 0.88, 0.32])
    ax_d.axis('off')
    dd_text = [
        'What is a degree distribution?',
        "",
        "P(k) = fraction of nodes with degree k.",
        "",
        "Our diamond graph:",
        "  P(2) = 2/4 = 0.50   (nodes A, D have degree 2)",
        "  P(3) = 2/4 = 0.50   (nodes B, C have degree 3)",
        "",
        "What is a Poisson distribution?",
        "",
        "If you do n independent trials each with small probability p,",
        'the number of successes follows Poisson(λ) where λ = n·p.',
        "",
        "Example: each node in ER has n-1 = 3 potential neighbors,",
        "each present with prob p = 0.833.",
        "Expected degree λ = 3 × 0.833 = 2.5",
        "P(degree=k) = e^(-λ) · λᵏ / k!",
        "  P(0) = 0.082,  P(1) = 0.205,  P(2) = 0.257,  P(3) = 0.214",
        "",
        "(For small n, the exact distribution is Binomial, not Poisson,",
        " but Poisson is the standard approximation for large graphs.)",
    ]
    for i, line in enumerate(dd_text):
        ax_d.text(0.01, 0.98 - i*0.048, line, fontsize=9.5, family='monospace',
                  fontweight='bold' if i in [0, 8] else 'normal')

    pdf.savefig(fig)
    plt.close()


def er_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))

    ax_t = fig.add_axes([0.06, 0.93, 0.88, 0.05])
    ax_t.axis('off')
    ax_t.text(0, 0.5, 'Model 1: Erdős-Rényi (ER)', fontsize=17, fontweight='bold', color=C_ER)

    # How it works
    ax_h = fig.add_axes([0.06, 0.74, 0.88, 0.18])
    ax_h.axis('off')
    lines = [
        "How it works:",
        "  1 parameter: p = edge probability",
        "  For every pair of nodes, independently flip a biased coin:",
        "    heads (prob p) → edge exists",
        "    tails (prob 1-p) → no edge",
        "",
        "  Set p = density = (actual edges) / (possible edges)",
        "  All pairs treated identically — no hubs, no structure.",
    ]
    for i, line in enumerate(lines):
        ax_h.text(0.01, 0.95 - i*0.12, line, fontsize=10, family='monospace')

    # Diamond example
    ax_g = fig.add_axes([0.06, 0.54, 0.30, 0.18])
    draw_diamond(ax_g, title='Diamond: n=4, m=5')

    ax_c = fig.add_axes([0.40, 0.54, 0.55, 0.18])
    ax_c.axis('off')
    calc = [
        "Calculation:",
        "  p = 5/6 = 0.833",
        "  6 possible edges: 5 present, 1 absent (A-D)",
        "",
        "  P(G) = p⁵ × (1-p)¹",
        "       = 0.833⁵ × 0.167¹",
        "       = 0.401 × 0.167 = 0.0670",
        "",
        "  bits = -log₂(0.0670) = 3.90",
        "  bits/edge = 3.90 / 5 = 0.78",
    ]
    for i, line in enumerate(calc):
        ax_c.text(0.01, 0.95 - i*0.10, line, fontsize=9.5, family='monospace')

    # Degree distribution comparison
    ax_d = fig.add_axes([0.10, 0.22, 0.80, 0.28])
    k = np.arange(0, 5)
    # Exact: Binomial(3, 0.833)
    er_probs = binom.pmf(k, 3, 5/6)
    # Empirical
    emp = {2: 0.5, 3: 0.5}

    bars1 = ax_d.bar(k - 0.15, er_probs, 0.3, color=C_ER, alpha=0.7, label='ER predicts: Binomial(3, 0.833)')
    emp_ks = [2, 3]
    emp_ps = [0.5, 0.5]
    ax_d.bar([ki + 0.15 for ki in emp_ks], emp_ps, 0.3, color='#555', alpha=0.5, label='Actual diamond')
    ax_d.set_xlabel('Degree k', fontsize=10)
    ax_d.set_ylabel('P(k)', fontsize=10)
    ax_d.set_title('What ER predicts vs what we actually have', fontsize=11)
    ax_d.legend(fontsize=9)
    ax_d.set_xticks(k)

    # Key point
    ax_k = fig.add_axes([0.06, 0.04, 0.88, 0.15])
    ax_k.axis('off')
    kp = [
        "Key point about ER:",
        "",
        "ER predicts a spread of degrees (0, 1, 2, 3 all possible).",
        "But our graph only has degrees 2 and 3.",
        "ER wastes probability on degrees that don't exist.",
        "For larger graphs with hubs (degree >> mean), this gets much worse.",
    ]
    for i, line in enumerate(kp):
        ax_k.text(0.01, 0.95 - i*0.16, line, fontsize=10, family='monospace',
                  fontweight='bold' if i == 0 else 'normal')

    pdf.savefig(fig)
    plt.close()


def pa_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))

    ax_t = fig.add_axes([0.06, 0.93, 0.88, 0.05])
    ax_t.axis('off')
    ax_t.text(0, 0.5, 'Model 2: Preferential Attachment (PA)', fontsize=17,
              fontweight='bold', color=C_PA)

    ax_h = fig.add_axes([0.06, 0.72, 0.88, 0.20])
    ax_h.axis('off')
    lines = [
        "How it works:",
        "  1 parameter: m₀ = edges each new node adds",
        "",
        "  Build the graph by adding nodes one at a time:",
        '    Each new node connects to m₀ existing nodes',
        "    Probability of picking node i ∝ degree(i)",
        '    → high-degree nodes attract more edges ("rich get richer")',
        "",
        "  This always produces: P(k) = 2·m₀·(m₀+1) / (k·(k+1)·(k+2))",
        "  The tail is always k⁻³ — you CANNOT change the slope.",
    ]
    for i, line in enumerate(lines):
        ax_h.text(0.01, 0.95 - i*0.10, line, fontsize=9.8, family='monospace')

    # Calculation
    ax_c = fig.add_axes([0.06, 0.44, 0.88, 0.26])
    ax_c.axis('off')
    calc = [
        "Diamond example:  degrees [2, 3, 3, 2], mean = 2.5",
        "  m₀ = round(2.5 / 2) = 1",
        "",
        "  PA predicts:",
        "    P(k=1) = 4/(1·2·3) = 0.667",
        "    P(k=2) = 4/(2·3·4) = 0.167",
        "    P(k=3) = 4/(3·4·5) = 0.067",
        "",
        "  Our degrees: [2, 2, 3, 3]",
        "  bits_degree = -2·log₂(0.167) - 2·log₂(0.067)",
        "              = 2·2.585 + 2·3.907",
        "              = 12.98 bits for the degree sequence",
        "",
        "  Then add wiring cost (same as other models, see Config page).",
        "  PA's total will be higher than ER here because the diamond",
        "  is dense — PA is designed for sparse, growing networks.",
    ]
    for i, line in enumerate(calc):
        ax_c.text(0.01, 0.97 - i*0.065, line, fontsize=9.5, family='monospace')

    # Plot PA distribution
    ax_p = fig.add_axes([0.10, 0.12, 0.80, 0.28])
    k = np.arange(1, 15)
    pa_probs = 2.0 * 1 * 2 / (k * (k + 1.0) * (k + 2.0))
    pa_probs = pa_probs / pa_probs.sum()
    ax_p.bar(k, pa_probs, 0.6, color=C_PA, alpha=0.7)
    ax_p.set_xlabel('Degree k', fontsize=10)
    ax_p.set_ylabel('P(k)', fontsize=10)
    ax_p.set_title('PA predicted degree distribution (m₀=1): always tail ~ k⁻³', fontsize=11)
    ax_p.set_xticks(range(1, 15))

    # Key point
    ax_k = fig.add_axes([0.06, 0.04, 0.88, 0.06])
    ax_k.axis('off')
    ax_k.text(0.01, 0.9, "Key: PA always predicts k⁻³. Good when the real graph is close to that.",
              fontsize=10, family='monospace', fontweight='bold')

    pdf.savefig(fig)
    plt.close()


def hollywood_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))

    ax_t = fig.add_axes([0.06, 0.93, 0.88, 0.05])
    ax_t.axis('off')
    ax_t.text(0, 0.5, 'Model 3: Hollywood (Pitman-Yor)', fontsize=17,
              fontweight='bold', color=C_HW)

    ax_h = fig.add_axes([0.06, 0.68, 0.88, 0.24])
    ax_h.axis('off')
    lines = [
        "How it works:",
        "  2 parameters: α (discount, 0 ≤ α < 1) and θ (concentration)",
        "",
        "  Imagine dealing out 2m = 10 edge endpoints to nodes:",
        "    First endpoint → create node A (1 endpoint so far)",
        "    Each next endpoint either:",
        "      Goes to existing node i with prob (dᵢ - α) / (t + θ)",
        "      Goes to a NEW node with prob    (θ + K·α) / (t + θ)",
        "      where t = step#, K = #nodes so far, dᵢ = endpoints at node i",
        "",
        "  α controls the power-law tail:  P(k) ~ k^(-(1+α))",
        "    α = 0   →  no power law (like uniform-ish)",
        "    α = 0.5 →  P(k) ~ k⁻¹·⁵  (heavier tail than PA)",
        "    α ≈ 1   →  very heavy tail",
        "",
        "  Unlike PA (fixed k⁻³), Hollywood can TUNE the slope.",
    ]
    for i, line in enumerate(lines):
        ax_h.text(0.01, 0.97 - i*0.065, line, fontsize=9.5, family='monospace')

    # Calculation
    ax_c = fig.add_axes([0.06, 0.36, 0.88, 0.30])
    ax_c.axis('off')
    calc = [
        "Diamond example: degrees [2,3,3,2], 2m=10 endpoints, n=4 nodes",
        "  Suppose we fit α = 0.5, θ = 1.0",
        "",
        "  The Pitman-Yor probability of this partition is:",
        "",
        "  Denominator: Γ(θ+2m) / Γ(θ) = Γ(11) / Γ(1) = 10! = 3628800",
        "",
        "  New-node factor: θ·(θ+α)·(θ+2α)·(θ+3α)",
        "    = 1.0 · 1.5 · 2.0 · 2.5 = 7.5",
        "",
        "  Growth factors per node:",
        "    degree 2: Γ(2-0.5)/Γ(1-0.5) = Γ(1.5)/Γ(0.5) = 0.5    (×2 nodes)",
        "    degree 3: Γ(3-0.5)/Γ(1-0.5) = Γ(2.5)/Γ(0.5) = 0.75   (×2 nodes)",
        "",
        "  P = 7.5 × 0.5² × 0.75² / 3628800",
        "    = 7.5 × 0.25 × 0.5625 / 3628800 = 0.000000291",
        "",
        "  bits = -log₂(0.000000291) = 21.7 bits  ← very expensive!",
    ]
    for i, line in enumerate(calc):
        ax_c.text(0.01, 0.97 - i*0.056, line, fontsize=9.3, family='monospace')

    # Tuning alpha plot
    ax_p = fig.add_axes([0.10, 0.10, 0.80, 0.22])
    k = np.arange(1, 30)
    for alpha_val, ls, lw in [(0.25, '--', 1.5), (0.5, '-', 2.5), (0.75, ':', 2)]:
        probs = np.exp(gammaln(k - alpha_val) - gammaln(1 - alpha_val) - gammaln(k + 1))
        probs = probs / probs.sum()
        ax_p.plot(k, probs, ls, color=C_HW, linewidth=lw,
                 label=f'α={alpha_val} → tail ~ k⁻{1+alpha_val:.2f}')
    ax_p.set_xscale('log')
    ax_p.set_yscale('log')
    ax_p.set_xlabel('Degree k')
    ax_p.set_ylabel('P(k)')
    ax_p.set_title('Hollywood: different α values change the slope', fontsize=11)
    ax_p.legend(fontsize=9)
    ax_p.grid(True, alpha=0.2)

    # Key
    ax_k = fig.add_axes([0.06, 0.03, 0.88, 0.06])
    ax_k.axis('off')
    ax_k.text(0.01, 0.9,
              "Key: tunable slope is nice, but the joint partition probability\n"
              "has a huge denominator that makes it expensive (see next page).",
              fontsize=9.5, family='monospace', fontweight='bold')

    pdf.savefig(fig)
    plt.close()


def hollywood_vs_pa_page(pdf):
    """Explain why Hollywood does worse than PA."""
    text_page(pdf, [
        "",
        "PA encodes each degree INDEPENDENTLY:",
        "  bits = Σᵢ -log₂( P_PA(dᵢ) )",
        "  = -log₂(P(2)) - log₂(P(3)) - log₂(P(3)) - log₂(P(2))",
        "",
        "  Each term is just a lookup in the k⁻³ formula.",
        "  The probabilities are reasonable: P(2)=0.167, P(3)=0.067.",
        "",
        "",
        "Hollywood encodes the ENTIRE partition JOINTLY:",
        "  bits = -log₂( P_PY(all degrees together | α, θ) )",
        "",
        "  This asks: what is the probability that a sequential",
        "  process (dealing out 10 endpoints one at a time) produces",
        "  EXACTLY the partition [2, 3, 3, 2]?",
        "",
        "  The denominator Γ(θ+2m)/Γ(θ) counts ALL possible ways",
        "  to assign 2m endpoints. For 2m=10: denominator = 10! = 3.6M",
        "  The numerator is much smaller → tiny probability → many bits.",
        "",
        "",
        "It's like the difference between:",
        '  PA:        "What\'s the chance of rolling a 3?" → ~17%',
        '  Hollywood: "What\'s the chance of this EXACT sequence',
        '              of 10 rolls?" → tiny',
        "",
        "",
        "This is why Hollywood scores worse in our comparison.",
        "It's not that the model is bad — it's that we're",
        "asking it a harder question (joint probability of the",
        "full partition vs marginal probability of each degree).",
        "",
        "In principle, Hollywood SHOULD be better than PA",
        "because it can tune the power-law slope. But the joint",
        "probability formulation makes it pay a steep price.",
    ], title="Why does Hollywood lose to PA?", title_color=C_HW)


def config_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))

    ax_t = fig.add_axes([0.06, 0.93, 0.88, 0.05])
    ax_t.axis('off')
    ax_t.text(0, 0.5, 'Model 4: Configuration Model', fontsize=17,
              fontweight='bold', color=C_CFG)

    ax_h = fig.add_axes([0.06, 0.74, 0.88, 0.18])
    ax_h.axis('off')
    lines = [
        "How it works:",
        "  Parameters: the FULL degree sequence [2, 3, 3, 2]  (n numbers)",
        "",
        "  Step 1: Give each node 'stubs' (half-edges) = its degree",
        "    A gets 2 stubs, B gets 3, C gets 3, D gets 2",
        "    Total: 10 stubs = 2 × 5 edges",
        "",
        "  Step 2: Randomly pair up stubs to form edges",
        "    The degree sequence is INPUT, not predicted.",
    ]
    for i, line in enumerate(lines):
        ax_h.text(0.01, 0.95 - i*0.11, line, fontsize=9.8, family='monospace')

    # Wiring explanation with stubs
    ax_w = fig.add_axes([0.06, 0.44, 0.88, 0.28])
    ax_w.axis('off')
    wiring = [
        "Cost = Part 1 (degree sequence) + Part 2 (wiring)",
        "",
        "Part 1: Encode the degree sequence",
        "  Build empirical distribution: P(2)=2/4=0.5, P(3)=2/4=0.5",
        "  bits = -2·log₂(0.5) - 2·log₂(0.5) = 4·1 = 4.0 bits",
        "  + table cost (encoding the distribution itself) ≈ 4 bits",
        "  → about 8 bits total",
        "",
        "Part 2: Encode the wiring",
        "  How many ways to pair 10 stubs? (2m-1)!! = 9!! = 9·7·5·3·1 = 945",
        "  How many give THIS graph? ∏dᵢ! = 2!·3!·3!·2! = 2·6·6·2 = 144",
        "  bits = log₂(945/144) = log₂(6.5625) = 2.71 bits",
        "",
        "Total: 8 + 2.71 = 10.71 bits",
        "bits/edge = 10.71 / 5 = 2.14",
    ]
    for i, line in enumerate(wiring):
        ax_w.text(0.01, 0.97 - i*0.067, line, fontsize=9.5, family='monospace')

    # Key insight box
    ax_k = fig.add_axes([0.06, 0.06, 0.88, 0.36])
    ax_k.axis('off')
    insight = [
        "The wiring cost is the SAME for all models.",
        "",
        "Every model that conditions on degrees uses the same formula:",
        "  wiring_bits = log₂((2m-1)!!) - Σᵢ log₂(dᵢ!)",
        "",
        "The ONLY difference between models is how they",
        "encode the degree sequence:",
        "",
        "  ER         Doesn't encode degrees separately.",
        "             Uses flat prob p for every edge.",
        "             Pays when degrees are non-uniform.",
        "",
        "  PA         Encodes each degree using P(k) ~ k⁻³",
        "             1 parameter. Good for power-law graphs.",
        "",
        "  Hollywood  Encodes degree partition jointly via",
        "             Pitman-Yor process. 2 parameters. Tunable",
        "             slope, but expensive joint probability.",
        "",
        "  Config     Encodes degrees using exact empirical",
        "             frequencies. n parameters. Always fits",
        "             perfectly, but costs n numbers to transmit.",
    ]
    for i, line in enumerate(insight):
        ax_k.text(0.01, 0.97 - i*0.046, line, fontsize=9.5, family='monospace',
                  fontweight='bold' if i == 0 else 'normal')

    pdf.savefig(fig)
    plt.close()


def comparison_page(pdf):
    """Final comparison with real data."""
    fig = plt.figure(figsize=(8.5, 11))

    ax_t = fig.add_axes([0.06, 0.93, 0.88, 0.05])
    ax_t.axis('off')
    ax_t.text(0, 0.5, 'Putting It Together', fontsize=17, fontweight='bold', color='#1a1a2e')

    # Diamond summary
    ax_d = fig.add_axes([0.06, 0.76, 0.88, 0.16])
    ax_d.axis('off')
    lines = [
        "Diamond graph summary (4 nodes, 5 edges):",
        "",
        "  Model          Params     How it encodes degrees",
        "  ─────────────────────────────────────────────────────",
        "  ER             1 (p)      Doesn't — flat probability",
        "  PA             1 (m₀)     Each degree via k⁻³",
        "  Hollywood      2 (α,θ)    Joint partition probability",
        "  Config         4 (dᵢ)     Empirical frequencies",
    ]
    for i, line in enumerate(lines):
        ax_d.text(0.01, 0.95 - i*0.12, line, fontsize=9.8, family='monospace')

    # USAir97 plot
    overlay_path = os.path.join(OUTPUT_DIR, "plots", "model_overlay_SZIP_USAir97.png")
    if os.path.exists(overlay_path):
        ax_img = fig.add_axes([0.02, 0.42, 0.96, 0.32])
        img = plt.imread(overlay_path)
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title('Real data: USAir97 (332 nodes, 2126 edges)', fontsize=11)

    # USAir results
    ax_r = fig.add_axes([0.06, 0.20, 0.88, 0.20])
    ax_r.axis('off')
    results = [
        "USAir97 results:",
        "",
        "  Model       Params   Bits/Edge   Winner?",
        "  ──────────────────────────────────────────",
        "  ER          1        6.12",
        "  PA          1        4.50        ← best here!",
        "  Config      332      4.61",
        "  Hollywood   2        17.27",
        "",
        "  PA wins because USAir97's degrees happen to follow k⁻³.",
        "  Config is close but pays for 332 degree values.",
    ]
    for i, line in enumerate(results):
        ax_r.text(0.01, 0.95 - i*0.09, line, fontsize=9.8, family='monospace')

    # Takeaway
    ax_ta = fig.add_axes([0.06, 0.04, 0.88, 0.14])
    ax_ta.axis('off')
    ta = [
        "Takeaway:",
        "",
        "  No single model is best for all graphs.",
        "  • If degrees are roughly uniform → ER is fine",
        "  • If degrees follow k⁻³ → PA wins",
        "  • If degrees are messy / multi-modal → Config wins",
        "  • Hollywood is theoretically appealing (tunable + only 2 params)",
        "    but the joint partition probability hurts it in practice.",
    ]
    for i, line in enumerate(ta):
        ax_ta.text(0.01, 0.95 - i*0.125, line, fontsize=9.8, family='monospace',
                   fontweight='bold' if i == 0 else 'normal')

    pdf.savefig(fig)
    plt.close()


def main():
    with PdfPages(PDF_PATH) as pdf:
        title_page(pdf)
        setup_page(pdf)
        er_page(pdf)
        pa_page(pdf)
        hollywood_page(pdf)
        hollywood_vs_pa_page(pdf)
        config_page(pdf)
        comparison_page(pdf)

    print(f"PDF saved to: {PDF_PATH}")


if __name__ == "__main__":
    main()
