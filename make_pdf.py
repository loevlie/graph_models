#!/usr/bin/env python3
"""
Generate a presentation-ready PDF explaining each model with worked examples.
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
import textwrap

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(OUTPUT_DIR, "edge_models_explained.pdf")

# Consistent colors
C_ER = '#e74c3c'
C_PA = '#e67e22'
C_HW = '#3498db'
C_CFG = '#2ecc71'

def wrapped_text(ax, x, y, text, fontsize=10, **kwargs):
    """Add wrapped text to an axis."""
    ax.text(x, y, text, fontsize=fontsize, verticalalignment='top',
            transform=ax.transAxes, family='sans-serif', **kwargs)


def title_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.text(0.5, 0.65, 'Edge Models for\nGraph Compression', fontsize=32,
            ha='center', va='center', fontweight='bold', color='#1a1a2e')
    ax.text(0.5, 0.48, 'A worked-example guide to ER, PA, Hollywood,\nand Configuration models',
            fontsize=14, ha='center', va='center', color='#555')
    ax.text(0.5, 0.38, 'Data from "Practical Shuffle Coding" (Kunze et al., NeurIPS 2024)',
            fontsize=11, ha='center', va='center', color='#888')

    # Draw the expressiveness arrow
    models = [("ER", C_ER), ("PA", C_PA), ("Hollywood", C_HW), ("Config", C_CFG)]
    for i, (name, color) in enumerate(models):
        x = 0.18 + i * 0.2
        ax.add_patch(plt.Rectangle((x - 0.06, 0.24), 0.12, 0.06, facecolor=color,
                                    alpha=0.2, edgecolor=color, linewidth=2, transform=ax.transAxes))
        ax.text(x, 0.27, name, ha='center', va='center', fontsize=11, fontweight='bold', color=color)
        if i < 3:
            ax.annotate('', xy=(x + 0.08, 0.27), xytext=(x + 0.14, 0.27),
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', color='#999', lw=2))

    ax.text(0.5, 0.20, '← fewer params, simpler          more params, better fit →',
            ha='center', fontsize=9, color='#999', style='italic')

    pdf.savefig(fig)
    plt.close()


def bits_explainer_page(pdf):
    """Page explaining what 'bits per edge' means."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.9])
    ax.axis('off')

    text = """What does "bits per edge" mean?

A model assigns a probability P(G) to a graph G.  The number of bits
needed to encode G under that model is:

    bits = −log₂( P(G) )

Higher probability → fewer bits → better compression.

"Bits per edge" divides by the number of edges m:

    bits/edge = −log₂( P(G) ) / m

This lets us compare graphs of different sizes.

Simple example: a fair coin

Suppose you flip a coin 10 times and get HHTTHHTTHH.
P(that exact sequence) = (1/2)¹⁰ = 1/1024
bits = −log₂(1/1024) = 10 bits  →  1 bit per flip

If the coin is biased (P(H) = 0.9):
P(HHTTHHTTHH) = 0.9⁷ × 0.1³ = 0.000478
bits = −log₂(0.000478) = 11.03 bits → 1.10 bits per flip

With P(H) = 0.7 (closer to the actual 7/10 ratio):
P(HHTTHHTTHH) = 0.7⁷ × 0.3³ = 0.00222
bits = −log₂(0.00222) = 8.82 bits → 0.88 bits per flip  ← best model

The model whose probability best matches the data wins.
Same idea for graphs: the edge model that best captures
the graph's structure assigns the highest probability."""

    ax.text(0.02, 0.98, text, fontsize=11, verticalalignment='top',
            family='monospace', linespacing=1.5)
    pdf.savefig(fig)
    plt.close()


def er_page(pdf):
    """ER model explanation with worked example."""
    fig = plt.figure(figsize=(8.5, 11))

    # Title
    ax_title = fig.add_axes([0.08, 0.92, 0.84, 0.06])
    ax_title.axis('off')
    ax_title.text(0, 0.5, 'Model 1: Erdős-Rényi (ER)', fontsize=18, fontweight='bold', color=C_ER)

    # Text explanation
    ax_text = fig.add_axes([0.08, 0.62, 0.84, 0.30])
    ax_text.axis('off')
    text = """How it works:
  • One parameter: p (edge probability)
  • Each of the n(n−1)/2 possible edges exists independently with probability p
  • Set p = (number of edges) / (number of possible edges) = density

Degree distribution:
  • Each node has n−1 potential neighbors, each with probability p
  • Degree ~ Binomial(n−1, p)  ≈  Poisson(mean_degree) for large n
  • This means most nodes have similar degree — no hubs

Bits per edge:
  • P(G) = p^m × (1−p)^(N−m)   where N = n(n−1)/2, m = edges
  • bits = −m·log₂(p) − (N−m)·log₂(1−p)

Worked example: 5 nodes, 4 edges"""

    ax_text.text(0.02, 0.98, text, fontsize=10.5, verticalalignment='top',
                family='monospace', linespacing=1.5)

    # Draw example graph
    ax_graph = fig.add_axes([0.08, 0.38, 0.35, 0.24])
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(0,1), (1,2), (2,3), (0,4)])
    pos = {0: (0,1), 1: (1,1.5), 2: (2,1), 3: (2.5,0), 4: (-0.5,0)}
    degrees = dict(G.degree())
    colors = [C_ER] * 5
    nx.draw_networkx(G, pos, ax=ax_graph, node_color=colors, node_size=400,
                     font_color='white', font_weight='bold',
                     labels={i: str(degrees[i]) for i in range(5)},
                     edgecolors='white', linewidths=1.5, width=2)
    ax_graph.set_title('Example: 5 nodes, 4 edges\n(labels = degree)', fontsize=9)
    ax_graph.axis('off')

    # Calculation
    ax_calc = fig.add_axes([0.48, 0.38, 0.48, 0.24])
    ax_calc.axis('off')
    calc = """Calculation:
  n = 5,  m = 4
  N = 5×4/2 = 10 possible edges
  p = 4/10 = 0.4

  bits = −4·log₂(0.4) − 6·log₂(0.6)
       = −4·(−1.322) − 6·(−0.737)
       = 5.288 + 4.422
       = 9.71 bits

  bits/edge = 9.71 / 4 = 2.43"""

    ax_calc.text(0.02, 0.98, calc, fontsize=10, verticalalignment='top',
                family='monospace', linespacing=1.5)

    # Degree distribution plot
    ax_deg = fig.add_axes([0.12, 0.06, 0.76, 0.28])
    k = np.arange(0, 5)
    er_probs = binom.pmf(k, 4, 0.4)  # Binom(n-1, p) = Binom(4, 0.4)
    ax_deg.bar(k - 0.15, er_probs, 0.3, color=C_ER, alpha=0.7, label='ER predicts: Binomial(4, 0.4)')
    # Empirical
    emp = Counter([degrees[i] for i in range(5)])
    emp_ks = sorted(emp.keys())
    emp_probs = [emp[ki]/5 for ki in emp_ks]
    ax_deg.bar([ki + 0.15 for ki in emp_ks], emp_probs, 0.3, color='black', alpha=0.5, label='Empirical')
    ax_deg.set_xlabel('Degree k')
    ax_deg.set_ylabel('P(k)')
    ax_deg.set_title('ER predicted vs empirical degree distribution')
    ax_deg.legend(fontsize=9)
    ax_deg.set_xticks(k)

    pdf.savefig(fig)
    plt.close()


def pa_page(pdf):
    """PA model explanation with worked example."""
    fig = plt.figure(figsize=(8.5, 11))

    ax_title = fig.add_axes([0.08, 0.92, 0.84, 0.06])
    ax_title.axis('off')
    ax_title.text(0, 0.5, 'Model 2: Preferential Attachment (PA)', fontsize=18,
                  fontweight='bold', color=C_PA)

    ax_text = fig.add_axes([0.08, 0.60, 0.84, 0.32])
    ax_text.axis('off')
    text = """How it works:
  • One parameter: m₀ (edges added per new node)
  • Start with a small seed graph
  • Each new node connects to m₀ existing nodes
  • Probability of connecting to node i:  P(i) = degree(i) / Σ degrees
  • "Rich get richer" — high-degree nodes attract more connections

Degree distribution:
  • P(k) = 2·m₀·(m₀+1) / ( k·(k+1)·(k+2) )   for k ≥ m₀
  • This is always a power law with exponent −3
  • The slope is FIXED — you can't tune it

Bits per edge:
  • Encode each node's degree using the PA-predicted distribution
  • bits_degree_seq = Σᵢ −log₂( P_PA(dᵢ) )
  • Then add wiring cost (same for all models — see Config page)

Worked example:  Same graph — 5 nodes, 4 edges, mean degree 1.6"""

    ax_text.text(0.02, 0.98, text, fontsize=10.5, verticalalignment='top',
                family='monospace', linespacing=1.5)

    # Show the growth process
    ax_grow = fig.add_axes([0.08, 0.34, 0.84, 0.24])
    ax_grow.axis('off')

    calc = """  m₀ = max(1, round(mean_degree / 2)) = max(1, round(0.8)) = 1
  PA predicts: P(k) = 2·1·2 / (k·(k+1)·(k+2))

  k=1: P(1) = 4/(1·2·3) = 0.667
  k=2: P(2) = 4/(2·3·4) = 0.167
  k=3: P(3) = 4/(3·4·5) = 0.067

  Our degree sequence: [2, 2, 2, 1, 1]
  bits_degree = −2·log₂(0.167) − 2·log₂(0.167) − 1·log₂(0.167)
                −1·log₂(0.667) − 1·log₂(0.667)
              = 3·(−log₂(0.167)) + 2·(−log₂(0.667))
              = 3·2.585 + 2·0.585 = 8.93 bits for the degree sequence"""

    ax_grow.text(0.02, 0.98, calc, fontsize=10, verticalalignment='top',
                family='monospace', linespacing=1.5)

    # Degree distribution comparison
    ax_deg = fig.add_axes([0.12, 0.06, 0.76, 0.26])
    k = np.arange(1, 10)
    pa_probs = 2.0 * 1 * 2 / (k * (k + 1.0) * (k + 2.0))
    pa_probs = pa_probs / pa_probs.sum()

    ax_deg.bar(k, pa_probs, 0.6, color=C_PA, alpha=0.7, label='PA predicts: P(k) ~ k⁻³')
    ax_deg.set_xlabel('Degree k')
    ax_deg.set_ylabel('P(k)')
    ax_deg.set_title('PA predicted degree distribution (m₀=1)')
    ax_deg.legend(fontsize=9)
    ax_deg.set_xticks(range(1, 10))

    pdf.savefig(fig)
    plt.close()


def hollywood_page(pdf):
    """Hollywood/PY model explanation."""
    fig = plt.figure(figsize=(8.5, 11))

    ax_title = fig.add_axes([0.08, 0.92, 0.84, 0.06])
    ax_title.axis('off')
    ax_title.text(0, 0.5, 'Model 3: Hollywood (Pitman-Yor)', fontsize=18,
                  fontweight='bold', color=C_HW)

    ax_text = fig.add_axes([0.08, 0.52, 0.84, 0.40])
    ax_text.axis('off')
    text = """How it works:
  • Two parameters: α (discount, 0 ≤ α < 1) and θ (concentration, θ > −α)
  • Model the 2m edge endpoints as a seating process:
    - First endpoint sits at node 1
    - Each next endpoint either:
        Goes to existing node i with prob:  (dᵢ − α) / (t + θ)
        Goes to a new node with prob:       (θ + K·α) / (t + θ)
      where t = step number, K = nodes used so far, dᵢ = current degree

  • α controls the power-law tail:  P(k) ~ k^(−(1+α))  for large k
    α = 0  →  no power law (Dirichlet process)
    α = 0.5 →  P(k) ~ k^(−1.5)
    α = 0.75 → P(k) ~ k^(−1.75)

  • θ controls how many nodes get used (higher θ → more nodes)

Unlike PA (fixed k⁻³), Hollywood can tune the power-law slope.

Probability of a degree sequence (d₁, ..., dₙ) from 2m endpoints:

              ⎡ n-1              ⎤   n  Γ(dᵢ − α)
  P(d|α,θ) = ⎢ ∏ (θ + k·α)     ⎥ · ∏ ──────────
              ⎣ k=0              ⎦  i=1  Γ(1 − α)
              ─────────────────────────────────────
                    Γ(θ + 2m) / Γ(θ)"""

    ax_text.text(0.02, 0.98, text, fontsize=10.5, verticalalignment='top',
                family='monospace', linespacing=1.5)

    # Worked example
    ax_calc = fig.add_axes([0.08, 0.26, 0.84, 0.25])
    ax_calc.axis('off')
    calc = """Worked example:  5 nodes, degrees [2, 2, 2, 1, 1], so 2m = 8 endpoints

  Suppose fitted α = 0.5, θ = 1.0

  Denominator: Γ(θ + 2m)/Γ(θ) = Γ(9)/Γ(1) = 8! = 40320

  New-node terms: θ·(θ+α)·(θ+2α)·(θ+3α)·(θ+4α)
                = 1.0 · 1.5 · 2.0 · 2.5 · 3.0 = 22.5

  Growth terms (d=2): Γ(2−0.5)/Γ(1−0.5) = Γ(1.5)/Γ(0.5) = 0.5  (×3 nodes)
  Growth terms (d=1): Γ(1−0.5)/Γ(1−0.5) = 1.0                   (×2 nodes)

  P = 22.5 · (0.5)³ · (1.0)² / 40320 = 22.5 · 0.125 / 40320 = 0.0000698

  bits_degree_seq = −log₂(0.0000698) = 13.81 bits"""

    ax_calc.text(0.02, 0.98, calc, fontsize=10, verticalalignment='top',
                family='monospace', linespacing=1.5)

    # Different alpha values
    ax_deg = fig.add_axes([0.12, 0.05, 0.76, 0.19])
    k = np.arange(1, 30)
    for alpha_val, ls in [(0.25, '--'), (0.5, '-'), (0.75, ':')]:
        probs = np.exp(gammaln(k - alpha_val) - gammaln(1 - alpha_val) - gammaln(k + 1))
        probs = probs / probs.sum()
        ax_deg.plot(k, probs, ls, color=C_HW, linewidth=2,
                   label=f'α={alpha_val} → tail ~ k⁻{1+alpha_val:.2f}', alpha=0.8)
    ax_deg.set_xscale('log')
    ax_deg.set_yscale('log')
    ax_deg.set_xlabel('Degree k')
    ax_deg.set_ylabel('P(k)')
    ax_deg.set_title('Hollywood: tuning α changes the power-law slope')
    ax_deg.legend(fontsize=8)
    ax_deg.grid(True, alpha=0.2)

    pdf.savefig(fig)
    plt.close()


def config_page(pdf):
    """Configuration model explanation."""
    fig = plt.figure(figsize=(8.5, 11))

    ax_title = fig.add_axes([0.08, 0.92, 0.84, 0.06])
    ax_title.axis('off')
    ax_title.text(0, 0.5, 'Model 4: Configuration Model', fontsize=18,
                  fontweight='bold', color=C_CFG)

    ax_text = fig.add_axes([0.08, 0.56, 0.84, 0.36])
    ax_text.axis('off')
    text = """How it works:
  • Parameters: the FULL degree sequence (d₁, d₂, ..., dₙ)  — n numbers
  • Give each node dᵢ "stubs" (half-edges)
  • Randomly pair up all 2m stubs to form m edges
  • The degree sequence is INPUT, not predicted

This is NOT a generative model like the others — it's a two-part code:
  Part 1: Encode the degree sequence (transmit n numbers)
  Part 2: Encode which specific wiring (stub pairing) was used

Cost breakdown:

  Part 1: Degree sequence
    • Build empirical distribution: P(k) = (count of nodes with degree k) / n
    • bits = Σᵢ −log₂( P(dᵢ) )  +  cost to transmit the distribution table

  Part 2: Wiring (same formula used by ALL models that condition on degrees)
    • Total stub pairings: (2m−1)!! = (2m)! / (2ᵐ · m!)
    • Pairings giving THIS specific graph: ∏ᵢ dᵢ!
    • bits_wiring = log₂((2m−1)!!) − Σᵢ log₂(dᵢ!)"""

    ax_text.text(0.02, 0.98, text, fontsize=10.5, verticalalignment='top',
                family='monospace', linespacing=1.5)

    # Stub pairing diagram
    ax_stub = fig.add_axes([0.08, 0.32, 0.40, 0.22])
    ax_stub.axis('off')
    ax_stub.set_xlim(-1, 6)
    ax_stub.set_ylim(-0.5, 3)
    # Draw nodes with stubs
    nodes_info = [(0, 2, 2, 'A'), (2.5, 2, 2, 'B'), (5, 2, 2, 'C'), (1.25, 0, 1, 'D'), (3.75, 0, 1, 'E')]
    for x, y, deg, label in nodes_info:
        circle = plt.Circle((x, y), 0.3, color=C_CFG, alpha=0.3, ec=C_CFG, lw=2)
        ax_stub.add_patch(circle)
        ax_stub.text(x, y, f'{label}\nd={deg}', ha='center', va='center', fontsize=8, fontweight='bold')
        # Draw stubs
        for s in range(deg):
            angle = np.pi/2 + s * np.pi / max(1, deg)
            dx = 0.4 * np.cos(angle)
            dy = 0.4 * np.sin(angle)
            ax_stub.plot([x, x+dx], [y, y+dy], '-', color=C_CFG, lw=3, solid_capstyle='round')
    ax_stub.set_title('Stubs before pairing\n(total: 8 stubs = 2×4 edges)', fontsize=9)

    # Worked example
    ax_calc = fig.add_axes([0.50, 0.26, 0.46, 0.28])
    ax_calc.axis('off')
    calc = """Worked example:
  Degrees: [2, 2, 2, 1, 1], m = 4

  Part 1: Degree sequence
    P(d=2) = 3/5 = 0.6
    P(d=1) = 2/5 = 0.4
    bits = −3·log₂(0.6) − 2·log₂(0.4)
         = 3·0.737 + 2·1.322
         = 2.211 + 2.644 = 4.86 bits
    + table cost ≈ 4 bits
    total ≈ 8.86 bits

  Part 2: Wiring
    (2·4−1)!! = 7!! = 7·5·3·1 = 105
    ∏dᵢ! = 2!·2!·2!·1!·1! = 8
    bits = log₂(105) − log₂(8)
         = 6.71 − 3.0 = 3.71 bits

  Total: 8.86 + 3.71 = 12.57 bits
  bits/edge = 12.57 / 4 = 3.14"""

    ax_calc.text(0.02, 0.98, calc, fontsize=9.5, verticalalignment='top',
                family='monospace', linespacing=1.4)

    # Key insight
    ax_insight = fig.add_axes([0.08, 0.06, 0.84, 0.18])
    ax_insight.axis('off')
    insight = """Key insight:  The wiring cost is the SAME for all models that condition on degrees.
  The only difference between models is how they encode the degree sequence:

  ER:          Doesn't encode degrees — uses flat probability p for every edge
  PA:          Encodes degrees using P(k) ~ k⁻³           (1 parameter)
  Hollywood:   Encodes degrees using Pitman-Yor EPPF       (2 parameters)
  Config:      Encodes degrees using empirical frequencies  (n parameters)

  More parameters → better fit to data → fewer bits for degrees
  But also → more bits to describe the model itself"""

    ax_insight.text(0.02, 0.98, insight, fontsize=10, verticalalignment='top',
                   family='monospace', linespacing=1.5,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8fde8', alpha=0.5))

    pdf.savefig(fig)
    plt.close()


def comparison_page(pdf):
    """Final comparison with the USAir97 plot."""
    fig = plt.figure(figsize=(8.5, 11))

    ax_title = fig.add_axes([0.08, 0.92, 0.84, 0.06])
    ax_title.axis('off')
    ax_title.text(0, 0.5, 'Putting It Together: USAir97', fontsize=18, fontweight='bold', color='#1a1a2e')

    # Embed the USAir97 overlay plot
    overlay_path = os.path.join(OUTPUT_DIR, "plots", "model_overlay_SZIP_USAir97.png")
    if os.path.exists(overlay_path):
        img = plt.imread(overlay_path)
        ax_img = fig.add_axes([0.04, 0.52, 0.92, 0.38])
        ax_img.imshow(img)
        ax_img.axis('off')

    # Summary table
    ax_table = fig.add_axes([0.08, 0.28, 0.84, 0.22])
    ax_table.axis('off')
    text = """USAir97: 332 nodes, 2,126 edges, density = 0.039

  Model         Params    Bits/Edge   Total Bits
  ─────────────────────────────────────────────────
  ER            1 (p)       6.12       13,007
  PA            1 (m₀)      4.50        9,563
  Config        332 (dᵢ)    4.61        9,800
  Hollywood     2 (α,θ)    17.27       36,700

  PA wins here!  Its k⁻³ prediction happens to match USAir97's
  degree distribution closely, and it only needs 1 parameter.

  Config is a close second — it always fits well, but pays for
  encoding all 332 degree values.

  Hollywood loses badly — with α ≈ 0, it degenerates to a model
  that can't capture the heavy tail at all."""

    ax_table.text(0.02, 0.98, text, fontsize=10, verticalalignment='top',
                 family='monospace', linespacing=1.5)

    # Bottom insight
    ax_bottom = fig.add_axes([0.08, 0.06, 0.84, 0.20])
    ax_bottom.axis('off')
    text2 = """The tradeoff:
  ┌────────────────────────────────────────────────────────────────┐
  │  Fewer params = cheaper model description                     │
  │                 but worse fit to data (more bits for degrees)  │
  │                                                                │
  │  More params  = better fit to data                             │
  │                 but expensive model description                │
  │                                                                │
  │  Best model = the one where (model cost + data cost) is       │
  │               minimized. This depends on the graph!            │
  └────────────────────────────────────────────────────────────────┘

  For compression (e.g., shuffle coding), you also want a model
  that's easy to sample from — which favors parametric models
  (ER, PA, Hollywood) over the configuration model."""

    ax_bottom.text(0.02, 0.98, text2, fontsize=10, verticalalignment='top',
                  family='monospace', linespacing=1.4)

    pdf.savefig(fig)
    plt.close()


def main():
    with PdfPages(PDF_PATH) as pdf:
        title_page(pdf)
        bits_explainer_page(pdf)
        er_page(pdf)
        pa_page(pdf)
        hollywood_page(pdf)
        config_page(pdf)
        comparison_page(pdf)

    print(f"PDF saved to: {PDF_PATH}")


if __name__ == "__main__":
    main()
