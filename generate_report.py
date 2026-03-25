#!/usr/bin/env python3
"""
Generate a self-contained HTML report with embedded images and interactive charts.
Also generates additional comparison visualizations.
"""

import os
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
CSV_PATH = os.path.join(OUTPUT_DIR, "graph_analysis_results.csv")

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def make_comparison_plots(df):
    """Generate additional comparison visualizations."""

    model_df = df.dropna(subset=["er_bits_per_edge", "pa_bits_per_edge", "config_bits_per_edge", "hollywood_bits_per_edge"])

    # 1. Model comparison bar chart (exclude Hollywood to keep scale readable)
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(model_df))
    width = 0.2
    ax.bar(x - 1.5*width, model_df["er_bits_per_edge"], width, label="ER", color="#e74c3c", alpha=0.85)
    ax.bar(x - 0.5*width, model_df["pa_bits_per_edge"], width, label="PA", color="#e67e22", alpha=0.85)
    ax.bar(x + 0.5*width, model_df["config_bits_per_edge"], width, label="Configuration", color="#2ecc71", alpha=0.85)
    ax.bar(x + 1.5*width, model_df["hollywood_bits_per_edge"], width, label="Hollywood (PY)", color="#3498db", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_df["dataset"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Bits per Edge")
    ax.set_title("Edge Model Comparison: Bits per Edge (lower is better)")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison_bars.png")
    plt.savefig(path, dpi=150)
    plt.close()

    # 1b. Same chart but zoomed to just ER/PA/Config (Hollywood is so much worse it distorts scale)
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.25
    ax.bar(x - width, model_df["er_bits_per_edge"], width, label="ER", color="#e74c3c", alpha=0.85)
    ax.bar(x, model_df["pa_bits_per_edge"], width, label="PA", color="#e67e22", alpha=0.85)
    ax.bar(x + width, model_df["config_bits_per_edge"], width, label="Configuration", color="#2ecc71", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_df["dataset"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Bits per Edge")
    ax.set_title("ER vs PA vs Configuration (zoomed — Hollywood excluded for scale)")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison_zoomed.png")
    plt.savefig(path, dpi=150)
    plt.close()

    # 2. Best parametric model vs Config savings
    fig, ax = plt.subplots(figsize=(10, 5))
    best_parametric = model_df[["er_bits_per_edge", "pa_bits_per_edge", "hollywood_bits_per_edge"]].min(axis=1)
    savings = best_parametric.values - model_df["config_bits_per_edge"].values
    colors = ['#e74c3c' if s > 0 else '#2ecc71' for s in savings]
    ax.barh(model_df["dataset"], savings, color=colors, alpha=0.85)
    ax.set_xlabel("Bits/Edge: Best Parametric - Config (negative = Config wins)")
    ax.set_title("Config vs Best Parametric Model (ER/PA/Hollywood)")
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "config_vs_best_parametric.png")
    plt.savefig(path, dpi=150)
    plt.close()

    # 3. Graph properties overview (scatter: nodes vs edges, colored by source)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 3a. Size scatter
    ax = axes[0]
    for source, color in [("SZIP", "#e74c3c"), ("REC", "#3498db"), ("TU", "#2ecc71")]:
        mask = df["source"] == source
        ax.scatter(df.loc[mask, "nodes"], df.loc[mask, "edges"],
                  c=color, s=60, alpha=0.7, label=source, edgecolors='white', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Edges")
    ax.set_title("Graph Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3b. Degree distribution stats
    ax = axes[1]
    for source, color in [("SZIP", "#e74c3c"), ("REC", "#3498db"), ("TU", "#2ecc71")]:
        mask = df["source"] == source
        ax.scatter(df.loc[mask, "deg_mean"], df.loc[mask, "deg_std"],
                  c=color, s=60, alpha=0.7, label=source, edgecolors='white', linewidth=0.5)
    ax.set_xlabel("Mean Degree")
    ax.set_ylabel("Degree Std Dev")
    ax.set_title("Degree Distribution Shape")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3c. Clustering vs density
    ax = axes[2]
    valid = df.dropna(subset=["clustering_coeff"])
    for source, color in [("SZIP", "#e74c3c"), ("REC", "#3498db"), ("TU", "#2ecc71")]:
        mask = valid["source"] == source
        ax.scatter(valid.loc[mask, "density"], valid.loc[mask, "clustering_coeff"],
                  c=color, s=60, alpha=0.7, label=source, edgecolors='white', linewidth=0.5)
    ax.set_xlabel("Density")
    ax.set_ylabel("Clustering Coefficient")
    ax.set_title("Clustering vs Density")
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "graph_properties_overview.png")
    plt.savefig(path, dpi=150)
    plt.close()

    # 4. Power-law alpha distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_pl = df.dropna(subset=["powerlaw_alpha"])
    colors = ['#2ecc71' if ht == 'Yes' else ('#e74c3c' if ht == 'No' else '#95a5a6')
              for ht in valid_pl["heavy_tailed"]]
    ax.barh(valid_pl["dataset"], valid_pl["powerlaw_alpha"], color=colors, alpha=0.85)
    ax.set_xlabel("Power-law Alpha Exponent")
    ax.set_title("Power-law Fit: Alpha Exponents (green=heavy-tailed, red=not)")
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5, label='alpha=2')
    ax.axvline(x=3, color='gray', linestyle=':', alpha=0.5, label='alpha=3')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "powerlaw_alpha.png")
    plt.savefig(path, dpi=150)
    plt.close()


def generate_html(df):
    """Generate self-contained HTML report."""
    make_comparison_plots(df)

    # Collect all plot images
    plot_files = sorted(Path(PLOTS_DIR).glob("*.png"))

    # Group plots by type
    mo_szip = sorted([p for p in plot_files if "model_overlay_SZIP" in p.name])
    mo_rec = sorted([p for p in plot_files if "model_overlay_REC" in p.name])
    mo_tu = sorted([p for p in plot_files if "model_overlay_TU" in p.name])
    dd_szip = [p for p in plot_files if "degree_dist_SZIP" in p.name]
    dd_rec = [p for p in plot_files if "degree_dist_REC" in p.name]
    dd_tu = [p for p in plot_files if "degree_dist_TU" in p.name]
    overview_plots = [p for p in plot_files if "degree_dist" not in p.name and "model_overlay" not in p.name]

    def img_gallery(paths, cols=3):
        html = '<div style="display:grid;grid-template-columns:' + ' '.join(['1fr']*cols) + ';gap:10px;">'
        for p in paths:
            b64 = img_to_base64(str(p))
            name = p.stem.replace("degree_dist_", "").replace("_", " / ")
            html += f'<div style="text-align:center;"><img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,0.1);"/><br><small>{name}</small></div>'
        html += '</div>'
        return html

    # Build the stats table
    model_df = df.dropna(subset=["er_bits_per_edge"])
    stats_df = df.copy()
    format_cols = {
        "density": ".8f", "deg_mean": ".2f", "deg_median": ".1f", "deg_std": ".2f",
        "deg_skew": ".2f", "powerlaw_alpha": ".3f", "powerlaw_vs_lognormal_p": ".4f",
        "clustering_coeff": ".4f", "er_bits_per_edge": ".2f",
        "config_bits_per_edge": ".2f", "hollywood_bits_per_edge": ".2f",
        "hollywood_alpha": ".4f", "hollywood_theta": ".1f",
    }

    def fmt_table(dataframe, highlight_model=False):
        rows = []
        for _, row in dataframe.iterrows():
            cells = []
            for col in dataframe.columns:
                val = row[col]
                if pd.isna(val):
                    cells.append('<td style="color:#999;">--</td>')
                elif col in format_cols and isinstance(val, (int, float)):
                    formatted = f"{val:{format_cols[col]}}"
                    style = ""
                    if highlight_model and col in ("er_bits_per_edge", "pa_bits_per_edge", "config_bits_per_edge", "hollywood_bits_per_edge"):
                        er = row.get("er_bits_per_edge", float('inf'))
                        pa = row.get("pa_bits_per_edge", float('inf'))
                        cfg = row.get("config_bits_per_edge", float('inf'))
                        hw = row.get("hollywood_bits_per_edge", float('inf'))
                        vals = [v for v in [er, pa, cfg, hw] if not pd.isna(v)]
                        if vals:
                            best = min(vals)
                            if val == best:
                                style = ' style="background:#d4edda;font-weight:bold;"'
                    cells.append(f'<td{style}>{formatted}</td>')
                elif isinstance(val, (int, np.integer)):
                    cells.append(f'<td>{val:,}</td>')
                else:
                    cells.append(f'<td>{val}</td>')
            rows.append('<tr>' + ''.join(cells) + '</tr>')
        header = '<tr>' + ''.join(f'<th>{c}</th>' for c in dataframe.columns) + '</tr>'
        return f'<table class="data-table"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>'

    # Display columns for the main stats table
    stats_cols = ["dataset", "source", "num_graphs_in_dataset", "nodes", "edges", "density",
                  "deg_mean", "deg_median", "deg_max", "deg_std", "deg_skew",
                  "powerlaw_alpha", "heavy_tailed", "clustering_coeff", "n_components"]
    model_cols = ["dataset", "source", "er_bits_per_edge", "pa_bits_per_edge",
                  "config_bits_per_edge", "hollywood_bits_per_edge"]

    stats_table = fmt_table(stats_df[[c for c in stats_cols if c in stats_df.columns]])
    model_table = fmt_table(stats_df[[c for c in model_cols if c in stats_df.columns]], highlight_model=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Graph Dataset Analysis - Practical Shuffle Coding Edge Models</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; color: #2c3e50; line-height: 1.6; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
    h1 {{ font-size: 1.8em; margin: 30px 0 10px; color: #1a1a2e; border-bottom: 3px solid #3498db; padding-bottom: 8px; }}
    h2 {{ font-size: 1.4em; margin: 25px 0 10px; color: #2c3e50; }}
    h3 {{ font-size: 1.1em; margin: 20px 0 8px; color: #34495e; }}
    p, li {{ font-size: 0.95em; margin-bottom: 8px; }}
    .hero {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; }}
    .hero h1 {{ color: white; border-bottom-color: #3498db; }}
    .hero p {{ color: #b8c6db; font-size: 1em; }}
    .card {{ background: white; border-radius: 10px; padding: 24px; margin: 15px 0; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; overflow-x: auto; display: block; }}
    .data-table th {{ background: #2c3e50; color: white; padding: 8px 10px; text-align: left; position: sticky; top: 0; white-space: nowrap; }}
    .data-table td {{ padding: 6px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }}
    .data-table tr:hover {{ background: #f0f7ff; }}
    .data-table tr:nth-child(even) {{ background: #fafafa; }}
    .data-table tr:nth-child(even):hover {{ background: #f0f7ff; }}
    .finding {{ background: #eaf4ff; border-left: 4px solid #3498db; padding: 15px 20px; margin: 15px 0; border-radius: 0 6px 6px 0; }}
    .finding.positive {{ background: #eafff0; border-left-color: #2ecc71; }}
    .finding.negative {{ background: #fff0f0; border-left-color: #e74c3c; }}
    .finding.neutral {{ background: #fff8ea; border-left-color: #f39c12; }}
    .img-full {{ width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
    @media (max-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }}
    .tag-szip {{ background: #fde8e8; color: #c0392b; }}
    .tag-rec {{ background: #e8f0fd; color: #2980b9; }}
    .tag-tu {{ background: #e8fde8; color: #27ae60; }}
    ul {{ padding-left: 20px; }}
    .toc {{ background: white; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }}
    .toc a {{ color: #3498db; text-decoration: none; }}
    .toc a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">

<div class="hero">
    <h1>Graph Dataset Analysis</h1>
    <p>Practical Shuffle Coding Edge Models &mdash; Kunze et al., NeurIPS 2024</p>
    <p style="margin-top:10px;font-size:0.9em;color:#8899aa;">
        Analyzing {len(df)} graph datasets across 3 sources (SZIP, REC, TU).
        Comparing Erdos-Renyi, Preferential Attachment, Configuration Model, and Hollywood (Pitman-Yor) edge models.
    </p>
</div>

<div class="toc">
    <h3>Contents</h3>
    <ul>
        <li><a href="#models-explained">The Four Edge Models (explainer)</a></li>
        <li><a href="#overview">Overview & Comparison Plots</a></li>
        <li><a href="#stats">Graph Statistics Table</a></li>
        <li><a href="#models">Model Comparison (bits per edge)</a></li>
        <li><a href="#degree">Degree Distributions: Empirical vs Models</a></li>
        <li><a href="#findings">Key Findings</a></li>
    </ul>
</div>

<h1 id="models-explained">The Four Edge Models</h1>

<div class="card">
    <p style="margin-bottom:15px;">We compare four models of increasing expressiveness for how edges form in a graph.
    Each model assigns a probability to any given graph &mdash; higher probability = fewer bits needed to encode it.
    The tradeoff: more expressive models fit the data better, but cost more bits to describe the model itself.</p>
    <img src="data:image/png;base64,{img_to_base64(os.path.join(PLOTS_DIR, 'explainer_expressiveness.png'))}" class="img-full" />
</div>

<div class="card">
    <h3 style="margin-bottom:12px;">How each model generates a graph (numbers = node degree)</h3>
    <img src="data:image/png;base64,{img_to_base64(os.path.join(PLOTS_DIR, 'explainer_mechanisms.png'))}" class="img-full" />
    <div style="margin-top:20px; display:grid; grid-template-columns:1fr 1fr; gap:15px;">
        <div style="background:#fde8e8;padding:12px;border-radius:8px;">
            <b style="color:#c0392b;">Erdos-Renyi (ER)</b><br>
            <code>P(edge) = p</code> for all pairs, independently.<br>
            Predicts <b>Poisson</b> degrees &mdash; symmetric bell curve, no hubs.
        </div>
        <div style="background:#fef0e0;padding:12px;border-radius:8px;">
            <b style="color:#e67e22;">Preferential Attachment (PA)</b><br>
            New nodes connect to existing ones with <code>P &prop; degree</code>.<br>
            Predicts <b>P(k) ~ k<sup>-3</sup></b> &mdash; power law, but slope is always -3.
        </div>
        <div style="background:#e8f0fd;padding:12px;border-radius:8px;">
            <b style="color:#2980b9;">Hollywood / Pitman-Yor (PY)</b><br>
            Edge endpoints assigned via Chinese Restaurant Process:<br>
            <code>P(existing node) &prop; degree - &alpha;</code>, &ensp; <code>P(new node) &prop; &theta; + K&alpha;</code><br>
            Predicts <b>P(k) ~ k<sup>-(1+&alpha;)</sup></b> &mdash; power law with tunable slope.
        </div>
        <div style="background:#e8fde8;padding:12px;border-radius:8px;">
            <b style="color:#27ae60;">Configuration Model</b><br>
            Takes the <em>exact</em> degree sequence as input, then randomly wires stubs.<br>
            <b>Does not predict</b> a degree distribution &mdash; it matches whatever you give it.
            The cost is encoding the full degree sequence.
        </div>
    </div>
</div>

<div class="card">
    <h3 style="margin-bottom:12px;">What each model predicts for the degree distribution</h3>
    <img src="data:image/png;base64,{img_to_base64(os.path.join(PLOTS_DIR, 'explainer_degree_dists.png'))}" class="img-full" />
    <p style="margin-top:10px;color:#555;font-size:0.9em;">
        <b>Left (linear)</b>: ER&rsquo;s Poisson is a symmetric hump. PA and Hollywood are heavily skewed toward low degrees with long tails.<br>
        <b>Right (log-log)</b>: Power laws appear as straight lines. ER curves downward (exponential tail cutoff).
        PA is always slope -3. Hollywood&rsquo;s slope is tunable via &alpha;.
        Config (green dots) can match any shape &mdash; it&rsquo;s the data itself.
    </p>
</div>

<h1 id="overview">Overview & Comparison Plots</h1>
<div class="card">
    <img src="data:image/png;base64,{img_to_base64(os.path.join(PLOTS_DIR, 'model_comparison_bars.png'))}" class="img-full" />
</div>
<div class="card">
    <p style="color:#555;font-size:0.9em;margin-bottom:8px;">Hollywood is so much worse it distorts the scale. Here&rsquo;s the same chart zoomed to just the three competitive models:</p>
    <img src="data:image/png;base64,{img_to_base64(os.path.join(PLOTS_DIR, 'model_comparison_zoomed.png'))}" class="img-full" />
</div>
<div class="grid2">
    <div class="card">
        <img src="data:image/png;base64,{img_to_base64(os.path.join(PLOTS_DIR, 'config_vs_best_parametric.png'))}" class="img-full" />
    </div>
    <div class="card">
        <img src="data:image/png;base64,{img_to_base64(os.path.join(PLOTS_DIR, 'powerlaw_alpha.png'))}" class="img-full" />
    </div>
</div>
<div class="card">
    <img src="data:image/png;base64,{img_to_base64(os.path.join(PLOTS_DIR, 'graph_properties_overview.png'))}" class="img-full" />
</div>

<h1 id="stats">Graph Statistics</h1>
<div class="card" style="overflow-x:auto;">
    {stats_table}
</div>

<h1 id="models">Model Comparison (bits per edge)</h1>
<div class="card" style="overflow-x:auto;">
    <p style="margin-bottom:12px;">Green highlights = best model for that dataset. Lower bits/edge = better compression.</p>
    {model_table}
</div>

<h1 id="degree">Degree Distributions: Empirical vs Model Predictions</h1>
<p style="margin:10px 0 5px;color:#555;">Each plot shows the empirical degree distribution (black/green dots) overlaid with predictions from four models:<br>
<b style="color:#e74c3c;">ER</b> (Binomial/Poisson &mdash; no heavy tail) &bull;
<b style="color:#e67e22;">PA</b> (Preferential Attachment, fixed k<sup>-3</sup> power law) &bull;
<b style="color:#3498db;">Hollywood</b> (Pitman-Yor, tunable k<sup>-(1+&alpha;)</sup>) &bull;
<b style="color:#2ecc71;">Config</b> (= empirical by construction).
Left panel: empirical + overlays. Right panel: model predictions compared.</p>

<h2><span class="tag tag-szip">SZIP</span> Single Large Graphs</h2>
{''.join(f'<div class="card"><img src="data:image/png;base64,{img_to_base64(str(p))}" class="img-full" /></div>' for p in mo_szip)}

<h2><span class="tag tag-rec">REC</span> Large Social & Non-Social Networks</h2>
{''.join(f'<div class="card"><img src="data:image/png;base64,{img_to_base64(str(p))}" class="img-full" /></div>' for p in mo_rec)}
{f'<div class="card"><p style="color:#888;">Large REC graphs (Foursquare, Digg, YouTube, Skitter) &mdash; degree distribution only (too large for PY simulation):</p>{img_gallery([p for p in dd_rec if not any(x in p.name for x in ["DBLP","Gowalla"])], cols=2)}</div>' if len([p for p in dd_rec if not any(x in p.name for x in ["DBLP","Gowalla"])]) > 0 else ''}

<h2><span class="tag tag-tu">TU</span> Graph Collections (aggregate degree distribution)</h2>
{''.join(f'<div class="card"><img src="data:image/png;base64,{img_to_base64(str(p))}" class="img-full" /></div>' for p in mo_tu)}

<h1 id="findings">Key Findings</h1>

<div class="finding positive">
    <h3>Configuration Model wins in 11/12 datasets</h3>
    <p>The configuration model consistently compresses better than ER by 0.5&ndash;5 bits/edge.
    It captures degree heterogeneity &mdash; real graphs have highly non-uniform degree distributions
    (high variance, skew, often heavy tails), and encoding the degree sequence explicitly pays off.</p>
</div>

<div class="finding negative">
    <h3>Hollywood (Pitman-Yor) model is too restrictive</h3>
    <p>With only 2 parameters (alpha, theta), the PY process cannot capture the specific shape of empirical
    degree distributions. The cost of a poor parametric fit to the degree sequence far outweighs the savings
    from having fewer parameters. For many datasets, the optimizer finds alpha&nbsp;&asymp;&nbsp;0 and
    very large theta, effectively defaulting to a Dirichlet process &mdash; a poor model for heterogeneous degrees.</p>
</div>

<div class="finding neutral">
    <h3>Preferential Attachment (PA): right idea, wrong exponent</h3>
    <p>PA predicts a fixed k<sup>-3</sup> power law. The overlay plots show it captures the right
    <em>shape</em> for heavy-tailed networks (straight line on log-log) but the slope is often wrong &mdash;
    real networks have exponents ranging from ~1.7 to ~3.0. PA also can&rsquo;t model the low-degree
    bump or high-degree cutoff seen in most empirical distributions. It sits between ER (too narrow)
    and Hollywood (tunable but still too rigid) in expressiveness.</p>
</div>

<div class="finding neutral">
    <h3>Most graphs are NOT convincingly power-law</h3>
    <p>When tested against the lognormal alternative using the Clauset et al. methodology, most datasets
    show low p-values (power-law is not significantly better than lognormal). Only 'geom', 'as' (SZIP),
    'Gowalla', and 'Skitter' (REC) show evidence of genuinely heavy-tailed degree distributions.</p>
</div>

<div class="finding">
    <h3>Dataset characteristics summary</h3>
    <ul>
        <li><strong>SZIP</strong> (6 graphs): Medium-sized single graphs (332&ndash;25K nodes). Moderate to sparse. Some heavy-tailed.</li>
        <li><strong>REC</strong> (6 graphs): Very large single graphs (196K&ndash;3.2M nodes). Extremely sparse. Social networks show extreme degree heterogeneity (max degree up to 106K).</li>
        <li><strong>TU</strong> (4 collections): Collections of small graphs. MUTAG (molecules): narrow bounded degrees. IMDB-BINARY: dense ego-networks. PROTEINS: biological. reddit_threads: star-like trees.</li>
    </ul>
</div>

<div class="card" style="text-align:center;color:#888;font-size:0.85em;margin-top:40px;">
    <p>Generated from the <a href="https://github.com/juliuskunze/shuffle-coding" style="color:#3498db;">shuffle-coding</a> datasets.
    Analysis code: <code>analyze_graphs.py</code> | Report: <code>generate_report.py</code></p>
</div>

</div>
</body>
</html>"""

    report_path = os.path.join(OUTPUT_DIR, "report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"Report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    generate_html(df)
