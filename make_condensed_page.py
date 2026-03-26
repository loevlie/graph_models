#!/usr/bin/env python3
"""Generate a condensed, clean index.html for GitHub Pages."""

import os
import base64
import pandas as pd
from pathlib import Path

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def b64_img(name):
    path = os.path.join(PLOTS_DIR, name)
    if not os.path.exists(path):
        return ""
    return f'<img src="data:image/png;base64,{img_to_base64(path)}" class="img-full" />'

def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "graph_analysis_results.csv"))

    # Build the compact model comparison table
    model_df = df.dropna(subset=["er_bits_per_edge", "pa_bits_per_edge", "config_bits_per_edge", "hollywood_bits_per_edge"])
    table_rows = ""
    for _, row in model_df.iterrows():
        er = row["er_bits_per_edge"]
        pa = row["pa_bits_per_edge"]
        cfg = row["config_bits_per_edge"]
        hw = row["hollywood_bits_per_edge"]
        best = min(er, pa, cfg, hw)
        def cell(v):
            s = f"{v:.1f}"
            if v == best:
                return f'<td class="best">{s}</td>'
            return f'<td>{s}</td>'
        table_rows += f'<tr><td><b>{row["dataset"]}</b></td><td>{row["source"]}</td><td>{int(row["nodes"]):,}</td><td>{int(row["edges"]):,}</td>{cell(er)}{cell(pa)}{cell(cfg)}{cell(hw)}</tr>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Edge Models for Graph Compression</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #fafafa; color: #333; line-height: 1.65; }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 24px 20px; }}
  h1 {{ font-size: 1.6em; margin: 0 0 4px; color: #1a1a2e; }}
  h2 {{ font-size: 1.2em; margin: 32px 0 12px; color: #2c3e50;
       border-bottom: 2px solid #3498db; padding-bottom: 4px; }}
  p {{ margin-bottom: 12px; font-size: 0.93em; }}
  .subtitle {{ color: #777; font-size: 0.9em; margin-bottom: 20px; }}
  .card {{ background: white; border-radius: 8px; padding: 20px;
           margin: 14px 0; box-shadow: 0 1px 6px rgba(0,0,0,0.06); }}
  .img-full {{ width: 100%; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
  th {{ background: #2c3e50; color: white; padding: 7px 10px; text-align: left; white-space: nowrap; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }}
  tr:hover {{ background: #f5f9ff; }}
  .best {{ background: #d4edda; font-weight: bold; }}
  .model-box {{ padding: 12px 16px; border-radius: 8px; margin-bottom: 10px; }}
  .model-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  @media (max-width: 700px) {{ .model-grid {{ grid-template-columns: 1fr; }} }}
  code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
  .arrow {{ display: flex; align-items: center; justify-content: center; gap: 8px;
            margin: 16px 0; flex-wrap: wrap; }}
  .arrow-box {{ padding: 8px 14px; border-radius: 8px; font-size: 0.85em;
                font-weight: 600; text-align: center; min-width: 120px; }}
  .arrow-sep {{ font-size: 1.2em; color: #999; }}
  a {{ color: #3498db; }}
</style>
</head>
<body>
<div class="wrap">

<h1>Edge Models for Graph Compression</h1>
<p class="subtitle">Comparing how well different random graph models capture real-world edge structure.<br>
Data from <a href="https://github.com/juliuskunze/shuffle-coding">Practical Shuffle Coding</a> (Kunze et al., NeurIPS 2024).</p>

<h2>The Four Models</h2>
<p>Each model assigns a probability to a graph. Higher probability = fewer bits to compress it.
The tradeoff: more parameters fit data better, but cost more bits to describe the model itself.</p>

<div class="arrow">
  <div class="arrow-box" style="background:#fde8e8;color:#c0392b;">ER<br><small>1 param</small></div>
  <span class="arrow-sep">&rarr;</span>
  <div class="arrow-box" style="background:#fef0e0;color:#e67e22;">PA<br><small>1 param</small></div>
  <span class="arrow-sep">&rarr;</span>
  <div class="arrow-box" style="background:#e8f0fd;color:#2980b9;">Hollywood<br><small>2 params</small></div>
  <span class="arrow-sep">&rarr;</span>
  <div class="arrow-box" style="background:#e8fde8;color:#27ae60;">Config<br><small>n params</small></div>
</div>

<div class="model-grid">
  <div class="model-box" style="background:#fde8e8;">
    <b style="color:#c0392b;">Erdos-Renyi</b> &mdash;
    Each possible edge exists with probability <code>p</code>. Predicts Poisson degree distribution. No hubs, no heavy tails.
  </div>
  <div class="model-box" style="background:#fef0e0;">
    <b style="color:#e67e22;">Preferential Attachment</b> &mdash;
    New nodes link to popular nodes: <code>P &prop; degree</code> ("rich get richer").
    Always predicts <code>P(k) ~ k<sup>-3</sup></code>.
  </div>
  <div class="model-box" style="background:#e8f0fd;">
    <b style="color:#2980b9;">Hollywood (Pitman-Yor)</b> &mdash;
    Edge endpoints are assigned to nodes sequentially: each goes to an existing node
    with probability proportional to <code>degree - &alpha;</code>, or to a new node with probability
    proportional to <code>&theta; + K&alpha;</code>.
    Predicts <code>P(k) ~ k<sup>-(1+&alpha;)</sup></code> &mdash; tunable power law.
  </div>
  <div class="model-box" style="background:#e8fde8;">
    <b style="color:#27ae60;">Configuration Model</b> &mdash;
    Takes the <em>exact</em> degree sequence as input, then randomly wires stubs.
    Degrees are not predicted &mdash; they are part of the encoding cost.
  </div>
</div>

<h2>What Each Model Predicts</h2>
<div class="card">
  {b64_img('explainer_degree_dists.png')}
  <p style="margin-top:8px;color:#555;font-size:0.88em;">
    On log-log scale, power laws are straight lines. ER curves downward (no heavy tail).
    PA is always slope -3. Hollywood's slope is tunable. Config matches the data exactly.
  </p>
</div>

<h2>Three Example Graphs</h2>
<p>We pick three diverse datasets: a molecule collection (MUTAG), an airline network (USAir97), and a social network (Gowalla).</p>

<div class="card">
  <h3 style="margin-bottom:8px;">USAir97 &mdash; US airline routes (332 nodes, 2,126 edges)</h3>
  {b64_img('model_overlay_SZIP_USAir97.png')}
  <p style="margin-top:6px;font-size:0.88em;color:#555;">Heavy-tailed degree distribution. PA wins here (4.50 vs Config's 4.61 bits/edge).</p>
</div>

<div class="card">
  <h3 style="margin-bottom:8px;">MUTAG &mdash; molecular graphs (188 molecules, avg 18 nodes)</h3>
  {b64_img('model_overlay_TU_MUTAG.png')}
  <p style="margin-top:6px;font-size:0.88em;color:#555;">Narrow, bounded degrees (max 4). Config wins (5.01 bits/edge).</p>
</div>

<div class="card">
  <h3 style="margin-bottom:8px;">Gowalla &mdash; social network (196K nodes, 950K edges)</h3>
  {b64_img('model_overlay_REC_Gowalla.png')}
  <p style="margin-top:6px;font-size:0.88em;color:#555;">Degrees span 4 orders of magnitude. Config wins (12.78 bits/edge).</p>
</div>

<h2>Full Comparison (bits per edge)</h2>
<p>Lower is better. <span class="best" style="padding:1px 6px;">Green</span> = best model for that dataset.</p>
<div class="card" style="overflow-x:auto;">
  <table>
    <thead>
      <tr><th>Dataset</th><th>Source</th><th>Nodes</th><th>Edges</th><th>ER</th><th>PA</th><th>Config</th><th>Hollywood</th></tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</div>

<h2>How These Models Work With Shuffle Coding</h2>
<div class="card">
  <p>All four models do the same thing: define a probability <code>P(G)</code> for any graph G.
  Shuffle coding just needs that probability &mdash; it doesn&rsquo;t care how you compute it.</p>
  <p>The encoder and decoder must agree on the model beforehand. The difference is
  what they need to agree on:</p>
  <table style="margin:10px 0;">
    <tr><th>Model</th><th>What encoder sends first</th><th>Then encodes</th></tr>
    <tr><td style="color:#c0392b;font-weight:bold;">ER</td><td>1 number: p</td><td>Full adjacency (each edge slot independently)</td></tr>
    <tr><td style="color:#e67e22;font-weight:bold;">PA</td><td>1 number: m₀</td><td>Each node&rsquo;s degree using P(k)~k⁻³, then wiring</td></tr>
    <tr><td style="color:#2980b9;font-weight:bold;">Hollywood</td><td>2 numbers: &alpha;, &theta;</td><td>Degree partition via Pitman-Yor, then wiring</td></tr>
    <tr><td style="color:#27ae60;font-weight:bold;">Config</td><td>n numbers: d₁...dₙ</td><td>Wiring only (degrees already sent)</td></tr>
  </table>
  <p>All four are valid codes. Config is &ldquo;generative&rdquo; too &mdash; given the degree sequence,
  you randomly pair stubs to sample a graph. The only difference is that Config&rsquo;s
  &ldquo;model description&rdquo; is n numbers instead of 1&ndash;2. Those n numbers are part of the
  bitstream, and their cost is included in the bits-per-edge numbers above.</p>
</div>

<h2>Key Takeaways</h2>
<div class="card">
  <p><b>No single model is best for all graphs.</b> Config has the lowest bits-per-edge
  in most cases, but uses n parameters. PA wins when degrees happen to follow k<sup>-3</sup>.
  Hollywood is theoretically appealing (tunable slope, only 2 params) but pays a steep
  price for encoding the degree sequence as a joint partition rather than independent draws.</p>
  <p><b>The tradeoff is model description cost vs data fit.</b> More parameters = better fit
  to the data = fewer bits for the graph itself, but more bits to describe the model.
  The best model minimizes the total.</p>
</div>

<p style="text-align:center;color:#aaa;font-size:0.82em;margin-top:30px;">
  <a href="report.html">Full detailed report</a> &bull;
  <a href="https://github.com/loevlie/graph_models">Source code</a> &bull;
  Data from <a href="https://github.com/juliuskunze/shuffle-coding">shuffle-coding</a>
</p>

</div>
</body>
</html>"""

    path = os.path.join(OUTPUT_DIR, "index.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
