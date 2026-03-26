"""
Edge Models for Graph Compression — Demo
=========================================
Compares 5 models on real graph datasets:
  ER, PA, Configuration, Decision Table, Decision Table + Common Neighbors

Uses the Rust backend (edge_models_rs) for speed.
Run: python demo.py
"""

import edge_models_rs as em
import time

DATA = "/Users/dennisloevlie/graphs/practical_shuffle_coding_edge_models/data"

DATASETS = [
    # (name, description)
    ("USAir97",  "US airline routes (small)"),
    ("Erdos",    "Math co-authorship (medium)"),
    ("DBLP",     "CS co-authorship (large)"),
    ("Gowalla",  "Social check-ins (large)"),
    ("YouTube",  "Social network (very large)"),
    ("Skitter",  "Internet topology (very large)"),
]

# ── Run all models ──────────────────────────────────────────────

results = []

for name, desc in DATASETS:
    print(f"Loading {name}...", end=" ", flush=True)
    g = em.load_dataset(f"{DATA}/{name}", name)[0]
    n, m = g.num_nodes(), g.num_edges()
    degs = g.degrees()
    print(f"{n:,} nodes, {m:,} edges")

    # Base models
    er  = em.er_bpe(n, m, g.density())
    pa  = em.pa_bpe(degs, n, m)
    cfg = em.config_bpe(degs, n, m)

    # Decision table (binary features, exact)
    dt = em.run_decision_table_exact(g, base="er")

    # Decision table + common neighbors (the good one)
    t0 = time.perf_counter()
    cn = em.run_decision_table_cn(g, base="er", num_deg_bins=12)
    cn_time = time.perf_counter() - t0

    results.append({
        "name": name, "desc": desc, "n": n, "m": m,
        "er": er, "pa": pa, "config": cfg,
        "dt": dt["bits_per_edge"],
        "cn": cn["bits_per_edge"],
        "cn_table": cn["table_entries"],
        "cn_time": cn_time,
    })

# ── Print results ───────────────────────────────────────────────

print()
print("=" * 90)
print("RESULTS: bits per edge (lower = better compression)")
print("=" * 90)
print()
print(f"{'Dataset':<12} {'Nodes':>10} {'Edges':>10}   {'ER':>6} {'PA':>6} {'Cfg':>6} {'DT':>6} {'DT+CN':>6}  {'Best':>6}")
print("-" * 90)

for r in results:
    models = {"ER": r["er"], "PA": r["pa"], "Cfg": r["config"],
              "DT": r["dt"], "DT+CN": r["cn"]}
    best_name = min(models, key=models.get)
    best_val = models[best_name]

    print(f"{r['name']:<12} {r['n']:>10,} {r['m']:>10,}  ",
          f"{r['er']:6.2f} {r['pa']:6.2f} {r['config']:6.2f}",
          f"{r['dt']:6.2f} {r['cn']:6.2f}  {best_name:>6}")

# ── Summary ─────────────────────────────────────────────────────

print()
print("=" * 90)
print("WHAT EACH MODEL DOES")
print("=" * 90)
print("""
  ER      Each edge exists with probability p = density.
          1 parameter. No structure at all.

  PA      Each node's degree drawn from P(k) ~ k^-3.
          1 parameter. Captures heavy tails, but fixed slope.

  Config  Encode the exact degree sequence, then random wiring.
          n parameters. Always fits degrees perfectly.

  DT      Decision table correction over ER, using binary
          neighbor-degree features. Learns per-feature-pair
          edge probabilities. ~100-300 table entries.

  DT+CN   Decision table using (degree_bin, degree_bin,
          common_neighbors_bin). Common neighbors = how many
          friends two nodes share. This captures community
          structure that degree-only models completely miss.
""")

print("KEY FINDING:")
print()
large = [r for r in results if r["m"] > 100000]
if large:
    avg_cn = sum(r["cn"] for r in large) / len(large)
    avg_cfg = sum(r["config"] for r in large) / len(large)
    pct = (1 - avg_cn / avg_cfg) * 100
    print(f"  On large graphs, DT+CN uses {pct:.0f}% fewer bits than Config")
    print(f"  with only ~250 table entries vs Config's n={large[0]['n']:,}+ parameters.")
    print()
    print(f"  The secret: common neighbors captures local community structure.")
    print(f"  Two nodes in the same cluster share many neighbors → likely connected.")
    print(f"  Two nodes in different clusters share zero → likely not.")
    print(f"  Degree-only models (ER, PA, Config) can't see this.")
