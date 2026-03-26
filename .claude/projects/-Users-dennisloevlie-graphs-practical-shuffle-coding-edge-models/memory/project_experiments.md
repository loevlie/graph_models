---
name: experiment_log
description: Running log of decision table model experiments, results, and next steps for the edge models project
type: project
---

Experiment log for edge models / decision table project.

## Experiment 1: Base models comparison (2026-03-25)
Compared ER, PA, Configuration, Hollywood on SZIP/REC/TU datasets.
- Config wins most datasets (encodes exact degree seq + wiring)
- PA wins USAir97 (degrees happen to follow k⁻³)
- Hollywood performs poorly (joint partition probability too expensive)
- Results in graph_analysis_results.csv

## Experiment 2: Decision table with ER base — Python (2026-03-25)
Binary features: neighbor degree thresholds [1,2,4,8,...] → binary vector per node.
Pair key = (feat_u, feat_v). Sampling-based.
- USAir97: DT+ER=5.32 (beats ER 6.12, loses to PA 4.50)
- DBLP: DT+ER=12.94 via sampling (appeared to beat Config 15.82) ← **sampling was misleading**
- Table learns real structure (hub-hub pairs 4x more likely than avg in USAir97)

## Experiment 3: Decision table with PA/Config base — Python (2026-03-25)
- DT+PA on USAir97: 5.32 (table hurts by -0.15, PA base already captures degree structure)
- DT+Config: also hurts, table overhead not justified
- Conclusion: table only helps on large graphs where cost amortizes

## Experiment 4: Rust implementation (2026-03-26)
Built edge_models_rs with PyO3 bindings. 70-82x faster than Python.
- DBLP (1M edges): 0.26s for DT, 0.04s for exact
- YouTube (9.4M edges): 1.04s sampling, 0.56s exact
- Skitter (11M edges): 1.05s sampling, 0.45s exact

## Experiment 5: Exact vs sampling results (2026-03-26)
**Sampling was giving false positives.** Exact results:
```
Dataset       ER      PA    Config   DT+ER(exact)  DT+PA(exact)
USAir97     6.12    4.50     4.61       5.32          5.32
DBLP       16.99   16.18    15.82      15.79         15.79  ← DT barely wins over Config
Gowalla    15.75   12.97    12.78      14.49         14.49
Skitter    18.43   14.39    14.16      17.05         17.05
YouTube    20.52   14.82    14.76      18.91         18.91
```
- DT only wins on DBLP, by 0.03 bpe
- Binary neighbor-degree features too coarse to beat Config/PA
- Sampling bias inflated DT results on large graphs

## Experiment 6: Common neighbors + degree bins (TODO)
Adding richer features: (deg_bin(u), deg_bin(v), common_neighbor_bin(u,v)).
- 3D lookup table, ~500 entries
- Common neighbors is strongest link prediction feature
- Compute CN for edges via CSR binary search: O(m * d_avg)
- Estimate CN distribution for non-edges from edge stats

**Why:** degree bin captures what Config has; common neighbors captures LOCAL
structure that no current model uses (community/cluster membership proxy).

**How to apply:** Implemented in `decision_table.rs` as `decision_table_exact_cn`.
```

## Next steps
- Common neighbors feature (Experiment 6)
- Batch mode: share one table across a graph collection (advisor suggestion)
- Richer per-node features: clustering coefficient, etc.
- Compare against a small gradient-boosted tree model
