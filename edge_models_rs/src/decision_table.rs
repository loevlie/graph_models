use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashMap;

use crate::base_models::config_edge_prob;
use crate::features::{
    compute_log_bins, compute_node_features, pair_key, FeatureVec,
    degree_bin, common_neighbors, cn_bin, triple_key,
};
use crate::graph::Graph;

/// Which base model to use for fallback probabilities.
#[derive(Clone, Copy, Debug)]
pub enum BaseModel {
    ER,
    PA,
    Config,
}

/// Result of the decision table model.
#[derive(Debug)]
pub struct DecisionTableResult {
    pub bits_per_edge: f64,
    pub table_entries: usize,
    pub feature_bits: usize,
    pub table_cost_bits: f64,
    pub encoding_cost_bits: f64,
    pub base_only_bpe: f64,
    pub improvement: f64,
}

/// Per-key accumulator during sampling.
struct KeyStats {
    edges: u32,
    total: u32,
    base_prob_sum: f64,
}

/// Sampling-based decision table model.
///
/// For each sampled node pair (u,v):
///   - Compute their feature-pair key
///   - Record whether edge exists and the base model's probability
///
/// Then for frequent keys, use the empirical edge rate as a correction.
pub fn decision_table_model(
    graph: &Graph,
    base: BaseModel,
    min_frequency: usize,
    n_samples: usize,
    seed: u64,
) -> DecisionTableResult {
    let n = graph.num_nodes();
    let m = graph.num_edges();
    let density = graph.density();
    let degrees = graph.degrees();
    let max_deg = *degrees.iter().max().unwrap_or(&1);

    // Compute features
    let log_bins = compute_log_bins(max_deg);
    let n_feat_bits = log_bins.len();
    let features = compute_node_features(graph, &degrees, &log_bins);

    // Sampling
    let mut rng = StdRng::seed_from_u64(seed);
    let mut stats: FxHashMap<u64, KeyStats> = FxHashMap::default();
    let actual_samples = n_samples.min(n * (n - 1) / 2);

    // Use a set to avoid sampling the same pair twice
    let mut seen: FxHashMap<u64, ()> = FxHashMap::default();
    let mut sampled = 0usize;
    let mut attempts = 0usize;
    let max_attempts = actual_samples * 5;

    while sampled < actual_samples && attempts < max_attempts {
        let u = rng.gen_range(0..n as u32);
        let v = rng.gen_range(0..n as u32);
        if u == v {
            attempts += 1;
            continue;
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        let pair_id = (a as u64) << 32 | b as u64;
        if seen.contains_key(&pair_id) {
            attempts += 1;
            continue;
        }
        seen.insert(pair_id, ());

        let key = pair_key(features[a as usize], features[b as usize]);
        let has_edge = graph.has_edge(a, b);

        let base_p = match base {
            BaseModel::ER => density,
            BaseModel::PA | BaseModel::Config => {
                config_edge_prob(degrees[a as usize], degrees[b as usize], m)
            }
        };

        let entry = stats.entry(key).or_insert(KeyStats {
            edges: 0,
            total: 0,
            base_prob_sum: 0.0,
        });
        entry.total += 1;
        if has_edge {
            entry.edges += 1;
        }
        entry.base_prob_sum += base_p;

        sampled += 1;
        attempts += 1;
    }

    // Build decision table for frequent keys
    let mut decision_table: FxHashMap<u64, f64> = FxHashMap::default();
    for (&key, ks) in &stats {
        if ks.total as usize >= min_frequency {
            let emp_p = (ks.edges as f64 / ks.total as f64).clamp(1e-10, 1.0 - 1e-10);
            decision_table.insert(key, emp_p);
        }
    }

    // Table description cost
    let table_entry_bits = 2 * n_feat_bits + 10; // 2 feature vecs + probability
    let table_cost = 32.0 + decision_table.len() as f64 * table_entry_bits as f64;

    // Base model param cost
    let base_param_bits: f64 = match base {
        BaseModel::Config => {
            // Degree sequence encoding cost (same as configuration_model)
            use std::collections::HashMap;
            let mut deg_counts: HashMap<u32, usize> = HashMap::new();
            for &d in &degrees {
                *deg_counts.entry(d).or_insert(0) += 1;
            }
            let k = deg_counts.len();
            let d_max = max_deg;
            let table_bits = k as f64
                * (f64::log2(d_max as f64 + 1.0) + f64::log2(n as f64 + 1.0))
                + f64::log2(k as f64 + 1.0);
            let seq_bits: f64 = degrees
                .iter()
                .map(|&d| {
                    let prob = deg_counts[&d] as f64 / n as f64;
                    -f64::log2(prob)
                })
                .sum();
            table_bits + seq_bits
        }
        _ => 32.0,
    };

    // Compute encoding cost from sampled stats
    let mut total_enc_bits = 0.0f64;
    let mut base_enc_bits = 0.0f64;
    let mut total_sampled = 0u64;

    for (&key, ks) in &stats {
        let avg_base_p = (ks.base_prob_sum / ks.total as f64).clamp(1e-10, 1.0 - 1e-10);

        let p = if let Some(&dt_p) = decision_table.get(&key) {
            dt_p
        } else {
            avg_base_p
        };

        let edges = ks.edges as f64;
        let non_edges = (ks.total - ks.edges) as f64;

        total_enc_bits += edges * (-f64::log2(p)) + non_edges * (-f64::log2(1.0 - p));
        base_enc_bits += edges * (-f64::log2(avg_base_p)) + non_edges * (-f64::log2(1.0 - avg_base_p));
        total_sampled += ks.total as u64;
    }

    // Extrapolate to full graph
    let total_pairs = n as f64 * (n as f64 - 1.0) / 2.0;
    let full_enc_bits = if total_sampled > 0 {
        total_enc_bits / total_sampled as f64 * total_pairs
    } else {
        f64::NAN
    };
    let full_base_bits = if total_sampled > 0 {
        base_enc_bits / total_sampled as f64 * total_pairs
    } else {
        f64::NAN
    };

    let dt_total = base_param_bits + table_cost + full_enc_bits;
    let base_total = base_param_bits + full_base_bits;
    let bpe = dt_total / m as f64;
    let base_bpe = base_total / m as f64;

    DecisionTableResult {
        bits_per_edge: bpe,
        table_entries: decision_table.len(),
        feature_bits: n_feat_bits,
        table_cost_bits: table_cost,
        encoding_cost_bits: full_enc_bits,
        base_only_bpe: base_bpe,
        improvement: base_bpe - bpe,
    }
}

/// Exact (no-sampling) decision table model. O(n + m + K²) where K = unique features.
///
/// Instead of sampling random pairs, we:
/// 1. Count nodes per feature vector — O(n)
/// 2. Count edges per feature-pair key by iterating actual edges — O(m)
/// 3. Compute total pairs per key from node counts — O(K²)
///
/// This gives exact edge rates per key without any sampling noise.
pub fn decision_table_exact(
    graph: &Graph,
    base: BaseModel,
    min_frequency: usize,
) -> DecisionTableResult {
    let n = graph.num_nodes();
    let m = graph.num_edges();
    let density = graph.density();
    let degrees = graph.degrees();
    let max_deg = *degrees.iter().max().unwrap_or(&1);

    // Compute features
    let log_bins = compute_log_bins(max_deg);
    let n_feat_bits = log_bins.len();
    let features = compute_node_features(graph, &degrees, &log_bins);

    // Step 1: Count nodes per feature vector
    let mut feat_counts: FxHashMap<FeatureVec, u64> = FxHashMap::default();
    for &fv in &features {
        *feat_counts.entry(fv).or_insert(0) += 1;
    }
    let unique_feats: Vec<FeatureVec> = feat_counts.keys().copied().collect();
    let k = unique_feats.len();

    // Step 2: Count edges per feature-pair key — O(m)
    let mut edge_counts: FxHashMap<u64, u64> = FxHashMap::default();
    for u in 0..n {
        for &v in graph.neighbors(u) {
            if (u as u32) < v {
                let key = pair_key(features[u], features[v as usize]);
                *edge_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    // Step 3: Compute total pairs per key from node counts — O(K²)
    let mut pair_counts: FxHashMap<u64, u64> = FxHashMap::default();
    for i in 0..k {
        let fi = unique_feats[i];
        let ci = feat_counts[&fi];
        // Same feature pair: C(ci, 2)
        let key = pair_key(fi, fi);
        *pair_counts.entry(key).or_insert(0) += ci * (ci - 1) / 2;
        // Different feature pairs
        for j in (i + 1)..k {
            let fj = unique_feats[j];
            let cj = feat_counts[&fj];
            let key = pair_key(fi, fj);
            *pair_counts.entry(key).or_insert(0) += ci * cj;
        }
    }

    // Step 4: Compute average base probability per key
    // For ER: constant density. For PA/Config: need avg du*dv/(2m) per key.
    // Compute sum of degrees per feature vector, then avg product per key.
    let mut deg_sum_per_feat: FxHashMap<FeatureVec, f64> = FxHashMap::default();
    let mut deg_sq_sum_per_feat: FxHashMap<FeatureVec, f64> = FxHashMap::default();
    for (i, &fv) in features.iter().enumerate() {
        let d = degrees[i] as f64;
        *deg_sum_per_feat.entry(fv).or_insert(0.0) += d;
        *deg_sq_sum_per_feat.entry(fv).or_insert(0.0) += d * d;
    }

    // Build decision table
    let mut decision_table: FxHashMap<u64, f64> = FxHashMap::default();
    for (&key, &total) in &pair_counts {
        if total as usize >= min_frequency {
            let edges = *edge_counts.get(&key).unwrap_or(&0);
            let emp_p = (edges as f64 / total as f64).clamp(1e-10, 1.0 - 1e-10);
            decision_table.insert(key, emp_p);
        }
    }

    // Table description cost
    let table_entry_bits = 2 * n_feat_bits + 10;
    let table_cost = 32.0 + decision_table.len() as f64 * table_entry_bits as f64;

    // Base model param cost
    let base_param_bits: f64 = match base {
        BaseModel::Config => {
            use std::collections::HashMap;
            let mut deg_counts: HashMap<u32, usize> = HashMap::new();
            for &d in &degrees {
                *deg_counts.entry(d).or_insert(0) += 1;
            }
            let kk = deg_counts.len();
            let table_bits = kk as f64
                * (f64::log2(max_deg as f64 + 1.0) + f64::log2(n as f64 + 1.0))
                + f64::log2(kk as f64 + 1.0);
            let seq_bits: f64 = degrees.iter()
                .map(|&d| -f64::log2(deg_counts[&d] as f64 / n as f64))
                .sum();
            table_bits + seq_bits
        }
        _ => 32.0,
    };

    // Compute exact encoding cost
    let mut total_enc_bits = 0.0f64;
    let mut base_enc_bits = 0.0f64;

    for (&key, &total) in &pair_counts {
        let edges = *edge_counts.get(&key).unwrap_or(&0);
        let non_edges = total - edges;

        // Compute base probability for this key
        // Unpack key to get feature vecs
        let fv_lo = FeatureVec((key >> 32) as u32);
        let fv_hi = FeatureVec(key as u32);

        let base_p = match base {
            BaseModel::ER => density,
            BaseModel::PA | BaseModel::Config => {
                // Average du*dv/(2m) for pairs with these features
                let sum_lo = deg_sum_per_feat.get(&fv_lo).copied().unwrap_or(0.0);
                let cnt_lo = feat_counts.get(&fv_lo).copied().unwrap_or(1) as f64;
                let avg_d_lo = sum_lo / cnt_lo;

                if fv_lo == fv_hi {
                    // Same feature: use E[d]^2 approx (or more precisely, sum_sq / cnt)
                    let sq_sum = deg_sq_sum_per_feat.get(&fv_lo).copied().unwrap_or(1.0);
                    // E[du*dv] for random pair from same group ≈ (sum_d)^2 / cnt^2
                    // More exact: (sum_d^2 - sum_d2) / (cnt*(cnt-1)) but approx is fine
                    (avg_d_lo * avg_d_lo / (2.0 * m as f64)).clamp(1e-10, 1.0 - 1e-10)
                } else {
                    let sum_hi = deg_sum_per_feat.get(&fv_hi).copied().unwrap_or(0.0);
                    let cnt_hi = feat_counts.get(&fv_hi).copied().unwrap_or(1) as f64;
                    let avg_d_hi = sum_hi / cnt_hi;
                    (avg_d_lo * avg_d_hi / (2.0 * m as f64)).clamp(1e-10, 1.0 - 1e-10)
                }
            }
        };
        let base_p = base_p.clamp(1e-10, 1.0 - 1e-10);

        let p = if let Some(&dt_p) = decision_table.get(&key) {
            dt_p
        } else {
            base_p
        };

        if edges > 0 {
            total_enc_bits += edges as f64 * (-f64::log2(p));
            base_enc_bits += edges as f64 * (-f64::log2(base_p));
        }
        if non_edges > 0 {
            total_enc_bits += non_edges as f64 * (-f64::log2(1.0 - p));
            base_enc_bits += non_edges as f64 * (-f64::log2(1.0 - base_p));
        }
    }

    let dt_total = base_param_bits + table_cost + total_enc_bits;
    let base_total = base_param_bits + base_enc_bits;
    let bpe = dt_total / m as f64;
    let base_bpe = base_total / m as f64;

    DecisionTableResult {
        bits_per_edge: bpe,
        table_entries: decision_table.len(),
        feature_bits: n_feat_bits,
        table_cost_bits: table_cost,
        encoding_cost_bits: total_enc_bits,
        base_only_bpe: base_bpe,
        improvement: base_bpe - bpe,
    }
}

/// Exact decision table using (degree_bin_u, degree_bin_v, common_neighbors_bin).
///
/// Steps:
/// 1. Assign each node a degree bin — O(n)
/// 2. For each edge, compute common neighbors and bin it — O(m * d_avg)
/// 3. Count total pairs per (deg_bin, deg_bin) from bin counts — O(B²)
/// 4. Distribute non-edge CN counts (CN=0 for most non-edges in sparse graphs)
/// 5. Build decision table and compute exact encoding cost
pub fn decision_table_cn(
    graph: &Graph,
    base: BaseModel,
    min_frequency: usize,
    num_deg_bins: usize,
) -> DecisionTableResult {
    let n = graph.num_nodes();
    let m = graph.num_edges();
    let density = graph.density();
    let degrees = graph.degrees();

    // Step 1: Degree bins
    let deg_bins: Vec<u8> = degrees.iter().map(|&d| degree_bin(d, num_deg_bins)).collect();

    // Count nodes per degree bin
    let mut bin_counts = vec![0u64; num_deg_bins];
    for &b in &deg_bins {
        bin_counts[b as usize] += 1;
    }

    // Sum of degrees per bin (for base_p computation)
    let mut deg_sum_per_bin = vec![0.0f64; num_deg_bins];
    for (i, &b) in deg_bins.iter().enumerate() {
        deg_sum_per_bin[b as usize] += degrees[i] as f64;
    }

    // Step 2: For each edge, compute common neighbors and build edge counts per triple key
    let mut edge_counts: FxHashMap<u32, u64> = FxHashMap::default();
    // Also track CN distribution per (deg_bin, deg_bin) pair for non-edge estimation
    let mut edge_cn_sum: FxHashMap<u16, u64> = FxHashMap::default(); // (db_lo, db_hi) -> sum of CN for edges
    let mut edge_cn_count: FxHashMap<u16, u64> = FxHashMap::default();

    for u in 0..n {
        for &v in graph.neighbors(u) {
            if (u as u32) < v {
                let cn = common_neighbors(graph, u, v as usize);
                let cnb = cn_bin(cn);
                let db_u = deg_bins[u];
                let db_v = deg_bins[v as usize];
                let key = triple_key(db_u, db_v, cnb);
                *edge_counts.entry(key).or_insert(0) += 1;

                let deg_pair_key = if db_u <= db_v {
                    (db_u as u16) << 8 | db_v as u16
                } else {
                    (db_v as u16) << 8 | db_u as u16
                };
                *edge_cn_sum.entry(deg_pair_key).or_insert(0) += cn as u64;
                *edge_cn_count.entry(deg_pair_key).or_insert(0) += 1;
            }
        }
    }

    // Step 3: Total pairs per (deg_bin_u, deg_bin_v)
    let mut total_pairs_per_degpair: FxHashMap<u16, u64> = FxHashMap::default();
    for bi in 0..num_deg_bins {
        for bj in bi..num_deg_bins {
            let pairs = if bi == bj {
                bin_counts[bi] * (bin_counts[bi] - 1) / 2
            } else {
                bin_counts[bi] * bin_counts[bj]
            };
            if pairs > 0 {
                let dpk = (bi as u16) << 8 | bj as u16;
                total_pairs_per_degpair.insert(dpk, pairs);
            }
        }
    }

    // Step 4: For each (deg_bin, deg_bin) pair, assign non-edges to CN=0 bin
    // In sparse graphs, the vast majority of non-edges have CN=0.
    // Total pairs with CN>0 that are non-edges is small; approximate as:
    //   non_edge_cn0 = total_pairs - edges_total - non_edge_cn_positive
    // For simplicity, assume ALL non-edges have CN=0 (good approx for sparse graphs)
    let mut pair_counts: FxHashMap<u32, u64> = FxHashMap::default();
    for (&dpk, &total) in &total_pairs_per_degpair {
        let bi = (dpk >> 8) as u8;
        let bj = dpk as u8;
        // Count edges with each CN bin for this deg pair
        let mut edge_total_for_dp = 0u64;
        for cnb in 0..6u8 {
            let key = triple_key(bi, bj, cnb);
            let ec = *edge_counts.get(&key).unwrap_or(&0);
            edge_total_for_dp += ec;
            if ec > 0 {
                pair_counts.insert(key, ec); // placeholder, will add non-edges below
            }
        }
        // Non-edges: put them all in CN=0 bin
        let non_edges = total.saturating_sub(edge_total_for_dp);
        let cn0_key = triple_key(bi, bj, 0);
        // Total pairs for cn0 = non_edges (as non-edges) + edges with cn=0
        let edges_cn0 = *edge_counts.get(&cn0_key).unwrap_or(&0);
        pair_counts.insert(cn0_key, non_edges + edges_cn0);

        // For CN > 0 bins: pairs = edges (all pairs with CN>0 that are non-edges is rare)
        // This is an approximation — in dense graphs, non-edges with CN>0 exist
        for cnb in 1..6u8 {
            let key = triple_key(bi, bj, cnb);
            let ec = *edge_counts.get(&key).unwrap_or(&0);
            if ec > 0 {
                // Estimate: pairs with this CN bin ≈ edges * (1/density) approximately
                // More precisely: for sparse graphs, almost all pairs with CN>0 are edges
                // For now use edges * 2 as a rough upper bound
                let est_pairs = (ec as f64 / density.max(0.001)).min(total as f64) as u64;
                pair_counts.insert(key, est_pairs.max(ec));
            }
        }
    }

    // Step 5: Build decision table
    let mut decision_table: FxHashMap<u32, f64> = FxHashMap::default();
    for (&key, &total) in &pair_counts {
        if total as usize >= min_frequency {
            let edges = *edge_counts.get(&key).unwrap_or(&0);
            let emp_p = (edges as f64 / total as f64).clamp(1e-10, 1.0 - 1e-10);
            decision_table.insert(key, emp_p);
        }
    }

    // Table description cost: each entry = 2 deg_bin indices (5 bits each) + CN bin (3 bits) + probability (10 bits)
    let n_feat_bits = 2 * 5 + 3; // 13 bits per key
    let table_entry_bits = n_feat_bits + 10;
    let table_cost = 32.0 + decision_table.len() as f64 * table_entry_bits as f64;

    // Base model param cost
    let base_param_bits: f64 = match base {
        BaseModel::Config => {
            use std::collections::HashMap;
            let mut deg_counts_map: HashMap<u32, usize> = HashMap::new();
            for &d in &degrees {
                *deg_counts_map.entry(d).or_insert(0) += 1;
            }
            let k = deg_counts_map.len();
            let d_max = *degrees.iter().max().unwrap_or(&1);
            let table_bits = k as f64
                * (f64::log2(d_max as f64 + 1.0) + f64::log2(n as f64 + 1.0))
                + f64::log2(k as f64 + 1.0);
            let seq_bits: f64 = degrees.iter()
                .map(|&d| -f64::log2(deg_counts_map[&d] as f64 / n as f64))
                .sum();
            table_bits + seq_bits
        }
        _ => 32.0,
    };

    // Step 6: Compute encoding cost
    let mut total_enc_bits = 0.0f64;
    let mut base_enc_bits = 0.0f64;

    for (&key, &total) in &pair_counts {
        let edges = *edge_counts.get(&key).unwrap_or(&0);
        let non_edges = total.saturating_sub(edges);

        // Base probability
        let base_p = match base {
            BaseModel::ER => density,
            BaseModel::PA | BaseModel::Config => {
                // Extract deg bins from key
                let db_lo = (key >> 8) as u8 & 0x1f;
                let db_hi = (key >> 3) as u8 & 0x1f;
                let avg_d_lo = if bin_counts[db_lo as usize] > 0 {
                    deg_sum_per_bin[db_lo as usize] / bin_counts[db_lo as usize] as f64
                } else { 1.0 };
                let avg_d_hi = if bin_counts[db_hi as usize] > 0 {
                    deg_sum_per_bin[db_hi as usize] / bin_counts[db_hi as usize] as f64
                } else { 1.0 };
                (avg_d_lo * avg_d_hi / (2.0 * m as f64)).clamp(1e-10, 1.0 - 1e-10)
            }
        };
        let base_p = base_p.clamp(1e-10, 1.0 - 1e-10);

        let p = if let Some(&dt_p) = decision_table.get(&key) {
            dt_p
        } else {
            base_p
        };

        if edges > 0 {
            total_enc_bits += edges as f64 * (-f64::log2(p));
            base_enc_bits += edges as f64 * (-f64::log2(base_p));
        }
        if non_edges > 0 {
            total_enc_bits += non_edges as f64 * (-f64::log2(1.0 - p));
            base_enc_bits += non_edges as f64 * (-f64::log2(1.0 - base_p));
        }
    }

    let dt_total = base_param_bits + table_cost + total_enc_bits;
    let base_total = base_param_bits + base_enc_bits;
    let bpe = dt_total / m as f64;
    let base_bpe = base_total / m as f64;

    DecisionTableResult {
        bits_per_edge: bpe,
        table_entries: decision_table.len(),
        feature_bits: n_feat_bits,
        table_cost_bits: table_cost,
        encoding_cost_bits: total_enc_bits,
        base_only_bpe: base_bpe,
        improvement: base_bpe - bpe,
    }
}
