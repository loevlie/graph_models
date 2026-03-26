use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashMap;

use crate::base_models::config_edge_prob;
use crate::features::{compute_log_bins, compute_node_features, pair_key};
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
