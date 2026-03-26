use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;

const LN2: f64 = std::f64::consts::LN_2;

fn log2(x: f64) -> f64 {
    x.ln() / LN2
}

/// ER model: bits per edge = (-log2(P_ER(G)) + 32) / m
pub fn er_bits_per_edge(n: usize, m: usize, density: f64) -> f64 {
    if density <= 0.0 || density >= 1.0 || m == 0 {
        return f64::NAN;
    }
    let max_edges = n as f64 * (n as f64 - 1.0) / 2.0;
    let ll_bits = m as f64 * log2(density) + (max_edges - m as f64) * log2(1.0 - density);
    (-ll_bits + 32.0) / m as f64
}

/// Configuration model edge probability: du * dv / (2m).
pub fn config_edge_prob(du: u32, dv: u32, m: usize) -> f64 {
    let p = (du as f64 * dv as f64) / (2.0 * m as f64);
    p.clamp(1e-10, 1.0 - 1e-10)
}

/// Wiring bits given degree sequence.
/// = log2((2m-1)!!) - sum(log2(d_i!))
/// = log2((2m)!) - m - log2(m!) - sum(log2(d_i!))
pub fn wiring_bits(degrees: &[u32], m: usize) -> f64 {
    let log2_2m_fact = ln_gamma(2.0 * m as f64 + 1.0) / LN2;
    let log2_m_fact = ln_gamma(m as f64 + 1.0) / LN2;
    let log2_di_fact_sum: f64 = degrees.iter()
        .map(|&d| ln_gamma(d as f64 + 1.0) / LN2)
        .sum();
    log2_2m_fact - m as f64 - log2_m_fact - log2_di_fact_sum
}

/// PA model: bits per edge using P(k) = 2*m0*(m0+1) / (k*(k+1)*(k+2))
pub fn pa_bits_per_edge(degrees: &[u32], n: usize, m: usize) -> f64 {
    if m == 0 || n == 0 {
        return f64::NAN;
    }
    let mean_deg: f64 = degrees.iter().map(|&d| d as f64).sum::<f64>() / n as f64;
    let m0 = (mean_deg / 2.0).round().max(1.0) as u32;
    let d_max = *degrees.iter().max().unwrap_or(&1);

    // Compute PA distribution over [1, d_max+1]
    let mut pa_probs: Vec<f64> = (1..=d_max + 1)
        .map(|k| {
            let k = k as f64;
            2.0 * m0 as f64 * (m0 as f64 + 1.0) / (k * (k + 1.0) * (k + 2.0))
        })
        .collect();
    let total: f64 = pa_probs.iter().sum();
    for p in &mut pa_probs {
        *p /= total;
    }

    // Encode each degree
    let mut seq_bits = 0.0;
    for &d in degrees {
        let idx = d.saturating_sub(1) as usize;
        let prob = if idx < pa_probs.len() { pa_probs[idx].max(1e-10) } else { 1e-10 };
        seq_bits += -log2(prob);
    }

    let w_bits = wiring_bits(degrees, m);
    (32.0 + seq_bits + w_bits) / m as f64
}

/// Configuration model: empirical degree encoding + wiring.
pub fn configuration_model_bits_per_edge(degrees: &[u32], n: usize, m: usize) -> f64 {
    if m == 0 {
        return f64::NAN;
    }

    // Empirical degree distribution
    let mut deg_counts: HashMap<u32, usize> = HashMap::new();
    for &d in degrees {
        *deg_counts.entry(d).or_insert(0) += 1;
    }
    let k = deg_counts.len();
    let d_max = *degrees.iter().max().unwrap_or(&1);

    // Table cost
    let table_bits = k as f64 * (log2(d_max as f64 + 1.0) + log2(n as f64 + 1.0)) + log2(k as f64 + 1.0);

    // Per-node degree encoding
    let mut seq_bits = 0.0;
    for &d in degrees {
        let count = deg_counts[&d];
        let prob = count as f64 / n as f64;
        seq_bits += -log2(prob);
    }

    let w_bits = wiring_bits(degrees, m);
    (table_bits + seq_bits + w_bits) / m as f64
}
