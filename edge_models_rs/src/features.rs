use crate::graph::Graph;

/// Compact binary feature vector stored as a u32 bitmask.
/// Bit i is set if the node has at least one neighbor with degree >= 2^i.
#[derive(Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FeatureVec(pub u32);

impl FeatureVec {
    pub fn num_bits(&self) -> usize {
        32 - self.0.leading_zeros() as usize
    }
}

/// Compute log-scale bins: [1, 2, 4, 8, ...] up to max_degree.
pub fn compute_log_bins(max_degree: u32) -> Vec<u32> {
    let mut bins = Vec::new();
    let mut t: u32 = 1;
    while t <= max_degree {
        bins.push(t);
        t = t.saturating_mul(2);
    }
    bins
}

/// Compute binary feature vectors for all nodes. O(m) total work.
pub fn compute_node_features(graph: &Graph, degrees: &[u32], log_bins: &[u32]) -> Vec<FeatureVec> {
    let n = graph.num_nodes();
    let mut features = vec![FeatureVec(0); n];

    for u in 0..n {
        let mut bits: u32 = 0;
        for &nb in graph.neighbors(u) {
            let nb_deg = degrees[nb as usize];
            for (b, &threshold) in log_bins.iter().enumerate() {
                if nb_deg >= threshold {
                    bits |= 1 << b;
                }
            }
        }
        features[u] = FeatureVec(bits);
    }

    features
}

/// Pack two FeatureVecs into a single u64 key (smaller first for symmetry).
pub fn pair_key(a: FeatureVec, b: FeatureVec) -> u64 {
    let (lo, hi) = if a <= b { (a.0, b.0) } else { (b.0, a.0) };
    (lo as u64) << 32 | hi as u64
}
