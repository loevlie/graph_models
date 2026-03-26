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

/// Compute log-bin index for a degree value. Returns 0..num_bins-1.
/// Bin 0: degree 0-1, Bin 1: degree 2-3, Bin 2: degree 4-7, etc.
pub fn degree_bin(degree: u32, num_bins: usize) -> u8 {
    if degree == 0 {
        return 0;
    }
    let bin = (32 - degree.leading_zeros()) as usize; // ~ floor(log2(degree)) + 1
    bin.min(num_bins - 1) as u8
}

/// Compute common neighbor count between two nodes using sorted neighbor lists.
/// O(d_u + d_v) via merge of sorted arrays.
pub fn common_neighbors(graph: &Graph, u: usize, v: usize) -> u32 {
    let nu = graph.neighbors(u);
    let nv = graph.neighbors(v);
    let mut count = 0u32;
    let (mut i, mut j) = (0, 0);
    while i < nu.len() && j < nv.len() {
        if nu[i] == nv[j] {
            count += 1;
            i += 1;
            j += 1;
        } else if nu[i] < nv[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    count
}

/// Bin a common-neighbor count. Bins: 0, 1, 2-3, 4-7, 8-15, 16+
pub fn cn_bin(cn: u32) -> u8 {
    match cn {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=7 => 3,
        8..=15 => 4,
        _ => 5,
    }
}

/// A 3D key: (deg_bin_u, deg_bin_v, cn_bin) packed into a u32.
/// deg_bins use 5 bits each (max 32 bins), cn_bin uses 3 bits (max 8 bins).
pub fn triple_key(db_u: u8, db_v: u8, cnb: u8) -> u32 {
    let (lo, hi) = if db_u <= db_v { (db_u, db_v) } else { (db_v, db_u) };
    (lo as u32) << 8 | (hi as u32) << 3 | cnb as u32
}
