/// CSR-format undirected simple graph.
/// Neighbors of node i are in `neighbors[offsets[i]..offsets[i+1]]`, sorted.
pub struct Graph {
    num_nodes: usize,
    offsets: Vec<usize>,
    neighbors: Vec<u32>,
    num_edges: usize,
}

impl Graph {
    /// Build from an edge list. Deduplicates, removes self-loops, stores undirected.
    pub fn from_edges(num_nodes: usize, edges: &[(u32, u32)]) -> Self {
        // Collect unique undirected edges (u < v)
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); num_nodes];
        for &(u, v) in edges {
            if u == v {
                continue;
            }
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            adj[a as usize].push(b);
            adj[b as usize].push(a);
        }

        // Sort and deduplicate each neighbor list
        let mut num_edges = 0;
        for list in &mut adj {
            list.sort_unstable();
            list.dedup();
            num_edges += list.len();
        }
        num_edges /= 2; // each edge counted twice

        // Build CSR
        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let mut neighbors = Vec::new();
        offsets.push(0);
        for list in &adj {
            neighbors.extend_from_slice(list);
            offsets.push(neighbors.len());
        }

        Graph { num_nodes, offsets, neighbors, num_edges }
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    pub fn degree(&self, u: usize) -> u32 {
        (self.offsets[u + 1] - self.offsets[u]) as u32
    }

    pub fn degrees(&self) -> Vec<u32> {
        (0..self.num_nodes).map(|i| self.degree(i)).collect()
    }

    pub fn neighbors(&self, u: usize) -> &[u32] {
        &self.neighbors[self.offsets[u]..self.offsets[u + 1]]
    }

    /// Binary search in sorted neighbor list. O(log d).
    pub fn has_edge(&self, u: u32, v: u32) -> bool {
        let nbs = self.neighbors(u as usize);
        nbs.binary_search(&v).is_ok()
    }

    pub fn density(&self) -> f64 {
        let n = self.num_nodes as f64;
        let max_edges = n * (n - 1.0) / 2.0;
        if max_edges == 0.0 { 0.0 } else { self.num_edges as f64 / max_edges }
    }
}
