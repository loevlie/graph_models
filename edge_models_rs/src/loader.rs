use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::graph::Graph;

/// Load a TU-format dataset. Returns one Graph per graph in the collection.
pub fn load_tu_dataset(dir: &Path, name: &str) -> Result<Vec<Graph>, String> {
    let edge_path = dir.join(format!("{}_A.txt", name));
    let indicator_path = dir.join(format!("{}_graph_indicator.txt", name));

    // Parse edges (1-indexed pairs, directed — each undirected edge appears twice)
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let file = File::open(&edge_path)
        .map_err(|e| format!("Cannot open {}: {}", edge_path.display(), e))?;
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| e.to_string())?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 2 {
            let u: u32 = parts[0].trim().parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
            let v: u32 = parts[1].trim().parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
            edges.push((u - 1, v - 1)); // convert to 0-indexed
        }
    }

    // Parse graph indicator (1-indexed graph ID per node)
    let mut node_to_graph: Vec<usize> = Vec::new();
    let file = File::open(&indicator_path)
        .map_err(|e| format!("Cannot open {}: {}", indicator_path.display(), e))?;
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| e.to_string())?;
        let g: usize = line.trim().parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        node_to_graph.push(g - 1);
    }

    let num_graphs = node_to_graph.iter().copied().max().unwrap_or(0) + 1;

    // Group nodes by graph
    let mut graph_nodes: Vec<Vec<u32>> = vec![Vec::new(); num_graphs];
    for (node, &g) in node_to_graph.iter().enumerate() {
        graph_nodes[g].push(node as u32);
    }

    // Group edges by graph
    let mut graph_edges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_graphs];
    for &(u, v) in &edges {
        let g = node_to_graph[u as usize];
        graph_edges[g].push((u, v));
    }

    // Build graphs with local node IDs
    let mut graphs = Vec::with_capacity(num_graphs);
    for g in 0..num_graphs {
        let nodes = &graph_nodes[g];
        if nodes.is_empty() {
            continue;
        }
        let min_node = *nodes.iter().min().unwrap();
        let n = nodes.len();
        let local_edges: Vec<(u32, u32)> = graph_edges[g]
            .iter()
            .map(|&(u, v)| (u - min_node, v - min_node))
            .collect();
        graphs.push(Graph::from_edges(n, &local_edges));
    }

    Ok(graphs)
}
