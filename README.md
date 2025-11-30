# Dijkstra's Algorithm Demo

A simple Python implementation of Dijkstra's shortest path algorithm with visualization.

## Setup

### 1. Install Dependencies
Run this command in the `dijkstra_demo` folder:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install matplotlib networkx
```

### 2. Run the Program
```bash
python dijkstra.py
```

## What the Program Does

- **Creates a sample graph** with 6 nodes and weighted edges
- **Runs Dijkstra's algorithm** starting from node 'A'
- **Calculates shortest distances** to all other nodes
- **Finds shortest paths** and displays them
- **Generates visualizations** as PNG files:
  - `dijkstra_graph.png` - Full graph with all nodes
  - `dijkstra_path.png` - Graph highlighting the path from A to F

## Output

The program prints:
- All edges in the graph
- Shortest distances from the starting node to every other node
- Example shortest paths with their distances
- Confirmations when PNG files are saved

## Program Structure

```
DijkstraGraph class:
├── __init__()           - Initialize empty graph
├── add_edge()           - Add weighted edge between nodes
├── dijkstra()           - Main algorithm (returns distances and parent pointers)
├── get_path()           - Reconstruct shortest path
└── visualize()          - Draw graph with matplotlib
```

## Algorithm Explanation

1. **Initialize** all distances to infinity except start node (distance 0)
2. **Use a priority queue** to always process the closest unvisited node
3. **For each node**, check all neighbors and update distances if a shorter path is found
4. **Mark nodes as visited** to avoid reprocessing
5. **Continue** until all reachable nodes are processed

## Time Complexity

- With priority queue (heap): **O((V + E) log V)**
- V = number of vertices, E = number of edges

## Files

- `dijkstra.py` - Main program
- `requirements.txt` - Python dependencies
- `README.md` - This file
- Generated: `dijkstra_graph.png`, `dijkstra_path.png`

## Example Output

```
Shortest distances from 'A':
  A → A: 0
  A → B: 4
  A → C: 2
  A → D: 8
  A → E: 10
  A → F: 13

Example Shortest Paths:
  A → F: A → C → B → D → F (distance: 13)
  A → E: A → C → D → E (distance: 10)
  A → D: A → C → B → D (distance: 8)
```

## Notes

- Graph is **undirected** (edges work both ways)
- All edge weights are **positive** (Dijkstra requirement)
- For negative weights, use Bellman-Ford algorithm instead
- Visualizations use spring layout for node positioning (random seed for consistency)
