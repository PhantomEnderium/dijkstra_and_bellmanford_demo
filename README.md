# Shortest Path Algorithms Comparison

A comprehensive Python implementation and comparison of **Dijkstra's Algorithm** and **Bellman-Ford Algorithm** for finding shortest paths in weighted graphs, with interactive visualization.

## Setup

### 1. Install Dependencies
Run this command in the `dijkstra_demo` folder:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install matplotlib networkx requests python-dotenv
```

### 2. Run the Program
```bash
python shortest-path-gen.py
```

## What the Program Does

- **Creates a weighted graph** from preset templates or custom input
- **Runs both Dijkstra's and Bellman-Ford algorithms** starting from a user-selected node
- **Calculates shortest distances** to all other nodes using both algorithms
- **Finds shortest paths** and compares results
- **Detects differences** when Dijkstra fails on graphs with negative edges
- **Generates detailed visualizations** as PNG files:
  - `shortest_path_graph.png` - Full graph with all nodes and edges
  - `shortest_path_dijkstra_path.png` - Dijkstra's algorithm highlighting the shortest path
  - `shortest_path_bellman_ford_path.png` - Bellman-Ford algorithm highlighting the shortest path
  - `shortest_path_comparison.png` - Side-by-side comparison of both algorithms
- **Optionally sends visualizations to Discord** via webhook (using `.env` configuration)
- **Saves algorithm output** to `algorithm_output.txt` and `algorithm_steps.txt`

## Usage

When you run the program, it will:

1. **Display preset graph options:**
   - Small graph (5 nodes, ideal for learning)
   - Medium graph (6 nodes, recommended)
   - Large graph (8 nodes)
   - Negative edge graph (demonstrates Bellman-Ford advantage)
   - Complete graph (dense connectivity)

2. **Ask for start and end nodes** (with sensible defaults)

3. **Execute both algorithms** and display:
   - Step-by-step algorithm execution
   - Shortest distances from start to all nodes
   - Shortest path with total cost
   - Execution statistics (operations/iterations)

4. **Generate comparison** showing if algorithms differ

5. **Create visualizations** with:
   - White circles for non-selected nodes with black text
   - Black circles for selected path nodes with white text
   - Solid black edges for non-path edges
   - **Dashed edges** (spaced out) for path edges

## Output

The program prints:
- All graph edges with weights
- Shortest distances and paths for both algorithms
- Detailed execution steps for each algorithm
- Comparison table showing any differences
- Confirmation when PNG files are saved
- Warning if negative edge differences are detected

## Program Structure

```
ShortestPathGraph class:
├── __init__()                      - Initialize empty graph
├── add_edge()                      - Add weighted edge between nodes
├── dijkstra()                      - Dijkstra's algorithm
├── bellman_ford()                  - Bellman-Ford algorithm
├── get_path()                      - Reconstruct shortest path from parent dict
├── visualize()                     - Draw graph with matplotlib
├── generate_comparison_image()     - Create side-by-side comparison
└── generate_comparison_table()     - Display results in table format

Helper Functions:
├── select_graph()                  - Menu for graph preset selection
├── get_preset_graphs()             - Define available graph templates
├── save_output_to_file()          - Write results to text file
└── send_to_discord()              - Upload visualizations to Discord webhook
```

## Algorithm Explanations

### Dijkstra's Algorithm
1. **Initialize** all distances to infinity except start node (distance 0)
2. **Use a priority queue** to always process the closest unvisited node
3. **For each node**, check all neighbors and update distances if a shorter path is found
4. **Mark nodes as visited** to avoid reprocessing
5. **Continue** until all reachable nodes are processed
6. **Limitation:** Cannot handle negative edge weights correctly

### Bellman-Ford Algorithm
1. **Initialize** all distances to infinity except start node (distance 0)
2. **Relax all edges** repeatedly (V-1 times where V = number of vertices)
3. **For each edge (u,v) with weight w**, update: `distance[v] = min(distance[v], distance[u] + w)`
4. **On the Vth iteration**, check if any distance decreases (indicates negative cycle)
5. **Advantage:** Works correctly with negative edge weights
6. **Trade-off:** Slower than Dijkstra but more robust

## Time Complexity

| Algorithm | Time Complexity | Best For |
|-----------|-----------------|----------|
| Dijkstra | O((V + E) log V) | Positive weights only, fast performance |
| Bellman-Ford | O(V × E) | Any weights, negative edge handling |

- V = number of vertices
- E = number of edges

## Files

- `shortest-path-gen.py` - Main program
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `.env` - Configuration for Discord webhook (optional)
- Generated outputs:
  - `shortest_path_graph.png` - Base graph visualization
  - `shortest_path_dijkstra_path.png` - Dijkstra path visualization
  - `shortest_path_bellman_ford_path.png` - Bellman-Ford path visualization
  - `shortest_path_comparison.png` - Side-by-side comparison
  - `algorithm_output.txt` - Detailed algorithm results
  - `algorithm_steps.txt` - Step-by-step execution log

## Example Output

```
============================================================
SHORTEST PATH ALGORITHMS COMPARISON
Dijkstra's Algorithm vs Bellman-Ford
============================================================

SELECT GRAPH PRESET
======= =========
  1. Small Graph (5 nodes)
  2. Medium Graph (6 nodes - Recommended)
  3. Large Graph (8 nodes)
  4. Negative Edge Graph
  5. Complete Graph

Enter your choice (1-5): 2

Selected: Medium Graph (6 nodes - Recommended)

Graph Edges:
  A -- B (weight: 4)
  A -- C (weight: 2)
  B -- D (weight: 5)
  C -- B (weight: 1)
  C -- D (weight: 8)
  D -- E (weight: 2)
  D -- F (weight: 6)

Available nodes: A, B, C, D, E, F

Enter start node (default: A): A
Enter end node (default: F): F

============================================================
DIJKSTRA'S ALGORITHM
============================================================

Shortest distances from 'A':
  A → A: 0
  A → B: 3
  A → C: 2
  A → D: 8
  A → E: 10
  A → F: 14

📋 Dijkstra Execution Steps: (15 operations)
  Visit A (distance: 0)
  Update B: 4
  Update C: 2
  Visit C (distance: 2)
  ...

Dijkstra path from A to F:
  A → C → B → D → F (cost: 14)

============================================================
BELLMAN-FORD ALGORITHM
============================================================

Shortest distances from 'A':
  A → A: 0
  A → B: 3
  A → C: 2
  A → D: 8
  A → E: 10
  A → F: 14

📋 Bellman-Ford Execution Steps: (3 iterations)
  Iteration 1: Relaxed edges...
  Iteration 2: Relaxed edges...
  Iteration 3: No improvements (converged)

Bellman-Ford path from A to F:
  A → C → B → D → F (cost: 14)

============================================================
Done! Check the generated PNG files for visualizations.
============================================================
```

## Discord Integration (Optional)

To enable Discord webhook integration:

1. Create a Discord webhook for your server
2. Create a `.env` file in the project root:
   ```
   DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
   ```
3. The program will automatically send visualizations to Discord after execution

## Visualization Features

- **Node Colors:**
  - White circles with black text: Non-selected nodes
  - Black circles with white text: Selected path nodes
  - Black border: All nodes for visibility

- **Edge Styles:**
  - Solid lines: Non-path edges
  - Dashed lines (spaced): Selected shortest path edges
  - Arrow heads: Show edge direction

- **Graph Layout:** Spring layout algorithm for natural positioning

## Key Differences Between Algorithms

| Feature | Dijkstra | Bellman-Ford |
|---------|----------|--------------|
| Negative edges | ❌ Fails | ✅ Works |
| Negative cycles | ❌ Undetected | ✅ Detects |
| Speed | ⚡ Faster | 🐢 Slower |
| Use case | General purpose | Negative weights, cycle detection |

## Notes

- **Graph Type:** Directed graph with weighted edges
- **Edge Weights:** Can be positive, negative, or zero (for Bellman-Ford)
- **Disconnected nodes:** Marked with ∞ (unreachable)
- **Non-interactive mode:** Auto-selects Medium preset (useful for automation/CI)
- All visualizations saved at 150 DPI for high quality

## Requirements

- Python 3.7+
- matplotlib 3.8.4
- networkx 3.3
- requests (for Discord integration)
- python-dotenv (for .env file support)
