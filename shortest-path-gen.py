"""
Shortest Path Algorithms - Dijkstra & Bellman-Ford
A comparison of Dijkstra's and Bellman-Ford shortest path algorithms
"""

import sys
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import requests
import os
import random
from dotenv import load_dotenv

# Fix Unicode encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() == 'utf-8':
    pass  # UTF-8 is fine
else:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class ShortestPathGraph:
    """Graph implementation for shortest path algorithms"""
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = set()
        self.edges_list = []  # Store edges for Bellman-Ford
    
    def add_edge(self, u, v, weight, directed=True):
        """Add a directed edge to the graph"""
        # Handle random weight
        if isinstance(weight, str) and weight.lower() == 'random':
            weight = random.randint(1, 10)

        self.graph[u].append((v, weight))
        self.edges_list.append((u, v, weight))
        self.nodes.add(u)
        self.nodes.add(v)

        # Ensure nodes are added even if they have no outgoing edges
        if v not in self.graph:
            self.graph[v] = []
    
    def dijkstra(self, start):
        """
        Dijkstra's algorithm implementation
        Returns: dictionary of shortest distances from start node
        """
        # Initialize distances and visited set
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        visited = set()
        parent = {node: None for node in self.nodes}
        visit_order = {}  # Track order nodes were visited
        visit_counter = 0
        
        # Priority queue: (distance, node)
        pq = [(0, start)]
        steps = []  # Track steps for visualization
        operations = 0  # Track number of operations
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            operations += 1
            
            # Skip if already visited
            if current_node in visited:
                continue
            
            visited.add(current_node)
            visit_order[current_node] = visit_counter
            visit_counter += 1
            steps.append(f"Visit {current_node} (distance: {current_distance})")
            
            # Check all neighbors
            for neighbor, weight in self.graph[current_node]:
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    
                    # If we found a shorter path, update
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        parent[neighbor] = current_node
                        steps.append(f"  Update {neighbor}: {distances[neighbor]}")
                        heapq.heappush(pq, (new_distance, neighbor))
                        operations += 1
        
        return distances, parent, steps, operations, visit_order
    
    def bellman_ford(self, start):
        """
        Bellman-Ford algorithm implementation
        Returns: distances, parent dict, negative_cycle flag, steps, and iteration count
        """
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        parent = {node: None for node in self.nodes}
        steps = []  # Track steps for visualization
        iterations = 0

        # Relax edges |V| - 1 times
        for iteration in range(len(self.nodes) - 1):
            updated = False
            for u, v, weight in self.edges_list:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    parent[v] = u
                    steps.append(f"Iteration {iteration + 1}: Relax {u}→{v}, update {v} to {distances[v]}")
                    updated = True
            iterations += 1
            # Early exit if no update
            if not updated:
                steps.append(f"Iteration {iteration + 1}: No updates, converged early")
                break

        # Check for negative cycle
        negative_cycle = False
        for u, v, weight in self.edges_list:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                steps.append(f"Negative cycle detected: {u}→{v}")
                negative_cycle = True
                break

        return distances, parent, negative_cycle, steps, iterations
    
    def get_path(self, parent, start, end):
        """Reconstruct the shortest path from start to end"""
        path = []
        current = end
        
        while current is not None:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        # Return path if it starts with start node, otherwise return empty
        if path and path[0] == start:
            return path
        else:
            # Path reconstruction failed or unreachable
            return []
    
    def generate_comparison_table(self, dijkstra_distances, bellman_distances):
        """Generate comparison table between two algorithms"""
        table = "COMPARISON TABLE\n"
        table += "=" * 50 + "\n"
        table += f"{'Node':<6} {'Dijkstra':<15} {'Bellman-Ford':<15} {'Match':<8}\n"
        table += "-" * 50 + "\n"
        
        all_nodes = sorted(set(list(dijkstra_distances.keys()) + list(bellman_distances.keys())))
        differences = 0
        
        for node in all_nodes:
            dij_dist = dijkstra_distances.get(node, float('inf'))
            bf_dist = bellman_distances.get(node, float('inf'))
            
            dij_str = str(dij_dist) if dij_dist != float('inf') else "∞"
            bf_str = str(bf_dist) if bf_dist != float('inf') else "∞"
            match = "✓" if dij_dist == bf_dist else "✗ DIFF"
            
            if dij_dist != bf_dist:
                differences += 1
            
            table += f"{node:<6} {dij_str:<15} {bf_str:<15} {match:<8}\n"
        
        table += "-" * 50 + "\n"
        if differences > 0:
            table += f"⚠️  {differences} NODE(S) WITH DIFFERENT RESULTS\n"
        else:
            table += "✓ All results match (no negative edges affecting paths)\n"
        
        return table
    
    def generate_comparison_image(self, start, dijkstra_distances, dijkstra_path, bellman_distances, bellman_path, dijkstra_visit):
        """Generate a side-by-side comparison of both algorithms"""
        G = nx.DiGraph()
        for u in self.graph:
            for v, weight in self.graph[u]:
                G.add_edge(u, v, weight=weight)
        
        try:
            pos = self.hierarchical_layout(G)
        except:
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.circular_layout(G)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='#f5f5f5')
        
        # ===== DIJKSTRA SIDE =====
        ax1.set_facecolor('#f5f5f5')
        
        # Draw edges FIRST
        if dijkstra_path:
            path_edges = [(dijkstra_path[i], dijkstra_path[i+1]) for i in range(len(dijkstra_path)-1)]
            all_edges = list(G.edges())
            non_path_edges = [e for e in all_edges if e not in path_edges]
            
            # Draw non-path edges as neutral light gray
            nx.draw_networkx_edges(G, pos, edgelist=non_path_edges, width=1.5, edge_color='#cccccc', ax=ax1, arrows=True, arrowsize=10, arrowstyle='->')
            
            # Draw path edges with gradient (dark to light green)
            num_edges = len(path_edges)
            for i, (u, v) in enumerate(path_edges):
                t = i / max(num_edges - 1, 1)
                r = int(0)
                g = int(0x44 + (0xff - 0x44) * t)
                b = int(0x00 + (0x88 - 0x00) * t)
                color_hex = f'#{r:02x}{g:02x}{b:02x}'
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3, edge_color=color_hex, style='dotted', ax=ax1, arrows=True, arrowsize=12, arrowstyle='->')
        else:
            for u, v in G.edges():
                weight = G[u][v]['weight']
                color = '#ff4444' if weight < 0 else '#00aa00'
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1.5, edge_color=color, ax=ax1, arrows=True, arrowsize=10, arrowstyle='->')
        
        # Draw nodes AFTER edges
        nx.draw_networkx_nodes(G, pos, node_color='#2e7d32', node_size=800, ax=ax1, edgecolors='#ffffff', linewidths=2)
        nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='#558b2f', node_size=800, ax=ax1, edgecolors='#ffffff', linewidths=2)
        
        if dijkstra_path:
            nx.draw_networkx_nodes(G, pos, nodelist=dijkstra_path, node_color='#f9a825', node_size=800, ax=ax1, edgecolors='#ffffff', linewidths=2)
        
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_color='#ffffff', ax=ax1)
        
        if dijkstra_visit:
            for node, order in dijkstra_visit.items():
                x, y = pos[node]
                ax1.text(x - 0.08, y + 0.08, str(order), fontsize=9, weight='bold',
                       bbox=dict(boxstyle='circle,pad=0.2', facecolor='#ffaa00', edgecolor='#000000', linewidth=1),
                       color='#000000', ha='center', va='center')
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        for (u, v), label in edge_labels.items():
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            ax1.text(x, y, str(label), fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='#cccccc', alpha=0.95, edgecolor='#2e7d32', linewidth=1.5),
                   color='#000000')
        
        ax1.set_title(f"Dijkstra's Algorithm", fontsize=14, fontweight='bold', color='#000000')
        ax1.axis('off')
        
        # ===== BELLMAN-FORD SIDE =====
        ax2.set_facecolor('#f5f5f5')
        
        # Draw edges FIRST
        if bellman_path:
            path_edges = [(bellman_path[i], bellman_path[i+1]) for i in range(len(bellman_path)-1)]
            all_edges = list(G.edges())
            non_path_edges = [e for e in all_edges if e not in path_edges]
            
            # Draw non-path edges as neutral light gray
            nx.draw_networkx_edges(G, pos, edgelist=non_path_edges, width=1.5, edge_color='#cccccc', ax=ax2, arrows=True, arrowsize=10, arrowstyle='->')
            
            # Draw path edges with gradient (dark to light green)
            num_edges = len(path_edges)
            for i, (u, v) in enumerate(path_edges):
                t = i / max(num_edges - 1, 1)
                r = int(0)
                g = int(0x44 + (0xff - 0x44) * t)
                b = int(0x00 + (0x88 - 0x00) * t)
                color_hex = f'#{r:02x}{g:02x}{b:02x}'
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3, edge_color=color_hex, style='dotted', ax=ax2, arrows=True, arrowsize=12, arrowstyle='->')
        else:
            for u, v in G.edges():
                weight = G[u][v]['weight']
                color = '#ff4444' if weight < 0 else '#00aa00'
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1.5, edge_color=color, ax=ax2, arrows=True, arrowsize=10, arrowstyle='->')
        
        # Draw nodes AFTER edges
        nx.draw_networkx_nodes(G, pos, node_color='#2e7d32', node_size=800, ax=ax2, edgecolors='#ffffff', linewidths=2)
        nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='#558b2f', node_size=800, ax=ax2, edgecolors='#ffffff', linewidths=2)
        
        if bellman_path:
            nx.draw_networkx_nodes(G, pos, nodelist=bellman_path, node_color='#f9a825', node_size=800, ax=ax2, edgecolors='#ffffff', linewidths=2)
        
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_color='#ffffff', ax=ax2)
        
        for (u, v), label in edge_labels.items():
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            ax2.text(x, y, str(label), fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='#cccccc', alpha=0.95, edgecolor='#2e7d32', linewidth=1.5),
                   color='#000000')
        
        ax2.set_title(f"Bellman-Ford Algorithm", fontsize=14, fontweight='bold', color='#000000')
        ax2.axis('off')
        
        plt.suptitle("Algorithm Comparison", fontsize=16, fontweight='bold', color='#000000', y=0.98)
        plt.tight_layout()
        return plt
    
    def draw_gradient_path(self, ax, pos, path, num_segments=10):
        """Draw path with gradient from dark to light green, keeping dotted style"""
        if len(path) < 2:
            return
        
        # Generate gradient from dark green to light green
        colors = []
        for i in range(num_segments):
            ratio = i / (num_segments - 1)
            # Dark green (0.1) to light green (0.9)
            r = 0.0 + (0.2 * ratio)  # 0 to 0.2
            g = 0.6 + (0.3 * ratio)  # 0.6 to 0.9
            b = 0.0 + (0.2 * ratio)  # 0 to 0.2
            colors.append((r, g, b))
        
        # Draw each segment of the path with gradient color
        for path_idx in range(len(path) - 1):
            u, v = path[path_idx], path[path_idx + 1]
            
            # Calculate which color in the gradient this edge should use
            segment_idx = int((path_idx / (len(path) - 1)) * (num_segments - 1))
            segment_idx = min(segment_idx, num_segments - 1)
            color = colors[segment_idx]
            
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color=color,
                linewidth=4,
                linestyle='dotted',
                zorder=2
            )
    
    def draw_edge_labels_with_offset(self, G, pos, ax, edge_labels, font_size=12):
        """Draw edge labels with background boxes"""
        for (u, v), label in edge_labels.items():
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            
            ax.text(x, y, str(label), fontsize=font_size, ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='#cccccc', alpha=0.95, edgecolor='#2e7d32', linewidth=1.5),
                   color='#000000')
    
    def hierarchical_layout(self, G, root='A'):
        """Create layered layout with barycenter ordering to minimize crossings"""
        pos = {}
        
        # BFS to calculate depth of each node
        from collections import deque
        
        depth = {node: 0 for node in G.nodes()}
        queue = deque([root])
        visited = {root}
        
        while queue:
            node = queue.popleft()
            for successor in G.successors(node):
                if successor not in visited:
                    depth[successor] = depth[node] + 1
                    visited.add(successor)
                    queue.append(successor)
        
        # Group nodes by depth
        depth_groups = defaultdict(list)
        for node in G.nodes():
            depth_groups[depth[node]].append(node)
        
        # Calculate x positions based on depth
        max_depth = max(depth.values()) if depth else 0
        if max_depth > 0:
            x_step = 20 / (max_depth + 1)
        else:
            x_step = 1
        
        # First pass: position nodes by depth (unsorted)
        temp_pos = {}
        for d in sorted(depth_groups.keys()):
            nodes_at_depth = depth_groups[d]
            x = -10 + (d + 1) * x_step
            
            num_nodes = len(nodes_at_depth)
            if num_nodes == 1:
                temp_pos[nodes_at_depth[0]] = (x, 0)
            else:
                y_range = 8
                y_step = y_range / (num_nodes - 1) if num_nodes > 1 else 1
                for i, node in enumerate(nodes_at_depth):
                    y = -4 + i * y_step
                    temp_pos[node] = (x, y)
        
        # Second pass: reorder nodes at each depth by barycenter of their children
        # to minimize edge crossings
        for d in sorted(depth_groups.keys()):
            nodes_at_depth = depth_groups[d]
            
            # Calculate barycenter (average y position of children)
            barycenters = {}
            for node in nodes_at_depth:
                children = [v for u, v in G.edges() if u == node]
                if children:
                    child_y_values = [temp_pos.get(child, (0, 0))[1] for child in children]
                    barycenters[node] = sum(child_y_values) / len(child_y_values)
                else:
                    barycenters[node] = temp_pos[node][1]
            
            # Sort nodes by barycenter
            sorted_nodes = sorted(nodes_at_depth, key=lambda n: barycenters[n])
            
            # Reposition sorted nodes
            x = -10 + (d + 1) * x_step
            num_nodes = len(sorted_nodes)
            if num_nodes == 1:
                pos[sorted_nodes[0]] = (x, 0)
            else:
                y_range = 8
                y_step = y_range / (num_nodes - 1) if num_nodes > 1 else 1
                for i, node in enumerate(sorted_nodes):
                    y = -4 + i * y_step
                    pos[node] = (x, y)
        
        return pos

    def visualize(self, start, distances=None, path=None, algorithm="Dijkstra", visit_order=None, generic_title=False):
        """Visualize the graph with matplotlib"""
        G = nx.DiGraph()  # Use directed graph
        
        # Add edges with weight attribute
        for u in self.graph:
            for v, weight in self.graph[u]:
                G.add_edge(u, v, weight=weight)
        
        # Create layout - use hierarchical for tree-like structures
        try:
            pos = self.hierarchical_layout(G)
        except:
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                # Fallback to circular layout if kamada_kawai fails
                pos = nx.circular_layout(G)
        
        # Draw graph with light mode
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='#f5f5f5')
        ax.set_facecolor('#f5f5f5')
        
        # Draw edges FIRST (so they go behind nodes)
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            weight = G[u][v]['weight']
            if weight < 0:
                edge_colors.append('#ff4444')  # Red for negative
                edge_widths.append(2)
            else:
                edge_colors.append('#00aa00')  # Green for positive
                edge_widths.append(1.5)
        
        # Highlight path if provided
        if path:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            # Draw all non-path edges as neutral light gray
            all_edges = list(G.edges())
            non_path_edges = [e for e in all_edges if e not in path_edges and (e[1], e[0]) not in path_edges]
            non_path_colors = ['#cccccc'] * len(non_path_edges)
            non_path_widths = [1.5] * len(non_path_edges)
            
            nx.draw_networkx_edges(G, pos, edgelist=non_path_edges, width=non_path_widths, 
                                 edge_color=non_path_colors, ax=ax, arrows=True, arrowsize=10, arrowstyle='->')
            
            # Draw path edges with dark-to-light green gradient and dotted style
            num_edges = len(path_edges)
            for i, (u, v) in enumerate(path_edges):
                # Gradient position (0 to 1)
                t = i / max(num_edges - 1, 1)
                
                # Gradient: dark green (#004400) to light green (#00ff88)
                r = int(0)
                g = int(0x44 + (0xff - 0x44) * t)
                b = int(0x00 + (0x88 - 0x00) * t)
                
                color_hex = f'#{r:02x}{g:02x}{b:02x}'
                
                # Draw with dotted style and gradient color
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3, edge_color=color_hex, 
                                     style='dotted', ax=ax, arrows=True, arrowsize=12, arrowstyle='->')
        else:
            nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), width=edge_widths, 
                                 edge_color=edge_colors, ax=ax, arrows=True, arrowsize=10, arrowstyle='->')
        
        # Draw nodes AFTER edges (so they appear on top)
        nx.draw_networkx_nodes(G, pos, node_color='#2e7d32', node_size=800, ax=ax, edgecolors='#ffffff', linewidths=2)
        
        # Highlight start node
        if start:
            nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='#558b2f', node_size=800, ax=ax, edgecolors='#ffffff', linewidths=2)
        
        # Highlight path if provided
        if path:
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='#f9a825', node_size=800, ax=ax, edgecolors='#ffffff', linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', font_color='#ffffff', ax=ax)
        
        # Draw visit order numbers if provided
        if visit_order:
            for node, order in visit_order.items():
                x, y = pos[node]
                ax.text(x - 0.08, y + 0.08, str(order), fontsize=10, weight='bold',
                       bbox=dict(boxstyle='circle,pad=0.2', facecolor='#ffaa00', edgecolor='#000000', linewidth=1),
                       color='#000000', ha='center', va='center')
        
        # Draw edge weights with offset to avoid overlap
        edge_labels = nx.get_edge_attributes(G, 'weight')
        self.draw_edge_labels_with_offset(G, pos, ax, edge_labels, font_size=12)
        
        if generic_title:
            title = "Dijkstra's Algorithm vs Bellman-Ford"
        else:
            title = f"{algorithm}'s Algorithm Visualization"
        ax.set_title(title, fontsize=16, fontweight='bold', color='#000000')
        ax.axis('off')
        plt.tight_layout()
        return plt


def get_preset_graphs():
    """Return preset graph configurations for testing"""
    presets = {
        '1': {
            'name': 'Easy - Simple Negative Edge',
            'edges': [
                ('A', 'B', 1),
                ('A', 'C', 5),
                ('B', 'D', 1),
                ('C', 'B', -8),
                ('C', 'D', 7),
            ]
        },
        '2': {
            'name': 'Medium - Multiple Negative Edges',
            'edges': [
                ('A', 'B', 1),
                ('A', 'C', 5),
                ('B', 'D', 1),
                ('C', 'B', -10),
                ('C', 'D', 4),
                ('B', 'E', 2),
                ('D', 'F', 3),
                ('E', 'F', -5),
            ]
        },
        '3': {
            'name': 'Hard - Complex with Multiple Paths',
            'edges': [
                ('A', 'B', 4),
                ('A', 'C', 2),
                ('B', 'C', -3),
                ('B', 'D', 2),
                ('C', 'D', 4),
                ('C', 'E', 2),
                ('D', 'E', -1),
                ('D', 'F', 1),
                ('E', 'F', 3),
            ]
        },
        '4': {
            'name': 'Extreme - 20 Nodes with Negative Edges',
            'edges': [
                ('A', 'B', 3), ('B', 'C', -2), ('C', 'D', 4), ('D', 'E', -1),
                ('E', 'F', 2), ('F', 'G', -3), ('G', 'H', 5), ('H', 'I', -4),
                ('I', 'J', 1), ('J', 'K', -2), ('K', 'L', 3), ('L', 'M', -1),
                ('M', 'N', 2), ('N', 'O', -3), ('O', 'P', 4), ('P', 'Q', -2),
                ('Q', 'R', 1), ('R', 'S', -1), ('S', 'T', 3), ('T', 'A', -2),
                ('A', 'C', 6), ('B', 'D', -4), ('E', 'G', 7), ('H', 'J', -5),
                ('K', 'M', 8), ('N', 'P', -6), ('Q', 'S', 9), ('T', 'B', -7),
                ('A', 'F', 10), ('C', 'H', -8), ('G', 'K', 12), ('M', 'Q', -9)
            ]
        },
        '5': {
            'name': 'Complex Graph',
            'edges': [
                ('A','Ba',1),('A','Bb',1),

                ('Ba','Ca',1),('Ba','Ia',1),
                ('Ca','Da',1),('Ca','Fa',1),
                ('Da','Fa',1),('Da','Ea',1),
                ('Fa','Ga',1),('Ea','Ga',1),
                ('Ea','Ha',1),('Ga','Ha',1),
                ('Ia','Ja',1),('Ia','La',1),
                ('Ja','La',1),('Ja','Ka',1),
                ('La','Ma',1),('Ka','Ma',1),
                ('Ka','Na',1),('Ma','Na',1),
                ('Ha','Oa',1),('Na','Oa',1),

                ('Bb','Cb',1),('Bb','Ib',1),
                ('Cb','Db',1),('Cb','Fb',1),
                ('Db','Fb',1),('Db','Eb',1),
                ('Fb','Gb',1),('Eb','Gb',1),
                ('Eb','Hb',1),('Gb','Hb',1),
                ('Ib','Jb',1),('Ib','Lb',1),
                ('Jb','Lb',1),('Jb','Kb',1),
                ('Lb','Mb',1),('Kb','Mb',1),
                ('Kb','Nb',1),('Mb','Nb',1),
                ('Hb','Ob',1),('Nb','Ob',1),

                ('Oa','P',1),('Ob','P',1)

            ]
        },
        '6': {
            'name': 'Simple Complex',
            'edges': [
                ('A', 'B', 1),
                ('A', 'C', 1),
                ('B', 'D', 1),
                ('C', 'D', 1),
                ('D', 'E', 1),
                ('D', 'F', 1),
                ('E', 'G', 1),
                ('F', 'G', 1)
            ]
        },
        '7': {
            'name': 'Custom - Enter your own edges',
            'edges': None  # Will prompt user
        }
    }
    return presets


def select_graph():
    """Display graph selection menu and return edge list"""
    # Check if running interactively
    if not sys.stdin.isatty():
        # Non-interactive mode - use Medium preset
        return get_preset_graphs()['2']['edges']

    presets = get_preset_graphs()

    print("\n" + "=" * 60)
    print("SELECT GRAPH PRESET")
    print("=" * 60)
    for key, preset in presets.items():
        print(f"  {key}. {preset['name']}")

    # Generate valid choices dynamically
    valid_choices = list(presets.keys())
    choice_range = f"1-{len(presets)}"

    while True:
        choice = input(f"\nEnter your choice ({choice_range}): ").strip()
        if choice in presets:
            preset = presets[choice]
            print(f"\nSelected: {preset['name']}")
            return preset['edges']
        else:
            print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")


def save_output_to_file(filename, content):
    """Save algorithm output to a text file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"✗ Error saving to file: {e}")
        return False


def send_to_discord(webhook_url, image_paths, message="", algorithm_data=""):
    """Send images and algorithm data to Discord webhook"""
    if not webhook_url:
        print("✗ No webhook URL provided")
        return False
    
    try:
        file_contents = {}  # Dictionary for file uploads
        
        # Prepare image files
        for i, path in enumerate(image_paths):
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    file_contents[f'files[{i}]'] = (os.path.basename(path), f.read())
                print(f"✓ Loaded image: {path}")
            else:
                print(f"✗ File not found: {path}")
        
        # Prepare algorithm data file if provided
        if algorithm_data:
            algorithm_file_path = "algorithm_steps.txt"
            try:
                with open(algorithm_file_path, 'w', encoding='utf-8') as f:
                    f.write(algorithm_data)
                
                with open(algorithm_file_path, 'rb') as f:
                    file_contents[f'files[{len(file_contents)}]'] = (os.path.basename(algorithm_file_path), f.read())
                print(f"✓ Loaded algorithm data: {algorithm_file_path}")
            except Exception as e:
                print(f"✗ Error preparing algorithm data: {e}")
        
        if not file_contents:
            print("✗ No files to send")
            return False
        
        print(f"\nPreparing to send {len(file_contents)} file(s) to Discord...")

        # Send just the summary message
        full_message = message if message else "Algorithm comparison completed. See attached files for details."
        payload = {'content': full_message}
        print(f"Message: {full_message}")

        # Send with files
        response = requests.post(webhook_url, data=payload, files=file_contents, timeout=30)

        if response.status_code in [200, 204]:
            print("✓ All files sent to Discord successfully!")
            return True
        else:
            print(f"✗ Failed to send to Discord: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error sending to Discord: {e}")
        return False


def main():
    """Main demonstration"""
    # Load environment variables from .env file
    load_dotenv()
    print("=" * 60)
    print("SHORTEST PATH ALGORITHMS COMPARISON")
    print("Dijkstra's Algorithm vs Bellman-Ford")
    print("=" * 60)
    
    # Create a sample graph
    g = ShortestPathGraph()
    
    # Get graph from preset or user input
    edges = select_graph()
    
    print("\nGraph Edges:")
    for u, v, w in edges:
        g.add_edge(u, v, w)
        # Get the actual weight (in case it was random)
        actual_weight = [weight for node, weight in g.graph[u] if node == v][0]
        display_weight = w if not (isinstance(w, str) and w.lower() == 'random') else f"random ({actual_weight})"
        print(f"  {u} -- {v} (weight: {display_weight})")
    
    # Ask user for start and end nodes
    available_nodes = sorted(list(g.nodes))
    print(f"\nAvailable nodes: {', '.join(available_nodes)}")
        
    # Use 'A' as default start if available, otherwise first node
    default_start = 'A' if 'A' in available_nodes else available_nodes[0]
    default_end = available_nodes[-1]  # largest label

    while True:
        user_start = input(f"Enter start node (default: {default_start}): ").strip().upper() or default_start
        if user_start in available_nodes:
            start_node = user_start
            break
        print(f"✗ Invalid node. Choose from: {', '.join(available_nodes)}")

    while True:
        user_end = input(f"Enter end node (default: {default_end}): ").strip().upper() or default_end
        if user_end in available_nodes:
            end_node = user_end
            break
        print(f"✗ Invalid node. Choose from: {', '.join(available_nodes)}")
    
    image_paths = []
    
    # ===== DIJKSTRA'S ALGORITHM =====
    print(f"\n{'=' * 60}")
    print("DIJKSTRA'S ALGORITHM")
    print(f"{'=' * 60}")
    
    dijkstra_distances, dijkstra_parent, dijkstra_steps, dijkstra_ops, dijkstra_visit = g.dijkstra(start_node)
    
    # Display results
    print(f"\nShortest distances from '{start_node}':")
    for node in sorted(dijkstra_distances.keys()):
        dist = dijkstra_distances[node]
        if dist == float('inf'):
            print(f"  {start_node} → {node}: ∞ (unreachable)")
        else:
            print(f"  {start_node} → {node}: {dist}")
    
    print(f"\n📋 Dijkstra Execution Steps: ({dijkstra_ops} operations)")
    for step in dijkstra_steps:
        print(f"  {step}")
    
    print(f"\nDijkstra path from {start_node} to {end_node}:")
    dijkstra_path = g.get_path(dijkstra_parent, start_node, end_node)
    if dijkstra_path:
        path_str = " → ".join(dijkstra_path)
        path_cost = dijkstra_distances[end_node]
        print(f"  {path_str} (cost: {path_cost})")
    else:
        print(f"  No path found")
    
    # Visualize graph (same for both algorithms)
    print(f"\nGenerating graph visualization...")
    plt_obj = g.visualize(start_node, algorithm="Dijkstra", visit_order=dijkstra_visit, generic_title=True)
    plt_obj.savefig('shortest_path_graph.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to 'shortest_path_graph.png'")
    image_paths.append('shortest_path_graph.png')
    
    # Visualize Dijkstra path
    print(f"Generating Dijkstra path visualization ({start_node} → {end_node})...")
    plt_obj = g.visualize(start_node, dijkstra_distances, dijkstra_path, algorithm="Dijkstra", visit_order=dijkstra_visit)
    plt_obj.savefig('shortest_path_dijkstra_path.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to 'shortest_path_dijkstra_path.png'")
    image_paths.append('shortest_path_dijkstra_path.png')
    
    # ===== BELLMAN-FORD ALGORITHM =====
    print(f"\n{'=' * 60}")
    print("BELLMAN-FORD ALGORITHM")
    print(f"{'=' * 60}")
    
    bellman_distances, bellman_parent, negative_cycle, bellman_steps, bellman_iters = g.bellman_ford(start_node)
    
    if negative_cycle:
        print("⚠ WARNING: Negative cycle detected! Path reconstruction skipped.")
        bellman_path = None
    else:
        bellman_path = g.get_path(bellman_parent, start_node, end_node)
    
    # Display results
    print(f"\nShortest distances from '{start_node}':")
    for node in sorted(bellman_distances.keys()):
        dist = bellman_distances[node]
        if dist == float('inf'):
            print(f"  {start_node} → {node}: ∞ (unreachable)")
        else:
            print(f"  {start_node} → {node}: {dist}")
    
    print(f"\n📋 Bellman-Ford Execution Steps: ({bellman_iters} iterations)")
    for step in bellman_steps:
        print(f"  {step}")
    
    print(f"\nBellman-Ford path from {start_node} to {end_node}:")
    if bellman_path:
        path_str = " → ".join(bellman_path)
        path_cost = bellman_distances[end_node]
        print(f"  {path_str} (cost: {path_cost})")
    else:
        print(f"  No path found")
    
    # Generate and display comparison table
    print(f"\n{'=' * 60}")
    comparison_table = g.generate_comparison_table(dijkstra_distances, bellman_distances)
    print(comparison_table)
    
    # Generate side-by-side comparison image
    print(f"\nGenerating side-by-side comparison image...")
    plt_obj = g.generate_comparison_image(start_node, dijkstra_distances, dijkstra_path, bellman_distances, bellman_path, dijkstra_visit)
    plt_obj.savefig('shortest_path_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to 'shortest_path_comparison.png'")
    image_paths.append('shortest_path_comparison.png')
    
    # Check if results differ and show warning banner
    has_differences = False
    for node in dijkstra_distances.keys():
        if dijkstra_distances.get(node) != bellman_distances.get(node):
            has_differences = True
            break
    
    if has_differences:
        print("\n" + "=" * 60)
        print("⚠️  ALGORITHMIC DIFFERENCE DETECTED!")
        print("=" * 60)
        print(f"Dijkstra found suboptimal paths due to negative edges.")
        print(f"Bellman-Ford correctly handled all negative weight edges.")
        print(f"Dijkstra operates greedily and cannot revisit nodes after committing.")
        print(f"Bellman-Ford relaxes edges iteratively, allowing correction.")
        print("=" * 60)
    
    # Visualize Bellman-Ford path
    print(f"Generating Bellman-Ford path visualization ({start_node} → {end_node})...")
    plt_obj = g.visualize(start_node, bellman_distances, bellman_path, algorithm="Bellman-Ford")
    plt_obj.savefig('shortest_path_bellman_ford_path.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to 'shortest_path_bellman_ford_path.png'")
    image_paths.append('shortest_path_bellman_ford_path.png')
    
    print(f"\n{'=' * 60}")
    print("Done! Check the generated PNG files for visualizations.")
    print("=" * 60)
    
    # Save output to file
    output_filename = "algorithm_output.txt"
    full_output = f"""SHORTEST PATH ALGORITHMS COMPARISON
Dijkstra's Algorithm vs Bellman-Ford
{'=' * 60}

GRAPH EDGES:
"""
    for u, v, w in edges:
        full_output += f"  {u} → {v} (weight: {w})\n"
    
    full_output += f"\n{'=' * 60}\nDIJKSTRA'S ALGORITHM\n{'=' * 60}\n"
    full_output += f"Shortest distances from '{start_node}':\n"
    for node in sorted(dijkstra_distances.keys()):
        dist = dijkstra_distances[node]
        if dist == float('inf'):
            full_output += f"  {start_node} → {node}: ∞ (unreachable)\n"
        else:
            full_output += f"  {start_node} → {node}: {dist}\n"
    
    full_output += f"\nExecution Steps: ({dijkstra_ops} operations)\n"
    for step in dijkstra_steps:
        full_output += f"  {step}\n"
    
    full_output += f"\nPath from {start_node} to {end_node}: "
    if dijkstra_path:
        full_output += f"{' → '.join(dijkstra_path)} (cost: {dijkstra_distances[end_node]})\n"
    else:
        full_output += "No path found\n"
    
    full_output += f"\n{'=' * 60}\nBELLMAN-FORD ALGORITHM\n{'=' * 60}\n"
    full_output += f"Shortest distances from '{start_node}':\n"
    for node in sorted(bellman_distances.keys()):
        dist = bellman_distances[node]
        if dist == float('inf'):
            full_output += f"  {start_node} → {node}: ∞ (unreachable)\n"
        else:
            full_output += f"  {start_node} → {node}: {dist}\n"
    
    full_output += f"\nExecution Steps: ({bellman_iters} iterations)\n"
    for step in bellman_steps:
        full_output += f"  {step}\n"
    
    full_output += f"\nPath from {start_node} to {end_node}: "
    if bellman_path:
        full_output += f"{' → '.join(bellman_path)} (cost: {bellman_distances[end_node]})\n"
    else:
        full_output += "No path found\n"
    
    full_output += f"\n{comparison_table}\n"
    
    if has_differences:
        full_output += f"\n{'=' * 60}\n⚠️  ALGORITHMIC DIFFERENCE DETECTED!\n{'=' * 60}\n"
        full_output += "Dijkstra found suboptimal paths due to negative edges.\n"
        full_output += "Bellman-Ford correctly handled all negative weight edges.\n"
        full_output += "Dijkstra operates greedily and cannot revisit nodes after committing.\n"
        full_output += "Bellman-Ford relaxes edges iteratively, allowing correction.\n"
    
    if save_output_to_file(output_filename, full_output):
        print(f"✓ Output saved to '{output_filename}'")
    
    # Send to Discord if webhook URL is configured
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL', '').strip()
    
    if not webhook_url:
        # Prompt if not in .env
        user_input = input("\nEnter Discord webhook URL (or press Enter to skip): ").strip()
        if user_input:
            webhook_url = user_input
    
    if webhook_url:
        print("\nSending all visualizations to Discord...")
        
        # Build comprehensive algorithm data string
        algorithm_output = "DIJKSTRA'S ALGORITHM\n"
        algorithm_output += "=" * 40 + f" ({dijkstra_ops} operations)\n"
        for step in dijkstra_steps:
            algorithm_output += step + "\n"
        algorithm_output += f"\nResult: {start_node} → {end_node} = {dijkstra_distances[end_node]}\n"
        if dijkstra_path:
            algorithm_output += f"Path: {' → '.join(dijkstra_path)}\n"
        
        algorithm_output += "\n\nBELLMAN-FORD ALGORITHM\n"
        algorithm_output += "=" * 40 + f" ({bellman_iters} iterations)\n"
        for step in bellman_steps:
            algorithm_output += step + "\n"
        algorithm_output += f"\nResult: {start_node} → {end_node} = {bellman_distances[end_node]}\n"
        if bellman_path:
            algorithm_output += f"Path: {' → '.join(bellman_path)}\n"
        
        # Create summary line
        summary_line = f"**Results: Dijkstra = {dijkstra_distances[end_node]} | Bellman-Ford = {bellman_distances[end_node]}**"
        if has_differences:
            diff = bellman_distances[end_node] - dijkstra_distances[end_node]
            summary_line += f" | Difference: {diff}"
        
        send_to_discord(webhook_url, image_paths, 
                       message=f"**Shortest Path Algorithms - Dijkstra vs Bellman-Ford**\n{summary_line}",
                       algorithm_data=algorithm_output)
    else:
        print("\n(Skipped Discord - no webhook URL provided)")


if __name__ == "__main__":
    main()
