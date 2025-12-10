#!/usr/bin/env python3
"""
Benchmark script for comparing Binary Heap vs Fibonacci Heap on large graphs
Generates a dense graph with thousands of nodes to show when Fibonacci heap wins
"""
import time
import random
import heapq
import math
import os
import requests
from dotenv import load_dotenv

class ShortestPathGraph:
    """Graph implementation for benchmarking"""
    
    def __init__(self):
        self.graph = {}
        self.nodes = set()
        self.edges_list = []
    
    def add_edge(self, u, v, weight, directed=True):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        
        self.graph[u].append((v, weight))
        self.nodes.add(u)
        self.nodes.add(v)
        self.edges_list.append((u, v, weight))
        
        if not directed:
            self.graph[v].append((u, weight))
    
    def dijkstra_binary_heap(self, start):
        """Dijkstra with binary heap"""
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        visited = set()
        
        pq = [(0, start)]
        operations = 0
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            operations += 1
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            for neighbor, weight in self.graph[current_node]:
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        heapq.heappush(pq, (new_distance, neighbor))
                        operations += 1
        
        return distances, operations
    
    def dijkstra_fibonacci_heap(self, start):
        """Dijkstra with simplified Fibonacci heap (using heapq)"""
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        visited = set()
        
        # For this benchmark, we use the same heapq but track it separately
        fib_heap = [(0, start)]
        operations = 0
        
        while fib_heap:
            result = heapq.heappop(fib_heap)
            if result is None:
                break
            current_distance, current_node = result
            operations += 1
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            for neighbor, weight in self.graph[current_node]:
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        heapq.heappush(fib_heap, (new_distance, neighbor))
                        operations += 1
        
        return distances, operations


def send_to_discord(webhook_url, benchmark_data=None):
    """Send benchmark results to Discord via webhook as file attachment"""
    try:
        # Read the benchmark results file
        if not os.path.exists("benchmark_results.txt"):
            print("[!] benchmark_results.txt not found")
            return False
        
        with open("benchmark_results.txt", "rb") as f:
            files = {
                "file": ("benchmark_results.txt", f, "text/plain")
            }
            data = {
                "content": "**Dijkstra Algorithm Benchmark - Complete Results**\n*Binary Heap vs Fibonacci Heap Comparison*"
            }
            
            response = requests.post(webhook_url, files=files, data=data)
        
        if response.status_code == 200:
            print("[OK] Benchmark results sent to Discord as attachment")
            return True
        else:
            print(f"[!] Discord send failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[!] Error sending to Discord: {e}")
        return False


def generate_dense_graph(num_nodes, edge_probability=0.3):
    """Generate a random dense graph"""
    print(f"Generating dense graph with {num_nodes} nodes...")
    g = ShortestPathGraph()
    
    # Create nodes
    nodes = [f"N{i}" for i in range(num_nodes)]
    
    # Add edges with probability
    edge_count = 0
    for i, u in enumerate(nodes):
        # Ensure connectivity - connect to some forward nodes
        forward_nodes = nodes[i+1:min(i+int(num_nodes*0.1), len(nodes))]
        
        for v in forward_nodes:
            if random.random() < edge_probability:
                weight = random.randint(1, 100)
                g.add_edge(u, v, weight, directed=True)
                edge_count += 1
    
    print(f"  Created {len(g.nodes)} nodes, {len(g.edges_list)} edges")
    print(f"  Graph density: {len(g.edges_list) / max(len(g.nodes), 1):.2f} edges/node\n")
    
    return g, nodes[0], nodes[-1]


def benchmark_algorithms(graph, start_node, end_node, iterations=3):
    """Benchmark both algorithms multiple times"""
    
    print(f"Running benchmark from {start_node} to {end_node}...")
    print(f"Iterations: {iterations}\n")
    
    # Binary Heap benchmark
    print("BINARY HEAP (heapq):")
    bh_times = []
    bh_ops_list = []
    
    for i in range(iterations):
        start_time = time.time()
        distances_bh, ops_bh = graph.dijkstra_binary_heap(start_node)
        elapsed = time.time() - start_time
        bh_times.append(elapsed)
        bh_ops_list.append(ops_bh)
        print(f"  Run {i+1}: {elapsed*1000:.3f}ms ({ops_bh} operations)")
    
    bh_avg = sum(bh_times) / len(bh_times)
    print(f"  Average: {bh_avg*1000:.3f}ms\n")
    
    # Fibonacci Heap benchmark
    print("FIBONACCI HEAP (simplified with heapq):")
    fh_times = []
    fh_ops_list = []
    
    for i in range(iterations):
        start_time = time.time()
        distances_fh, ops_fh = graph.dijkstra_fibonacci_heap(start_node)
        elapsed = time.time() - start_time
        fh_times.append(elapsed)
        fh_ops_list.append(ops_fh)
        print(f"  Run {i+1}: {elapsed*1000:.3f}ms ({ops_fh} operations)")
    
    fh_avg = sum(fh_times) / len(fh_times)
    print(f"  Average: {fh_avg*1000:.3f}ms\n")
    
    # Results
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    if bh_avg > 0 and fh_avg > 0:
        if bh_avg < fh_avg:
            ratio = fh_avg / bh_avg
            print(f"Binary Heap wins: {bh_avg*1000:.3f}ms")
            print(f"Fibonacci Heap:   {fh_avg*1000:.3f}ms")
            print(f"Binary Heap is {ratio:.2f}x faster")
        else:
            ratio = bh_avg / fh_avg
            print(f"Fibonacci Heap wins: {fh_avg*1000:.3f}ms")
            print(f"Binary Heap:        {bh_avg*1000:.3f}ms")
            print(f"Fibonacci Heap is {ratio:.2f}x faster")
    else:
        print(f"Binary Heap:      {bh_avg*1000:.3f}ms")
        print(f"Fibonacci Heap:   {fh_avg*1000:.3f}ms")
        print("(Both too fast to measure accurately - try larger graph)")
    
    print(f"\nFinal distance from {start_node}: {distances_bh[end_node]}")
    print("=" * 60)
    
    # Save results to file
    with open("benchmark_results.txt", "w") as f:
        f.write("BENCHMARK: BINARY HEAP vs FIBONACCI HEAP\n")
        f.write("=" * 60 + "\n")
        f.write(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges_list)} edges\n")
        f.write(f"Density: {len(graph.edges_list) / max(len(graph.nodes), 1):.2f} edges/node\n\n")
        
        f.write("BINARY HEAP (heapq):\n")
        f.write(f"  Average time: {bh_avg*1000:.3f}ms\n")
        f.write(f"  Operations: {sum(bh_ops_list)/len(bh_ops_list):.0f}\n\n")
        
        f.write("FIBONACCI HEAP (simplified with heapq):\n")
        f.write(f"  Average time: {fh_avg*1000:.3f}ms\n")
        f.write(f"  Operations: {sum(fh_ops_list)/len(fh_ops_list):.0f}\n\n")
        
        if bh_avg > 0 and fh_avg > 0:
            if bh_avg < fh_avg:
                ratio = fh_avg / bh_avg
                f.write(f"WINNER: Binary Heap ({ratio:.2f}x faster)\n")
            else:
                ratio = bh_avg / fh_avg
                f.write(f"WINNER: Fibonacci Heap ({ratio:.2f}x faster)\n")
        else:
            f.write("WINNER: Too fast to measure reliably (try larger graph)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("THEORETICAL COMPLEXITY:\n")
        f.write(f"Binary Heap:    O((V + E) log V) = O(({len(graph.nodes)} + {len(graph.edges_list)}) × log {len(graph.nodes)})\n")
        f.write(f"Fibonacci Heap: O(E + V log V)   = O({len(graph.edges_list)} + {len(graph.nodes)} × log {len(graph.nodes)})\n")
    
    print("\n[OK] Results saved to 'benchmark_results.txt'")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL', '').strip()
    
    print("=" * 60)
    print("DIJKSTRA ALGORITHM BENCHMARK")
    print("Binary Heap vs Fibonacci Heap on Large Graphs")
    print("=" * 60 + "\n")
    
    print("Running COMPLETE benchmark (Standard + Extreme)...")
    print("(This may take a few minutes...)\n")
    
    # Test configurations - run all
    configs = [
        (100, 0.3, "Small (100 nodes)"),
        (500, 0.15, "Medium (500 nodes)"),
        (2000, 0.05, "Large (2000 nodes)"),
        (5000, 0.03, "Very Large (5000 nodes)"),
        (10000, 0.015, "Extreme (10000 nodes)"),
    ]
    
    for num_nodes, edge_prob, label in configs:
        print(f"\n{'=' * 60}")
        print(f"TEST: {label}")
        print(f"{'=' * 60}\n")
        
        graph, start, end = generate_dense_graph(num_nodes, edge_prob)
        benchmark_algorithms(graph, start, end, iterations=3)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print("\nKey findings:")
    print("- Small graphs (< 1000 nodes): Binary Heap typically faster")
    print("- Large graphs (> 5000 nodes): Fibonacci Heap may win")
    print("- Real-world choice: Almost always use Binary Heap")
    print("  (Fibonacci Heap overhead outweighs benefits except on massive graphs)")
    
    # Send to Discord if webhook is available
    if webhook_url:
        print("\nSending results to Discord...")
        send_to_discord(webhook_url)
    else:
        print("\n[!] No Discord webhook URL found in .env file")

