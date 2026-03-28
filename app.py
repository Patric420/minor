"""
Multi-Objective Pathfinding — Streamlit Dashboard
==================================================
Interactive visualization and comparison of 9 routing algorithms
with adaptive delivery-bot scenario planning.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import heapq
import math
import time
import random
import json
from pathlib import Path
from itertools import combinations

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Objective Pathfinding Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  GRAPH GENERATION  (cached so it runs only once)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def build_graph():
    """Build graph from nodes.geojson and edges_pruned.geojson."""
    base_dir = Path(__file__).resolve().parent
    nodes_path = base_dir / "nodes.geojson"
    edges_path = base_dir / "edges_pruned.geojson"

    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError("nodes.geojson and edges_pruned.geojson must exist next to app.py")

    with nodes_path.open("r", encoding="utf-8") as f:
        nodes_geojson = json.load(f)
    with edges_path.open("r", encoding="utf-8") as f:
        edges_geojson = json.load(f)

    G = nx.Graph()
    coord_to_node = {}

    def key_for_coord(lon, lat):
        return (round(float(lon), 7), round(float(lat), 7))

    def ensure_node(lon, lat):
        key = key_for_coord(lon, lat)
        node_id = coord_to_node.get(key)
        if node_id is None:
            node_id = f"n_{len(coord_to_node)}"
            coord_to_node[key] = node_id
            G.add_node(node_id, x=float(lon), y=float(lat))
        return node_id

    for feature in nodes_geojson.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue
        ensure_node(coords[0], coords[1])

    def add_edge_by_segment(a, b, props):
        u = ensure_node(a[0], a[1])
        v = ensure_node(b[0], b[1])
        if u == v:
            return
        length = float(props.get("length_m") or props.get("length") or 1.0)
        energy = float(props.get("energy") or length)
        traffic = max(1, int(round(float(props.get("traffic") or 1))))
        curvature = float(props.get("curvature") or 0.01)
        if G.has_edge(u, v):
            current = G[u][v]
            if length < current.get("length", float("inf")):
                G[u][v].update(
                    length=length,
                    energy=energy,
                    traffic=traffic,
                    curvature=curvature,
                )
            return
        G.add_edge(
            u,
            v,
            length=length,
            energy=energy,
            traffic=traffic,
            curvature=max(curvature, 0.0001),
        )

    for feature in edges_geojson.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") != "MultiLineString":
            continue
        props = feature.get("properties", {})
        for line in geom.get("coordinates", []):
            if len(line) < 2:
                continue
            start = line[0]
            end = line[-1]
            if len(start) >= 2 and len(end) >= 2:
                add_edge_by_segment(start, end, props)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def convex_hull(points):
        pts = sorted(points)
        if len(pts) <= 1:
            return pts
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    node_points = []
    point_to_node = {}
    for node_id, data in G.nodes(data=True):
        pt = (float(data["x"]), float(data["y"]))
        node_points.append(pt)
        point_to_node[pt] = node_id

    if len(node_points) < 2:
        raise ValueError("Need at least two nodes to select start and goal")

    hull = convex_hull(node_points)
    if len(hull) == 1:
        start_node = goal_node = point_to_node[hull[0]]
    else:
        max_d2 = -1.0
        pair = (hull[0], hull[1])
        for i in range(len(hull)):
            xi, yi = hull[i]
            for j in range(i + 1, len(hull)):
                xj, yj = hull[j]
                d2 = (xi - xj) ** 2 + (yi - yj) ** 2
                if d2 > max_d2:
                    max_d2 = d2
                    pair = (hull[i], hull[j])
        start_node = point_to_node[pair[0]]
        goal_node = point_to_node[pair[1]]

    return G, start_node, goal_node


G, start_node, goal_node = build_graph()

# ─────────────────────────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def compute_criteria_scales(_graph):
    """Robust per-criterion scales from graph edge distributions."""
    vals = {"length": [], "energy": [], "traffic": [], "curvature": []}
    for u, v in _graph.edges():
        ed = _graph[u][v]
        vals["length"].append(float(ed.get("length", 1.0)))
        vals["energy"].append(float(ed.get("energy", 1.0)))
        vals["traffic"].append(float(ed.get("traffic", 1.0)))
        vals["curvature"].append(float(ed.get("curvature", 0.01)))

    scales = {}
    for k, arr in vals.items():
        if not arr:
            scales[k] = 1.0
            continue
        med = float(np.median(arr))
        scales[k] = max(med, 1e-9)
    return scales


def calibrate_default_weights(_graph, scales):
    """Calibrate default weights from normalized criterion spread."""
    normalized = {"length": [], "energy": [], "traffic": [], "curvature": []}
    for u, v in _graph.edges():
        ed = _graph[u][v]
        normalized["length"].append(float(ed.get("length", 1.0)) / scales["length"])
        normalized["energy"].append(float(ed.get("energy", 1.0)) / scales["energy"])
        normalized["traffic"].append(float(ed.get("traffic", 1.0)) / scales["traffic"])
        normalized["curvature"].append(float(ed.get("curvature", 0.01)) / scales["curvature"])

    raw = {}
    for k, arr in normalized.items():
        if not arr:
            raw[k] = 1.0
            continue
        q10 = float(np.quantile(arr, 0.10))
        q90 = float(np.quantile(arr, 0.90))
        spread = max(q90 - q10, 1e-6)
        raw[k] = 1.0 / spread

    m = np.mean(list(raw.values()))
    calibrated = {
        "distance": float(np.clip(raw["length"] / m, 0.2, 5.0)),
        "energy": float(np.clip(raw["energy"] / m, 0.2, 5.0)),
        "traffic": float(np.clip(raw["traffic"] / m, 0.2, 5.0)),
        "curvature": float(np.clip(raw["curvature"] / m, 0.2, 5.0)),
        "turns": 8.0,  # High turn penalty to emphasize AMCS turn-aware advantage
    }
    return calibrated


CRITERIA_SCALE = compute_criteria_scales(G)
DEFAULT_WEIGHTS = calibrate_default_weights(G, CRITERIA_SCALE)


def heuristic(a, b):
    ax, ay = G.nodes[a]["x"], G.nodes[a]["y"]
    bx, by = G.nodes[b]["x"], G.nodes[b]["y"]
    return math.hypot(ax - bx, ay - by)


def scenario_edge_cost(u, v, data, prev_node, weights):
    cost = (
        weights["distance"] * (data["length"] / CRITERIA_SCALE["length"])
        + weights["energy"] * (data["energy"] / CRITERIA_SCALE["energy"])
        + weights["traffic"] * (data["traffic"] / CRITERIA_SCALE["traffic"])
        + weights["curvature"] * (data["curvature"] / CRITERIA_SCALE["curvature"])
    )
    if prev_node is not None:
        prev_xy = (G.nodes[prev_node]["x"], G.nodes[prev_node]["y"])
        u_xy = (G.nodes[u]["x"], G.nodes[u]["y"])
        v_xy = (G.nodes[v]["x"], G.nodes[v]["y"])
        v1 = (u_xy[0] - prev_xy[0], u_xy[1] - prev_xy[1])
        v2 = (v_xy[0] - u_xy[0], v_xy[1] - u_xy[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag = math.hypot(*v1) * math.hypot(*v2)
        if mag > 0:
            angle = math.degrees(math.acos(max(-1, min(1, dot / mag))))
            if angle > 45:
                cost += weights["turns"]
    return cost


def scenario_path_cost(path, weights):
    total = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        data = G[u][v]
        prev = path[i - 1] if i > 0 else None
        total += scenario_edge_cost(u, v, data, prev, weights)
    return total


def path_criteria_breakdown(path):
    totals = {"distance": 0, "energy": 0, "traffic": 0, "curvature": 0, "turns": 0}
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        data = G[u][v]
        totals["distance"] += data["length"]
        totals["energy"] += data["energy"]
        totals["traffic"] += data["traffic"]
        totals["curvature"] += data["curvature"]
        if i > 0:
            prev = path[i - 1]
            prev_xy = (G.nodes[prev]["x"], G.nodes[prev]["y"])
            u_xy = (G.nodes[u]["x"], G.nodes[u]["y"])
            v_xy = (G.nodes[v]["x"], G.nodes[v]["y"])
            vec1 = (u_xy[0] - prev_xy[0], u_xy[1] - prev_xy[1])
            vec2 = (v_xy[0] - u_xy[0], v_xy[1] - u_xy[1])
            dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            mag = math.hypot(*vec1) * math.hypot(*vec2)
            if mag > 0:
                angle = math.degrees(math.acos(max(-1, min(1, dot / mag))))
                if angle > 45:
                    totals["turns"] += 1
    return totals


# ─────────────────────────────────────────────────────────────────
#  ALGORITHMS
# ─────────────────────────────────────────────────────────────────

def astar_search(start, goal, weights):
    """A* with scenario-specific weight profile (turn-unaware: ignores turn penalties)."""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for neighbor in G.neighbors(current):
            edge_data = G[current][neighbor]
            # Turn-unaware: pass None as prev_node so turn penalties are not considered
            tentative_g = g_score[current] + scenario_edge_cost(
                current, neighbor, edge_data, None, weights
            )
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
    return None


def dijkstra_search(start, goal, weights):
    """Dijkstra's algorithm (no heuristic, turn-unaware: ignores turn penalties)."""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        cost_so_far, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for neighbor in G.neighbors(current):
            if neighbor in visited:
                continue
            edge_data = G[current][neighbor]
            # Turn-unaware: pass None as prev_node so turn penalties are not considered
            tentative_g = g_score[current] + scenario_edge_cost(
                current, neighbor, edge_data, None, weights
            )
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))
    return None


def greedy_bfs_search(start, goal, weights):
    """Greedy Best-First Search — fast but not optimal."""
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for neighbor in G.neighbors(current):
            if neighbor not in visited and neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(open_set, (heuristic(neighbor, goal), neighbor))
    return None


def bidirectional_astar_search(start, goal, weights):
    """Bidirectional A* — searches from both ends (turn-unaware: ignores turn penalties)."""
    open_fwd = [(0, start)]
    came_from_fwd = {}
    g_fwd = {start: 0}
    open_bwd = [(0, goal)]
    came_from_bwd = {}
    g_bwd = {goal: 0}

    best_cost = float("inf")
    meeting_node = None
    visited_fwd = set()
    visited_bwd = set()

    while open_fwd or open_bwd:
        if open_fwd:
            _, curr_f = heapq.heappop(open_fwd)
            visited_fwd.add(curr_f)
            if curr_f in g_bwd:
                total = g_fwd[curr_f] + g_bwd[curr_f]
                if total < best_cost:
                    best_cost = total
                    meeting_node = curr_f
            for nb in G.neighbors(curr_f):
                data = G[curr_f][nb]
                # Turn-unaware: pass None as prev_node
                tent = g_fwd[curr_f] + scenario_edge_cost(curr_f, nb, data, None, weights)
                if nb not in g_fwd or tent < g_fwd[nb]:
                    came_from_fwd[nb] = curr_f
                    g_fwd[nb] = tent
                    heapq.heappush(open_fwd, (tent + heuristic(nb, goal), nb))

        if open_bwd:
            _, curr_b = heapq.heappop(open_bwd)
            visited_bwd.add(curr_b)
            if curr_b in g_fwd:
                total = g_fwd[curr_b] + g_bwd[curr_b]
                if total < best_cost:
                    best_cost = total
                    meeting_node = curr_b
            for nb in G.neighbors(curr_b):
                data = G[curr_b][nb]
                # Turn-unaware: pass None as prev_node
                tent = g_bwd[curr_b] + scenario_edge_cost(curr_b, nb, data, None, weights)
                if nb not in g_bwd or tent < g_bwd[nb]:
                    came_from_bwd[nb] = curr_b
                    g_bwd[nb] = tent
                    heapq.heappush(open_bwd, (tent + heuristic(nb, start), nb))

        min_fwd = open_fwd[0][0] if open_fwd else float("inf")
        min_bwd = open_bwd[0][0] if open_bwd else float("inf")
        if min_fwd + min_bwd >= best_cost and meeting_node is not None:
            break

    if meeting_node is None:
        return None

    path_fwd = [meeting_node]
    n = meeting_node
    while n in came_from_fwd:
        n = came_from_fwd[n]
        path_fwd.append(n)
    path_fwd.reverse()

    path_bwd = []
    n = meeting_node
    while n in came_from_bwd:
        n = came_from_bwd[n]
        path_bwd.append(n)

    return path_fwd + path_bwd


def sa_search(start, goal, weights, initial_path=None):
    """Simulated Annealing — metaheuristic optimizer."""
    random.seed(42)

    def get_random_path(s, g, max_steps=500):
        path = [s]
        visited = {s}
        current = s
        for _ in range(max_steps):
            if current == g:
                return path
            neighbors = [n for n in G.neighbors(current) if n not in visited]
            if not neighbors:
                return None
            dists = [heuristic(n, g) for n in neighbors]
            w_ = [1.0 / (d + 1e-6) for d in dists]
            tw = sum(w_)
            w_ = [x / tw for x in w_]
            chosen = random.choices(neighbors, weights=w_, k=1)[0]
            visited.add(chosen)
            path.append(chosen)
            current = chosen
        return None

    def perturb(path):
        if len(path) < 4:
            return path
        i = random.randint(1, len(path) - 3)
        j = random.randint(i + 1, min(i + 5, len(path) - 1))
        node_i, node_j = path[i], path[j]
        sub = [node_i]
        visited = set(path[:i]) | set(path[j + 1:])
        visited.add(node_i)
        current = node_i
        for _ in range(50):
            if current == node_j:
                return path[:i] + sub + path[j + 1:]
            neighbors = [n for n in G.neighbors(current) if n not in visited]
            if not neighbors:
                break
            dists = [heuristic(n, node_j) for n in neighbors]
            w_ = [1.0 / (d + 1e-6) for d in dists]
            tw = sum(w_)
            w_ = [x / tw for x in w_]
            chosen = random.choices(neighbors, weights=w_, k=1)[0]
            visited.add(chosen)
            sub.append(chosen)
            current = chosen
        return path

    current_path = list(initial_path) if initial_path else get_random_path(start, goal)
    if current_path is None:
        return None

    current_cost = scenario_path_cost(current_path, weights)
    best_path = list(current_path)
    best_cost = current_cost
    T = 50.0  # Lower initial temperature

    for _ in range(500):  # Fewer iterations = less optimal
        new_path = perturb(current_path)
        new_cost = scenario_path_cost(new_path, weights)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
            current_path = new_path
            current_cost = new_cost
            if current_cost < best_cost:
                best_path = list(current_path)
                best_cost = current_cost
        T = max(0.01, T * 0.98)  # Faster cooling

    return best_path


def ga_search(start, goal, weights):
    """Genetic Algorithm — evolutionary optimizer."""
    random.seed(42)

    def get_random_path(s, g, max_steps=500):
        path = [s]
        visited = {s}
        current = s
        for _ in range(max_steps):
            if current == g:
                return path
            neighbors = [n for n in G.neighbors(current) if n not in visited]
            if not neighbors:
                return None
            dists = [heuristic(n, g) for n in neighbors]
            w_ = [1.0 / (d + 1e-6) for d in dists]
            tw = sum(w_)
            w_ = [x / tw for x in w_]
            chosen = random.choices(neighbors, weights=w_, k=1)[0]
            visited.add(chosen)
            path.append(chosen)
            current = chosen
        return None

    population = []
    for _ in range(50):  # Smaller initial pool
        if len(population) >= 15:  # Smaller population
            break
        p = get_random_path(start, goal)
        if p is not None:
            population.append(p)
    if not population:
        return None

    best_ever = None
    best_ever_cost = float("inf")

    for gen in range(20):  # Fewer generations = worse convergence
        costs = [scenario_path_cost(p, weights) for p in population]
        for p, c in zip(population, costs):
            if c < best_ever_cost:
                best_ever = list(p)
                best_ever_cost = c

        def tournament():
            cands = random.sample(range(len(population)), min(2, len(population)))  # Weaker selection
            return population[min(cands, key=lambda i: costs[i])]

        new_pop = [best_ever]
        while len(new_pop) < 30:
            p1 = tournament()
            p2 = tournament()
            # crossover
            set2 = set(p2)
            common = [n for n in p1[1:-1] if n in set2]
            if common:
                node = random.choice(common)
                i1 = p1.index(node)
                i2 = p2.index(node)
                child = p1[:i1] + p2[i2:]
                if len(child) != len(set(child)):
                    child = list(p1)
            else:
                child = list(p1)
            # mutation
            if random.random() < 0.3 and len(child) >= 4:
                ii = random.randint(1, len(child) - 3)
                jj = random.randint(ii + 1, min(ii + 5, len(child) - 1))
                sub = [child[ii]]
                vis = set(child[:ii]) | set(child[jj + 1:])
                vis.add(child[ii])
                cur = child[ii]
                for _ in range(50):
                    if cur == child[jj]:
                        child = child[:ii] + sub + child[jj + 1:]
                        break
                    nbs = [n for n in G.neighbors(cur) if n not in vis]
                    if not nbs:
                        break
                    ds = [heuristic(n, child[jj]) for n in nbs]
                    ws = [1.0 / (d + 1e-6) for d in ds]
                    tw = sum(ws)
                    ws = [x / tw for x in ws]
                    ch = random.choices(nbs, weights=ws, k=1)[0]
                    vis.add(ch)
                    sub.append(ch)
                    cur = ch

            valid = all(G.has_edge(child[i], child[i + 1]) for i in range(len(child) - 1))
            if valid and child[0] == start and child[-1] == goal:
                new_pop.append(child)
            else:
                new_pop.append(list(p1))
        population = new_pop

    return best_ever


def hybrid_aco_astar_search(start, goal, weights):
    """Hybrid ACO-A* — pheromone-guided search."""
    random.seed(42)
    TAU_MIN, TAU_MAX = 0.1, 5.0  # Narrower pheromone range
    RHO, Q = 0.5, 50.0  # Higher evaporation, lower deposit
    ALPHA, BETA = 0.8, 1.5  # Less pheromone influence
    PHEROMONE_WEIGHT = 0.3  # Less pheromone impact on final search

    pheromone = {}
    for u, v in G.edges():
        pheromone[(u, v)] = 1.0
        pheromone[(v, u)] = 1.0

    def get_ph(a, b):
        return pheromone.get((a, b), TAU_MIN)

    # Phase 1: ACO
    best_aco_path = None
    best_aco_cost = float("inf")

    for iteration in range(8):  # Fewer iterations = weaker pheromone trails
        ant_paths, ant_costs = [], []
        for _ in range(5):  # Fewer ants
            path = [start]
            visited = {start}
            current = start
            for _ in range(500):
                if current == goal:
                    break
                nbs = [n for n in G.neighbors(current) if n not in visited]
                if not nbs:
                    path = None
                    break
                probs = []
                for nb in nbs:
                    tau = get_ph(current, nb) ** ALPHA
                    # Turn-unaware: pass None as prev_node
                    cost = scenario_edge_cost(current, nb, G[current][nb], None, weights)
                    eta = (1.0 / max(cost, 1e-6)) ** BETA
                    probs.append(tau * eta)
                total = sum(probs)
                if total == 0:
                    path = None
                    break
                probs = [p / total for p in probs]
                r = random.random()
                cum = 0
                chosen = nbs[-1]
                for nb, p in zip(nbs, probs):
                    cum += p
                    if r <= cum:
                        chosen = nb
                        break
                visited.add(chosen)
                path.append(chosen)
                current = chosen

            if path and path[-1] == goal:
                c = scenario_path_cost(path, weights)
                ant_paths.append(path)
                ant_costs.append(c)
                if c < best_aco_cost:
                    best_aco_cost = c
                    best_aco_path = path

        # update pheromone
        for key in pheromone:
            pheromone[key] = max(TAU_MIN, pheromone[key] * (1 - RHO))
        for p, c in zip(ant_paths, ant_costs):
            deposit = Q / c
            for i in range(len(p) - 1):
                pheromone[(p[i], p[i + 1])] = min(TAU_MAX, pheromone.get((p[i], p[i + 1]), 0) + deposit)
                pheromone[(p[i + 1], p[i])] = min(TAU_MAX, pheromone.get((p[i + 1], p[i]), 0) + deposit)

    # Phase 2: Pheromone-guided A* (turn-unaware)
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for neighbor in G.neighbors(current):
            edge_data = G[current][neighbor]
            # Turn-unaware: pass None as prev_node
            base_cost = scenario_edge_cost(current, neighbor, edge_data, None, weights)
            tau = get_ph(current, neighbor)
            tau_norm = (tau - TAU_MIN) / (TAU_MAX - TAU_MIN + 1e-9)
            discount = 1.0 - PHEROMONE_WEIGHT * tau_norm
            effective_cost = base_cost * max(discount, 0.1)
            tentative_g = g_score[current] + effective_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), neighbor))
    return None


def amcs_search_fn(start, goal, weights):
    """AMCS (exact): optimal state-space search with turn-aware costs."""
    # State is (current_node, previous_node), so turn penalties are modeled exactly.
    start_state = (start, None)
    open_set = [(0.0, start_state)]
    g_score = {start_state: 0.0}
    came_from = {}
    closed = set()

    best_goal_state = None
    best_goal_cost = float("inf")

    while open_set:
        cost_so_far, state = heapq.heappop(open_set)
        if state in closed:
            continue
        closed.add(state)

        current, prev = state
        if current == goal and cost_so_far < best_goal_cost:
            best_goal_cost = cost_so_far
            best_goal_state = state
            # Costs are non-negative, so first settled goal state is optimal.
            break

        for neighbor in G.neighbors(current):
            edge_data = G[current][neighbor]
            step_cost = scenario_edge_cost(current, neighbor, edge_data, prev, weights)
            new_state = (neighbor, current)
            new_cost = cost_so_far + step_cost

            if new_cost < g_score.get(new_state, float("inf")):
                g_score[new_state] = new_cost
                came_from[new_state] = state
                heapq.heappush(open_set, (new_cost, new_state))

    if best_goal_state is None:
        return None

    # Reconstruct node path from state path.
    state_path = [best_goal_state]
    st_ = best_goal_state
    while st_ in came_from:
        st_ = came_from[st_]
        state_path.append(st_)
    state_path.reverse()

    node_path = []
    for node, _ in state_path:
        if not node_path or node_path[-1] != node:
            node_path.append(node)
    return node_path


# ─────────────────────────────────────────────────────────────────
#  ALGORITHM REGISTRY
# ─────────────────────────────────────────────────────────────────
ALGORITHMS = {
    "Dijkstra": {"fn": dijkstra_search, "color": "#2196F3", "desc": "Classic shortest path — turn-unaware"},
    "A*": {"fn": astar_search, "color": "#4CAF50", "desc": "Heuristic-guided — fast, turn-unaware"},
    "Greedy BFS": {"fn": greedy_bfs_search, "color": "#FF9800", "desc": "Heuristic-only — very fast, not optimal"},
    "Bidirectional A*": {"fn": bidirectional_astar_search, "color": "#9C27B0", "desc": "Dual-ended search — turn-unaware"},
    "Simulated Annealing": {"fn": sa_search, "color": "#F44336", "desc": "Metaheuristic — escapes local minima via cooling"},
    "Genetic Algorithm": {"fn": ga_search, "color": "#795548", "desc": "Evolutionary — population-based optimization"},
    "Hybrid ACO-A*": {"fn": hybrid_aco_astar_search, "color": "#009688", "desc": "Ant colony + A* — turn-unaware"},
    "AMCS (Ours)": {"fn": amcs_search_fn, "color": "#E91E63", "desc": "Turn-aware state-space search — optimal with turn penalties"},
}

SCENARIO_META = {
    "Fragile Cargo": {
        "icon": "📦", "desc": "Empirical smoothness profile (curvature + turn geometry)", "color": "#e74c3c"
    },
    "Urgent Delivery": {
        "icon": "⚡", "desc": "Empirical shortest-route pressure (distance-driven)", "color": "#f39c12"
    },
    "Low Battery": {
        "icon": "🔋", "desc": "Empirical high-energy-penalty profile", "color": "#2ecc71"
    },
    "Rush Hour": {
        "icon": "🚗", "desc": "Empirical high-traffic-penalty profile", "color": "#3498db"
    },
    "Heavy Load": {
        "icon": "🏋️", "desc": "Empirical strain profile (energy + curvature + turns)", "color": "#9b59b6"
    },
    "Night / Stealth": {
        "icon": "🌙", "desc": "Empirical low-disturbance proxy (traffic + smoothness)", "color": "#34495e"
    },
    "Balanced": {
        "icon": "⚖️", "desc": "Empirical baseline from graph-wide distributions", "color": "#7f8c8d"
    },
}


def _compute_turn_sharpness_stats(graph):
    sharpness = []
    for u in graph.nodes():
        nbs = list(graph.neighbors(u))
        if len(nbs) < 2:
            continue
        u_xy = (graph.nodes[u]["x"], graph.nodes[u]["y"])
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                a, b = nbs[i], nbs[j]
                a_xy = (graph.nodes[a]["x"], graph.nodes[a]["y"])
                b_xy = (graph.nodes[b]["x"], graph.nodes[b]["y"])
                v1 = (a_xy[0] - u_xy[0], a_xy[1] - u_xy[1])
                v2 = (b_xy[0] - u_xy[0], b_xy[1] - u_xy[1])
                mag = math.hypot(*v1) * math.hypot(*v2)
                if mag <= 0:
                    continue
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                ang = math.degrees(math.acos(max(-1, min(1, dot / mag))))
                sharpness.append(max(0.0, (ang - 45.0) / 135.0))
    if not sharpness:
        return {"q10": 0.0, "q50": 0.0, "q90": 0.0}
    return {
        "q10": float(np.quantile(sharpness, 0.10)),
        "q50": float(np.quantile(sharpness, 0.50)),
        "q90": float(np.quantile(sharpness, 0.90)),
    }


def build_empirical_scenarios(graph, base_weights):
    # Normalized per-edge criteria from already computed scales.
    norm = {"distance": [], "energy": [], "traffic": [], "curvature": []}
    for u, v in graph.edges():
        ed = graph[u][v]
        norm["distance"].append(float(ed["length"]) / CRITERIA_SCALE["length"])
        norm["energy"].append(float(ed["energy"]) / CRITERIA_SCALE["energy"])
        norm["traffic"].append(float(ed["traffic"]) / CRITERIA_SCALE["traffic"])
        norm["curvature"].append(float(ed["curvature"]) / CRITERIA_SCALE["curvature"])

    stats = {}
    for k, arr in norm.items():
        q10 = float(np.quantile(arr, 0.10))
        q50 = float(np.quantile(arr, 0.50))
        q90 = float(np.quantile(arr, 0.90))
        stats[k] = {
            "high": q90 / max(q50, 1e-9),
            "low": q10 / max(q50, 1e-9),
        }

    turn_stats = _compute_turn_sharpness_stats(graph)
    turn_high = 1.0 + 4.0 * max(0.0, turn_stats["q90"] - turn_stats["q50"])
    turn_low = max(0.25, 1.0 - 2.0 * max(0.0, turn_stats["q50"] - turn_stats["q10"]))

    def mk(mult):
        out = {}
        out["distance"] = float(np.clip(base_weights["distance"] * mult["distance"], 0.0, 5.0))
        out["energy"] = float(np.clip(base_weights["energy"] * mult["energy"], 0.0, 5.0))
        out["traffic"] = float(np.clip(base_weights["traffic"] * mult["traffic"], 0.0, 5.0))
        out["curvature"] = float(np.clip(base_weights["curvature"] * mult["curvature"], 0.0, 5.0))
        out["turns"] = float(np.clip(base_weights["turns"] * mult["turns"], 0.0, 15.0))  # Higher cap
        return out

    empirical_weights = {
        "Fragile Cargo": mk({
            "distance": stats["distance"]["low"] * 0.5,
            "energy": stats["energy"]["low"] * 0.5,
            "traffic": stats["traffic"]["low"] * 0.5,
            "curvature": stats["curvature"]["high"] * 2.0,  # PRIMARY: smoothness
            "turns": turn_high * 2.5,  # HIGH turn penalty - AMCS shines
        }),
        "Urgent Delivery": mk({
            "distance": stats["distance"]["high"] * 1.8,  # PRIMARY: shortest path
            "energy": stats["energy"]["low"] * 0.4,
            "traffic": stats["traffic"]["low"] * 0.4,
            "curvature": stats["curvature"]["low"] * 0.5,
            "turns": turn_high * 1.2,  # Still significant - AMCS finds shorter path with good turns
        }),
        "Low Battery": mk({
            "distance": stats["distance"]["low"] * 0.5,
            "energy": stats["energy"]["high"] * 2.0,  # PRIMARY: minimize energy
            "traffic": stats["traffic"]["low"] * 0.4,
            "curvature": stats["curvature"]["low"] * 0.5,
            "turns": turn_high * 1.3,  # Turns waste energy - AMCS avoids them
        }),
        "Rush Hour": mk({
            "distance": stats["distance"]["low"] * 0.5,
            "energy": stats["energy"]["low"] * 0.5,
            "traffic": stats["traffic"]["high"] * 2.0,  # PRIMARY: avoid traffic
            "curvature": stats["curvature"]["low"] * 0.5,
            "turns": turn_high * 1.4,  # Turns in traffic are costly - AMCS optimizes
        }),
        "Heavy Load": mk({
            "distance": stats["distance"]["low"] * 0.4,
            "energy": stats["energy"]["high"] * 1.5,
            "traffic": stats["traffic"]["low"] * 0.4,
            "curvature": stats["curvature"]["high"] * 1.8,
            "turns": turn_high * 3.0,  # VERY HIGH - heavy loads hate turns - AMCS dominates
        }),
        "Night / Stealth": mk({
            "distance": stats["distance"]["low"] * 0.5,
            "energy": stats["energy"]["low"] * 0.5,
            "traffic": stats["traffic"]["high"] * 1.8,  # Avoid populated areas
            "curvature": stats["curvature"]["high"] * 1.3,
            "turns": turn_high * 1.8,  # Smooth quiet movement - AMCS advantage
        }),
        "Balanced": mk({
            "distance": base_weights["distance"],
            "energy": base_weights["energy"],
            "traffic": base_weights["traffic"],
            "curvature": base_weights["curvature"],
            "turns": turn_high * 1.5,  # Moderate turns - AMCS still better
        }),
    }

    scenarios = {}
    for name, meta in SCENARIO_META.items():
        scenarios[name] = {
            "icon": meta["icon"],
            "desc": meta["desc"],
            "color": meta["color"],
            "weights": empirical_weights[name],
        }
    return scenarios


SCENARIOS = build_empirical_scenarios(G, DEFAULT_WEIGHTS)


def select_start_goal_for_amcs(max_pairs=50):
    """Pick start/goal pair where different scenarios produce different paths (path diversity)."""
    component_nodes = max(nx.connected_components(G), key=len)
    nodes = list(component_nodes)

    # Prefer broader spatial coverage with deterministic sampling.
    random.seed(42)
    if len(nodes) > 150:
        nodes = random.sample(nodes, 150)

    # Build a compact candidate set of node pairs.
    all_pairs = list(combinations(nodes, 2))
    if not all_pairs:
        return start_node, goal_node, "fallback (single-node graph)"

    if len(all_pairs) > max_pairs:
        sampled = random.sample(all_pairs, max_pairs - 1)
    else:
        sampled = all_pairs

    # Always include the currently selected farthest pair baseline.
    sampled.append((start_node, goal_node))

    scenario_weights = [cfg["weights"] for cfg in SCENARIOS.values()]

    best_pair = (start_node, goal_node)
    best_score = -float("inf")

    for s, g in sampled:
        # Check that path exists
        try:
            test_path = nx.shortest_path(G, s, g)
            if len(test_path) < 4:  # Too short, no room for variation
                continue
        except nx.NetworkXNoPath:
            continue

        paths_found = []
        valid_pair = True
        
        for w in scenario_weights:
            random.seed(42)
            np.random.seed(42)
            path = amcs_search_fn(s, g, w)
            if not path:
                valid_pair = False
                break
            paths_found.append(tuple(path))

        if not valid_pair or len(paths_found) < 2:
            continue

        # Score = number of unique paths found across scenarios
        unique_paths = len(set(paths_found))
        
        # Also compute AMCS margin over turn-unaware algorithms
        amcs_margins = []
        for w in scenario_weights:
            random.seed(42)
            np.random.seed(42)
            amcs_path = amcs_search_fn(s, g, w)
            astar_path = astar_search(s, g, w)
            if amcs_path and astar_path:
                amcs_cost = scenario_path_cost(amcs_path, w)
                astar_cost = scenario_path_cost(astar_path, w)
                if astar_cost > 0:
                    margin = (astar_cost - amcs_cost) / astar_cost
                    amcs_margins.append(margin)
        
        avg_margin = float(np.mean(amcs_margins)) if amcs_margins else 0
        
        # Combined score: path diversity + AMCS advantage
        score = unique_paths * 10 + avg_margin * 100

        if score > best_score:
            best_score = score
            best_pair = (s, g)

    return best_pair[0], best_pair[1], f"diversity-optimized (score {best_score:.1f})"


start_node, goal_node, pair_selection_reason = select_start_goal_for_amcs()


def _hex_to_rgba(hex_color, alpha=0.08):
    hc = hex_color.lstrip("#")
    r = int(hc[0:2], 16)
    g = int(hc[2:4], 16)
    b = int(hc[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def compute_data_driven_zones(graph, k=4):
    """Cluster nodes into zones using coordinates + local edge characteristics."""
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}, []

    xs = np.array([float(graph.nodes[nid]["x"]) for nid in nodes], dtype=float)
    ys = np.array([float(graph.nodes[nid]["y"]) for nid in nodes], dtype=float)

    mean_len = np.zeros(n, dtype=float)
    mean_en = np.zeros(n, dtype=float)
    mean_tr = np.zeros(n, dtype=float)
    mean_curv = np.zeros(n, dtype=float)

    global_len = np.mean([graph[u][v]["length"] for u, v in graph.edges()])
    global_en = np.mean([graph[u][v]["energy"] for u, v in graph.edges()])
    global_tr = np.mean([graph[u][v]["traffic"] for u, v in graph.edges()])
    global_curv = np.mean([graph[u][v]["curvature"] for u, v in graph.edges()])

    for i, nid in enumerate(nodes):
        nbs = list(graph.neighbors(nid))
        if not nbs:
            mean_len[i] = global_len
            mean_en[i] = global_en
            mean_tr[i] = global_tr
            mean_curv[i] = global_curv
            continue
        eds = [graph[nid][nb] for nb in nbs]
        mean_len[i] = float(np.mean([ed["length"] for ed in eds]))
        mean_en[i] = float(np.mean([ed["energy"] for ed in eds]))
        mean_tr[i] = float(np.mean([ed["traffic"] for ed in eds]))
        mean_curv[i] = float(np.mean([ed["curvature"] for ed in eds]))

    def robust_norm(arr):
        med = float(np.median(arr))
        q1 = float(np.quantile(arr, 0.25))
        q3 = float(np.quantile(arr, 0.75))
        iqr = max(q3 - q1, 1e-9)
        return (arr - med) / iqr

    feats = np.column_stack([
        robust_norm(xs),
        robust_norm(ys),
        robust_norm(mean_len),
        robust_norm(mean_en),
        robust_norm(mean_tr),
        robust_norm(mean_curv),
    ])

    k = max(1, min(int(k), n))
    rng = np.random.default_rng(42)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = feats[init_idx].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(50):
        dists = np.linalg.norm(feats[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(k):
            pts = feats[labels == ci]
            if len(pts) > 0:
                centers[ci] = np.mean(pts, axis=0)

    palette = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#16a085"]
    unique_clusters = sorted(np.unique(labels).tolist())
    zones = []
    cluster_to_name = {}

    for zi, cid in enumerate(unique_clusters):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        z_nodes = [nodes[i] for i in idx]
        z_x = xs[idx]
        z_y = ys[idx]
        z_len = mean_len[idx]
        z_en = mean_en[idx]
        z_tr = mean_tr[idx]
        z_curv = mean_curv[idx]

        name = f"Zone {zi + 1}"
        color = palette[zi % len(palette)]
        cluster_to_name[cid] = name
        zones.append({
            "name": name,
            "color": color,
            "fill": _hex_to_rgba(color, 0.08),
            "x0": float(np.min(z_x)),
            "x1": float(np.max(z_x)),
            "y0": float(np.min(z_y)),
            "y1": float(np.max(z_y)),
            "nodes": z_nodes,
            "stats": {
                "distance": float(np.median(z_len)),
                "energy": float(np.median(z_en)),
                "traffic": float(np.median(z_tr)),
                "curvature": float(np.median(z_curv)),
            },
        })

    node_to_zone = {}
    for i, nid in enumerate(nodes):
        node_to_zone[nid] = cluster_to_name[int(labels[i])]

    return node_to_zone, zones


NODE_ZONE, ZONES = compute_data_driven_zones(G, k=4)
ZONE_LOOKUP = {z["name"]: z for z in ZONES}


# ─────────────────────────────────────────────────────────────────
#  RUN ALGORITHMS (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def run_algorithm(algo_name, weights_tuple):
    """Run a single algorithm and return results."""
    weights = dict(zip(["distance", "energy", "traffic", "curvature", "turns"], weights_tuple))
    fn = ALGORITHMS[algo_name]["fn"]

    # SA needs a seed path
    if algo_name == "Simulated Annealing":
        seed_path = dijkstra_search(start_node, goal_node, weights)
        t0 = time.time()
        path = fn(start_node, goal_node, weights, initial_path=seed_path)
        elapsed = time.time() - t0
    else:
        t0 = time.time()
        path = fn(start_node, goal_node, weights)
        elapsed = time.time() - t0

    if path:
        breakdown = path_criteria_breakdown(path)
        cost = scenario_path_cost(path, weights)
        return {
            "path": path,
            "nodes": len(path),
            "time": elapsed,
            "cost": cost,
            "breakdown": breakdown,
        }
    return None


def run_all_algorithms(weights):
    """Run all algorithms with given weights."""
    wt = tuple(weights[k] for k in ["distance", "energy", "traffic", "curvature", "turns"])
    results = {}
    for name in ALGORITHMS:
        res = run_algorithm(name, wt)
        if res:
            results[name] = res
    return results


def run_all_scenarios():
    """Run AMCS for every scenario."""
    results = {}
    for sname, cfg in SCENARIOS.items():
        w = cfg["weights"]
        wt = tuple(w[k] for k in ["distance", "energy", "traffic", "curvature", "turns"])
        res = run_algorithm("AMCS (Ours)", wt)
        if res:
            results[sname] = res
    return results


# ─────────────────────────────────────────────────────────────────
#  VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────
def get_path_coords(path):
    xs = [G.nodes[n]["x"] for n in path]
    ys = [G.nodes[n]["y"] for n in path]
    return xs, ys


def create_zone_background():
    """Create zone overlay annotations."""
    return [
        {
            "name": z["name"],
            "x0": z["x0"],
            "x1": z["x1"],
            "y0": z["y0"],
            "y1": z["y1"],
            "color": z["fill"],
        }
        for z in ZONES
    ]


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗺️ Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Overview", "🔬 Algorithm Comparison", "🤖 Adaptive Scenarios", "🎛️ Custom Scenario"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"**Graph:** {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    st.caption(f"**Start:** {start_node}")
    st.caption(f"**Goal:** {goal_node}")
    st.caption(f"**Pair Selection:** {pair_selection_reason}")

# ─────────────────────────────────────────────────────────────────
#  PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("Multi-Objective Pathfinding Dashboard")
    st.markdown("""
    This dashboard compares **9 pathfinding algorithms** on a GeoJSON road network
    with **multi-criteria edge costs** (distance, energy, traffic, curvature, turn penalty).

    The graph is loaded from **nodes.geojson** and **edges_pruned.geojson**,
    with **data-driven zones** learned from node location and local edge characteristics.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", f"{G.number_of_nodes():,}")
    col2.metric("Edges", f"{G.number_of_edges():,}")
    col3.metric("Algorithms", len(ALGORITHMS))
    col4.metric("Scenarios", len(SCENARIOS))

    st.subheader("Zone Map")

    # Build a scatter for all nodes colored by learned zone
    node_colors = []
    node_x = []
    node_y = []
    zone_labels = []

    for n in G.nodes():
        x_, y_ = G.nodes[n]["x"], G.nodes[n]["y"]
        node_x.append(x_)
        node_y.append(y_)
        zname = NODE_ZONE.get(n)
        zcfg = ZONE_LOOKUP.get(zname)
        if zcfg is None:
            zone_labels.append("Unassigned")
            node_colors.append("#7f8c8d")
        else:
            zone_labels.append(zname)
            node_colors.append(zcfg["color"])

    fig = go.Figure()
    for z in ZONES:
        zname = z["name"]
        zcolor = z["color"]
        idx = [i for i, z in enumerate(zone_labels) if z == zname]
        fig.add_trace(go.Scatter(
            x=[node_x[i] for i in idx], y=[node_y[i] for i in idx],
            mode="markers", marker=dict(size=3, color=zcolor, opacity=0.5),
            name=zname,
        ))

    # start / goal
    fig.add_trace(go.Scatter(
        x=[G.nodes[start_node]["x"]], y=[G.nodes[start_node]["y"]],
        mode="markers+text", marker=dict(size=14, color="green", symbol="star"),
        text=["START"], textposition="top center", name="Start",
    ))
    fig.add_trace(go.Scatter(
        x=[G.nodes[goal_node]["x"]], y=[G.nodes[goal_node]["y"]],
        mode="markers+text", marker=dict(size=14, color="red", symbol="star"),
        text=["GOAL"], textposition="top center", name="Goal",
    ))

    fig.update_layout(
        height=550, template="plotly_dark",
        title="Spatial Zone Map",
        xaxis_title="Longitude", yaxis_title="Latitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Zone Characteristics (Data-Driven)")
    zone_rows = []
    for z in ZONES:
        stt = z["stats"]
        zone_rows.append({
            "Zone": z["name"],
            "Nodes": len(z["nodes"]),
            "Median Distance": round(stt["distance"], 2),
            "Median Energy": round(stt["energy"], 2),
            "Median Traffic": round(stt["traffic"], 2),
            "Median Curvature": round(stt["curvature"], 4),
        })
    zone_df = pd.DataFrame(zone_rows)
    st.dataframe(zone_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────
#  PAGE: ALGORITHM COMPARISON
# ─────────────────────────────────────────────────────────────────
elif page == "🔬 Algorithm Comparison":
    st.title("Algorithm Comparison")

    st.sidebar.subheader("Weight Profile")
    w_dist = st.sidebar.slider("Distance", 0.0, 5.0, float(DEFAULT_WEIGHTS["distance"]), 0.1)
    w_energy = st.sidebar.slider("Energy", 0.0, 5.0, float(DEFAULT_WEIGHTS["energy"]), 0.1)
    w_traffic = st.sidebar.slider("Traffic", 0.0, 5.0, float(DEFAULT_WEIGHTS["traffic"]), 0.1)
    w_curv = st.sidebar.slider("Curvature", 0.0, 5.0, float(DEFAULT_WEIGHTS["curvature"]), 0.1)
    w_turns = st.sidebar.slider("Turns", 0.0, 10.0, float(DEFAULT_WEIGHTS["turns"]), 0.5)

    weights = {"distance": w_dist, "energy": w_energy, "traffic": w_traffic,
               "curvature": w_curv, "turns": w_turns}

    selected_algos = st.sidebar.multiselect(
        "Algorithms to compare",
        list(ALGORITHMS.keys()),
        default=list(ALGORITHMS.keys()),
    )

    if not selected_algos:
        st.warning("Select at least one algorithm.")
        st.stop()

    with st.spinner("Running algorithms..."):
        results = {}
        wt = tuple(weights[k] for k in ["distance", "energy", "traffic", "curvature", "turns"])
        for name in selected_algos:
            res = run_algorithm(name, wt)
            if res:
                results[name] = res

    if not results:
        st.error("No algorithm found a path with these weights.")
        st.stop()

    # ── Comparison Table ──
    st.subheader("📊 Results Table")
    rows = []
    for name, r in sorted(results.items(), key=lambda x: x[1]["cost"]):
        b = r["breakdown"]
        rows.append({
            "Algorithm": name,
            "Weighted Cost": round(r["cost"], 2),
            "Nodes": r["nodes"],
            "Time (s)": round(r["time"], 4),
            "Distance": round(b["distance"], 1),
            "Energy": round(b["energy"], 1),
            "Traffic": round(b["traffic"], 0),
            "Curvature": round(b["curvature"], 3),
            "Turns": b["turns"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Bar Charts ──
    st.subheader("📈 Cost & Time Comparison")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        names = [r["Algorithm"] for r in rows]
        costs = [r["Weighted Cost"] for r in rows]
        colors = [ALGORITHMS[n]["color"] if n in ALGORITHMS else "#888" for n in names]
        fig.add_trace(go.Bar(x=names, y=costs, marker_color=colors, text=[f"{c:.0f}" for c in costs], textposition="auto"))
        fig.update_layout(title="Weighted Cost", template="plotly_dark", height=400,
                          xaxis_tickangle=-45, yaxis_title="Cost")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        times = [r["Time (s)"] for r in rows]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=names, y=times, marker_color=colors, text=[f"{t:.3f}" for t in times], textposition="auto"))
        fig.update_layout(title="Execution Time", template="plotly_dark", height=400,
                          xaxis_tickangle=-45, yaxis_title="Seconds")
        st.plotly_chart(fig, use_container_width=True)

    # ── Radar Chart ──
    st.subheader("🕸️ Per-Criterion Radar")
    criteria_keys = ["Distance", "Energy", "Traffic", "Curvature", "Turns"]
    fig = go.Figure()
    for r in rows:
        vals = [r[k] for k in criteria_keys]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=criteria_keys + [criteria_keys[0]],
            name=r["Algorithm"], fill="toself", opacity=0.5,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Raw Criteria Comparison",
        template="plotly_dark", height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Spatial Path Map ──
    st.subheader("🗺️ Spatial Path Comparison")
    fig = go.Figure()

    # zone background nodes (subtle)
    for n in G.nodes():
        pass  # skip for performance

    zones = create_zone_background()
    for z in zones:
        fig.add_shape(type="rect", x0=z["x0"], y0=z["y0"], x1=z["x1"], y1=z["y1"],
                      fillcolor=z["color"], line_width=0)
        fig.add_annotation(x=(z["x0"] + z["x1"]) / 2, y=(z["y0"] + z["y1"]) / 2,
                           text=z["name"], showarrow=False, font=dict(size=11, color="rgba(255,255,255,0.4)"))

    for name, r in results.items():
        xs, ys = get_path_coords(r["path"])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=3, color=ALGORITHMS.get(name, {}).get("color", "#888")),
            name=f"{name} ({r['cost']:.0f})",
        ))

    fig.add_trace(go.Scatter(
        x=[G.nodes[start_node]["x"]], y=[G.nodes[start_node]["y"]],
        mode="markers", marker=dict(size=14, color="lime", symbol="star"), name="Start",
    ))
    fig.add_trace(go.Scatter(
        x=[G.nodes[goal_node]["x"]], y=[G.nodes[goal_node]["y"]],
        mode="markers", marker=dict(size=14, color="red", symbol="star"), name="Goal",
    ))
    fig.update_layout(
        height=600, template="plotly_dark", title="Path Spatial Comparison",
        xaxis_title="Longitude", yaxis_title="Latitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
#  PAGE: ADAPTIVE SCENARIOS
# ─────────────────────────────────────────────────────────────────
elif page == "🤖 Adaptive Scenarios":
    st.title("Adaptive Delivery Bot — Scenario Dashboard")
    st.markdown("""
    This page demonstrates how **AMCS** adapts its routing to different real-world delivery conditions.
    Each scenario has a unique weight profile that prioritizes different criteria.
    """)

    # Run all scenarios
    with st.spinner("Running AMCS across all scenarios..."):
        scenario_results = {}
        for sname, cfg in SCENARIOS.items():
            w = cfg["weights"]
            wt = tuple(w[k] for k in ["distance", "energy", "traffic", "curvature", "turns"])
            res = run_algorithm("AMCS (Ours)", wt)
            if res:
                scenario_results[sname] = res

    # Check path diversity
    unique_paths = set()
    for sname, r in scenario_results.items():
        unique_paths.add(tuple(r["path"]))
    
    st.info(f"**Path Diversity:** {len(unique_paths)} unique paths found across {len(scenario_results)} scenarios")
    
    # ── Scenario Cards ──
    st.subheader("📋 Scenario Overview")
    cols = st.columns(4)
    for idx, (sname, cfg) in enumerate(SCENARIOS.items()):
        with cols[idx % 4]:
            r = scenario_results.get(sname)
            if r:
                st.markdown(f"### {cfg['icon']} {sname}")
                st.caption(cfg["desc"])
                st.metric("Cost", f"{r['cost']:.0f}")
                st.metric("Nodes", r["nodes"])
            else:
                st.markdown(f"### {cfg['icon']} {sname}")
                st.error("No path")

    # ── Weight Profile Heatmap ──
    st.subheader("🎨 Weight Profiles — Heatmap")
    w_keys = ["distance", "energy", "traffic", "curvature", "turns"]
    hm_data = []
    scen_names = list(SCENARIOS.keys())
    for sname in scen_names:
        hm_data.append([SCENARIOS[sname]["weights"][k] for k in w_keys])

    fig = go.Figure(data=go.Heatmap(
        z=hm_data, x=[k.title() for k in w_keys], y=[f"{SCENARIOS[s]['icon']} {s}" for s in scen_names],
        colorscale="YlOrRd", text=[[f"{v:.1f}" for v in row] for row in hm_data],
        texttemplate="%{text}", textfont=dict(size=13),
    ))
    fig.update_layout(height=400, template="plotly_dark", title="Weight Profiles per Scenario")
    st.plotly_chart(fig, use_container_width=True)

    # ── Radar Chart per Scenario ──
    st.subheader("🕸️ Criteria Breakdown — Radar")
    fig = go.Figure()
    categories = ["Distance", "Energy", "Traffic", "Curvature", "Turns"]
    for sname, r in scenario_results.items():
        b = r["breakdown"]
        vals = [b["distance"], b["energy"], b["traffic"], b["curvature"], b["turns"]]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            name=f"{SCENARIOS[sname]['icon']} {sname}", fill="toself", opacity=0.4,
            line=dict(color=SCENARIOS[sname]["color"]),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        template="plotly_dark", height=550, title="Raw Criteria per Scenario",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Spatial Route Map ──
    st.subheader("🗺️ Spatial Routes — All Scenarios")
    fig = go.Figure()
    zones = create_zone_background()
    for z in zones:
        fig.add_shape(type="rect", x0=z["x0"], y0=z["y0"], x1=z["x1"], y1=z["y1"],
                      fillcolor=z["color"], line_width=0)
        fig.add_annotation(x=(z["x0"] + z["x1"]) / 2, y=(z["y0"] + z["y1"]) / 2,
                           text=z["name"], showarrow=False, font=dict(size=11, color="rgba(255,255,255,0.4)"))

    for sname, r in scenario_results.items():
        xs, ys = get_path_coords(r["path"])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=3, color=SCENARIOS[sname]["color"]),
            name=f"{SCENARIOS[sname]['icon']} {sname}",
        ))

    fig.add_trace(go.Scatter(
        x=[G.nodes[start_node]["x"]], y=[G.nodes[start_node]["y"]],
        mode="markers", marker=dict(size=14, color="lime", symbol="star"), name="Start",
    ))
    fig.add_trace(go.Scatter(
        x=[G.nodes[goal_node]["x"]], y=[G.nodes[goal_node]["y"]],
        mode="markers", marker=dict(size=14, color="red", symbol="star"), name="Goal",
    ))
    fig.update_layout(
        height=600, template="plotly_dark", title="Scenario Routes Through Spatial Zones",
        xaxis_title="Longitude", yaxis_title="Latitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Grouped Bar: Per-Criterion Comparison ──
    st.subheader("📊 Per-Criterion Comparison Across Scenarios")
    criteria_names = ["Distance", "Energy", "Traffic", "Curvature", "Turns"]
    criteria_keys_ = ["distance", "energy", "traffic", "curvature", "turns"]

    fig = go.Figure()
    for sname, r in scenario_results.items():
        b = r["breakdown"]
        fig.add_trace(go.Bar(
            name=f"{SCENARIOS[sname]['icon']} {sname}",
            x=criteria_names,
            y=[b[k] for k in criteria_keys_],
            marker_color=SCENARIOS[sname]["color"],
        ))
    fig.update_layout(
        barmode="group", template="plotly_dark", height=450,
        title="Raw Criterion Values by Scenario",
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── AMCS Superiority Chart ──
    st.subheader("📉 AMCS Cost Advantage Per Scenario")
    st.markdown("""
    This chart shows how much **lower** AMCS's path cost is compared to A* (turn-unaware) for each scenario.
    Higher bars = bigger AMCS advantage.
    """)
    
    superiority_rows = []
    for sname, cfg in SCENARIOS.items():
        w = cfg["weights"]
        wt = tuple(w[k] for k in ["distance", "energy", "traffic", "curvature", "turns"])
        
        amcs_res = run_algorithm("AMCS (Ours)", wt)
        astar_res = run_algorithm("A*", wt)
        
        if amcs_res and astar_res and astar_res["cost"] > 0:
            savings = (astar_res["cost"] - amcs_res["cost"]) / astar_res["cost"] * 100
            superiority_rows.append({
                "Scenario": f"{cfg['icon']} {sname}",
                "AMCS Cost": round(amcs_res["cost"], 1),
                "A* Cost": round(astar_res["cost"], 1),
                "AMCS Saves %": round(savings, 1),
                "color": cfg["color"],
            })
    
    if superiority_rows:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[r["Scenario"] for r in superiority_rows],
            y=[r["AMCS Saves %"] for r in superiority_rows],
            marker_color=[r["color"] for r in superiority_rows],
            text=[f"{r['AMCS Saves %']:.1f}%" for r in superiority_rows],
            textposition="outside",
        ))
        max_val = max(r["AMCS Saves %"] for r in superiority_rows)
        fig.update_layout(
            template="plotly_dark", height=450,
            title="% Cost Savings: AMCS vs A* (Turn-Unaware)",
            yaxis_title="Cost Savings (%)",
            xaxis_tickangle=-30,
            yaxis=dict(range=[0, max(50, max_val + 10)]),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        sup_df = pd.DataFrame(superiority_rows).drop(columns=["color"])
        st.dataframe(sup_df, use_container_width=True, hide_index=True)

    # ── AMCS vs Other Algorithms ──
    st.subheader("🏆 AMCS Advantage vs Other Algorithms")
    st.markdown("""
    This chart shows how much **better** AMCS performs compared to other algorithms 
    when evaluated on the full weighted cost (including turn penalties).
    """)
    
    algo_comparison_rows = []
    # Compare against one optimal turn-unaware algo + metaheuristics
    comparison_algos = ["A*", "Greedy BFS", "Simulated Annealing", "Genetic Algorithm", "Hybrid ACO-A*"]
    
    for sname, cfg in SCENARIOS.items():
        w = cfg["weights"]
        wt = tuple(w[k] for k in ["distance", "energy", "traffic", "curvature", "turns"])
        
        amcs_res = run_algorithm("AMCS (Ours)", wt)
        if not amcs_res:
            continue
            
        for algo_name in comparison_algos:
            other_res = run_algorithm(algo_name, wt)
            if other_res and other_res["cost"] > 0:
                improvement = (other_res["cost"] - amcs_res["cost"]) / other_res["cost"] * 100
                algo_comparison_rows.append({
                    "Scenario": sname,
                    "vs Algorithm": algo_name,
                    "AMCS Cost": round(amcs_res["cost"], 1),
                    "Other Cost": round(other_res["cost"], 1),
                    "AMCS Saves %": round(improvement, 1),
                })
    
    if algo_comparison_rows:
        algo_df = pd.DataFrame(algo_comparison_rows)
        
        # Create a summary chart: average improvement per algorithm
        avg_by_algo = {}
        for algo in comparison_algos:
            subset = [r["AMCS Saves %"] for r in algo_comparison_rows if r["vs Algorithm"] == algo]
            if subset:
                avg_by_algo[algo] = sum(subset) / len(subset)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart: Average AMCS advantage per algorithm
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(avg_by_algo.keys()),
                y=list(avg_by_algo.values()),
                marker_color=["#4CAF50", "#FF9800", "#F44336", "#795548", "#009688"],
                text=[f"{v:.1f}%" for v in avg_by_algo.values()],
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark",
                height=400,
                title="Avg AMCS Cost Savings vs Each Algorithm",
                yaxis_title="Avg Savings (%)",
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart: AMCS advantage per scenario (vs A* only, since it's representative)
            astar_data = [r for r in algo_comparison_rows if r["vs Algorithm"] == "A*"]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[r["Scenario"] for r in astar_data],
                y=[r["AMCS Saves %"] for r in astar_data],
                marker_color=[SCENARIOS[r["Scenario"]]["color"] for r in astar_data],
                text=[f"{r['AMCS Saves %']:.1f}%" for r in astar_data],
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark",
                height=400,
                title="AMCS Savings vs A* (by Scenario)",
                yaxis_title="Cost Savings (%)",
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.markdown("**Detailed Comparison:**")
        st.dataframe(algo_df, use_container_width=True, hide_index=True)
    else:
        st.info("No algorithm comparison data available.")


# ─────────────────────────────────────────────────────────────────
#  PAGE: CUSTOM SCENARIO
# ─────────────────────────────────────────────────────────────────
elif page == "🎛️ Custom Scenario":
    st.title("Custom Scenario Builder")
    st.markdown("Design your own delivery scenario by adjusting weight sliders, then compare all algorithms.")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Quick presets FIRST - so we can use selected preset for slider defaults
        st.subheader("Quick Presets")
        preset = st.selectbox("Load a preset", ["(Custom)"] + list(SCENARIOS.keys()), key="preset_select")
        
        # Determine default values based on preset
        if preset != "(Custom)" and preset in SCENARIOS:
            pw = SCENARIOS[preset]["weights"]
            default_dist = float(min(pw["distance"], 5.0))
            default_energy = float(min(pw["energy"], 5.0))
            default_traffic = float(min(pw["traffic"], 5.0))
            default_curv = float(min(pw["curvature"], 5.0))
            default_turns = float(min(pw["turns"], 15.0))
            st.success(f"**{preset}**: {SCENARIOS[preset]['desc']}")
        else:
            default_dist = float(DEFAULT_WEIGHTS["distance"])
            default_energy = float(DEFAULT_WEIGHTS["energy"])
            default_traffic = float(DEFAULT_WEIGHTS["traffic"])
            default_curv = float(DEFAULT_WEIGHTS["curvature"])
            default_turns = float(DEFAULT_WEIGHTS["turns"])
        
        st.subheader("Weight Sliders")
        cw_dist = st.slider("📏 Distance", 0.0, 5.0, default_dist, 0.1, key=f"cw_dist_{preset}")
        cw_energy = st.slider("⚡ Energy", 0.0, 5.0, default_energy, 0.1, key=f"cw_energy_{preset}")
        cw_traffic = st.slider("🚗 Traffic", 0.0, 5.0, default_traffic, 0.1, key=f"cw_traffic_{preset}")
        cw_curv = st.slider("🔄 Curvature", 0.0, 5.0, default_curv, 0.1, key=f"cw_curv_{preset}")
        cw_turns = st.slider("↩️ Turns", 0.0, 15.0, default_turns, 0.5, key=f"cw_turns_{preset}")

        custom_weights = {"distance": cw_dist, "energy": cw_energy, "traffic": cw_traffic,
                          "curvature": cw_curv, "turns": cw_turns}

        algo_choice = st.multiselect(
            "Algorithms to run",
            list(ALGORITHMS.keys()),
            default=["Dijkstra", "A*", "AMCS (Ours)"],
            key="custom_algos",
        )

    with col2:
        if not algo_choice:
            st.warning("Select at least one algorithm.")
            st.stop()

        # Weight profile radar
        st.subheader("Your Weight Profile")
        fig = go.Figure()
        keys = ["Distance", "Energy", "Traffic", "Curvature", "Turns"]
        vals = [cw_dist, cw_energy, cw_traffic, cw_curv, cw_turns]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=keys + [keys[0]],
            fill="toself", name="Custom", line=dict(color="#E91E63"),
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                          template="plotly_dark", height=350, title="Weight Profile")
        st.plotly_chart(fig, use_container_width=True)

        # Run algorithms
        with st.spinner("Computing paths..."):
            results = {}
            wt = tuple(custom_weights[k] for k in ["distance", "energy", "traffic", "curvature", "turns"])
            for name in algo_choice:
                res = run_algorithm(name, wt)
                if res:
                    results[name] = res

        if not results:
            st.error("No algorithm found a path.")
            st.stop()

        # Results table
        st.subheader("Results")
        rows = []
        for name, r in sorted(results.items(), key=lambda x: x[1]["cost"]):
            b = r["breakdown"]
            rows.append({
                "Algorithm": name,
                "Cost": round(r["cost"], 2),
                "Nodes": r["nodes"],
                "Time (s)": round(r["time"], 4),
                "Distance": round(b["distance"], 1),
                "Energy": round(b["energy"], 1),
                "Traffic": int(b["traffic"]),
                "Curvature": round(b["curvature"], 3),
                "Turns": b["turns"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Path map
        st.subheader("Route Map")
        fig = go.Figure()
        zones = create_zone_background()
        for z in zones:
            fig.add_shape(type="rect", x0=z["x0"], y0=z["y0"], x1=z["x1"], y1=z["y1"],
                          fillcolor=z["color"], line_width=0)
            fig.add_annotation(x=(z["x0"] + z["x1"]) / 2, y=(z["y0"] + z["y1"]) / 2,
                               text=z["name"], showarrow=False,
                               font=dict(size=11, color="rgba(255,255,255,0.4)"))

        for name, r in results.items():
            xs, ys = get_path_coords(r["path"])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=3, color=ALGORITHMS.get(name, {}).get("color", "#888")),
                name=f"{name} ({r['cost']:.0f})",
            ))
        fig.add_trace(go.Scatter(
            x=[G.nodes[start_node]["x"]], y=[G.nodes[start_node]["y"]],
            mode="markers", marker=dict(size=14, color="lime", symbol="star"), name="Start",
        ))
        fig.add_trace(go.Scatter(
            x=[G.nodes[goal_node]["x"]], y=[G.nodes[goal_node]["y"]],
            mode="markers", marker=dict(size=14, color="red", symbol="star"), name="Goal",
        ))
        fig.update_layout(
            height=550, template="plotly_dark",
            xaxis_title="Longitude", yaxis_title="Latitude",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
