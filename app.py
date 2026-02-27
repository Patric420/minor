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
def build_graph(grid_size=45):
    """Build the synthetic 45×45 grid graph with zone-based attributes."""
    np.random.seed(42)
    random.seed(42)

    # bounding box
    cx, cy = 77.595, 12.972
    w, h = 0.03, 0.02
    min_x, max_x = cx - 1.5 * w, cx + 1.5 * w
    min_y, max_y = cy - 1.5 * h, cy + 1.5 * h

    grid_xs = np.linspace(min_x, max_x, grid_size)
    grid_ys = np.linspace(min_y, max_y, grid_size)

    G = nx.Graph()
    grid_map = {}
    nid = 0

    for i in range(grid_size):
        for j in range(grid_size):
            label = f"n_{nid}"
            px_ = grid_xs[i] + np.random.normal(0, w / grid_size * 0.08)
            py_ = grid_ys[j] + np.random.normal(0, h / grid_size * 0.08)
            G.add_node(label, x=float(px_), y=float(py_))
            grid_map[(i, j)] = label
            nid += 1

    def zone_edge_attrs(i1, j1, i2, j2):
        ci = (i1 + i2) / 2 / (grid_size - 1)
        cj = (j1 + j2) / 2 / (grid_size - 1)
        L = np.random.uniform(30, 80)
        E = L * np.random.uniform(0.8, 1.2)
        T = np.random.uniform(2, 6)
        C = np.random.uniform(0.02, 0.15)

        if ci < 0.35 and cj > 0.65:        # Urban Core
            L *= 0.5; E *= 0.6; T *= 5.0; C *= 0.4
        elif ci > 0.65 and cj > 0.65:      # Suburban
            L *= 1.6; E *= 1.8; T *= 0.25; C *= 1.8
        elif ci < 0.35 and cj < 0.35:      # Hilly
            L *= 1.1; E *= 0.35; T *= 1.5; C *= 6.0
        elif ci > 0.65 and cj < 0.35:      # Industrial
            L *= 1.3; E *= 2.8; T *= 1.2; C *= 0.15
        else:                                # Mixed
            L *= 1.0; E *= 1.0; T *= 2.0; C *= 1.0

        return {
            "length": round(max(5, L), 2),
            "energy": round(max(1, E), 2),
            "traffic": max(1, round(T)),
            "curvature": round(max(0.001, C), 4),
        }

    # 4-connected grid
    for i in range(grid_size):
        for j in range(grid_size):
            if i + 1 < grid_size:
                G.add_edge(grid_map[(i, j)], grid_map[(i + 1, j)],
                           **zone_edge_attrs(i, j, i + 1, j))
            if j + 1 < grid_size:
                G.add_edge(grid_map[(i, j)], grid_map[(i, j + 1)],
                           **zone_edge_attrs(i, j, i, j + 1))

    # diagonal shortcuts
    for _ in range(int(grid_size * grid_size * 0.15)):
        i, j = np.random.randint(0, grid_size), np.random.randint(0, grid_size)
        di, dj = np.random.choice([-1, 1]), np.random.choice([-1, 1])
        ni, nj = i + di, j + dj
        if 0 <= ni < grid_size and 0 <= nj < grid_size:
            u, v = grid_map[(i, j)], grid_map[(ni, nj)]
            if not G.has_edge(u, v):
                attrs = zone_edge_attrs(i, j, ni, nj)
                attrs["length"] = round(attrs["length"] * 1.41, 2)
                G.add_edge(u, v, **attrs)

    # highway skip edges
    for _ in range(int(grid_size * 1.2)):
        i, j = np.random.randint(0, grid_size), np.random.randint(0, grid_size)
        skip = np.random.randint(3, 7)
        for di, dj in [(skip, 0), (0, skip)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                u, v = grid_map[(i, j)], grid_map[(ni, nj)]
                if not G.has_edge(u, v):
                    G.add_edge(u, v,
                               length=round(skip * 52 + np.random.normal(0, 8), 2),
                               energy=round(skip * 75 + np.random.normal(0, 12), 2),
                               traffic=np.random.choice([1, 2]),
                               curvature=round(np.random.uniform(0.005, 0.02), 4))

    start_node = grid_map[(0, 0)]
    goal_node = grid_map[(grid_size - 1, grid_size - 1)]

    return G, grid_map, grid_size, start_node, goal_node


G, grid_map, GRID_SIZE, start_node, goal_node = build_graph()

# ─────────────────────────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {"distance": 1.0, "energy": 0.6, "traffic": 0.8, "curvature": 0.4, "turns": 2.0}


def heuristic(a, b):
    ax, ay = G.nodes[a]["x"], G.nodes[a]["y"]
    bx, by = G.nodes[b]["x"], G.nodes[b]["y"]
    return math.hypot(ax - bx, ay - by)


def scenario_edge_cost(u, v, data, prev_node, weights):
    cost = (
        weights["distance"] * data["length"]
        + weights["energy"] * data["energy"]
        + weights["traffic"] * data["traffic"]
        + weights["curvature"] * data["curvature"]
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
    """A* with scenario-specific weight profile."""
    open_set = []
    heapq.heappush(open_set, (0, start, None))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current, prev = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for neighbor in G.neighbors(current):
            edge_data = G[current][neighbor]
            tentative_g = g_score[current] + scenario_edge_cost(
                current, neighbor, edge_data, prev, weights
            )
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor, current))
    return None


def dijkstra_search(start, goal, weights):
    """Dijkstra's algorithm (no heuristic, guaranteed optimal)."""
    open_set = []
    heapq.heappush(open_set, (0, start, None))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        cost_so_far, current, prev = heapq.heappop(open_set)
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
            tentative_g = g_score[current] + scenario_edge_cost(
                current, neighbor, edge_data, prev, weights
            )
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor, current))
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
    """Bidirectional A* — searches from both ends."""
    open_fwd = [(0, start, None)]
    came_from_fwd = {}
    g_fwd = {start: 0}
    open_bwd = [(0, goal, None)]
    came_from_bwd = {}
    g_bwd = {goal: 0}

    best_cost = float("inf")
    meeting_node = None
    visited_fwd = {}
    visited_bwd = {}

    while open_fwd or open_bwd:
        if open_fwd:
            _, curr_f, prev_f = heapq.heappop(open_fwd)
            visited_fwd[curr_f] = prev_f
            if curr_f in g_bwd:
                total = g_fwd[curr_f] + g_bwd[curr_f]
                if total < best_cost:
                    best_cost = total
                    meeting_node = curr_f
            for nb in G.neighbors(curr_f):
                data = G[curr_f][nb]
                tent = g_fwd[curr_f] + scenario_edge_cost(curr_f, nb, data, prev_f, weights)
                if nb not in g_fwd or tent < g_fwd[nb]:
                    came_from_fwd[nb] = curr_f
                    g_fwd[nb] = tent
                    heapq.heappush(open_fwd, (tent + heuristic(nb, goal), nb, curr_f))

        if open_bwd:
            _, curr_b, prev_b = heapq.heappop(open_bwd)
            visited_bwd[curr_b] = prev_b
            if curr_b in g_fwd:
                total = g_fwd[curr_b] + g_bwd[curr_b]
                if total < best_cost:
                    best_cost = total
                    meeting_node = curr_b
            for nb in G.neighbors(curr_b):
                data = G[curr_b][nb]
                tent = g_bwd[curr_b] + scenario_edge_cost(curr_b, nb, data, prev_b, weights)
                if nb not in g_bwd or tent < g_bwd[nb]:
                    came_from_bwd[nb] = curr_b
                    g_bwd[nb] = tent
                    heapq.heappush(open_bwd, (tent + heuristic(nb, start), nb, curr_b))

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
    T = 100.0

    for _ in range(2000):
        new_path = perturb(current_path)
        new_cost = scenario_path_cost(new_path, weights)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
            current_path = new_path
            current_cost = new_cost
            if current_cost < best_cost:
                best_path = list(current_path)
                best_cost = current_cost
        T = max(0.01, T * 0.995)

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
    for _ in range(100):
        if len(population) >= 30:
            break
        p = get_random_path(start, goal)
        if p is not None:
            population.append(p)
    if not population:
        return None

    best_ever = None
    best_ever_cost = float("inf")

    for gen in range(50):
        costs = [scenario_path_cost(p, weights) for p in population]
        for p, c in zip(population, costs):
            if c < best_ever_cost:
                best_ever = list(p)
                best_ever_cost = c

        def tournament():
            cands = random.sample(range(len(population)), min(3, len(population)))
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
    TAU_MIN, TAU_MAX = 0.1, 10.0
    RHO, Q = 0.3, 100.0
    ALPHA, BETA = 1.0, 2.0
    PHEROMONE_WEIGHT = 0.5

    pheromone = {}
    for u, v in G.edges():
        pheromone[(u, v)] = 1.0
        pheromone[(v, u)] = 1.0

    def get_ph(a, b):
        return pheromone.get((a, b), TAU_MIN)

    # Phase 1: ACO
    best_aco_path = None
    best_aco_cost = float("inf")

    for iteration in range(15):  # reduced iterations for speed
        ant_paths, ant_costs = [], []
        for _ in range(10):
            path = [start]
            visited = {start}
            current = start
            prev = None
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
                    cost = scenario_edge_cost(current, nb, G[current][nb], prev, weights)
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
                prev = current
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

    # Phase 2: Pheromone-guided A*
    open_set = [(0, start, None)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current, prev = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for neighbor in G.neighbors(current):
            edge_data = G[current][neighbor]
            base_cost = scenario_edge_cost(current, neighbor, edge_data, prev, weights)
            tau = get_ph(current, neighbor)
            tau_norm = (tau - TAU_MIN) / (TAU_MAX - TAU_MIN + 1e-9)
            discount = 1.0 - PHEROMONE_WEIGHT * tau_norm
            effective_cost = base_cost * max(discount, 0.1)
            tentative_g = g_score[current] + effective_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), neighbor, current))
    return None


def amcs_search_fn(start, goal, weights):
    """AMCS — Adaptive Multi-Criteria Corridor Search."""
    CORRIDOR_DISCOUNT = 0.4
    LOOKAHEAD_DEPTH = 2
    ADAPT_STRENGTH = 0.3

    # Phase 1: Single-objective corridors
    def dijkstra_single(criterion):
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        visited = set()
        while open_set:
            csf, cur = heapq.heappop(open_set)
            if cur in visited:
                continue
            visited.add(cur)
            if cur == goal:
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                return path[::-1]
            for nb in G.neighbors(cur):
                if nb in visited:
                    continue
                tent = g_score[cur] + G[cur][nb][criterion]
                if nb not in g_score or tent < g_score[nb]:
                    came_from[nb] = cur
                    g_score[nb] = tent
                    heapq.heappush(open_set, (tent, nb))
        return None

    criteria_map = {"length": weights["distance"], "energy": weights["energy"],
                    "traffic": weights["traffic"], "curvature": weights["curvature"]}
    corridor_paths = {c: dijkstra_single(c) for c in criteria_map}

    # Phase 2: Corridor reinforcement
    corridor_score = {}
    for crit, w in criteria_map.items():
        p = corridor_paths[crit]
        if not p:
            continue
        for i in range(len(p) - 1):
            u_, v_ = p[i], p[i + 1]
            corridor_score[(u_, v_)] = corridor_score.get((u_, v_), 0) + w
            corridor_score[(v_, u_)] = corridor_score.get((v_, u_), 0) + w

    if corridor_score:
        ms = max(corridor_score.values())
        for k in corridor_score:
            corridor_score[k] /= ms

    # Precompute global means
    global_means = {
        "length": sum(G[u][v]["length"] for u, v in G.edges()) / G.number_of_edges(),
        "energy": sum(G[u][v]["energy"] for u, v in G.edges()) / G.number_of_edges(),
        "traffic": sum(G[u][v]["traffic"] for u, v in G.edges()) / G.number_of_edges(),
        "curvature": sum(G[u][v]["curvature"] for u, v in G.edges()) / G.number_of_edges(),
    }

    weight_cache = {}

    def adaptive_weights(node):
        if node in weight_cache:
            return weight_cache[node]
        stats = {"length": [], "energy": [], "traffic": [], "curvature": []}
        frontier = [node]
        vis = {node}
        for _ in range(LOOKAHEAD_DEPTH):
            nf = []
            for n in frontier:
                for nb in G.neighbors(n):
                    if nb not in vis:
                        vis.add(nb)
                        nf.append(nb)
                        data = G[n][nb]
                        for k in stats:
                            stats[k].append(data[k])
            frontier = nf
            if not frontier:
                break
        local = {k: (sum(v) / len(v) if v else 0) for k, v in stats.items()}

        base_w = {"length": weights["distance"], "energy": weights["energy"],
                  "traffic": weights["traffic"], "curvature": weights["curvature"]}
        adapted = {}
        for k in base_w:
            if global_means[k] > 0:
                ratio = local[k] / global_means[k]
                adapt = 1.0 + ADAPT_STRENGTH * (ratio - 1.0)
                adapted[k] = base_w[k] * max(0.5, min(2.0, adapt))
            else:
                adapted[k] = base_w[k]
        weight_cache[node] = adapted
        return adapted

    # Phase 3: Corridor-guided A*
    open_set = [(0, start, None)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current, prev = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        aw = adaptive_weights(current)
        for neighbor in G.neighbors(current):
            ed = G[current][neighbor]
            cost = (aw["length"] * ed["length"] + aw["energy"] * ed["energy"]
                    + aw["traffic"] * ed["traffic"] + aw["curvature"] * ed["curvature"])
            if prev is not None:
                prev_xy = (G.nodes[prev]["x"], G.nodes[prev]["y"])
                u_xy = (G.nodes[current]["x"], G.nodes[current]["y"])
                v_xy = (G.nodes[neighbor]["x"], G.nodes[neighbor]["y"])
                v1 = (u_xy[0] - prev_xy[0], u_xy[1] - prev_xy[1])
                v2 = (v_xy[0] - u_xy[0], v_xy[1] - u_xy[1])
                dt = v1[0] * v2[0] + v1[1] * v2[1]
                mg = math.hypot(*v1) * math.hypot(*v2)
                if mg > 0:
                    ang = math.degrees(math.acos(max(-1, min(1, dt / mg))))
                    if ang > 45:
                        cost += weights["turns"]

            c_sc = corridor_score.get((current, neighbor), 0)
            cost *= (1.0 - CORRIDOR_DISCOUNT * c_sc)

            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), neighbor, current))
    return None


# ─────────────────────────────────────────────────────────────────
#  ALGORITHM REGISTRY
# ─────────────────────────────────────────────────────────────────
ALGORITHMS = {
    "Dijkstra": {"fn": dijkstra_search, "color": "#2196F3", "desc": "Classic shortest path — guaranteed optimal"},
    "A*": {"fn": astar_search, "color": "#4CAF50", "desc": "Heuristic-guided search — fast + optimal"},
    "Greedy BFS": {"fn": greedy_bfs_search, "color": "#FF9800", "desc": "Heuristic-only — very fast, not optimal"},
    "Bidirectional A*": {"fn": bidirectional_astar_search, "color": "#9C27B0", "desc": "Searches from both ends simultaneously"},
    "Simulated Annealing": {"fn": sa_search, "color": "#F44336", "desc": "Metaheuristic — escapes local minima via cooling"},
    "Genetic Algorithm": {"fn": ga_search, "color": "#795548", "desc": "Evolutionary — population-based optimization"},
    "Hybrid ACO-A*": {"fn": hybrid_aco_astar_search, "color": "#009688", "desc": "Ant colony + A* — pheromone-guided search"},
    "AMCS (Ours)": {"fn": amcs_search_fn, "color": "#E91E63", "desc": "Adaptive Multi-Criteria Corridor Search"},
}

SCENARIOS = {
    "Fragile Cargo": {
        "icon": "📦", "desc": "Glass/electronics — smooth ride, minimal turns",
        "color": "#e74c3c",
        "weights": {"distance": 0.3, "energy": 0.3, "traffic": 0.4, "curvature": 3.0, "turns": 8.0},
    },
    "Urgent Delivery": {
        "icon": "⚡", "desc": "Time-critical — shortest distance, ignore comfort",
        "color": "#f39c12",
        "weights": {"distance": 3.0, "energy": 0.2, "traffic": 0.3, "curvature": 0.1, "turns": 0.5},
    },
    "Low Battery": {
        "icon": "🔋", "desc": "Battery < 20% — minimize energy at all costs",
        "color": "#2ecc71",
        "weights": {"distance": 0.4, "energy": 3.5, "traffic": 0.2, "curvature": 0.3, "turns": 0.5},
    },
    "Rush Hour": {
        "icon": "🚗", "desc": "Peak traffic — avoid congested roads",
        "color": "#3498db",
        "weights": {"distance": 0.8, "energy": 0.4, "traffic": 4.0, "curvature": 0.3, "turns": 0.5},
    },
    "Heavy Load": {
        "icon": "🏋️", "desc": "50 kg payload — minimize strain and sharp turns",
        "color": "#9b59b6",
        "weights": {"distance": 0.5, "energy": 2.5, "traffic": 0.5, "curvature": 2.0, "turns": 5.0},
    },
    "Night / Stealth": {
        "icon": "🌙", "desc": "Late night — quiet roads, low residential disturbance",
        "color": "#34495e",
        "weights": {"distance": 0.5, "energy": 0.5, "traffic": 3.5, "curvature": 1.5, "turns": 1.0},
    },
    "Balanced": {
        "icon": "⚖️", "desc": "Default — equal priority across all criteria",
        "color": "#7f8c8d",
        "weights": {"distance": 1.0, "energy": 1.0, "traffic": 1.0, "curvature": 1.0, "turns": 2.0},
    },
}


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
    all_x = [G.nodes[n]["x"] for n in G.nodes()]
    all_y = [G.nodes[n]["y"] for n in G.nodes()]
    mx, Mx = min(all_x), max(all_x)
    my, My = min(all_y), max(all_y)
    cx = (mx + Mx) / 2
    cy = (my + My) / 2

    zones = [
        {"name": "Urban Core", "x0": mx, "x1": cx, "y0": cy, "y1": My,
         "color": "rgba(231,76,60,0.08)"},
        {"name": "Suburban", "x0": cx, "x1": Mx, "y0": cy, "y1": My,
         "color": "rgba(46,204,113,0.08)"},
        {"name": "Hilly", "x0": mx, "x1": cx, "y0": my, "y1": cy,
         "color": "rgba(52,152,219,0.08)"},
        {"name": "Industrial", "x0": cx, "x1": Mx, "y0": my, "y1": cy,
         "color": "rgba(155,89,182,0.08)"},
    ]
    return zones


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
    st.caption(f"**Grid:** {GRID_SIZE}×{GRID_SIZE}")
    st.caption(f"**Start:** {start_node} (bottom-left)")
    st.caption(f"**Goal:** {goal_node} (top-right)")

# ─────────────────────────────────────────────────────────────────
#  PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("Multi-Objective Pathfinding Dashboard")
    st.markdown("""
    This dashboard compares **9 pathfinding algorithms** on a synthetic road network
    with **multi-criteria edge costs** (distance, energy, traffic, curvature, turn penalty).

    The graph features **5 spatial zones** with different characteristics, creating genuine
    multi-criteria trade-offs.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", f"{G.number_of_nodes():,}")
    col2.metric("Edges", f"{G.number_of_edges():,}")
    col3.metric("Algorithms", len(ALGORITHMS))
    col4.metric("Scenarios", len(SCENARIOS))

    st.subheader("Zone Map")

    # Build a scatter for all nodes colored by zone
    node_colors = []
    node_x = []
    node_y = []
    zone_labels = []
    all_x = [G.nodes[n]["x"] for n in G.nodes()]
    all_y = [G.nodes[n]["y"] for n in G.nodes()]
    mx, Mx = min(all_x), max(all_x)
    my, My = min(all_y), max(all_y)
    cx_m = (mx + Mx) / 2
    cy_m = (my + My) / 2

    for n in G.nodes():
        x_, y_ = G.nodes[n]["x"], G.nodes[n]["y"]
        node_x.append(x_)
        node_y.append(y_)
        if x_ < cx_m and y_ > cy_m:
            zone_labels.append("Urban Core")
            node_colors.append("#e74c3c")
        elif x_ > cx_m and y_ > cy_m:
            zone_labels.append("Suburban")
            node_colors.append("#2ecc71")
        elif x_ < cx_m and y_ <= cy_m:
            zone_labels.append("Hilly")
            node_colors.append("#3498db")
        elif x_ > cx_m and y_ <= cy_m:
            zone_labels.append("Industrial")
            node_colors.append("#9b59b6")
        else:
            zone_labels.append("Mixed")
            node_colors.append("#7f8c8d")

    fig = go.Figure()
    for zname, zcolor in [("Urban Core", "#e74c3c"), ("Suburban", "#2ecc71"),
                           ("Hilly", "#3498db"), ("Industrial", "#9b59b6")]:
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

    st.subheader("Zone Characteristics")
    zone_df = pd.DataFrame({
        "Zone": ["Urban Core (TL)", "Suburban (TR)", "Hilly (BL)", "Industrial (BR)", "Mixed (Center)"],
        "Distance": ["Short ⬇️", "Long ⬆️", "Medium", "Medium-Long", "Medium"],
        "Energy": ["Low ⬇️", "High ⬆️", "Very Low ⬇️⬇️", "Very High ⬆️⬆️", "Medium"],
        "Traffic": ["Very High ⬆️⬆️", "Very Low ⬇️⬇️", "Medium", "Medium", "High ⬆️"],
        "Curvature": ["Low ⬇️", "High ⬆️", "Very High ⬆️⬆️", "Very Low ⬇️⬇️", "Medium"],
    })
    st.dataframe(zone_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────
#  PAGE: ALGORITHM COMPARISON
# ─────────────────────────────────────────────────────────────────
elif page == "🔬 Algorithm Comparison":
    st.title("Algorithm Comparison")

    st.sidebar.subheader("Weight Profile")
    w_dist = st.sidebar.slider("Distance", 0.0, 5.0, 1.0, 0.1)
    w_energy = st.sidebar.slider("Energy", 0.0, 5.0, 0.6, 0.1)
    w_traffic = st.sidebar.slider("Traffic", 0.0, 5.0, 0.8, 0.1)
    w_curv = st.sidebar.slider("Curvature", 0.0, 5.0, 0.4, 0.1)
    w_turns = st.sidebar.slider("Turns", 0.0, 10.0, 2.0, 0.5)

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

    # ── Improvement vs Balanced ──
    st.subheader("📉 Improvement vs Balanced Scenario")
    if "Balanced" in scenario_results:
        bal = scenario_results["Balanced"]["breakdown"]
        imp_rows = []
        for sname, r in scenario_results.items():
            if sname == "Balanced":
                continue
            b = r["breakdown"]
            # find the primary criterion for this scenario (highest weight)
            sw = SCENARIOS[sname]["weights"]
            primary = max(sw, key=sw.get)
            bal_val = bal[primary]
            scen_val = b[primary]
            if bal_val > 0:
                pct = (bal_val - scen_val) / bal_val * 100
            else:
                pct = 0
            imp_rows.append({"Scenario": f"{SCENARIOS[sname]['icon']} {sname}",
                             "Primary Criterion": primary.title(),
                             "Balanced Value": round(bal_val, 2),
                             "Scenario Value": round(scen_val, 2),
                             "Improvement %": round(pct, 1),
                             "color": SCENARIOS[sname]["color"]})

        imp_df = pd.DataFrame(imp_rows)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[r["Scenario"] for r in imp_rows],
            y=[r["Improvement %"] for r in imp_rows],
            marker_color=[r["color"] for r in imp_rows],
            text=[f"{r['Improvement %']:.1f}%" for r in imp_rows],
            textposition="auto",
        ))
        fig.update_layout(
            template="plotly_dark", height=400,
            title="% Improvement on Primary Criterion vs Balanced",
            yaxis_title="Improvement %",
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(imp_df.drop(columns=["color"]), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────
#  PAGE: CUSTOM SCENARIO
# ─────────────────────────────────────────────────────────────────
elif page == "🎛️ Custom Scenario":
    st.title("Custom Scenario Builder")
    st.markdown("Design your own delivery scenario by adjusting weight sliders, then compare all algorithms.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Weight Sliders")
        cw_dist = st.slider("📏 Distance", 0.0, 5.0, 1.0, 0.1, key="cw_dist")
        cw_energy = st.slider("⚡ Energy", 0.0, 5.0, 1.0, 0.1, key="cw_energy")
        cw_traffic = st.slider("🚗 Traffic", 0.0, 5.0, 1.0, 0.1, key="cw_traffic")
        cw_curv = st.slider("🔄 Curvature", 0.0, 5.0, 1.0, 0.1, key="cw_curv")
        cw_turns = st.slider("↩️ Turns", 0.0, 10.0, 2.0, 0.5, key="cw_turns")

        custom_weights = {"distance": cw_dist, "energy": cw_energy, "traffic": cw_traffic,
                          "curvature": cw_curv, "turns": cw_turns}

        # Quick presets
        st.subheader("Quick Presets")
        preset = st.selectbox("Load a preset", ["(none)"] + list(SCENARIOS.keys()))
        if preset != "(none)":
            pw = SCENARIOS[preset]["weights"]
            st.info(f"**{preset}**: {SCENARIOS[preset]['desc']}")
            st.json(pw)

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
