import streamlit as st
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from queue import PriorityQueue, Queue
import types

# -----------------------------
# CONFIG
# -----------------------------
GRID_SIZE = 20

st.set_page_config(layout="wide", page_title="PathLab – Path Finding Visualizer")

# -----------------------------
# GRID UTILS
# -----------------------------
def create_grid(size):
    return np.zeros((size, size), dtype=int)

def neighbors(pos, grid):
    x, y = pos
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx, ny] != 1:
                yield (nx, ny)

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# -----------------------------
# BASE CLASS
# -----------------------------
class PathFindingAlgorithm:
    name = "Base"

    def find_path(self, grid, start, goal):
        raise NotImplementedError

# -----------------------------
# ALGORITHMS
# -----------------------------
class BFS(PathFindingAlgorithm):
    name = "BFS"

    def find_path(self, grid, start, goal):
        q = Queue()
        q.put(start)
        came_from = {start: None}
        visited = 0

        while not q.empty():
            current = q.get()
            visited += 1
            if current == goal:
                break
            for n in neighbors(current, grid):
                if n not in came_from:
                    came_from[n] = current
                    q.put(n)

        return reconstruct(came_from, start, goal), visited

class DFS(PathFindingAlgorithm):
    name = "DFS"

    def find_path(self, grid, start, goal):
        stack = [start]
        came_from = {start: None}
        visited = 0

        while stack:
            current = stack.pop()
            visited += 1
            if current == goal:
                break
            for n in neighbors(current, grid):
                if n not in came_from:
                    came_from[n] = current
                    stack.append(n)

        return reconstruct(came_from, start, goal), visited

class Dijkstra(PathFindingAlgorithm):
    name = "Dijkstra"

    def find_path(self, grid, start, goal):
        pq = PriorityQueue()
        pq.put((0, start))
        came_from = {start: None}
        cost = {start: 0}
        visited = 0

        while not pq.empty():
            _, current = pq.get()
            visited += 1
            if current == goal:
                break
            for n in neighbors(current, grid):
                new_cost = cost[current] + 1
                if n not in cost or new_cost < cost[n]:
                    cost[n] = new_cost
                    pq.put((new_cost, n))
                    came_from[n] = current

        return reconstruct(came_from, start, goal), visited

class AStar(PathFindingAlgorithm):
    name = "A*"

    def find_path(self, grid, start, goal):
        pq = PriorityQueue()
        pq.put((0, start))
        came_from = {start: None}
        g = {start: 0}
        visited = 0

        while not pq.empty():
            _, current = pq.get()
            visited += 1
            if current == goal:
                break
            for n in neighbors(current, grid):
                new_g = g[current] + 1
                if n not in g or new_g < g[n]:
                    g[n] = new_g
                    f = new_g + heuristic(n, goal)
                    pq.put((f, n))
                    came_from[n] = current

        return reconstruct(came_from, start, goal), visited

# -----------------------------
# HELPERS
# -----------------------------
def reconstruct(came_from, start, goal):
    if goal not in came_from:
        return []
    cur = goal
    path = []
    while cur:
        path.append(cur)
        cur = came_from[cur]
    return path[::-1]

def plot_grid(grid, path, start, goal):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="gray_r")

    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, color="blue")

    ax.scatter(start[1], start[0], color="green", s=100)
    ax.scatter(goal[1], goal[0], color="red", s=100)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

# -----------------------------
# REGISTRY
# -----------------------------
if "algorithms" not in st.session_state:
    st.session_state.algorithms = {
        "BFS": BFS(),
        "DFS": DFS(),
        "Dijkstra": Dijkstra(),
        "A*": AStar()
    }

# -----------------------------
# UI – GRID SETUP
# -----------------------------
st.sidebar.title("PathLab Controls")

grid = create_grid(GRID_SIZE)

start = st.sidebar.slider("Start X", 0, GRID_SIZE-1, 0), st.sidebar.slider("Start Y", 0, GRID_SIZE-1, 0)
goal = st.sidebar.slider("Goal X", 0, GRID_SIZE-1, GRID_SIZE-1), st.sidebar.slider("Goal Y", 0, GRID_SIZE-1, GRID_SIZE-1)

obstacle_count = st.sidebar.slider("Random Obstacles", 0, 80, 30)
np.random.seed(0)
for _ in range(obstacle_count):
    x, y = np.random.randint(0, GRID_SIZE, 2)
    if (x, y) not in [start, goal]:
        grid[x, y] = 1

# -----------------------------
# RUN ALGORITHMS
# -----------------------------
results = {}

for name, algo in st.session_state.algorithms.items():
    t0 = time.time()
    path, visited = algo.find_path(grid, start, goal)
    t1 = time.time()
    results[name] = {
        "path": path,
        "visited": visited,
        "time_ms": (t1 - t0)*1000,
        "path_len": len(path)
    }

# -----------------------------
# VISUALIZATION TABS
# -----------------------------
st.title("2D Path Finding Algorithm Visualizer")

tabs = st.tabs(list(results.keys()))

for tab, name in zip(tabs, results.keys()):
    with tab:
        st.pyplot(plot_grid(grid, results[name]["path"], start, goal))
        st.write(results[name])

# -----------------------------
# DASHBOARD
# -----------------------------
st.subheader("Algorithm Comparison Dashboard")

df = pd.DataFrame([
    {
        "Algorithm": name,
        "Path Length": r["path_len"],
        "Visited Nodes": r["visited"],
        "Time (ms)": r["time_ms"]
    }
    for name, r in results.items()
])

st.dataframe(df)

st.bar_chart(df.set_index("Algorithm")[["Visited Nodes", "Time (ms)"]])

# -----------------------------
# ADD ALGORITHM SECTION
# -----------------------------
st.subheader("➕ Add Custom Algorithm")

code = st.text_area(
    "Paste algorithm code here (must define a class inheriting PathFindingAlgorithm)",
    height=250
)

if st.button("Validate & Add Algorithm"):
    try:
        local_env = {}
        exec(code, globals(), local_env)

        new_algo = None
        for obj in local_env.values():
            if isinstance(obj, type) and issubclass(obj, PathFindingAlgorithm):
                new_algo = obj()

        if new_algo:
            st.session_state.algorithms[new_algo.name] = new_algo
            st.success(f"Algorithm '{new_algo.name}' added successfully!")
        else:
            st.error("No valid algorithm class found.")

    except Exception as e:
        st.error(str(e))
