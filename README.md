# 🧭 PathLab – Interactive 2D Path Finding Visualizer

PathLab is a **Streamlit-based interactive application** for visualizing and comparing classical **path finding algorithms** on a 2D grid world.
The project is designed as a **learning, experimentation, and demonstration tool** for path planning concepts used in robotics, autonomous navigation, and AI.

---

## 🚀 Features

* 🗺️ **2D Grid-Based Environment**

  * User-defined start and goal positions
  * Randomly generated obstacles
  * Simple abstraction of a navigable environment

* 🧠 **Multiple Path Finding Algorithms**

  * Breadth-First Search (BFS)
  * Depth-First Search (DFS)
  * Dijkstra’s Algorithm
  * A* (A-Star) Algorithm

* 👀 **Algorithm-wise Visualization**

  * Separate tabs for each algorithm
  * Path highlighted from source to destination
  * Easy visual comparison of behavior and performance

* 📊 **Comparison Dashboard**

  * Path length
  * Number of visited nodes
  * Execution time (ms)
  * Tabular and graphical comparison

* ➕ **Add Custom Algorithms**

  * Paste your own path finding algorithm code
  * Automatically validated and added at runtime
  * Encourages experimentation and extensibility

---

## 🖥️ Live Demo

👉 **Streamlit App URL:**
`[PLACEHOLDER – Add Streamlit Cloud App URL Here]`

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **NumPy**
* **Pandas**
* **Matplotlib**

---

## ▶️ How to Run Locally

```bash
pip install streamlit numpy pandas matplotlib
streamlit run app.py
```

Then open the browser at:

```
http://localhost:8501
```

---

## 📁 Project Structure (Current)

```text
app.py        # Single-file Streamlit application
README.md     # Project documentation
```

> Note: The project is intentionally kept as a **single-file application** for simplicity and clarity.

---

## ⚠️ Limitations (Current Version)

This project is **intentionally kept simple** to focus on core algorithmic understanding and visualization.

Current limitations include:

* Uses a **synthetic 2D grid**, not a real-world map
* Obstacles are **randomly generated**, not dynamic
* No traffic, road constraints, or real-time updates
* No robot kinematics or motion constraints
* No ROS / SLAM integration yet

---

## 🔮 Future Enhancements (Planned)

* Replace grid world with **real map data (OpenStreetMap / Map APIs)**
* Integrate **traffic and dynamic cost data** using Traffic APIs
* Support **dynamic replanning (D* Lite / Anytime Planning)**
* ROS2 integration with **occupancy grids and costmaps**
* Step-by-step animation of algorithm execution
* Extension to **3D navigation and multi-goal planning**

---

## 🎯 Motivation

This project was built to:

* Strengthen fundamentals of **path planning algorithms**
* Bridge theory with **visual intuition**
* Serve as a foundation for **robotics, autonomous systems, and research**
* Act as a base for future academic and industry-level extensions

---

## 📜 License

This project is released for **educational and research purposes**.
You are free to fork, modify, and extend it.

