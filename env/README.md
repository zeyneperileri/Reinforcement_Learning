# Grid World Environment

A reusable, customizable Grid World environment designed to match the examples in *Mathematical Foundations of Reinforcement Learning* by Shiyu Zhao.

This module acts as a shared foundation across chapters—supporting both **Model-Based** algorithms (Dynamic Programming: Value Iteration, Policy Iteration) and **Model-Free** algorithms (Monte Carlo, Temporal-Difference, Q-Learning), along with textbook-style Matplotlib visualizations.

---

## Environment Specifications

* **State Space ($\mathcal{S}$):** 2D grid coordinates `(row, col)` where `(0, 0)` is the top-left corner. Helper methods `state_to_idx` and `idx_to_state` map coordinates to flat indices `0 .. (rows * cols - 1)` for matrix-vector Bellman operations.
* **Action Space ($\mathcal{A}$):** 5 discrete actions:

| Action ID | Movement | Symbol |
| :---: | :--- | :---: |
| `0` | Up | `↑` |
| `1` | Right | `→` |
| `2` | Down | `↓` |
| `3` | Left | `←` |
| `4` | Stay (in place) | `○` |

* **Cell Types & Default Rewards:**
  * **Target cell (Blue):** `+1.0`
  * **Forbidden cell (Orange):** `-1.0`
  * **Boundary collision:** `-1.0` (agent stays in its current cell)
  * **Normal cell (White):** `0.0`

---

## Configuration Parameters

You do **not** need to modify `grid_world.py` to change the grid layout or reward structure. Pass your desired parameters when instantiating `GridWorld`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `rows`, `cols` | `int` | `5`, `5` | Dimensions of the grid |
| `start_state` | `tuple` | `(0, 0)` | Initial agent position `(row, col)` |
| `target_state` | `tuple` | `(3, 2)` | Target cell coordinates |
| `forbidden_states` | `list[tuple]` | `None` | List of forbidden/obstacle cells |
| `r_target` | `float` | `1.0` | Reward for entering/staying in the target cell |
| `r_forbidden` | `float` | `-1.0` | Penalty for entering a forbidden cell |
| `r_boundary` | `float` | `-1.0` | Penalty for attempting to move outside the grid |
| `r_other` | `float` | `0.0` | Reward for moving into a regular cell |

---

## Usage Examples

### 1. Creating a Custom Grid
```python
from env.grid_world import GridWorld

env = GridWorld(
    rows=4,
    cols=4,
    start_state=(0, 0),
    target_state=(3, 3),
    forbidden_states=[(1, 1), (2, 1), (2, 2)],
    r_target=10.0,
    r_forbidden=-10.0,
    r_boundary=-1.0,
    r_other=-0.1,
)
```

### 2. Model-Based Usage (Dynamic Programming / Bellman Equations)
Query transitions for any `(state, action)` pair without altering the agent's current position:
```python
next_state, reward = env.get_next_state_and_reward(state=(0, 0), action=1)
```

### 3. Model-Free Usage (Monte Carlo / TD Learning)
Interact with the environment step by step to generate episodes:
```python
state = env.reset()
next_state, reward = env.step(action=1)
```

### 4. Visualizing State Values & Policy
Plot the grid alone, or overlay computed state values (`values`) and policy arrows (`policy`):
```python
# Plot empty grid with agent
env.plot(title="Initial Grid World")

# Plot learned state values and optimal policy (1D or 2D arrays)
env.plot(
    values=state_values,
    policy=optimal_policy,
    show_agent=False,
    title="Optimal Policy & State Values",
    save_path="optimal_policy.png",  # Optional: save figure to file
)
```

### Quick Preview
Run the file directly from the project root to preview the default 5×5 layout:
```bash
python env/grid_world.py
```