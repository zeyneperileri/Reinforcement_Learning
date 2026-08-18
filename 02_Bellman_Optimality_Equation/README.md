# Value Iteration & Contraction Mapping Solver

This repository contains a clean, general-purpose Python implementation of the **Value Iteration** algorithm for Bellman Optimality Equations (BOE). Rather than relying on a hardcoded grid environment, this script is built to handle arbitrary state and action dimensions using pure matrix-vector operations. 

The primary objective of this project is to connect the **Contraction Mapping Theorem** with practical numerical computation.

---

## Mathematical Background

In reinforcement learning, the Bellman Optimality Equation defines the relationship between the optimal value of a state and its successors. 

1. **The Bellman Optimality Operator ($f$):** For a given set of actions with reward vectors ($r_a$) and transition probability matrices ($P_a$), the value update rule is defined as:
   $$v_{k} = \max_a \left( r_a + \gamma P_a v_k \right)$$

2. **Contraction Property:** Because the discount factor $\gamma \in (0, 1)$, the right hand side of the equation is a contraction mapping under the maximum norm ($\|\cdot\|_\infty$). 

3. **Guaranteed Convergence:** By the Contraction Mapping Theorem, successive applications of the operator are mathematically guaranteed to shrink the maximum error exponentially, converging uniquely to the optimal state value vector ($v^*$) regardless of the initial starting guess ($v_0$).

---

## Project Structure

* `contraction_mapping_boe.py`: The core script containing the interactive solver. It prompts for dimensions ($n$ states and $m$ actions), collects reward vectors and transition matrices, and iteratively computes $v^*$.

---

## Code Features

* **Dynamic Dimensions:** Accepts user-defined state ($n$) and action ($m$) sizes.
* **Vectorized Operations:** Uses `numpy` for efficient matrix multiplications (`np.dot`), stacking (`np.vstack`), and maximum extraction (`np.max` / `np.maximum`).
* **Infinity Norm Convergence Check:** Continuously monitors the maximum absolute change between consecutive iterations ($\delta = \|v_{new} - v\|_\infty$).
* **Automatic Termination:** Stops the loop once the error falls below the strict convergence threshold ($\theta = 10^{-6}$), signaling that the algorithm has locked onto the fixed point.

---

## How to Run

Make sure you have Python and `numpy` installed. Run the script directly from your terminal:

```bash
python contraction_mapping_boe.py