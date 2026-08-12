# Iterative Policy Evaluation

This directory contains a Python implementation of the iterative solution for the General Bellman Equation. The algorithm algorithmically evaluates the state-value function $v_\pi$ for a given policy $\pi$ in a Markov Decision Process (MDP).

## Mathematical Framework

The core of the algorithm relies on the iterative update rule derived directly from the Bellman Equation's matrix form:

$$v_{k+1} = r_\pi + \gamma P_\pi v_k$$

*   **$v_k$**: State-value vector at iteration $k$.
*   **$r_\pi$**: Expected immediate reward vector.
*   **$P_\pi$**: State transition probability matrix.
*   **$\gamma$**: Discount rate 

## Convergence Criterion

To determine when the algorithm has reached a stable point, the implementation checks the maximum absolute change between sequential iterations. In mathematical terms, this distance is evaluated using the $L_\infty$ norm:

$$||v_{k+1} - v_k||_\infty = \max_s |v_{k+1}(s) - v_k(s)|$$

The loop naturally terminates when this maximum difference falls below a small threshold ($\theta = 10^{-6}$), indicating that the values have successfully converged.
