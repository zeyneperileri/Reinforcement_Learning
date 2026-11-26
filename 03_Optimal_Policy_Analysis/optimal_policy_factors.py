import os
import sys
import numpy as np

# Ensure the project root is in Python's search path so `env` can be imported anywhere
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.grid_world import GridWorld

# Create an images folder inside 03_Optimal_Policy_Analysis for saving plots
IMG_DIR = os.path.join(CURRENT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)


def solve_boe(env, gamma=0.9, theta=1e-10, max_iters=10000):
    """
    Solve the Bellman Optimality Equation via Contraction Mapping (Value Iteration).
    Returns optimal state values (v_star), optimal policy (pi_star), and iteration count.
    """
    v = np.zeros(env.num_states)
    policy = np.zeros(env.num_states, dtype=int)

    for iteration in range(1, max_iters + 1):
        v_new = np.zeros_like(v)

        for s_idx in range(env.num_states):
            state = env.idx_to_state(s_idx)
            q_values = np.zeros(env.num_actions)

            for a in range(env.num_actions):
                next_state, reward = env.get_next_state_and_reward(state, a)
                next_idx = env.state_to_idx(next_state)
                q_values[a] = reward + gamma * v[next_idx]

            v_new[s_idx] = np.max(q_values)
            policy[s_idx] = int(np.argmax(np.round(q_values, 7)))

        if np.max(np.abs(v_new - v)) < theta:
            return v_new, policy, iteration

        v = v_new

    return v, policy, max_iters


if __name__ == "__main__":
    forbidden = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]
    gamma_default = 0.9

    # Experiment 1: Baseline
    env_base = GridWorld(
        rows=5, cols=5,
        target_state=(3, 2),
        forbidden_states=forbidden,
        r_target=1.0, r_forbidden=-1.0, r_boundary=-1.0, r_other=0.0,
    )
    v_base, pi_base, iters_base = solve_boe(env_base, gamma=gamma_default)
    print(f"[Exp 1 - Baseline] Converged in {iters_base} iterations.")
    env_base.plot(
        values=v_base,
        policy=pi_base,
        show_agent=False,
        title="Baseline: γ = 0.9, r_forbidden = -1 (Takes Risky Shortcut)",
        save_path=os.path.join(IMG_DIR, "exp1_baseline.png"),
    )

    # Experiment 2: High Forbidden Penalty
    env_strict = GridWorld(
        rows=5, cols=5,
        target_state=(3, 2),
        forbidden_states=forbidden,
        r_target=1.0, r_forbidden=-10.0, r_boundary=-1.0, r_other=0.0,
    )
    v_strict, pi_strict, iters_strict = solve_boe(env_strict, gamma=gamma_default)
    print(f"[Exp 2 - High Penalty] Converged in {iters_strict} iterations.")
    env_strict.plot(
        values=v_strict,
        policy=pi_strict,
        show_agent=False,
        title="High Penalty: γ = 0.9, r_forbidden = -10 (Takes Safe Detour)",
        save_path=os.path.join(IMG_DIR, "exp2_high_penalty.png"),
    )

    # Experiment 3: Lower Discount Rates
    v_mid, pi_mid, iters_mid = solve_boe(env_base, gamma=0.5)
    print(f"[Exp 3a - gamma=0.5] Converged in {iters_mid} iterations.")
    env_base.plot(
        values=v_mid,
        policy=pi_mid,
        show_agent=False,
        title="Short-Sighted: γ = 0.5, r_forbidden = -1 (Avoids Shortcut)",
        save_path=os.path.join(IMG_DIR, "exp3a_gamma_05.png"),
    )

    v_myopic, pi_myopic, iters_myopic = solve_boe(env_base, gamma=0.0)
    print(f"[Exp 3b - gamma=0.0] Converged in {iters_myopic} iterations.")
    env_base.plot(
        values=v_myopic,
        policy=pi_myopic,
        show_agent=False,
        title="Completely Myopic: γ = 0.0 (Only Immediate Rewards Matter)",
        save_path=os.path.join(IMG_DIR, "exp3b_gamma_00.png"),
    )

    # Experiment 4: Positive Affine Reward Transformation Invariance
    a, b = 2.0, 3.0
    env_affine = GridWorld(
        rows=5, cols=5,
        target_state=(3, 2),
        forbidden_states=forbidden,
        r_target=a * 1.0 + b,
        r_forbidden=a * (-1.0) + b,
        r_boundary=a * (-1.0) + b,
        r_other=a * 0.0 + b,
    )
    v_affine, pi_affine, iters_affine = solve_boe(env_affine, gamma=gamma_default)

    v_theoretical = a * v_base + b / (1.0 - gamma_default)
    values_match = np.allclose(v_affine, v_theoretical, atol=1e-6)
    policy_identical = np.array_equal(pi_base, pi_affine)

    print("\n[Exp 4 - Affine Transformation (r' = 2r + 3)]")
    print(f"  v'* == a*v* + b/(1-gamma) holds? : {values_match}")
    print(f"  Optimal policy unchanged (pi'* == pi*)? : {policy_identical}")
    print(f"  Max absolute value error: {np.max(np.abs(v_affine - v_theoretical)):.2e}")