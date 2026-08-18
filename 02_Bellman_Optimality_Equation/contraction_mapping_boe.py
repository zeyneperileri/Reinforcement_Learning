import numpy as np


def value_iteration():
    print("=== Bellman Optimality Equation Calculator ===")

    n = int(input("Enter the total number of states (n): "))
    m = int(input("Enter the total number of actions (m): "))
    gamma = float(input("Enter the discount factor (gamma) (ae.g., 0.9): "))

    rewards = []
    transitions = []

    for a in range(m):
        print(f"\n--- Action {a + 1} Parameters ---")

        # Reward vector for action 'a'
        r_input = input(f"Enter {n} reward values for Action {a + 1} (space-separated): ")
        r = np.array([float(x) for x in r_input.split()])
        rewards.append(r)

        # Transition probability matrix for action 'a'
        print(f"Enter the {n}x{n} transition probability matrix row by row:")
        P_rows = []
        for i in range(n):
            row = [float(x) for x in input(f"Row {i + 1}: ").split()]
            P_rows.append(row)
        transitions.append(np.array(P_rows))

    #Initialize state value guess v_0 as a zero vector
    v = np.zeros(n)
    theta = 1e-6
    iteration = 0

    print("\nStarting value iteration loops...")

    #Iterative loop for Bellman Optimality Operator
    while True:
        action_values = []

        # Calculate the expected values for each action using the current state values v: r_a + gamma * P_a * v
        for a in range(m):
            val = rewards[a] + gamma * np.dot(transitions[a], v)
            action_values.append(val)

        # Apply the maximum operator across all actions for each state: v_{k+1} = max_a (r_a + gamma * P_a * v_k)
        v_new = np.max(np.vstack(action_values), axis=0)

        # Calculate the maximum norm for convergence check: ||v_new - v||_infinity
        delta = np.max(np.abs(v_new - v))

        # Update state values for the next iteration step
        v = v_new
        iteration += 1

        # Break the loop if the error falls below the threshold (convergence to unique fixed point v*)
        if delta < theta:
            break

    print(f"\nComputation complete! Converged in {iteration} iterations.")
    print("Final Optimal State Values (v*):")
    for i in range(n):
        print(f"v*(s{i + 1}) = {v[i]:.4f}")


if __name__ == "__main__":
    value_iteration()