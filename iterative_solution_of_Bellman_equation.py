import numpy as np
def bellman_calculator():
    print("--- General Bellman (Iterative Policy Evaluation) Calculator ---")
    try:
        n = int(input("\nPlease enter the total number of states (e.g., 4): "))
        gamma = float(input("Please input the discount rate: (between 0 and 1): "))
        print(f"\nPlease enter the expected rewards (r_pi) for {n} states, separated by spaces.")
        r_input = input("Example (0 1 1 1): ")
        r_pi = np.array([float(x) for x in r_input.split()])
        if len(r_pi) != n:
            print("Error: The number of rewards entered does not match the number of states!")
            return

        print(f"\nPlease enter the {n}x{n} transition probability matrix (P_pi) row by row.")
        print("Note: The probabilities in each row must sum up to 1. Separate numbers with spaces.")
        P_pi = []
        for i in range(n):
            row_input = input(f"Transition probabilities for state s{i+1}: ")
            row = [float(x) for x in row_input.split()]
            if len(row) != n:
                print("Error: The number of probabilities does not match the number of states!")
                return
            P_pi.append(row)
        P_pi = np.array(P_pi)

        theta = 1e-6
        # Initial guess v_0. We use a zero vector, but any arbitrary vector would work
        # since convergence to a unique fixed point is guaranteed by the Contraction Mapping Theorem.
        v = np.zeros(n)
        v = np.zeros(n)
        iteration = 0

        print("\nStarting iterations...")

        while True:
            # The Bellman Equation Update: v_{k+1} = r_pi + gamma * P_pi * v_k
            v_new = r_pi + gamma * np.dot(P_pi, v)

            # Calculate the maximum change between iterations using the L-infinity norm (Supremum norm)
            delta = np.max(np.abs(v_new - v))

            # Update the value vector
            v = v_new
            iteration += 1

            # Check for convergence (We will consider the algorithm to have converged when the difference between two steps falls below 0.000001)
            if delta < theta:
                break


        print("-" * 40)
        print(f"COMPUTATION COMPLETE! Algorithm converged in {iteration} steps.")
        print("Final State Values (v_pi):")
        for i, val in enumerate(v):
            print(f"v(s{i + 1}) = {val:.4f}") #To avoid unsightly clutter of digits on the screen, only four decimal places of the number will be displayed.

    except ValueError:
        print("\nError: Please enter valid numerical values only!")

bellman_calculator()