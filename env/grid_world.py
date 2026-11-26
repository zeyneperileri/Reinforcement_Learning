import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GridWorld:
    """
    Standard Grid World environment following Shiyu Zhao's
    'Mathematical Foundations of Reinforcement Learning'.
    Supports both model-based (DP) and model-free (MC, TD) algorithms.
    """

    ACTIONS = {
        0: (-1, 0),  # Up
        1: (0, 1),   # Right
        2: (1, 0),   # Down
        3: (0, -1),  # Left
        4: (0, 0),   # Stay (a_5 in the book)
    }

    ACTION_SYMBOLS = {0: "↑", 1: "→", 2: "↓", 3: "←", 4: "○"}

    def __init__(
        self,
        rows=5,
        cols=5,
        start_state=(0, 0),
        target_state=(3, 2),
        forbidden_states=None,
        r_target=1.0,
        r_forbidden=-1.0,
        r_boundary=-1.0,
        r_other=0.0,
    ):
        self.rows = rows
        self.cols = cols
        self.num_states = rows * cols
        self.num_actions = len(self.ACTIONS)

        self.start_state = start_state
        self.target_state = target_state
        self.forbidden_states = set(forbidden_states) if forbidden_states else set()

        self.r_target = r_target
        self.r_forbidden = r_forbidden
        self.r_boundary = r_boundary
        self.r_other = r_other

        self.current_state = self.start_state

    def state_to_idx(self, state):
        """Convert (row, col) coordinates to a single state index 0..(N-1)."""
        row, col = state
        return row * self.cols + col

    def idx_to_state(self, idx):
        """Convert a single state index to (row, col) coordinates."""
        return divmod(idx, self.cols)

    def get_next_state_and_reward(self, state, action):
        """Return (next_state, reward) without changing self.current_state."""
        row, col = state
        d_row, d_col = self.ACTIONS[action]
        next_row, next_col = row + d_row, col + d_col

        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            return state, self.r_boundary

        next_state = (next_row, next_col)

        if next_state == self.target_state:
            return next_state, self.r_target
        if next_state in self.forbidden_states:
            return next_state, self.r_forbidden

        return next_state, self.r_other

    def reset(self, start_state=None):
        """Reset the environment to the starting state."""
        self.current_state = start_state if start_state is not None else self.start_state
        return self.current_state

    def step(self, action):
        """Take one step in the environment and return (next_state, reward)."""
        next_state, reward = self.get_next_state_and_reward(self.current_state, action)
        self.current_state = next_state
        return next_state, reward

    def plot(self, values=None, policy=None, show_agent=True, title="Grid World", save_path=None):
        """
        Visualize the Grid World in Shiyu Zhao's textbook style.
        - values: optional 1D (N,) or 2D (rows, cols) state-value array
        - policy: optional 1D (N,) or 2D (rows, cols) array of action IDs (0..4)
        """
        fig, ax = plt.subplots(figsize=(self.cols * 1.15, self.rows * 1.15))

        val_grid = None
        if values is not None:
            val_grid = np.asarray(values, dtype=float).reshape(self.rows, self.cols)

        pol_grid = None
        if policy is not None:
            pol_grid = np.asarray(policy, dtype=int).reshape(self.rows, self.cols)

        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)
                if pos == self.target_state:
                    facecolor = "#5DADE2"  # Target (Blue)
                elif pos in self.forbidden_states:
                    facecolor = "#F5B041"  # Forbidden (Orange/Amber)
                else:
                    facecolor = "#FFFFFF"  # Normal (White)

                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    facecolor=facecolor,
                    edgecolor="#2C3E50",
                    linewidth=1.5,
                )
                ax.add_patch(rect)

                # State values
                if val_grid is not None:
                    y_text = r + 0.80 if pol_grid is not None else r + 0.50
                    ax.text(
                        c + 0.5, y_text, f"{val_grid[r, c]:.2f}",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color="#1B2631",
                    )

                # Policy arrows
                if pol_grid is not None:
                    action = pol_grid[r, c]
                    y_center = r + 0.36 if val_grid is not None else r + 0.50
                    if action == 4:  # Stay
                        circle = patches.Circle(
                            (c + 0.5, y_center), 0.11,
                            fill=False, edgecolor="#154360", linewidth=2.2,
                        )
                        ax.add_patch(circle)
                    else:
                        d_row, d_col = self.ACTIONS[action]
                        ax.annotate(
                            "",
                            xy=(c + 0.5 + d_col * 0.28, y_center + d_row * 0.28),
                            xytext=(c + 0.5 - d_col * 0.18, y_center - d_row * 0.18),
                            arrowprops=dict(arrowstyle="->", color="#154360", lw=2.2),
                        )

        # Agent marker
        if show_agent:
            ar, ac = self.current_state
            agent_dot = patches.Circle(
                (ac + 0.5, ar + 0.5), 0.22,
                facecolor="#2E86C1", edgecolor="#1B2631", linewidth=1.5, zorder=5,
            )
            ax.add_patch(agent_dot)
            ax.text(
                ac + 0.5, ar + 0.5, "A",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=6,
            )

        ax.set_xlim(0, self.cols)
        ax.set_ylim(self.rows, 0)
        ax.set_aspect("equal")

        ax.set_xticks(np.arange(self.cols) + 0.5)
        ax.set_yticks(np.arange(self.rows) + 0.5)
        ax.set_xticklabels(range(self.cols), fontsize=10)
        ax.set_yticklabels(range(self.rows), fontsize=10)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    # Classic 5x5 grid layout from Shiyu Zhao's book
    forbidden = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]
    env = GridWorld(
        rows=5,
        cols=5,
        start_state=(0, 0),
        target_state=(3, 2),
        forbidden_states=forbidden,
    )

    env.plot(title="Shiyu Zhao 5x5 Grid World")