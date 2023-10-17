import matplotlib.pyplot as plt
import torch


def generate_maze(grid_size: int) -> torch.Tensor:
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    maze = torch.ones((2 * grid_size + 1, 2 * grid_size + 1), dtype=torch.int8)
    maze[1::2, 1::2] = 0

    visited = torch.zeros((grid_size, grid_size), dtype=torch.bool)
    stack: list[tuple[int, int]] = []

    current_cell = (
        torch.randint(0, grid_size, (1,)).item(),
        torch.randint(0, grid_size, (1,)).item(),
    )
    stack.append(current_cell)
    visited[current_cell] = True

    while stack:
        x, y = current_cell
        neighbors = [
            (x + dx, y + dy)
            for dx, dy in directions
            if 0 <= x + dx < grid_size
            and 0 <= y + dy < grid_size
            and not visited[x + dx, y + dy]
        ]

        if neighbors:
            next_x, next_y = neighbors[torch.randint(0, len(neighbors), (1,)).item()]
            maze[2 * x + 1 + (next_x - x), 2 * y + 1 + (next_y - y)] = 0
            stack.append(current_cell)
            current_cell = (next_x, next_y)
            visited[next_x, next_y] = True
        else:
            current_cell = stack.pop()

    return maze[1:-1, 1:-1]


def maze_to_state_action(maze: torch.Tensor) -> torch.Tensor:
    grid_size = maze.shape[0] // 2 + 1
    actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    state_action_walls = torch.zeros((grid_size, grid_size, 4), dtype=torch.bool)

    for x in range(grid_size):
        for y in range(grid_size):
            for a, (dx, dy) in enumerate(actions):
                maze_x, maze_y = 2 * x + dx, 2 * y + dy
                if (
                    0 <= maze_x < maze.shape[0]
                    and 0 <= maze_y < maze.shape[1]
                    and maze[maze_x, maze_y] == 1
                ):
                    state_action_walls[x, y, a] = True

    return state_action_walls


def visualize(state_action_walls: torch.Tensor) -> None:  # noqa: Vulture
    grid_size = state_action_walls.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))

    for x in range(grid_size):
        for y in range(grid_size):
            ax.plot([x, x], [y, y + 1], "k-", linewidth=0.5)
            ax.plot([x, x + 1], [y, y], "k-", linewidth=0.5)

    for x in range(grid_size):
        for y in range(grid_size):
            if state_action_walls[x, y, 0]:
                ax.plot([x, x + 1], [y, y], "b-", linewidth=3)
            if state_action_walls[x, y, 1]:
                ax.plot([x + 1, x + 1], [y, y + 1], "b-", linewidth=3)
            if state_action_walls[x, y, 2]:
                ax.plot([x, x + 1], [y + 1, y + 1], "b-", linewidth=3)
            if state_action_walls[x, y, 3]:
                ax.plot([x, x], [y, y + 1], "b-", linewidth=3)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.savefig("maze.png")
