import pygame
import numpy as np
from mesa import Model, Agent
from mesa.time import RandomActivation

# Initialize Pygame
pygame.init()

# Set the width and height of the screen [width, height]
screen_width = 500
screen_height = 500
size = (screen_width, screen_height)
screen = pygame.display.set_mode(size)

# Colors
WHITE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
MAGENTA = (0, 128, 255)

# Load the maze image
maze_image = pygame.image.load("maze.png")
maze_image = pygame.transform.scale(maze_image, size)

# Get the size of the image
image_size = maze_image.get_size()

# Convert the maze image to a matrix representation
maze_matrix = np.zeros(image_size)
for row in range(image_size[1]):
    for col in range(image_size[0]):
        if maze_image.get_at((col, row)) == (0, 0, 0):
            maze_matrix[row][col] = 1

# Set the starting and ending positions of the maze
start_pos = (0, 0)
end_pos = (size[0]-1, size[1]-1)

# Represents the model of the maze and manages the agent's movement through the maze.
class MazeModel(Model):
    def __init__(self, maze_matrix, start_pos, end_pos, screen):
        super().__init__()
        self.maze_matrix = maze_matrix
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.schedule = RandomActivation(self)
        self.agent = MazeAgent(1, self)
        self.renderer = MazeRenderer(2, self, screen)
        self.schedule.add(self.agent)
        self.schedule.add(self.renderer)

    def step(self):
        self.schedule.step()

# Represents the agent that navigates through the maze using depth-first search (DFS).
class MazeAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.stack = [model.start_pos]
        self.visited = set()
        self.visited.add(model.start_pos)
        self.path = {model.start_pos: None}

    def step(self):
        if self.stack:
            current_pos = self.stack.pop()
            if current_pos == self.model.end_pos:
                self.model.schedule.remove(self)
                return
            for direction in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                next_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                if next_pos[0] < 0 or next_pos[0] >= self.model.maze_matrix.shape[0] or next_pos[1] < 0 or next_pos[1] >= self.model.maze_matrix.shape[1]:
                    continue
                if self.model.maze_matrix[next_pos[0]][next_pos[1]] == 1 or next_pos in self.visited:
                    continue
                self.visited.add(next_pos)
                self.stack.append(next_pos)
                self.path[next_pos] = current_pos
                self.model.renderer.update_agent_position(current_pos, next_pos)  # Notify the renderer
                # Draw the new position on the screen
                pygame.draw.rect(screen, GREEN, (next_pos[1], next_pos[0], 1, 1))
                pygame.display.update()

# Draw the maze on the screen
# Renders the maze and the path on the screen using Pygame
# Handles user input and events, such as quitting the program or resizing the window
for row in range(size[1]):
    for col in range(size[0]):
        if maze_matrix[row][col] == 1:
            pygame.draw.rect(screen, (0, 0, 0), (col, row, 1, 1))
        else:
            pygame.draw.rect(screen, (255, 255, 255), (col, row, 1, 1))


# This function extracts the path from the path dictionary by traversing it backwards from the end position to the
# start position
def extract_path(path_dict, start_pos, end_pos):
    path = []
    current_pos = end_pos
    while current_pos != start_pos:
        path.append(current_pos)
        current_pos = path_dict[current_pos]
    path.reverse()
    return path

# MazeRenderer is responsible for rendering the agent's movements on the screen.
# This improves multi-agency aspect and agent communication
class MazeRenderer(Agent):
    def __init__(self, unique_id, model, screen):
        super().__init__(unique_id, model)
        self.screen = screen

    def update_agent_position(self, old_pos, new_pos):
        print(f"OLD Position: {old_pos[1]} & {old_pos[0]}")
        print(f"NEW Position: {new_pos[1]} & {new_pos[0]}")

# Create the MESA model
maze_model = MazeModel(maze_matrix, start_pos, end_pos, screen)

# Run the MESA model until the agent finds a path
while maze_model.agent in maze_model.schedule.agents:
    maze_model.step()

# Get the path from the MESA agent
path = extract_path(maze_model.agent.path, start_pos, end_pos)

# Maze dimensions
maze_rows = len(maze_matrix)
maze_columns = len(maze_matrix[0])

# Calculate cell size
cell_size = min(screen_width // maze_columns, screen_height // maze_rows)

# Draw the path
for pos in path:
    pygame.draw.rect(screen, RED, (pos[1], pos[0], 1, 1))
    pygame.display.update()

# Update the display
pygame.display.update()

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
