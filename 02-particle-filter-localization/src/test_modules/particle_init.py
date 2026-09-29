import numpy as np
import matplotlib.pyplot as plt
import random

# Map parameters (from .yaml)
MAP_PATH = "../../maps/binary_map.npy"  # Adjust path if needed
MAP_RESOLUTION = 0.001  # meters per pixel
MAP_ORIGIN = [-1.25, -1.25]  # world coordinates

NUM_PARTICLES = 500

# 1. Load binary map
map_data = np.load(MAP_PATH)

# 2. Get indices of free space (value == 1)
free_cells = np.argwhere(map_data == 1)

# 3. Randomly sample from free cells
sample_indices = np.random.choice(len(free_cells), NUM_PARTICLES)
sampled_cells = free_cells[sample_indices]

# 4. Convert to world coordinates and assign random orientation
particles = []
for cell in sampled_cells:
    i, j = cell
    x = MAP_ORIGIN[0] + j * MAP_RESOLUTION
    y = MAP_ORIGIN[1] + i * MAP_RESOLUTION
    theta = random.uniform(-np.pi, np.pi)
    particles.append((x, y, theta))

particles = np.array(particles)  # shape: (500, 3)

# 5. Optional: Save or plot
np.save("../../maps/initial_particles.npy", particles)

# 6. Plot for debug
plt.figure(figsize=(8, 8))
plt.imshow(map_data, cmap='gray', origin='lower')
plt.scatter((particles[:,0] - MAP_ORIGIN[0]) / MAP_RESOLUTION,
            (particles[:,1] - MAP_ORIGIN[1]) / MAP_RESOLUTION,
            s=3, c='red', label='Particles')
plt.title("Sampled Particles in Free Space")
plt.legend()
plt.grid(True)
plt.show()
