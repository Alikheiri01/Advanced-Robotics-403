import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def map(image_path="/home/nvacenter/Desktop/anki/old/cropped_binary_map.pgm",
        output_file="map_matrix.txt", size=(500,500)):
    # Open and convert to grayscale
    im = Image.open(image_path).convert("L")

    # Threshold: make it binary (1 for obstacle, 0 for free)
    binary = im.point(lambda p: 0 if p > 128 else 1)

    # Resize
    binary = binary.resize(size)
    w, h = binary.size

    # Convert to numpy array
    nim = np.array(binary, dtype=np.uint8)

    # Save to text file
    np.savetxt(output_file, nim, fmt='%d')

    return nim

# Example usage
matrix = map()
print(matrix)
free = np.sum(matrix == 0)
obstacles = np.sum(matrix == 1)
print(f"Free cells: {free}, Obstacles: {obstacles}")

# Flip vertically for display
matrix_flipped = np.flipud(matrix)

plt.imshow(matrix_flipped, cmap="gray", origin="lower")
plt.title("Binary Map (Flipped Vertically)")
plt.show()
