
import numpy as np
import json

# Load the tensor
tensor = np.load("/Volumes/2TB/test-burn/pytorch/dumps/qwen25_20250924_191143/arrays/embeddings.npy")

# Print some basic info
print(f"Shape: {tensor.shape}")
print(f"Data type: {tensor.dtype}")

# Print the first 5 elements of the first row
print("First 5 elements of the first row:")
print(tensor[0, 0, :5])

# Also, let\'s look at the corresponding stats in stats.json
with open("/Volumes/2TB/test-burn/pytorch/dumps/qwen25_20250924_191143/stats.json", "r") as f:
    stats = json.load(f)

print("\n--- From stats.json ---")
print(json.dumps(stats["stats"]["embeddings"], indent=2))
