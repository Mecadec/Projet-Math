import numpy as np
import statistics
import math
from collections import OrderedDict
import matplotlib.pyplot as plt

# --- Data ------------------------------------------------------------
points = [(1, 1), (1, 2), (1, 5), (3, 4), (4, 3), (6, 2), (0, 4)]
x = [p[0] for p in points]
y = [p[1] for p in points]
n = len(points)

# --- Helper ----------------------------------------------------------
def descriptive(arr):
    mean = sum(arr) / n
    median = statistics.median(arr)
    variance_pop = sum((v - mean) ** 2 for v in arr) / n
    stdev_pop = math.sqrt(variance_pop)
    # unbiased (sample) estimators
    variance_sample = variance_pop * n / (n - 1)
    stdev_sample = math.sqrt(variance_sample)
    minimum = min(arr)
    maximum = max(arr)
    data_range = maximum - minimum
    return OrderedDict([
        ("mean", mean),
        ("median", median),
        ("variance_pop", variance_pop),
        ("stdev_pop", stdev_pop),
        ("variance_sample", variance_sample),
        ("stdev_sample", stdev_sample),
        ("min", minimum),
        ("max", maximum),
        ("range", data_range),
    ])

# --- Calculations ----------------------------------------------------
stats_x = descriptive(x)
stats_y = descriptive(y)

print("=== Descriptive Statistics ===\n")
print("Axis X:")
for k, v in stats_x.items():
    print(f"  {k:15}: {v:.4f}")

print("\nAxis Y:")
for k, v in stats_y.items():
    print(f"  {k:15}: {v:.4f}")

# --- Visualization ---------------------------------------------------
plt.scatter(x, y)
plt.title("Scatter plot of points")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
