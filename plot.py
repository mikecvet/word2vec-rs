import argparse
import matplotlib.pyplot as plt
import numpy as np

# run via $ python3 plot.py -f ./entropy.out

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True, help='data file, floats separated by newlines')
args = parser.parse_args()

# Read data from file
with open(args.file, 'r') as f:
    numbers = [float(line.strip()) for line in f.readlines()]

# Set up x-axis values
x_values = list(range(1, len(numbers) + 1))

# Compute polynomial fit (degree 3)
z = np.polyfit(x_values, numbers, 3)
p = np.poly1d(z)

# Plot the raw data points
plt.plot(x_values, numbers, marker='o', label='Data Points')

# Plot the smoothed trend line
plt.plot(x_values, p(x_values), 'r-', label='Trend Line')

plt.title("Visualization of cross-entropy error")
plt.xlabel("Epochs")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
