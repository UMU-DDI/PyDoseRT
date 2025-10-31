import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(-x / 2)
y4 = np.log(x + 1)
y5 = np.tan(x)
y6 = np.sqrt(x)
y7 = np.power(x, 2)
y8 = np.arctan(x)

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each line with the specified colors
ax.plot(x, y1, label="Red", color="red")
ax.plot(x, y2, label="Green", color="green")
ax.plot(x, y3, label="Blue", color="blue")
ax.plot(x, y4, label="Purple", color="purple")
ax.plot(x, y5, label="Teal", color="teal")
ax.plot(x, y6, label="Indigo", color="indigo")
ax.plot(
    x, y7, label="Deep Orange", color="darkorange"
)  # Replaced 'amber' with 'darkorange'
ax.plot(x, y8, label="Black", color="black")  # Added 'Cyan'

# Customize the plot
ax.set_title("Material Design-Inspired Line Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
