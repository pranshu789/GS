import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data
u = np.linspace(0, 2 * np.pi, 100)
R = 1.5
r = np.linspace(0, R, 100)
z = np.piecewise(r, [r <= R], [np.pi/8, 0])

# Plot the data using scatter
sc = ax.scatter(r * np.cos(u), r * np.sin(u), z, cmap=cm.coolwarm)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(sc, shrink=0.5, aspect=5)

plt.show()
