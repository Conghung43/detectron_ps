import numpy as np
import matplotlib.pyplot as plt
  
y, x = np.mgrid[:5, 1:6]
poly_coords = [
    (0.25, 2.75), (3.25, 2.75),
    (2.25, 0.75), (0.25, 0.75)
]
fig, ax = plt.subplots()
  
cells = ax.plot(x, y, x + y, color ='green')
ax.add_patch(
    plt.Polygon(poly_coords, color ='forestgreen',
                alpha = 0.5)
    )
ax.margins(x = 0.1, y = 0.05)
ax.set_aspect('equal')
  
fig.suptitle('matplotlib.axes.Axes.add_patch() \
function Example\n\n', fontweight ="bold")
plt.show()