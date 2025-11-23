import numpy as np
from matplotlib import pyplot as plt

x_values = np.arange(1, 5, 1)
y_values = x_values ** 2

plt.scatter(x_values, y_values)
plt.show()