import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-5, 5, 100)
y = 0.5*x + 3
plt.plot(x, y, c='orange')
plt.title('Straight Line')
plt.show()
