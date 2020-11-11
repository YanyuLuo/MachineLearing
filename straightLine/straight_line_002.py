import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = [[1], [4]]
y = [3, 5]
lr = LinearRegression().fit(X, y)
z = np.linspace(0, 5, 20)
plt.scatter(X, y, s=80)
plt.plot(z, lr.predict(z.reshape(-1, 1)), c='k')
plt.title('Straight Line')
print('y = {:.3f}'.format(lr.coef_[0]), 'x', '+{:.3f}'.format(lr.intercept_))
plt.show()
