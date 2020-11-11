from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X_new = np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]])
knn = KNeighborsClassifier(n_neighbors=1)
wine_dataset = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine_dataset['data'], wine_dataset['target'], random_state=0
)

knn.fit(X_train, y_train)
# print('测试数据集得分:{:.2f}'.format(knn.score(X_test, y_test)))
prediction = knn.predict(X_new)
print('预测红酒分类为：{}'.format(wine_dataset['target_names'][prediction]))
