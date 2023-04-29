import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib. pyplot as plt

iris = datasets.load_iris()

X= iris.data[:, :2]
y= iris.target

C=1.0
# Define a meshgrid of points that spans the feature space
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
X_plot = np.c_[xx.ravel(), yy.ravel()]


svc_classifier=svm.SVC(kernel='linear', C=C).fit(X , y)
Z= svc_classifier.predict(X_plot)
Z= Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contour(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with linear kernel')
plt.show()