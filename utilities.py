import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


from utility import visualize_classifier

#Samples with 2D Vectors and their labels

X= np.array ([
    [3.1 ,7.2],[4 ,6.7],[2.9 ,8], [5.1 ,4.5], [6, 5],       [5.6, 5], [3.3, 0.4], [3.9 , 0.9], [2.8, 1],[0.5, 3.4], [1,4], [0.6, 4.9]
    ])
y=np.array([0,0,0,1,1,1,2,2,2,3,3,3])

#Training logisic regression classifier 
classifier= linear_model.LogisticRegression(solver='liblinear', C=1)

#Train 
classifier.fit(X,y)

#Visualize the performance of classfier
visualize_classifier(classifier, X, y)


classifier= linear_model.LogisticRegression(solver='liblinear', C=100)

