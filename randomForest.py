import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

headernames=['sepal-length','sepal-width', 'petal-length', 'petal-width', 'Class']

#Run the classofier on the mesh grid
output= classifier.predict(np.c_[x_vals.ravel(),t_vals.ravel()])
#Reshape the output array
output = output.reshape(x_vals.shape)

dataset=pd.read_csv(path, names=headernames)
dataset.head()