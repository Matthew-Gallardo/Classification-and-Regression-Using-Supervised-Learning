import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X ,y, title= ''):
    #Minimuum and Maximum values for X and Y
    #that will used in mesh grid
    minX , maxX = X[:,0].min() -1.0, X[:,0].max()+1.0
    minY , maxY = X[:,1].min() -1.0, X[:,1].max()+1.0

    #Step size to use in plotting the mesh grid
    meshStep_size=0.01

    #Mesh grid of X and Y values
    xVals, yVals= np.meshgrid(np.arange(minX, maxX, meshStep_size), np.arange(minY, maxY, meshStep_size))

    #Run the classifier on the mesh grid
    output=classifier.predict(np.c_[xVals.ravel(), yVals.ravel()])

    #Reshape the output array'
    output=output.reshape(xVals.shape)

    #Create a plot
    plt.figure()

    #Specify the title
    plt.title(title)

    #Choose the color scheme for the plot
    plt.pcolormesh(xVals, yVals, output, cmap=plt.cm.gray)

    #Overlay the training points on the plot
    plt.scatter(X[:, 0], X[:,1], c=y, s=75, edgecolors='black', linewidths=1, cmap=plt.cm.Paired)

    #Specify the boundaries of the plot
    plt.xlim(xVals.min(), xVals.max())
    plt.ylim(yVals.min(), yVals.max())

    #Specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(X[:, 0].min()-1),int(X[:,0].max()+1),1.0)))
    plt.yticks((np.arange(int(X[:, 1].min()-1),int(X[:,1].max()+1),1.0)))           

    plt.show()

