import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Sample labels
trueLabels= [2,0,0,2,4,4,1,0,3,3,3]
predLabels= [2,1,0,2,4,3,1,0,1,3,3]

#Confusion Matrix
confusionMat=confusion_matrix(trueLabels, predLabels)

#Visualize confusion matrix
plt.imshow(confusionMat, interpolation='nearest',cmap=plt.cm.gray)
plt.title('Confusion Matrix')
plt.colorbar()
ticks=np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True Labels')
plt.xlabel('Predicted labels')
plt.show()

#Classification report
targets= ['Class-0','Class-1','Class-2','Class-3','Class-4',]
print('\n', classification_report(trueLabels,predLabels, target_names=targets))
