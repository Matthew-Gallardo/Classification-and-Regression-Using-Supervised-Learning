import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from utility import visualize_classifier

#Load input data
input_file='data_dec_trees.txt'
data= np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1],  data [:, -1]

#Seperate input data into 2 classesbased on labels
class0 =np.array(X[y==0])
class1 =np.array(X[y==1])

#Visualize input data
plt.figure()
plt.scatter(class0[:,0], class0[:,1], s=75, facecolors='black', edgecolors='black', linewidth=1, marker='x')
plt.scatter(class1[:,0], class1[:,1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')

#Split data into training and testing datasets
X_train , X_test, y_train, y_test= model_selection.train_test_split( X,y , test_size=0.25, random_state=5)

#Decision Trees classifier
params = {'random_state': 0 , 'max_depth' : 4}
classifier= DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset')

#Evaluate classifier performance
class_names= ['Class-0', 'Class-1']
print("\n"+"#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train),target_names=class_names))
print("#"*40+"\n")
print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred,target_names=class_names))
print("#"*40+"\n")

plt.show()




