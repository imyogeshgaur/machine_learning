# Import the Modules

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading datasets

iris = datasets.load_iris()

# Printing description

print(iris.DESCR)

# Added the Features and Labels

features = iris.data
label = iris.target
print(features[0],label[0])

# Built and Train the Classifier

clf= KNeighborsClassifier()
clf.fit(features,label)
KNeighborsClassifier()

# Making prediction from sample data

preds = clf.predict([[9.1,9.5,6.4,0.2]])
print(preds)
