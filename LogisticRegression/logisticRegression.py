# Importing The Modules

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np

# Loading Iris dataset

iris = datasets.load_iris()
x = iris['data'][:,3:]
y = (iris['target']==2).astype(np.int)

# Training logistic Regression Classifier

clf = LogisticRegression()
clf.fit(x,y)

# Predict from Sample

sample = clf.predict(([[1.6]]))
sample
array([0])

# Plot the Visulaization

x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1],"r-")
plt.show()
