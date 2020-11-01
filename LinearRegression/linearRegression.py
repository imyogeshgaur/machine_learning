# Import Libraries

import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

# Load BuiltIn DataSets

diabetes= datasets.load_diabetes()

# Make Taining and Testing Data

diabetes_x = diabetes.data[:,np.newaxis,2]

diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# Make Linaer Regression Model

model = linear_model.LinearRegression()

model.fit(diabetes_x_train,diabetes_x_train)

diabetes_y_predict = model.predict(diabetes_x_test)


print("Mean Square error is : ",mean_squared_error(diabetes_y_test,diabetes_y_predict))

print("Slope is :", model.coef_)
print("Intercept is : ",model.intercept_)

#plot The Graph

plt.scatter(diabetes_x_test,diabetes_y_test)
plt.show()