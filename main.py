import matplotlib as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
diabetes = datasets.load_diabetes()

#print(diabetes.DESCR)
#dict_keys(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])

diabetes_X = diabetes.data[:,np.newaxis,2]

print(diabetes_X)

diabetes_test = diabetes_X[:-30]
diabetes_train = diabetes_X[-20:]


diabetes_Y_train = diabetes.target[-20:]
diabetes_Y_test = diabetes.target[:-30]

model = linear_model.LinearRegression()
model.fit(diabetes_train,diabetes_Y_train)

diabetes_Y_predict = model.predict(diabetes_test)

print("Mean squared error is: ", mean_squared_error(diabetes_Y_test,diabetes_Y_predict))

