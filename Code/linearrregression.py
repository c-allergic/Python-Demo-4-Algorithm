import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd

data = pd.read_csv('../../Datasets/LinearRegression/train.csv')

data_train = data.dropna().to_numpy()
X = data_train[:, 0]
y = data_train[:, 1]


X = np.hstack([np.ones((X.shape[0], 1)), X.reshape(-1, 1)])

linearmodel = linear_model.LinearRegression()

X_train = X
y_train = y
linearmodel.fit(X_train, y_train)

data1 = pd.read_csv('../../Datasets/LinearRegression/test.csv')
data_test = data1.dropna().to_numpy()
X_test = data_test[:, 0]
y_test = data_test[:, 1]

X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test.reshape(-1, 1)])

y_pred_test = linearmodel.predict(X_test)
error_test = np.mean((np.abs(y_test - y_pred_test) / y_test) ** 2) 
print(f"Test Mean Squared Error: {error_test} %")