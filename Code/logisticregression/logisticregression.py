import numpy as np
import sklearn
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt


data = pd.read_csv('../../Datasets/framingham.csv')
# remove the null values
data = data.dropna()
# convert the data to np array
data_array = data.to_numpy()

X = data_array[:,0:15]
Y = data_array[:,15]
scaler = StandardScaler()
X = scaler.fit_transform(X)  # 标准化特征
X = np.hstack((np.ones((X.shape[0], 1)), X))
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2,random_state=42)

# parameters initialization
w = np.random.randn(X.shape[1], 1) * 0.01 
   
# sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1/(1+np.exp(-z))

# loss function
def loss(X,Y,w,lambda_=0.001):
    m = X.shape[0]
    z = np.dot(X, w)
    y_pred = sigmoid(z)
    epsilon = 1e-15
    loss = -np.sum(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred)) + lambda_ * np.sum(w[1:]**2) / 2
    return loss

# gradient function
def compute_gradient(X, Y, w, lambda_=0.001):
    m = X.shape[0]
    z = np.dot(X, w)
    y_pred = sigmoid(z)
    # 确保Y是二维列向量
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    # 确保y_pred是二维列向量
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
        
    dw = np.dot(X.T, (y_pred - Y)) / m
    dw[1:] += (lambda_ / m) * w[1:]  # 对权重添加正则化项，偏置项不进行正则化
    return dw

max_iterations = 150
learning_rate = 0.08

# train
for i in range(max_iterations):
    gradient = compute_gradient(X_train,Y_train,w)
    w = w - learning_rate*gradient
    current_loss = loss(X_train,Y_train,w)
    if i % 10 == 0:
        print(f"Iteration {i}: Loss = {current_loss}")

current_loss = loss(X_train,Y_train,w)
print(f"Optimal parameters: {w.T}. Loss: {current_loss}")

# predict
def predict(X, w):
    y_pred = sigmoid(np.dot(X, w))
    y_pred_class = np.where(y_pred >= 0.5, 1, 0)
    return y_pred_class

Y_pred = predict(X_test, w)
accuracy = np.mean(Y_pred == Y_test.flatten())
print("Test Accuracy of handmade model:", accuracy)

# sklearn model

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
accuracy = np.mean(Y_pred == Y_test.flatten())
print("Test Accuracy of sklearn model:", accuracy)