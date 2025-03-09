import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import OneHotEncoder

# binary classification (-1 and 1)
X = np.array([[1,-7],[1,-5],[1,1],[1,5]])
y = np.array([[-1],[-1],[1],[1]])
#linear regression for classification
w = inv(X.T @ (X)) @ (X.T) @ (y)
print(w,"\n")

#predict
X_new = np.array([[1,2]])
y_predict_new = np.sign(X_new @ w)
print(y_predict_new)
# expected output: [[-1.]]


# multi-class classification
# manually encode the label matrix
X = np.array([[1,1,1],[1,-1,1],[1,1,3],[1,1,0]])
Y_class = np.array([[1],[2],[1],[3]])
Y = np.array([[1,0,0],[0,1,0],[1,0,0],[0,0,1]])

# one-hot encoding function
Y_onehot = OneHotEncoder().fit_transform(Y_class).toarray()

# learning
W = inv(X.T @ X) @ X.T @ Y # could use Y_onehot instead of Y

# predicting
X_new = np.array([[1,0,-1]])
Y_predict = X_new @ W
print(np.argmax(Y_predict)+1)