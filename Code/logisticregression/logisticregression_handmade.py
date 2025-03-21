import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
np.random.seed(42)

class LogisticRegressionModel:
    def __init__(self,w=None):
        self.w = w
    def get_w(self):
        return (self.w[:][0], self.w[:][1])
        
# 生成两个类别的数据
# 类别 0 的数据
mean0 = [2, 3]  # 类别 0 的均值
cov0 = [[1, 0.5], [0.5, 1]]  # 类别 0 的协方差矩阵
data0 = np.random.multivariate_normal(mean0, cov0, 100)

# 类别 1 的数据
mean1 = [5, 7]  # 类别 1 的均值
cov1 = [[1, -0.5], [-0.5, 1]]  # 类别 1 的协方差矩阵
data1 = np.random.multivariate_normal(mean1, cov1, 100)

# 合并数据和标签
X = np.vstack((data0, data1))
y = np.hstack((np.zeros(100), np.ones(100)))

# 打乱数据
shuffle_index = np.random.permutation(len(X))
X_shuffled = X[shuffle_index]
y_shuffled = y[shuffle_index]

# 添加偏置项
X_shuffled = np.hstack((np.ones((X_shuffled.shape[0], 1)), X_shuffled))
scaler = StandardScaler()
X_shuffled = scaler.fit_transform(X_shuffled)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(X, y, w, lambda_=0.001):
    m = X.shape[0]
    z = np.dot(X, w)
    y_pred = sigmoid(z)
    epsilon = 1e-15
    loss = -np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon)) + lambda_ * np.sum(w[1:]**2) / 2
    return loss

def compute_gradient(X, y, w, lambda_=0.001):
    m = X.shape[0]
    z = np.dot(X, w)
    y_pred = sigmoid(z)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    dw = np.dot(X.T, (y_pred - y)) / m
    # dw[1:] += (lambda_ / m) * w[1:]
    return dw

def train(model:LogisticRegressionModel, X_train, y_train, max_iterations=1000, learning_rate=0.001):
    w = model.w
    last_loss = loss(X_train, y_train, w)
    for i in range(max_iterations):
        gradient = compute_gradient(X_train, y_train, w)
        w = w - learning_rate * gradient
        current_loss = loss(X_train, y_train, w)
        if i % 100 == 0:
          print(f"Iteration {i}: Loss = {current_loss}")
            
def plot_decision_boundary(model, X, y):
    # 创建网格点
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # 获取模型参数
    w = model.w
    
    # 计算决策边界
    # 决策边界方程：w0 + w1*x1 + w2*x2 = 0
    # 解方程得到 x2 = (-w0 - w1*x1)/w2
    x1 = np.arange(x_min, x_max, 0.01)
    x2 = (-w[0] - w[1] * x1) / w[2]
    
    # 绘制散点图和决策边界
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.plot(x1, x2, color='red', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary of Logistic Regression Model')
    plt.legend()
    plt.show()

# initialize parameters matrix
input_size = X_train.shape[1]
w = np.array([[0.1],[.1],[.6]])

# initialize model
model = LogisticRegressionModel(w)
train(model, X_shuffled, y_shuffled)
model.get_w()

# 绘制决策边界和散点图
plot_decision_boundary(model, X_shuffled, y_shuffled)