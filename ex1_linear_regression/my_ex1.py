# coding=utf-8
# @Time    2021\11\27 0027 11:39
# @File    my_ex1.py
# @Author  ZZP
# ------------------

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# %%
# 单变量线性回归
path = 'D:/360MoveData/Users/Administrator/Desktop/Python_work/ML_learning_Andrew_Ng/ex1_linear_regression/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())

# %%
print(data.describe())

# %%
# 看下数据分布图
plt.figure(figsize=(12, 8), dpi=80)
plt.scatter(data['Population'], data['Profit'])
plt.title('Scatter plot of training data')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()


# %%
# 使用梯度下降来实现线性回归，以最小化成本函数
# 首先，我们将创建一个以参数θ为特征函数的代价函数
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# %%
# 在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度
data.insert(0, 'Ones', 1)

# %%
# 变量初始化
# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]  # X是所有行，去掉最后一列
y = data.iloc[:, cols - 1:cols]  # y是所有行，最后一列

# %%
# 观察下 X (训练集) and y (目标变量)是否正确
print(X.head())
print(y.head())
print(type(X))
print(type(y))

# %%
# 代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。 我们还需要初始化theta。
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))

# %%
# 看下它们的纬度
print(X.shape)
print(y.shape)
print(theta.shape)

# %%
# 计算代价函数(theta初始值为0)
computeCost(X, y, theta)


# %%
# batch gradient descent（批量梯度下降）
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameter = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameter):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


# %%
# 初始化一些附加变量 - 学习速率α和要执行的迭代次数
alpha = 0.01
iters = 1000

# %%
# 运行梯度下降算法来将我们的参数θ适合于训练集
g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g)

# %%
# 我们可以使用我们拟合的参数计算训练模型的代价函数（误差）
print(computeCost(X, y, g))

# %%
# 绘制线性模型以及数据，直观地看出它的拟合
plt.figure(figsize=(12, 8), dpi=80)
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + g[0, 1] * x

plt.plot(x, f, 'r', label='Prediction')
plt.scatter(data.Population, data.Profit, label='Traning Data')
plt.legend(loc=2)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Predicted Profit vs. Population Size')
plt.show()

# %%
# 由于梯度方程式函数也在每个训练迭代中输出一个代价的向量，所以我们也可以绘制。 请注意，代价总是降低 - 这是凸优化问题的一个例子。
plt.figure(figsize=(12, 8), dpi=80)
plt.plot(np.arange(iters), cost, 'r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Error vs. Training Epoch')
plt.show()

# %%
# 多变量线性回归
path2 = 'D:/360MoveData/Users/Administrator/Desktop/Python_work/ML_learning_Andrew_Ng/ex1_linear_regression/ex1data2.txt'
data2 = pd.read_csv(path2, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data2.head())
print(data2.describe())

# %%
# 对于此任务，我们添加了另一个预处理步骤 - 特征归一化。
data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())

# %%
# 重复第一部分-单变量线性回归的步骤
# 增加全为1的1列
data2.insert(0, 'Ones', 1)

# %%
# 变量初始化
# set X (training data) and y (target variable)
cols2 = data2.shape[1]
X2 = data2.iloc[:, 0:cols2 - 1]  # X是所有行，去掉最后一列
y2 = data2.iloc[:, cols2 - 1:cols2]  # y是所有行，最后一列

# %%
# 观察下 X (训练集) and y (目标变量)是否正确
print(X2.head())
print(y2.head())
print(type(X2))
print(type(y2))

# %%
# 将他们都转换成矩阵格式
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

# %%
# 看下它们的纬度
print(X2.shape)
print(y2.shape)
print(theta2.shape)

# %%
# 计算代价函数(theta初始值为0)
computeCost(X2, y2, theta2)

# %%
# perform linear regression on the data set
# 运行梯度下降算法来将我们的参数θ适合于训练集
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
print(g2)
# get the cost (error) of the model
print(computeCost(X2, y2, g2))

# %%
# 快速查看这一个的训练进程
plt.figure(figsize=(12, 8), dpi=80)
plt.plot(np.arange(iters), cost2, 'r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Error vs. Training Epoch')
plt.show()

# %%
# 使用scikit-learn的线性回归函数，而不是从头开始实现这些算法。
# 我们将scikit-learn的线性回归算法应用于第1部分的数据，并看看它的表现。
model = linear_model.LinearRegression()
model.fit(X, y)

# %%
x = np.array(X[:, 1])
f = model.predict(X).flatten()
fig, ax = plt.subplots(figsize=(12, 8), dpi=80)
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()


# %%
# normal equation（正规方程）
def normal_equation(X, y):
    theta3 = np.linalg.inv(X.T @ X) @ X.T @ y  # X.T@X等价于X.T.dot(X)
    return theta3


# %%
final_theta3 = normal_equation(X, y)
print(final_theta3)
