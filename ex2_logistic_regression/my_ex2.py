# coding=utf-8
# @Time    2021\12\7 0007 21:04
# @File    my_ex2.py
# @Author  ZZP
# ------------------

# %%
from sklearn.metrics import classification_report
from sklearn import linear_model
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import pandas as pd

# %% 检查数据
path = 'ML_learning_Andrew_Ng/ex2_logistic_regression/ex2data1.txt'
data1 = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'Admitted'])
print(data1.head())

# %% 创建两个分数的散点图，并使用颜色编码来可视化，如果样本是正的（被接纳）或负的（未被接纳）
positive = data1[data1['Admitted'].isin([1])]
negative = data1[data1['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
ax.legend()
plt.show()


# %% sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# %% 快速的检查，来确保它可以工作
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()


# %% 编写代价函数来评估结果
def costfunction(theta, X, y):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    first = np.multiply((-y), np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / len(X)


# %% 做一些设置，和我们在练习1在线性回归的练习很相似
# add a ones column
data1.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data1.shape[1]
X = data1.iloc[:, 0:cols - 1]
y = data1.iloc[:, cols - 1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

# %% 查看X,y,theta的数据类型
print(X.shape)
print(y.shape)
print(theta.shape)

# %% 计算初始化参数的代价函数(theta为0)
print(costfunction(theta, X, y))


# %% 需要一个函数来计算我们的训练数据、标签和一些参数theta的梯度:gradient descent(梯度下降)
def gradient_descent(theta, X, y):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)

    paramaters = int(theta.ravel().shape[1])
    grad = np.zeros(paramaters)

    error = sigmoid(X * theta.T) - y

    for i in range(paramaters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad


# 注意，我们实际上没有在这个函数中执行梯度下降，我们仅仅在计算一个梯度步长。
# 在练习中，一个称为“fminunc”的Octave函数是用来优化函数来计算成本和梯度参数。
# 由于我们使用Python，我们可以用SciPy的“optimize”命名空间来做同样的事情。

# %% 用我们的数据和初始参数为0的梯度下降法的结果
print(gradient_descent(theta, X, y))

# %% 可以用SciPy's truncated newton（TNC）实现寻找最优参数。
result = opt.fmin_tnc(func=costfunction, x0=theta, fprime=gradient_descent, args=(X, y))
print(result)

# %% 看看在这个结论下代价函数计算结果是什么个样子
print(costfunction(result[0], X, y))


# %% 编写一个函数，用我们所学的参数theta来为数据集X输出预测。然后，我们可以使用这个函数来给我们的分类器的训练精度打分。
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x > 0.5 else 0 for x in probability]


# 当hθ大于0.5时，预测 y=1
# 当hθ小于0.5时，预测 y=0

# %%
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(correct)
print(accuracy)
# 用 sklearn 中的方法来检验
# from sklearn.metrics import classification_report
# print(classification_report(y, predictions))

# %% 决策边界
x = np.arange(20, 100, step=0.01)
y_bd = (theta_min[0, 0] + (theta_min[0, 1] * x)) / (-theta_min[0, 2])
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='*', label='Not Admitted')
ax.plot(x, y_bd)
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
ax.legend()
plt.show()

# %% 正则化逻辑回归
path = 'ML_learning_Andrew_Ng/ex2_logistic_regression/ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['test1', 'test2', 'accept'])
print(data2.head())

# %% 数据可视化
accepted = data2[data2['accept'].isin([1])]
rejected = data2[data2['accept'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(accepted['test1'], accepted['test2'], s=50, c='y', marker='o', label='Accepted')
ax.scatter(rejected['test1'], rejected['test2'], s=50, c='b', marker='x', label='Rejected')
ax.set_xlabel('Microchip Test 1')
ax.set_ylabel('Microchip Test 2')
ax.legend()
plt.show()


# %% 增加特征，进行特征映射
def feature_map(x1, x2, power):
    data3 = {}
    for i in np.arange(power + 1):
        for j in np.arange(i + 1):
            data3['f{}{}'.format(i - j, j)] = np.power(x1, i - j) * np.power(x2, j)
    return pd.DataFrame(data3)


# 进行映射
x1 = data2['test1'].values
x2 = data2['test2'].values
data3 = feature_map(x1, x2, 6)
print(data3.head())


# %% 正则化代价函数
def costreg(theta, X, y, learningrate):
    _theta = theta[1:]
    first = (-y) * np.log(sigmoid(X @ theta))
    second = (1 - y) * np.log(1 - sigmoid(X @ theta))
    reg = (learningrate / (2 * len(X))) * (_theta @ _theta)
    return np.mean(first - second) + reg


# %% 正则化梯度
def gradientreg(theta, X, y, learningrate):
    first = (X.T @ (sigmoid(X @ theta) - y)) / len(X)
    reg = (learningrate / len(X)) * theta
    reg[0] = 0  # 不惩罚θ0
    return first + reg


# %% 训练参数θ
X = data3.values
y = data2['accept'].values
theta = np.zeros(X.shape[1])
print(X.shape, y.shape, theta.shape)
print(costreg(theta, X, y, 1))

# %% 采用fmin_tnc函数训练
result1 = opt.fmin_tnc(func=costreg, x0=theta, fprime=gradientreg, args=(X, y, 2))
print(result1)

# %% 采用minimize函数训练
result2 = opt.minimize(fun=costreg, x0=theta, args=(X, y, 100), method='TNC', jac=gradientreg)
print(result2)


# %% 编写一个函数，用我们所学的参数theta来为数据集X输出预测。然后，我们可以使用这个函数来给我们的分类器的训练精度打分。
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x > 0.5 else 0 for x in probability]


# 当hθ大于0.5时，预测 y=1
# 当hθ小于0.5时，预测 y=0

# %% 评估逻辑回归
final_theta1 = np.matrix(result1[0])
predictions1 = predict(final_theta1, X)
correct1 = [1 if a == b else 0 for (a, b) in zip(predictions1, y)]
accuracy1 = sum(correct1) / len(X)
print(accuracy1)

# %%
final_theta2 = np.matrix(result2.x)
predictions2 = predict(final_theta2, X)
correct2 = [1 if a == b else 0 for (a, b) in zip(predictions2, y)]
accuracy2 = sum(correct2) / len(X)
print(accuracy2)

# %%
print(classification_report(y, predictions1))
print(classification_report(y, predictions2))

# %%决策边界
x = np.linspace(-1, 1.5, 250)
xx, yy = np.meshgrid(x, x)

z = feature_map(xx.ravel(), yy.ravel(), 6).values
z = z @ final_theta2.T
z = z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(accepted['test1'], accepted['test2'], s=50, c='r', marker='o', label='Accepted')
ax.scatter(rejected['test1'], rejected['test2'], s=50, c='y', marker='x', label='Rejected')
ax.set_xlabel('Microchip test 1')
ax.set_ylabel('Microchip test 2')
ax.legend()
plt.contour(xx, yy, z, 0)
plt.show()

# %%还可以使用高级Python库像scikit-learn来解决这个问题。
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X, y.ravel())
print(model.score(X, y))
