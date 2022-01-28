# coding=utf-8
# @Time    2022\1\10 0010 15:39
# @File    my_ex5.py
# @Author  ZZP
# ------------------

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio
import seaborn as sns
import scipy.optimize as opt


# %%
def load_data():
    d = sio.loadmat('ML_learning_Andrew_Ng/ex5_bias vs variance/ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])


X, y, Xval, yval, Xtest, ytest = load_data()

# %%
df = pd.DataFrame({'water level': X, 'flow': y})
sns.lmplot(x='water level', y='flow', data=df, fit_reg=False, height=7)
plt.show()

# %%
# X数据集X, Xval,Xtest增加一列
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]


# %% cost_function
def cost(theta, X, y):
    m = X.shape[0]
    inner = X @ theta - y
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)
    return cost


# %% theta initialized
theta = np.ones(X.shape[1])
print(cost(theta, X, y))


# %% cost_function_grad
def cost_grad(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)
    return inner / m


# %% compute cost_grad for theta=1
print(cost_grad(theta, X, y))


# %% reg_cost
def reg_cost(theta, X, y, l):
    m = X.shape[0]
    reg_term = (l / m) * np.power(theta[1:], 2).sum()
    return cost(theta, X, y) + reg_term


# %% compute reg_cost for theta=1
print(reg_cost(theta, X, y, 1))


# %% reg_cost_grad
def reg_cost_grad(theta, X, y, l):
    m = X.shape[0]
    reg_term = theta.copy()
    reg_term[0] = 0
    reg_term = (l / m) * reg_term
    return cost_grad(theta, X, y) + reg_term


# %% compute reg_cost_grad
print(reg_cost_grad(theta, X, y, 1))


# %% fitting linear regression
def linear_regre(X, y, l):
    # init theta
    theta = np.ones(X.shape[1])

    # train it
    res = opt.minimize(fun=reg_cost, x0=theta, args=(X, y, l), method='TNC', jac=reg_cost_grad, options={'disp': True})

    return res


# %% get theta
theta = np.ones(X.shape[0])
final_theta = linear_regre(X, y, 1).get('x')

# %% plot
b = final_theta[0]
k = final_theta[1]
plt.scatter(X[:, 1], y, label='Training data')
plt.plot(X[:, 1], X[:, 1] * k + b, label='Prediction data')
plt.legend()
plt.show()

# %% define training_cost, cv_cost
training_cost, cv_cost = [], []

# %% compute training_cost, cv_cost
m = X.shape[0]
for i in range(1, m + 1):
    res = linear_regre(X[:i], y[:i], 0)

    tc = reg_cost(res.x, X[:i], y[:i], 0)
    ct = reg_cost(res.x, Xval, yval, 0)

    training_cost.append(tc)
    cv_cost.append(ct)

# %% plot learning curve
plt.plot(np.arange(1, m + 1), training_cost, label='Training cost')
plt.plot(np.arange(1, m + 1), cv_cost, label='cross validation cost')
plt.legend()
plt.show()


# %% Polynomial regression
# 准备多项式回归数据
# 1. 扩展特征到 8阶,或者你需要的阶数
# 2. 使用 **归一化** 来合并 $x^n$
# 3. don't forget intercept term
def prepare_poly_data(*args, power):
    """
    args: keep feeding in X, Xval, or Xtest
        will return in the same order
    """

    def prepare(x):
        # expand features
        df = poly_features(x, power=power)

        # normalization
        ndarr = normalize_features(df).values

        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]


def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.values if as_ndarray else df


def normalize_features(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


# %% generate poly_features=8 data
X, y, Xval, yval, Xtest, ytest = load_data()
X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)


# %% plot learning curve for l=0
def plot_learning_curve(X, Xinit, y, Xval, yval, l):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        res = linear_regre(X[:i], y[:i], l=l)

        tc = reg_cost(res.x, X[:i], y[:i], 0)
        ct = reg_cost(res.x, Xval, yval, 0)

        training_cost.append(tc)
        cv_cost.append(ct)

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    ax[0].plot(np.arange(1, m + 1), training_cost, label='Training data cost')
    ax[0].plot(np.arange(1, m + 1), cv_cost, label='Cross validation cost')
    ax[0].set_xlabel('Number of training data')
    ax[0].set_ylabel('Error')
    ax[0].legend()

    fitx = np.linspace(-50, 50, 100)
    fitxtmp = prepare_poly_data(fitx, power=8)
    fity = np.dot(fitxtmp[0], linear_regre(X, y, l).x.T)
    ax[1].plot(fitx, fity, c='r', label='fitcurve')
    ax[1].scatter(Xinit, y, c='b', label='initial_Xy')
    ax[1].set_xlabel('water_level')
    ax[1].set_ylabel('flow')
    ax[1].legend()


# %% plot learning curve for l=0
plot_learning_curve(X_poly, X, y, Xval_poly, yval, 0)
plt.show()

# %% plot learning curve for l=1
plot_learning_curve(X_poly, X, y, Xval_poly, yval, 1)
plt.show()

# %% plot learning curve for l=100
plot_learning_curve(X_poly, X, y, Xval_poly, yval, 100)
plt.show()

# %% find the beat lambda
l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []
for l in l_candidate:
    res = linear_regre(X_poly, y, l)

    tc = reg_cost(res.x, X_poly, y, 0)
    ct = reg_cost(res.x, Xval_poly, yval, 0)

    training_cost.append(tc)
    cv_cost.append(ct)

# %%
plt.plot(l_candidate, training_cost, label='Training data cost')
plt.plot(l_candidate, cv_cost, label='Cross validation cost')
plt.xlabel('Change of the lambda')
plt.ylabel('Error')
plt.legend()
plt.show()

# %% best cv I got from all those candidates
print(l_candidate[np.argmin(cv_cost)])

# %% use test data to compute the cost
for l in l_candidate:
    theta = linear_regre(X_poly, y, l).x
    print('test cost (l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))

# %% plot learning curve for l=0.3
plot_learning_curve(X_poly, X, y, Xval_poly, yval, 0.3)
plt.show()
