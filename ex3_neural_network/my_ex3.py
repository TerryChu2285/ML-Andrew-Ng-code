# coding=utf-8
# @Time    2021\12\11 0011 10:31
# @File    my_ex3.py
# @Author  ZZP
# ------------------


# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

# %%将mat转换为python的数据
data = loadmat('ML_learning_Andrew_Ng/ex3_neural_network/ex3data1.mat')
print(data['X'].shape, data['y'].shape)

# 图像在matrix X中表示为400维向量（其中有5,000个）。
# 400维“特征”是原始20 x 20图像中每个像素的灰度强度。类标签在向量y中作为表示图像中数字的数字类。


# %%绘图
# 绘图函数，画100张图片
def plot_100_images(X):
    size = int(np.sqrt(X.shape[1]))

    # sample 100 image, reshape, reorg it
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            ax_array[i, j].matshow(sample_images[10 * i + j].reshape((size, size)), cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


# %%绘制并显示图形
plot_100_images(data['X'])
plt.show()


# %%向量化
# 第一个任务是将我们的逻辑回归实现修改为完全向量化（即没有“for”循环）。
# 这是因为向量化代码除了简洁外，还能够利用线性代数优化，并且通常比迭代代码快得多。
# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# %%代价函数
def cost(theta, X, y, learningrate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(sigmoid(1 - (X * theta.T))))
    reg = (learningrate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


# %%梯度下降的向量化表示
def gradient(theta, X, y, learningrate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    paramater = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningrate / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()


# %%一对多分类
# 实现一对一全分类方法，其中具有k个不同类的标签就有k个分类器，每个分类器在“类别 i”和“不是 i”之间决定。
# 我们将把分类器训练包含在一个函数中，该函数计算10个分类器中的每个分类器的最终权重，并将权重返回为k X（n + 1）数组，其中n是参数数量。
def one_vs_all(X, y, num_labels, learningrate):
    rows = X.shape[0]
    params = X.shape[1]

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # k * (n + 1) array for the parameters of each of the k classifiers
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learningrate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x

    return all_theta


# %%
rows = data['X'].shape[0]
params = data['X'].shape[1]

all_theta = np.zeros((10, params + 1))

X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

theta = np.zeros(params + 1)

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))

print(X.shape, y_0.shape, theta.shape, all_theta.shape)
print(np.unique(data['y']))

# %%
all_theta = one_vs_all(data['X'], data['y'], 10, 1)
print(all_theta)


# %%使用训练完毕的分类器预测每个图像的标签。
# 对于这一步，我们将计算每个类的类概率，对于每个训练样本（使用当然的向量化代码），并将输出类标签为具有最高概率的类。
def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax


# %%使用predict_all函数为每个实例生成类预测，看看我们的分类器是如何工作的。
y_pred = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(data['y'], y_pred)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))
