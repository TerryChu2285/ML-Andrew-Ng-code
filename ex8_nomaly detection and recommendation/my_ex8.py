# coding=utf-8
# @Time    2022\2\10 0010 16:17
# @File    my_ex8.py
# @Author  ZZP
# ------------------

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
from scipy.optimize import minimize

# %%
# anomaly detection
# input data1
data = loadmat('ML_learning_Andrew_Ng/ex8_nomaly detection and recommendation/data/ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']
print(X.shape, Xval.shape, yval.shape)

# %% visualize data1
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()


# %% def estimate-gaussian function
def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma


# %% test estimate-gaussian function for data1
mu, sigma = estimate_gaussian(X)
print(mu, sigma)

# %% compute probability of each data points in normal distribution
dist = stats.norm(mu[0], sigma[0])
print(dist.pdf(15))
print(dist.pdf(X[:, 0])[:50])

# %% compute the probability of X
p = np.zeros((X.shape[0], X.shape[1]))
p[:, 0] = stats.norm(mu[0], sigma[0]).pdf(X[:, 0])
p[:, 1] = stats.norm(mu[1], sigma[1]).pdf(X[:, 1])

# %% compute the probability of Xval
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:, 0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:, 0])
pval[:, 1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:, 1])


# %% def select-threshold function
def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon

        tp = np.sum(np.logical_and(preds == 1, yval == 1).astype(float))
        fp = np.sum(np.logical_and(preds == 0, yval == 1).astype(float))
        fn = np.sum(np.logical_and(preds == 1, yval == 0).astype(float))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1


# %% test select-threshold for Xval and yval
best_epsilon, best_f1 = select_threshold(pval, yval)
print(best_epsilon, best_f1)

# %% apply the best-epsilon for X to find anomaly data points
outliers = np.where(p < best_epsilon)
print(outliers)

# %% visualize the anomaly points
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
ax.scatter(X[outliers[0], 0], X[outliers[0], 1], s=80, color='r', marker='o')
plt.show()

# %% input data2
data2 = loadmat('ML_learning_Andrew_Ng/ex8_nomaly detection and recommendation/data/ex8data2.mat')
X2 = data2['X']
Xval2 = data2['Xval']
yval2 = data2['yval']
print(X2.shape, Xval2.shape, yval2.shape)

# %% find epsilon of data2
mu2, sigma2 = estimate_gaussian(X2)
pval2 = np.zeros((Xval2.shape[0], Xval2.shape[1]))
for i in range(11):
    pval2[:, i] = stats.norm(mu2[i], sigma2[i]).pdf(Xval2[:, i])
epsilon, f1 = select_threshold(pval2, yval2)
print(epsilon, f1)

# %% collaborative filtering learning algorithm
# input movies.mat
data = loadmat('ML_learning_Andrew_Ng/ex8_nomaly detection and recommendation/data/ex8_movies.mat')
Y = data['Y']
R = data['R']
print(Y.shape, R.shape)

# %% compute the average rating of the first movie
print(Y[1, np.where(R[1, :] == 1)[0]].mean())

# %% visualize the movies data
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(Y)
ax.set_xlabel('Users')
ax.set_ylabel('Movies')
fig.tight_layout()
plt.show()


# %% def cost function
def cost(params, Y, R, num_features):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]

    # reshape the parameter array into parameter matrix
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))

    # initializations
    J = 0

    # compute the cost
    error = np.multiply((X * theta.T) - Y, R)
    square_error = np.power(error, 2)
    J = (1 / 2) * np.sum(square_error)

    return J


# %% input movieParams.mat
params_data = loadmat('ML_learning_Andrew_Ng/ex8_nomaly detection and recommendation/data/ex8_movieParams.mat')
X = params_data['X']
theta = params_data['Theta']
print(X.shape, theta.shape)

# %% test cost-function
users = 4
movies = 5
features = 3
X_sub = X[:movies, :features]
theta_sub = theta[:users, :features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]
params = np.concatenate((np.ravel(X_sub), np.ravel(theta_sub)))
print(cost(params, Y_sub, R_sub, features))


# %% def collaborative-filtering-gradient function
def collaborative_filtering_gradient(params, Y, R, num_features):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    # reshape the parameter array into parameter matrix
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))

    # initializations
    J = 0
    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)

    # compute the cost
    error = np.multiply((X * theta.T) - Y, R)
    square_error = np.power(error, 2)
    J = (1 / 2) * np.sum(square_error)

    # calculate the gradients
    X_grad = error * theta
    theta_grad = error.T * X

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(theta_grad)))

    return J, grad


# %% test collaborative-filtering-gradient function
J, grad = collaborative_filtering_gradient(params, Y_sub, R_sub, features)
print(J, grad)


# %% def collaborative-filtering-gradient-regularized function
def collaborative_filtering_grad_reg(params, Y, R, num_features, learning_rate):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    # reshape the parameter array into parameter matrix
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))

    # initializations
    J = 0
    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)

    # compute the cost
    error = np.multiply((X * theta.T) - Y, R)
    square_error = np.power(error, 2)
    J = ((1 / 2) * np.sum(square_error)) + ((learning_rate / 2) * np.sum(np.power(theta, 2))) + \
        ((learning_rate / 2) * np.sum(np.power(X, 2)))

    # calculate the gradients
    X_grad = (error * theta) + (learning_rate * X)
    theta_grad = (error.T * X) + (learning_rate * theta)

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(theta_grad)))

    return J, grad


# %% test collaborative-filtering-gradient-regularized function
J, grad = collaborative_filtering_grad_reg(params, Y_sub, R_sub, features, 1.5)
print(J, grad)

# %% input movies_ids.txt
movies_ids = {}
f = open('ML_learning_Andrew_Ng/ex8_nomaly detection and recommendation/data/movie_ids.txt', encoding='gbk')
for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movies_ids[int(tokens[0]) - 1] = ' '.join(tokens[1:])
print(movies_ids[0])

# %% use the exercise rating
ratings = np.zeros((1682, 1))
ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5
print('Rated {0} with {1} stars.'.format(movies_ids[0], str(int(ratings[0]))))
print('Rated {0} with {1} stars.'.format(movies_ids[6], str(int(ratings[6]))))
print('Rated {0} with {1} stars.'.format(movies_ids[11], str(int(ratings[11]))))
print('Rated {0} with {1} stars.'.format(movies_ids[53], str(int(ratings[53]))))
print('Rated {0} with {1} stars.'.format(movies_ids[63], str(int(ratings[63]))))
print('Rated {0} with {1} stars.'.format(movies_ids[65], str(int(ratings[65]))))
print('Rated {0} with {1} stars.'.format(movies_ids[68], str(int(ratings[68]))))
print('Rated {0} with {1} stars.'.format(movies_ids[97], str(int(ratings[97]))))
print('Rated {0} with {1} stars.'.format(movies_ids[182], str(int(ratings[182]))))
print('Rated {0} with {1} stars.'.format(movies_ids[225], str(int(ratings[225]))))
print('Rated {0} with {1} stars.'.format(movies_ids[354], str(int(ratings[354]))))

# %% add the exercise rating to the data
Y = np.append(Y, ratings, axis=1)
R = np.append(R, ratings != 0, axis=1)
print(Y.shape, R.shape, ratings.shape)

# %% normalize the data
movies = Y.shape[0]
users = Y.shape[1]
features = 10
learning_rate = 10

X = np.random.random(size=(movies, features))
theta = np.random.random(size=(users, features))
params = np.concatenate((np.ravel(X), np.ravel(theta)))
print(X.shape, theta.shape, params.shape)

Ymean = np.zeros((movies, 1))
Ynorm = np.zeros((movies, users))
for i in range(movies):
    idx = np.where(R[i, :] == 1)[0]
    Ymean[i] = Y[i, idx].mean()
    Ynorm[i, idx] = Y[i, idx] - Ymean[i]
print(Ynorm.mean())

# %% minimize the cost
fmin = minimize(fun=collaborative_filtering_grad_reg, x0=params, args=(Ynorm, R, features, learning_rate),
                method='CG', jac=True, options={'maxiter': 100})
print(fmin)

# %% output X and theta
X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))
print(X.shape, theta.shape)

# %% get our predictions
predictions = X * theta.T
my_pred = predictions[:, 0] + Ymean
print(my_pred.shape)

# sort the rating
sorted_preds = np.sort(my_pred, axis=0)[::-1]
sorted_preds[:10]

# find the index
idx = np.argsort(my_pred, axis=0)[::-1]
print(idx)

#%% output top 10 predictions results
print('Top 10 movie predictions:')
for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_pred[j])), movies_ids[j]))