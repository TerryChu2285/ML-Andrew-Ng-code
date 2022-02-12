# coding=utf-8
# @Time    2022\1\28 0028 21:17
# @File    my_ex7.py
# @Author  ZZP
# ------------------

# %%
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from IPython.display import Image


# %% define find-closest-centroids function
def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_distance = 1000000
        for j in range(k):
            distance = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if distance < min_distance:
                min_distance = distance
                idx[i] = j

    return idx


# %% test find-closest-centroids function
data = loadmat('ML_learning_Andrew_Ng/ex7_kmeans and PCA/data/ex7data2.mat')
X = data['X']
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = find_closest_centroids(X, initial_centroids)
print(idx[:3])

# %% import data2
data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
print(data2.head())

# %% plot data2 scatter
sb.set(context='notebook', style='white')
sb.lmplot(x='X1', y='X2', data=data2, fit_reg=False)
plt.show()


# %% define compute-centroids function
def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()

    return centroids


# %% test compute-centroids function
print(compute_centroids(X, idx, 3))


# %% define run-K-means function
def run_K_means(X, initial_centroids, max_iter):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iter):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids


# %% test run-K-means function for data2
idx, centroids = run_K_means(X, initial_centroids, 10)

# %% plot cluster
cluster1 = X[np.where(idx == 0)[0], :]
cluster2 = X[np.where(idx == 1)[0], :]
cluster3 = X[np.where(idx == 2)[0], :]
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, c='b', label='Cluster1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, c='r', label='Cluster2')
ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, c='g', label='Cluster3')
ax.scatter(centroids[:, 0], centroids[:, 1], s=80, c='k', marker='x', label='Centroids')
ax.legend()
plt.show()


# %% define init-centroids function
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i, :] = X[idx[i], :]

    return centroids


# %% test init-centroids function
print(init_centroids(X, 3))

# %% compress bird-small.png
# import png
# Image(filename='ML_learning_Andrew_Ng/ex7_kmeans and PCA/data/bird_small.png')
image_data = loadmat('ML_learning_Andrew_Ng/ex7_kmeans and PCA/data/bird_small.mat')
A = image_data['A']
print(A.shape)

# %% pre-process image-data
# normalize value range
A = A / 255

# reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
print(X.shape)

# %% run K-means for bird-small.mat
# randomly initialize the centroids
initial_centroids = init_centroids(X, 16)

# run the algorithm
idx, centroids = run_K_means(X, initial_centroids, 10)

# get the closest centroids one last time
idx = find_closest_centroids(X, centroids)

# map each pixel to the centroids value
X_recover = centroids[idx.astype(int), :]
print(X_recover.shape)

# reshape to the original dimensions
X_recover = np.reshape(X_recover, (A.shape[0], A.shape[1], A.shape[2]))
print(X_recover.shape)

# %% plot recovered png
plt.imshow(X_recover)
plt.show()

# %% use scikit-learn to implement K-means
# cast to float, you need to do this otherwise the color would be weird after clustering
pic = io.imread('ML_learning_Andrew_Ng/ex7_kmeans and PCA/data/bird_small.png') / 255
plt.imshow(pic)
plt.show()

# %% pre-process pic data
# serialize data
print(pic.shape)
data = pic.reshape(128 * 128, 3)
print(data.shape)

# %% run algorithm
model = KMeans(n_clusters=16, n_init=100)
model.fit(data)
centroids = model.cluster_centers_
print(centroids.shape)
C = model.predict(data)
print(C.shape)
print(centroids[C].shape)
compressed_pic = centroids[C].reshape(128, 128, 3)

# %% plot compressed-pic data
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()

# %% Principal Component Analysis-PCA
# import data1
data = loadmat('ML_learning_Andrew_Ng/ex7_kmeans and PCA/data/ex7data1.mat')
X = data['X']

# %% plot data1 scatter
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()


# %% define PCA function
def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    sigma = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(sigma)

    return U, S, V


# %% test PCA function for data1
U, S, V = pca(X)


# %% define project data function
def project_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)


# %% project data1
Z = project_data(X, U, 1)


# %% define recover data function
def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)


# %% recover project-data1
X_recover = recover_data(Z, U, 1)

# %% plot 1D scatter
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(list(X_recover[:, 0]), list(X_recover[:, 1]))
plt.show()

# %% Apply PCA to faces.mat
# import faces.mat
faces = loadmat('ML_learning_Andrew_Ng/ex7_kmeans and PCA/data/ex7faces.mat')
X = faces['X']


# %% define plot-n-images function
def plot_n_images(X, n):
    # n must to be a square number
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size, sharex=True, sharey=True, figsize=(8, 8))

    for i in range(grid_size):
        for j in range(grid_size):
            ax_array[i, j].imshow((n_images[(i * grid_size) + j, :].reshape((pic_size, pic_size))).T)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


# %% plot 100 images
plot_n_images(X, 100)
plt.show()

# %% run PCA algorithm
U, S, V = pca(X)
plot_n_images(U, 36)
plt.show()

# %% project images-data
Z = project_data(X, U, 100)
plot_n_images(Z, 36)
plt.show()

# %% recover project-images-data
X_recover = recover_data(Z, U, 100)
plot_n_images(X_recover, 100)
plt.show()


#%% scikit-learn PCA
model = PCA(n_components=100)
Z = model.fit_transform(X)
plot_n_images(Z, 64)
X_recover = model.inverse_transform(Z)
plot_n_images(X_recover, 64)
plt.show()