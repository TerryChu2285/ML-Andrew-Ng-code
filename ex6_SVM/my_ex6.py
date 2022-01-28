# coding=utf-8
# @Time    2022\1\12 0012 20:05
# @File    my_ex6.py
# @Author  ZZP
# ------------------

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import svm
import re
from nltk.stem import PorterStemmer

# %% load data1
raw_data = loadmat('ML_learning_Andrew_Ng/ex6_SVM/data/ex6data1.mat')
data = pd.DataFrame(raw_data['X'], columns=('X1', 'X2'))
data['y'] = raw_data['y']
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]


# %% plot scatter function
def plot_scatter(x1, y1, x2, y2):
    plt.figure(figsize=(12, 8), dpi=80)
    plt.scatter(x1, y1, s=50, marker='x', label='Positive')
    plt.scatter(x2, y2, s=50, marker='o', label='Negative')
    plt.legend()


# %% plot data scatter
plot_scatter(positive['X1'], positive['X2'], negative['X1'], negative['X2'])
plt.show()


# %% decision boundary plot
def plot_decision_boundary(model):
    w = model.coef_[0]
    b = model.intercept_[0]
    x1 = np.linspace(0.0, 4.5, 200)
    x2 = (- b / w[1]) - ((w[0] / w[1]) * x1)
    plot_scatter(positive['X1'], positive['X2'], negative['X1'], negative['X2'])
    plt.plot(x1, x2, 'b-.')


# %% use the svm of scikit-learn to separate
# linear SVC
# C=1
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
svc.fit(data[['X1', 'X2']], data['y'])
print(svc.score(data[['X1', 'X2']], data['y']))
plot_decision_boundary(svc)
plt.show()

# %% C=100
svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
svc2.fit(data[['X1', 'X2']], data['y'])
print(svc2.score(data[['X1', 'X2']], data['y']))
plot_decision_boundary(svc2)
plt.show()


# %% gaussian function construct
def gaussian(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * (sigma ** 2)))


# %% test gaussian function
print(gaussian(np.array([1.0, 2.0, 1.0]), np.array([0.0, 4.0, -1.0]), 2))

# %% load data2
raw_data2 = loadmat('ML_learning_Andrew_Ng/ex6_SVM/data/ex6data2.mat')
data2 = pd.DataFrame(raw_data2['X'], columns=('X1', 'X2'))
data2['y'] = raw_data2['y']
positive2 = data2[data2['y'].isin([1])]
negative2 = data2[data2['y'].isin([0])]

# %% plot data2 scatter
plot_scatter(positive2['X1'], positive2['X2'], negative2['X1'], negative2['X2'])
plt.show()

# %% SVC-rbf
svc3 = svm.SVC(C=100, gamma=10, probability=True)
svc3.fit(data2[['X1', 'X2']], data2['y'])
print(svc3.score(data2[['X1', 'X2']], data2['y']))


# %% non-linear decision boundary function
def plot_non_linear_db(model, axis, positive1, positive2, negative1, negative2):
    x1, x2 = np.meshgrid(np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
                         np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1))
    x_new = np.c_[x1.ravel(), x2.ravel()]
    y_predict = model.predict(x_new)
    zz = y_predict.reshape(x1.shape)
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.figure(figsize=(12, 8), dpi=80)
    plt.contourf(x1, x2, zz, cmap=custom_cmap)
    plt.scatter(positive1, positive2, s=50, marker='x', label='Positive')
    plt.scatter(negative1, negative2, s=50, marker='o', label='Negative')
    plt.legend()


# %% plot data2 decision boundary
plot_non_linear_db(svc3, axis=[0, 1.05, 0.35, 1.05],
                   positive1=positive2['X1'], positive2=positive2['X2'],
                   negative1=negative2['X1'], negative2=negative2['X2'])
plt.show()

# %% load data3 and init C and gamma
raw_data3 = loadmat('ML_learning_Andrew_Ng/ex6_SVM/data/ex6data3.mat')
X3 = raw_data3['X']
y3 = raw_data3['y'].ravel()
Xval3 = raw_data3['Xval']
yval3 = raw_data3['yval'].ravel()
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

# %% find the best C and gamma
best_score = 0
best_params = {'C': None, 'gamma': None}
for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X3, y3)
        score = svc.score(Xval3, yval3)

        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

print(best_score, best_params)

# %% plot data3 db and scatter
data3 = pd.DataFrame(Xval3, columns=['X1', 'X2'])
data3['y'] = yval3
positive3 = data3[data3['y'].isin([1])]
negative3 = data3[data3['y'].isin([0])]
svc4 = svm.SVC(C=best_params['C'], gamma=best_params['gamma'])
svc4.fit(X3, y3)
plot_non_linear_db(svc4, axis=[-0.6, 0.3, -0.8, 0.6],
                   positive1=positive3['X1'], positive2=positive3['X2'],
                   negative1=negative3['X1'], negative2=negative3['X2'])
plt.show()

# %% email preprocessing
em = 'ML_learning_Andrew_Ng/ex6_SVM/data/emailSample1.txt'
f = open(em, 'r', encoding='utf-8')
email = f.read()
print(email)
f.close()


# %% email processing function
def processing(email):
    """Besides Word Stemming and Removal of non-words"""
    # Lower-casing
    email = email.lower()
    # Stripping HTML
    email = re.sub(r'<.*>', '', email)
    # Normalize URLs
    email = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email)
    # Normalize Dollars
    email = re.sub(r'[\$][0-9]+', 'dollar number', email)
    email = re.sub(r'\$', 'dollar number', email)
    # Normalize numbers
    email = re.sub(r'[0-9]+', 'number', email)
    # Normalize email addresses
    email = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email)
    return email


def processing2(email):
    """Word Stemming and Removal of non-words"""
    stemmer = PorterStemmer()
    email = processing(email)

    # split email to every single words
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)

    # traversing every single words
    tokenlist = []
    for token in tokens:
        # delete any non-letter character
        token = re.sub('[^a-zA-Z0-9]', '', token)
        # extract the root
        stemmed = stemmer.stem(token)
        # remove empty string''
        if not len(token): continue
        tokenlist.append(stemmed)

    return tokenlist


def vocabindex(email, vocab):
    """提取存在单词的索引"""
    tokenlist = processing2(email)
    index = []
    for i in range(len(tokenlist)):
        for j in range(len(vocab)):
            if tokenlist[i] == vocab[j]:
                index.append(j + 1)
    return index


# %% extract features
def featuresvector(email):
    df = pd.read_table('ML_learning_Andrew_Ng/ex6_SVM/data/vocab.txt', names=['words'])
    vocab = df.values
    vector = np.zeros(len(vocab))
    voc_index = vocabindex(email, vocab)
    for j in voc_index:
        vector[j] = 1
    return vector


# %% test
df = pd.read_table('ML_learning_Andrew_Ng/ex6_SVM/data/vocab.txt', names=['words'])
vocab = df.values
print(processing2(email))
print(vocabindex(email, vocab))

# %% for email sample1
vector = featuresvector(email)
print(vector)
print('length of vector = {}\nnum of non-zero = {}'.format(len(vector), int(vector.sum())))

# %% training SVM for spam classification
# train data load
spam_train = loadmat('ML_learning_Andrew_Ng/ex6_SVM/data/spamTrain.mat')
spam_train_X = spam_train['X']
spam_train_y = spam_train['y'].ravel()
# test data load
spam_test = loadmat('ML_learning_Andrew_Ng/ex6_SVM/data/spamTest.mat')
spam_test_X = spam_test['Xtest']
spam_test_y = spam_test['ytest'].ravel()
# svm
svc = svm.SVC()
svc.fit(spam_train_X, spam_train_y)
print(svc.score(spam_train_X, spam_train_y))
print(svc.score(spam_test_X, spam_test_y))
