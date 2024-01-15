import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
Xo = data[['Pregnancies', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']
X = X.values
Xo = Xo.values
y = y.values.astype(np.bool_)
test_portion = 0.15
ntest = int(test_portion * len(data))
ntrain = len(data) - ntest
r = np.random.permutation(len(data))
test_indices = r[:ntest]
train_indices = r[ntest:]
X_test = X[test_indices]
X_train = X[train_indices]
Xo_test = Xo[test_indices]
Xo_train = Xo[train_indices]
y_train = y[train_indices]
y_test = y[test_indices]

def knn(X_train, X_test, y_train):
    new_n = len(X_train)
    n = len(X_test)
    for v in range(5):
        start = int(v * new_n / 7)
        end = int((v + 1) * new_n / 7)
        X_v= X_train[start:end]
        y_v = y_train[start:end]
        X_tr = np.delete(X_train, np.s_[start:end], 0)
        y_tr = np.delete(y_train, np.s_[start:end], 0)
        nv = len(y_v)
        prediction = np.zeros(nv)
        avg_errors = np.zeros(24)
        for k in range(1, 25):
            for i in range(nv):
                distances = np.linalg.norm(X_tr - X_v[i], axis=1)
                neigh_idx = np.argpartition(distances, k, axis=None)[:k]

                prediction[i] = np.sum(y_tr[neigh_idx]) / k
            avg_errors[k - 1] += cross_entropy(prediction, y_v) / 5
    best_k = np.argmin(avg_errors) + 1
    yhat = np.zeros(n)

    for i in range(n):
        distances = np.linalg.norm(X_train - X_test[i], axis=1)
        neigh_idx = np.argpartition(distances, best_k)[:best_k]
        yhat[i] = np.sum(y_train[neigh_idx]) / k
    yhat = np.minimum(2 * yhat, 1)
    return yhat

def logistic_regression(X_train, X_test, y_train, n_iters=1000, learning_rate=0.001):

    multiplier = np.where(y_train == 1, 3, 1)
    n = X_train.shape[0]
    p = X_train.shape[1]
    w = np.zeros(p)
    b = 0

    for i in range(n_iters):
        z = np.dot(w, X_train.T) + b
        sigmoids = np.divide(1, 1 + np.exp(-1 * z))

        ## Gradient descent of log loss function

        dw = (1/n) * np.dot(np.multiply(multiplier, sigmoids - y_train), X_train)
        db = (1/n) * np.sum(np.multiply(multiplier, sigmoids - y_train))
        # print(db)
        w -= learning_rate * dw
        b -= learning_rate * db

    yhat = np.divide(1, 1 + np.exp(-1 * (w @ X_test.T + b)))
    # plt.plot(np.sort(yhat))
    # indices = np.argsort(yhat)
    # plt.scatter(np.arange(ntest), y_test[indices])
    # plt.show()
    return yhat
def lr2(X_train, X_test, y_train):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    yhat = lr.predict(X_test)
    return yhat

def truncated_svd(X_train, X_test, y_train):
    U, S, Vt = np.linalg.svd(X_train)
    Uk = U[:, :2]
    Sk = np.diag(S)[:2, :2]
    Vtk = Vt[:2, :]
    svd_k_inv = Vtk.T @ np.linalg.inv(Sk) @ Uk.T
    w_hat = svd_k_inv @ y_train
    y_hat = X_test @ w_hat
    print(y_hat.shape)
    return y_hat
def decision_tree(X_train, X_test, y_train):
    decision = DecisionTreeClassifier(criterion='entropy')
    decision.fit(X_train, y_train)
    yhat = decision.predict(X_test)
    return yhat

def cross_entropy(px, y):
    ppx = np.where(px == 0, 0.0000001, px)
    ppx = np.where(ppx == 1, 0.9999999, ppx)
    ppx /= sum(ppx)
    return (1/len(y)) * np.sum(- 3 * np.multiply(y, np.log(ppx)) - np.multiply(1 - y, np.log(np.ones(len(y)) - ppx)))

def fraction_correct(x, y):
    bool_array = np.zeros(len(x), dtype=bool)
    bool_array[x > 0.5] = 1
    return np.sum(np.logical_not(np.logical_xor(bool_array, y))) / len(x)

def false_positive_rate(x, y):
    bool_array = np.zeros(len(x), dtype=bool)
    bool_array[x > 0.5] = 1
    return 1 - (np.sum(np.logical_and(bool_array, y)) / np.sum(1 - y))
pca = PCA(n_components=3)
X_transform = pca.fit_transform(Xo)
color_map = {1: "red", 0: "blue"}
colors = [color_map[i] for i in y]
f = plt.figure()
ax = f.subplots(1, 3)
ax[0].scatter(X_transform[:, 0], X_transform[:, 1], color=colors)
ax[0].set(xlabel="PC0", ylabel="PC1")

ax[1].scatter(X_transform[:, 0], X_transform[:, 2], color=colors)
ax[1].set(xlabel="PC0", ylabel="PC2")
ax[2].scatter(X_transform[:, 1], X_transform[:, 2], color=colors)
ax[2].set(xlabel="PC1", ylabel="PC2")
plt.title("PCA on Observable Data")
plt.show()


yhat_knn = knn(X_train, X_test, y_train)
yhat_dt = decision_tree(X_train, X_test, y_train)
yhat_lsq = X_test @ np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
yhat_svd = truncated_svd(X_train, X_test, y_train)
yhat_lr = logistic_regression(X_train, X_test, y_train)
yhat_lr2 = lr2(X_train, X_test, y_train)

yohat_lsq = Xo_test @ np.linalg.inv(Xo_train.T @ Xo_train) @ Xo_train.T @ y_train
yohat_svd = truncated_svd(Xo_train, Xo_test, y_train)
yohat_knn = knn(Xo_train, Xo_test, y_train)
yohat_dt = decision_tree(Xo_train, Xo_test, y_train)
yohat_lr = logistic_regression(Xo_train, Xo_test, y_train)
print("KNN Loss: ", cross_entropy(yhat_knn, y_test))
print("Logistic Regression Loss: ", cross_entropy(yhat_lr, y_test))
print("Decision Tree Loss: ", cross_entropy(yhat_dt, y_test))
print("SVD Loss: ", cross_entropy(yhat_svd, y_test))

print(yhat_dt)
# print(fraction_correct(yhat_lr2, y_test))
# print(fraction_correct(yhat_lsq, y_test))
# print(fraction_correct(yhat_lr, y_test))
# print(fraction_correct(yhat_knn, y_test))
# print(fraction_correct(yhat_dt, y_test))
# print(fraction_correct(yohat_lr, y_test))
# print(fraction_correct(yohat_knn, y_test))
# print(fraction_correct(yohat_dt, y_test))
print(cross_entropy(np.zeros(len(y_test)), y_test))
models = [yhat_lsq, yhat_knn, yhat_lr, yhat_dt, yhat_svd]
o_models = [yohat_lsq, yohat_knn, yohat_lr, yohat_dt, yhat_svd]

fractions = [false_positive_rate(m, y_test) for m in models]
o_fractions = [false_positive_rate(m, y_test) for m in o_models]
entropies = [cross_entropy(m, y_test) for m in models]
o_entropies = [cross_entropy(m, y_test) for m in o_models]
labels = ["Least Squares", "K Nearest Neighbors", "Logistic Regression", "Decision Tree", "Truncated SVD" ]
plt.bar(labels, fractions)
plt.title("False Positive Rate: All Data")
plt.show()

plt.bar(labels, o_fractions)
plt.title("False Positive Rate: Observable Data")
plt.show()

print("Fraction KNN: ", fraction_correct(yhat_knn, y_test))
print("Fraction LR: ", fraction_correct(yhat_lr, y_test))

plt.bar(labels, entropies)
plt.title("Cross Entropy: All Data")
plt.show()

plt.bar(labels, o_entropies)
plt.title("Cross Entropy: Observable Data")
plt.show()

