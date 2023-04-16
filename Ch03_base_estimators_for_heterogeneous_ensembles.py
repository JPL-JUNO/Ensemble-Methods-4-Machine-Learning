"""
@Description: Base estimators for heterogeneous ensembles
@Author: Stephen CUI
@Time: 2023-04-15 22:21:20
"""

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from visualization import plot_2d_data
from visualization import plot_2d_classifier, get_colors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from numpy import ndarray
from typing import TypeVar, List, Tuple
T = TypeVar('T')

X, y = make_moons(600, noise=.25, random_state=13)
X, X_val, y, y_val = train_test_split(X, y, test_size=.25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plot_2d_data(ax, X, y, alpha=.95, xlabel='$x_1$', ylabel='$x_2$',
             title='Two moons data', colormap='RdBu')
plt.tight_layout()

estimators: List[Tuple[str, object]] = [('dt', DecisionTreeClassifier(max_depth=5)),
                                        ('svm', SVC(gamma=1.0, C=1.0, probability=True)),
                                        ('gp', GaussianProcessClassifier(RBF(1.0))),
                                        ('3nn', KNeighborsClassifier(n_neighbors=3)),
                                        ('rf', RandomForestClassifier(
                                            max_depth=3, n_estimators=3)),
                                        ('gnb', GaussianNB())]


def fit(estimators: List[Tuple[str, object]], X: ndarray, y: ndarray) -> List[Tuple[str, object]]:
    for _, estimator in estimators:
        estimator.fit(X, y)
    return estimators


estimators = fit(estimators, X_train, y_train)

n_estimators = len(estimators)
n_rows, n_cols = n_estimators // 3, 3

fig, ax = plt.subplots(n_rows, n_cols, figsize=(9, 6))
for i, (model, estimator) in enumerate(estimators):
    r, c = divmod(i, 3)
    test_err = 1 - accuracy_score(y_test, estimator.predict(X_test))
    title = '{0} (test error = {1:4.2f}%)'.format(model, test_err * 100)
    plot_2d_classifier(ax[r, c], X, y, colormap='RdBu', alpha=.5,
                       predict_function=estimator.predict_proba, predict_proba=True,
                       xlabel='$x_1$', ylabel='$x_2$', title=title)
    ax[r, c].set_xticks([])
    ax[r, c].set_yticks([])
fig.tight_layout()


def predict_individual(X: ndarray, estimators: object, proba: bool = False) -> ndarray:
    """_summary_

    Args:
        X (ndarray): _description_
        estimators (object): _description_
        proba (bool, optional): _description_. Defaults to False.

    Returns:
        ndarray: _description_
    """
    n_estimators = len(estimators)
    n_samples = X.shape[0]

    y = np.zeros(shape=(n_samples, n_estimators))
    for i, (_, estimator) in enumerate(estimators):
        if proba:
            y[:, i] = estimator.predict_proba(X)[:, 1]
        else:
            y[:, i] = estimator.predict(X)
    return y


y_individual = predict_individual(X_test, estimators, proba=False)


from scipy.stats import mode


def combine_using_majority_vote(X: ndarray, estimators: object) -> ndarray:
    y_individual = predict_individual(X, estimators, proba=False)
    y_final = mode(y_individual, axis=1, keepdims=False)
    return y_final[0].reshape(-1,)


y_pred = combine_using_majority_vote(X_test, estimators)
test_error = 1 - accuracy_score(y_test, y_pred)
print(
    'Majority Vote Error of heterogeneous Estimator: {:5.4f}%'.format(test_error * 100))


def combine_using_accuracy_weighting(X: ndarray, estimators: object, X_val: ndarray, y_val: ndarray) -> ndarray:
    n_estimators = len(estimators)
    y_val_individual = predict_individual(X_val, estimators, proba=False)

    wts = [accuracy_score(y_val, y_val_individual[:, col])
           for col in range(n_estimators)]

    wts /= np.sum(wts)
    y_pred_individual = predict_individual(X, estimators, proba=False)
    y_final = np.dot(y_pred_individual, wts)
    return np.round(y_final)


y_pred_accuracy_weight = combine_using_accuracy_weighting(
    X_test, estimators, X_val, y_val)
test_error_accuracy_weighting = 1 - \
    accuracy_score(y_test, y_pred_accuracy_weight)
print('Accuracy Weighting Error: {0:5.4f}%'.format(
    test_error_accuracy_weighting * 100))


def entropy(y: ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = np.array(counts.astype('float') / len(y))
    ent = - p.T @ np.log2(p)
    return ent


def combine_using_entropy_weighting(X: ndarray, estimator: object, X_val: ndarray, y_val) -> ndarray:
    n_estimators = len(estimator)
    y_val_individuals = predict_individual(X_val, estimators, proba=False)

    wts = [1 / entropy(y_val_individuals[:, i])
           for i in range(n_estimators)]
    wts /= np.sum(wts)

    y_pred_individual = predict_individual(X, estimators, proba=False)
    y_final = np.dot(y_pred_individual, wts)
    return np.round(y_final)


y_pred_entropy = combine_using_entropy_weighting(
    X_test, estimators, X_val, y_val)
test_error_entropy = 1 - accuracy_score(y_test, y_pred_entropy)
print('Entropy Weighting Error: {0:4.4f}%'.format(test_error_entropy * 100))


def combine_using_Dempster_Shafer(X, estimators):
    p_individual = predict_individual(X, estimators, proba=True)
    bpa0 = 1 - np.prod(p_individual, axis=1) - 1e-7
    bpa1 = 1 - np.prod(1 - p_individual, axis=1) - 1e-7
    belief = np.vstack([bpa0 / (1 - bpa0),
                        bpa1 / (1 - bpa1)]).T

    y_final = np.argmax(belief, axis=1)
    return y_final


y_pred_DST = combine_using_Dempster_Shafer(X_test, estimators)
test_error_DST = 1 - accuracy_score(y_test, y_pred_DST)
print('DST Error : {0:4.4f}%'.format(test_error_DST * 100))


combination_methods = [('Majority vote', combine_using_majority_vote),
                       ('Dempster-Shafer', combine_using_Dempster_Shafer),
                       ('Accuracy Weighting', combine_using_accuracy_weighting),
                       ('Entropy Weighting', combine_using_entropy_weighting)]

nrows, ncols = 2, 2
cm = get_colors(colormap='RdBu')
xMin, xMax = X_train[:, 0].min() - .25, X_train[:, 0].max() + .25
yMin, yMax = X_train[:, 1].min() - .25, X_train[:, 1].max() + .25

xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, .05),
                           np.arange(yMin, yMax, .05))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 6))
for (ax, (method, combiner)) in zip(axes.ravel(), combination_methods):
    if method == 'Majority vote' or method == 'Dempster-Shafer':
        zMesh = combiner(np.c_[xMesh.ravel(), yMesh.ravel()], estimators)
        y_pred = combiner(X_test, estimators)
    if method == 'Accuracy Weighting' or method == 'Entropy Weighting':
        zMesh = combiner(
            np.c_[xMesh.ravel(), yMesh.ravel()], estimators, X_val, y_val)
        y_pred = combiner(X_test, estimators, X_val, y_val)
    zMesh = zMesh.reshape(xMesh.shape)
    ax.contourf(xMesh, yMesh, zMesh, cmap='RdBu', alpha=.65)
    ax.contour(xMesh, yMesh, zMesh, [.5], colors='k', linewidths=2.5)
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train ==
               0, 1], marker='o', c=cm[0], edgecolor='k')
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train ==
               1, 1], marker='s', c=cm[1], edgecolor='k')

    title = '{}'.format(method)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.savefig('Figures/fig3-10.png', dpi=300)
