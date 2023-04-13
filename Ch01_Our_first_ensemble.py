"""
@Description: Our first ensemble
@Author: Stephen CUI
@Time: 2023-04-13 14:29:24
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_friedman1

X, y = make_friedman1(n_samples=500, n_features=15, noise=.3, random_state=23)
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=.25)

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

estimators = {'krr': KernelRidge(kernel='rbf'),
              'svr': SVR(gamma=.5),
              'dtr': DecisionTreeRegressor(max_depth=3),
              'knn': KNeighborsRegressor(n_neighbors=4),
              'gpr': GaussianProcessRegressor(alpha=.1),
              'mlp': MLPRegressor(alpha=25, max_iter=10_000)}

for name, estimator in estimators.items():
    estimator = estimator.fit(X_trn, y_trn)


import numpy as np
n_estimators, n_test_samples = len(estimators), X_tst.shape[0]
y_individual = np.zeros(shape=(n_test_samples, n_estimators))
for i, (model, estimator) in enumerate(estimators.items()):
    y_individual[:, i] = estimator.predict(X_tst)

y_final = np.mean(y_individual, axis=1)


import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from itertools import combinations

models = list(estimators.keys())
combinations_mean = np.zeros((len(estimators), 1))
combinations_std = np.zeros((len(estimators), 1))

for n_ensemble in range(1, len(estimators) + 1):
    combination = combinations(estimators, n_ensemble)

    average_predictions = [np.mean(np.array(
        [y_individual[:, models.index(e)] for e in list(c)]), axis=0) for c in combination]
    average_r2 = [r2_score(y_tst, y_pred) for y_pred in average_predictions]
    n_combinations = len(average_r2)

    plt.scatter(np.full((n_combinations,), n_ensemble),
                average_r2, color='steelblue', alpha=.5)
    combinations_mean[n_ensemble - 1] = np.mean(average_r2)
    combinations_std[n_ensemble - 1] = np.std(average_r2)

    if n_ensemble == 1:
        for r2, name in zip(average_r2, estimators):
            plt.text(1.1, r2, name)


fig = plt.figure()
plt.fill_between(np.arange(1, len(estimators) + 1),
                 (combinations_mean - combinations_std).ravel(),
                 (combinations_mean + combinations_std).ravel(),
                 color='orange', alpha=.3, linewidth=2)
plt.plot(np.arange(1, len(estimators) + 1), combinations_mean.ravel(), marker='o',
         markersize=8, markeredgecolor='k', linewidth=2)
plt.xlabel('Number of Model Ensembled', fontsize=12)
plt.ylabel('Coefficient of Determination, $R^2$', fontsize=12)
