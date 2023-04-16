"""
@Description: Case Study: Breast cancer diagnosis
@Author: Stephen CUI
@Time: 2023-04-14 23:19:16
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


dataset = load_breast_cancer()
X, y = dataset['data'], dataset['target']
rng = np.random.RandomState(seed=4190)

MAX_leaf_nodes = 8
N_runs = 20

n_estimators_range = range(2, 20, 1)
bag_train_error = np.zeros(shape=(N_runs, len(n_estimators_range)))
rf_train_error = np.zeros(shape=(N_runs, len(n_estimators_range)))
xt_train_error = np.zeros(shape=(N_runs, len(n_estimators_range)))

bag_test_error = np.zeros(shape=(N_runs, len(n_estimators_range)))
rf_test_error = np.zeros(shape=(N_runs, len(n_estimators_range)))
xt_test_error = np.zeros(shape=(N_runs, len(n_estimators_range)))

if not os.path.exists('Data/ErrorVsNumEstimators.pickle'):
    for run in range(0, N_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.25, random_state=rng)

        for j, n_estimator in enumerate(n_estimators_range):
            tree = DecisionTreeClassifier(max_leaf_nodes=MAX_leaf_nodes)
            bag = BaggingClassifier(estimator=tree, n_estimators=n_estimator,
                                    max_samples=.5, n_jobs=-1,
                                    random_state=rng)
            bag.fit(X_train, y_train)
            bag_train_error[run, j] = 1 - \
                accuracy_score(y_train, bag.predict(X_train))
            bag_test_error[run, j] = 1 - \
                accuracy_score(y_test, bag.predict(X_test))

            rf = RandomForestClassifier(max_leaf_nodes=MAX_leaf_nodes,
                                        n_estimators=n_estimator,
                                        bootstrap=True, n_jobs=-1, random_state=rng)
            rf.fit(X_train, y_train)
            rf_train_error[run, j] = 1 - \
                accuracy_score(y_train, rf.predict(X_train))
            rf_test_error[run, j] = 1 - \
                accuracy_score(y_test, rf.predict(X_test))

            xt = ExtraTreesClassifier(max_leaf_nodes=MAX_leaf_nodes,
                                      n_estimators=n_estimator,
                                      bootstrap=True, n_jobs=-1,
                                      random_state=rng)
            xt.fit(X_train, y_train)
            xt_train_error[run, j] = 1 - \
                accuracy_score(y_train, xt.predict(X_train))
            xt_test_error[run, j] = 1 - \
                accuracy_score(y_test, xt.predict(X_test))
    results = (bag_train_error, bag_test_error, rf_train_error,
               rf_test_error, xt_train_error, xt_test_error)
    with open('Data/ErrorVsNumEstimators.pickle', 'wb') as result_file:
        pickle.dump(results, result_file)
else:
    with open('Data/ErrorVsNumEstimators.pickle', 'rb') as result_file:
        (bag_train_error, bag_test_error, rf_train_error, rf_test_error,
         xt_train_error, xt_test_error) = pickle.load(result_file)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey=True)
m = np.mean(bag_train_error * 100, axis=0)

for ls, ens in zip(['--', '-.', ':'], (bag_train_error, rf_train_error, xt_train_error)):
    m = np.mean(ens * 100, axis=0)
    ax[0].plot(n_estimators_range, m, linestyle=ls, linewidth=2, marker='o')
    ax[0].legend(['Bagging', 'Random Forest', 'Extra Trees'])
    ax[0].set_xlabel('Number of Estimators')
    ax[0].set_ylabel('Training Error (%)')
    ax[0].axis([min(n_estimators_range) - .5,
               max(n_estimators_range) + .5, 0, 10])
for ls, ens in zip(['--', '-.', ':'], (bag_test_error, rf_test_error, xt_test_error)):
    m = np.mean(ens * 100, axis=0)
    ax[1].plot(n_estimators_range, m, linestyle=ls, linewidth=2, marker='o')
    ax[1].legend(['Bagging', 'Random Forest', 'Extra Trees'])
    ax[1].set_xlabel('Number of Estimators')
    ax[1].set_ylabel('Hold-out Test Error (%)')
    ax[1].axis([min(n_estimators_range) - .5,
               max(n_estimators_range) + .5, 0, 10])
fig.tight_layout()
