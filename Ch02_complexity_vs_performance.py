"""
@Description: Base learner complexity vs ensemble performance
@Author: Stephen CUI
@Time: 2023-04-15 15:41:20
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

if not os.path.exists('Data/ErrorVsNumLeaves.pickle'):
    N_runs = 20
    N_leaf_range = [2, 4, 8, 16, 24, 32]

    n_estimators = 10
    bag_train_error = np.zeros(shape=(N_runs, len(N_leaf_range)))
    rf_train_error = np.zeros(shape=(N_runs, len(N_leaf_range)))
    xt_train_error = np.zeros(shape=(N_runs, len(N_leaf_range)))

    bag_test_error = np.zeros(shape=(N_runs, len(N_leaf_range)))
    rf_test_error = np.zeros(shape=(N_runs, len(N_leaf_range)))
    xt_test_error = np.zeros(shape=(N_runs, len(N_leaf_range)))

    for run in range(0, N_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.25, random_state=rng)

        for j, max_leaf_nodes in enumerate(N_leaf_range):
            tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
            bag = BaggingClassifier(estimator=tree, n_estimators=n_estimators,
                                    max_samples=.5, n_jobs=-1,
                                    random_state=rng)
            bag.fit(X_train, y_train)
            bag_train_error[run, j] = 1 - \
                accuracy_score(y_train, bag.predict(X_train))
            bag_test_error[run, j] = 1 - \
                accuracy_score(y_test, bag.predict(X_test))

            rf = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,
                                        n_estimators=n_estimators,
                                        bootstrap=True, n_jobs=-1, random_state=rng)
            rf.fit(X_train, y_train)
            rf_train_error[run, j] = 1 - \
                accuracy_score(y_train, rf.predict(X_train))
            rf_test_error[run, j] = 1 - \
                accuracy_score(y_test, rf.predict(X_test))

            xt = ExtraTreesClassifier(max_leaf_nodes=max_leaf_nodes,
                                      n_estimators=n_estimators,
                                      bootstrap=True, n_jobs=-1,
                                      random_state=rng)
            xt.fit(X_train, y_train)
            xt_train_error[run, j] = 1 - \
                accuracy_score(y_train, xt.predict(X_train))
            xt_test_error[run, j] = 1 - \
                accuracy_score(y_test, xt.predict(X_test))
    results = (bag_train_error, bag_test_error, rf_train_error,
               rf_test_error, xt_train_error, xt_test_error)
    with open('Data/ErrorVsNumLeaves.pickle', 'wb') as result_file:
        pickle.dump(results, result_file)
else:
    with open('Data/ErrorVsNumLeaves.pickle', 'rb') as result_file:
        (bag_train_error, bag_test_error, rf_train_error, rf_test_error,
         xt_train_error, xt_test_error) = pickle.load(result_file)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey=True)
m = np.mean(bag_train_error * 100, axis=0)

for ls, ens in zip(['--', '-.', ':'], (bag_train_error, rf_train_error, xt_train_error)):
    m = np.mean(ens * 100, axis=0)
    ax[0].plot(N_leaf_range, m, linestyle=ls, linewidth=2, marker='o')
    ax[0].legend(['Bagging', 'Random Forest', 'Extra Trees'])
    ax[0].set_xlabel('Depth of Trees in Ensemble')
    ax[0].set_ylabel('Training Error (%)')
    ax[0].set_xticks(N_leaf_range)
    ax[0].grid()
    ax[0].axis([min(N_leaf_range) - .5, max(N_leaf_range) + .5, 0, 15])
for ls, ens in zip(['--', '-.', ':'], (bag_test_error, rf_test_error, xt_test_error)):
    m = np.mean(ens * 100, axis=0)
    ax[1].plot(N_leaf_range, m, linestyle=ls, linewidth=2, marker='o')
    ax[1].legend(['Bagging', 'Random Forest', 'Extra Trees'])
    ax[1].set_xlabel('Depth of Trees in Ensemble')
    ax[1].set_ylabel('Hold-out Test Error (%)')
    ax[1].set_xticks(N_leaf_range)
    ax[1].grid()
    ax[1].axis([min(N_leaf_range) - .5, max(N_leaf_range) + .5, 0, 15])
fig.tight_layout()
