"""
@Description: Bagging with sklearn
@Author: Stephen CUI
@Time: 2023-04-14 10:59:38
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from make_dataset import *
from visualization import plot_2d_classifier
import matplotlib.pyplot as plt
import numpy as np

base_estimator = DecisionTreeClassifier(max_depth=10)
bag_ens = BaggingClassifier(estimator=base_estimator,
                            n_estimators=500, max_samples=100, oob_score=True)
bag_ens.fit(X_train, y_train)
y_pred = bag_ens.predict(X_test)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
trees_to_plot = np.random.choice(500, size=5, replace=True)
title = 'Bagging Ensemble (acc = {0:4.2f} %)'.format(
    accuracy_score(y_test, y_pred) * 100)
plot_2d_classifier(axes[0, 0], X, y, colormap='RdBu', alpha=.3,
                   predict_function=bag_ens.predict,
                   xlabel='$x_1$', ylabel='$x_2$', title=title)
for i, ax in enumerate(axes.ravel()[1:]):
    j = trees_to_plot[i]
    test_acc_clf = accuracy_score(y_test, bag_ens[j].predict(X_test))
    bag_samples = bag_ens.estimators_samples_[j]
    X_bag = X[bag_samples, :]
    y_bag = y[bag_samples]
    title = 'Decision Tree {1} (acc = {0:4.2f}%)'.format(test_acc_clf * 100, j)
    plot_2d_classifier(ax, X, y, colormap='RdBu', alpha=.3,
                       predict_function=bag_ens[j].predict,
                       xlabel='$x_1$', ylabel='$x_2$', title=title)
fig.tight_layout()
