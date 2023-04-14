"""
@Description: Random forests with scikit-learn
@Author: Stephen CUI
@Time: 2023-04-14 16:00:06
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from make_dataset import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from visualization import plot_2d_classifier
from Ch02_bagging_with_sklearn import trees_to_plot

rng = np.random.RandomState(seed=4109)

rf_ens = RandomForestClassifier(n_estimators=500,
                                max_depth=10,
                                oob_score=True,
                                n_jobs=-1)

rf_ens.fit(X_train, y_train)
y_pred = rf_ens.predict(X_test)

if __name__ == '__main__':
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    title = 'Random Forest acc {0:4.2f}%'.format(
        accuracy_score(y_test, y_pred) * 100)
    plot_2d_classifier(axes[0, 0], X, y, colormap='RdBu',
                       alpha=.3,
                       predict_function=rf_ens.predict,
                       xlabel='$x_1$', ylabel='$x_2$', title=title)

    for i, ax in enumerate(axes.ravel()[1:]):
        j = trees_to_plot[i]
        test_acc_clf = accuracy_score(y_test, rf_ens[j].predict(X_test))

        title = 'Randomized Tree {1} (acc = {0:4.2f}%)'.format(
            100 * test_acc_clf, j)
        plot_2d_classifier(ax, X, y, colormap='RdBu', alpha=.3,
                           predict_function=rf_ens[j].predict,
                           xlabel='$x_1$', ylabel='$x_2$', title=title)
        plt.tight_layout()

    for i, score in enumerate(rf_ens.feature_importances_):
        print('Feature x{0}: {1:6.5f}'.format(i, score))
