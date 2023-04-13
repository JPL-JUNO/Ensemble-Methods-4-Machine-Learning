from sklearn.datasets import make_friedman1
from train_vs_test import plot_train_vs_test

X, y = make_friedman1(n_samples=500,
                      n_features=15,
                      noise=.3,
                      random_state=23)
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

subsets = ShuffleSplit(n_splits=5, test_size=.33, random_state=23)
model = DecisionTreeRegressor()
trn_scores, tst_scores = validation_curve(model, X, y,
                                          param_name='max_depth',
                                          param_range=range(1, 11),
                                          cv=subsets, scoring='r2')
mean_train_score = trn_scores.mean(axis=1)
mean_test_score = tst_scores.mean(axis=1)

col = plt.colormaps.get_cmap('RdBu')

fig = plt.figure()
plt.plot(range(1, 11), mean_train_score, linewidth=2,
         marker='o', markersize=10, mfc='none', label='Train Score')
plt.plot(range(1, 11), mean_test_score, linewidth=2,
         marker='s', markersize=10, mfc='none', label='Test Score')
plt.xticks(range(1, 11))
plt.legend()
plt.ylabel('$R^2$ coefficient')
plt.xlabel('Decision tree complexity (maximum tree depth)')
plt.title('Decision Tree Regression')
plt.show()


from sklearn.svm import SVR
from sklearn.metrics import r2_score

n_syn = 100
X_syn = np.linspace(-10.0, 10.0, n_syn).reshape(-1, 1)
y_true = np.sin(X_syn) / X_syn
y_true = y_true.ravel()
y_syn = y_true + .125 * np.random.normal(0, 1, y_true.shape)
y_syn[-1] = -.5  # add one very noisy point to illustrate the impact of overfitting

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, C in zip(axes.ravel(), 10.0**np.arange(-3, 3)):
    ax.scatter(X_syn[:, 0], y_syn, edgecolors='k', alpha=.5)
    ax.plot(X_syn[:, 0], y_syn, linewidth=1, linestyle='--', label='true')

    model = SVR(C=C, kernel='rbf', gamma=.75)
    model.fit(X_syn, y_syn)
    y_pred = model.predict(X_syn)

    ax.plot(X_syn[:, 0], y_pred, linewidth=2, linestyle='--', label='learned')
    trn_score = r2_score(y_syn, y_pred)
    ax.set_title('C=$10^{{ {0} }}$, trn score = {1:3.2f}'.format(
        int(np.log10(C)), trn_score))

handles, labels = axes[0, 0].get_legend_handles_labels()
axes[0, 0].legend(handles, labels, loc='upper left', fontsize=10)
fig.tight_layout()

from sklearn.svm import SVR
model = SVR(kernel='rbf', gamma=.1)
trn_scores, tst_scores = validation_curve(model, X, y.ravel(),
                                          param_name='C',
                                          param_range=np.logspace(-2, 4, 7),
                                          cv=subsets, scoring='r2')

mean_test_score = tst_scores.mean(axis=1)
mean_train_score = trn_scores.mean(axis=1)


plot_train_vs_test(np.logspace(-2, 4, 7),
                   mean_train_score, mean_test_score, semilogx=True)
