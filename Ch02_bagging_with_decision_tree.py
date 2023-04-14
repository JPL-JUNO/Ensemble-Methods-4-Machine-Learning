import numpy as np
from numpy import ndarray
from sklearn.tree import DecisionTreeClassifier
from typing import List

rng = np.random.RandomState(seed=4190)


def bagging_fit(X: ndarray, y: ndarray,
                n_estimators: int,
                max_depth: int = 5,
                max_samples: int = 200) -> List[DecisionTreeClassifier]:
    """实现 bagging 拟合，串行实现方式

    Args:
        X (ndarray): 输入数据
        y (ndarray): 标签数据
        n_estimators (int): 基估计器数量（这里是决策树）
        max_depth (int, optional): 决策树的参数，决策树深度. Defaults to 5.
        max_samples (int, optional): bagging sample 参数，重复抽样的样本数. Defaults to 200.

    Returns:
        _type_: 有决策树模型组成的 list
    """

    assert X.shape[0] == y.shape[0]
    n_examples = len(y)
    estimators = [DecisionTreeClassifier(max_depth=max_depth)
                  for _ in range(n_estimators)]

    for tree in estimators:
        bag = np.random.choice(n_examples, max_samples,
                               replace=True)
        tree.fit(X[bag, :], y[bag])
    return estimators


from scipy.stats import mode


def bagging_predict(X, estimators):
    all_predictions = np.array([tree.predict(X)for tree in estimators])
    y_pred, _ = mode(all_predictions, axis=0, keepdims=False)
    return np.squeeze(y_pred)


# 测试上面写的 ensemble （bagging with decision tree）

if __name__ == '__main__':

    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_moons(n_samples=300, noise=.25, random_state=rng)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33,
                                                        random_state=rng)
    bag_ens = bagging_fit(X_train, y_train,
                          n_estimators=500,
                          max_depth=12, max_samples=100)
    y_pred = bagging_predict(X_test, bag_ens)
    ensemble_acc = accuracy_score(y_test, y_pred)
    print(ensemble_acc)

    tree = DecisionTreeClassifier(max_depth=12)
    y_pred_single = tree.fit(X_train, y_train).predict(X_test)
    tree_acc = accuracy_score(y_test, y_pred_single)
    print('Single Decision Tree: Holdout tes accuracy = {0:4.2f}'.format(
        tree_acc * 100))

    import matplotlib.pyplot as plt
    from visualization import plot_2d_classifier

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    title = 'Single Decision Tree (acc = {0:4.2f}%)'.format(tree_acc * 100)
    plot_2d_classifier(ax[0], X, y, colormap='RdBu', alpha=.3,
                       predict_function=tree.predict, xlabel='$x_1$', ylabel='$x_2$',
                       title=title)
    title = 'Bagging Ensemble (acc = {0:4.2f}%)'.format(ensemble_acc * 100)
    plot_2d_classifier(ax[1], X, y, colormap='RdBu', alpha=.3,
                       predict_function=bagging_predict, predict_args=(
                           bag_ens), xlabel='$x_1$', ylabel='$x_2$', title=title)
