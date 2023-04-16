"""
@Description: Visualization
@Author: Stephen CUI
@Time: 2023-04-14 10:57:54
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as col
import numpy as np
from typing import List
from numpy import ndarray
from matplotlib.pyplot import Axes

from sklearn.datasets import make_moons
from sklearn.svm import SVC


def get_colors(colormap: str = 'viridis', n_colors: int = 2, bounds: tuple = (0, 1)) -> List[str]:
    """获取颜色

    Args:
        colormap (str, optional): 颜色映射. Defaults to 'viridis'.
        n_colors (int, optional): 颜色的个数. Defaults to 2.
        bounds (tuple, optional): 范围. Defaults to (0, 1).

    Returns:
        _type_: _description_
    """
    cmap = mpl.colormaps.get_cmap(colormap)

    colors_rgb = cmap(np.linspace(bounds[0], bounds[1], num=n_colors))
    colors_hex = [col.rgb2hex(c) for c in colors_rgb]
    return colors_hex


def plot_2d_data(ax: Axes, X: ndarray, y: ndarray, alpha: float = .95, s: int | ndarray = 20,
                 xlabel: str = None, ylabel: str = None, title: str = None, legend: str = None,
                 colormap: str = 'viridis') -> None:
    n_examples, n_features = X.shape

    assert n_features == 2, 'Data Set is not 2D!'
    assert n_examples == len(y), 'Length of X is not equal to the length of y!'

    unique_labels = np.sort(np.unique(y))
    n_classes = len(unique_labels)

    markers = ['o', 's', '^', 'v', '<', '>', 'p']

    cmap = mpl.colormaps.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, num=n_classes))

    if isinstance(s, ndarray):
        if len(s) != n_examples:
            raise ValueError('Length of s is not equal to length of y')
    else:
        s = np.full_like(y, fill_value=s)

    for i, label in enumerate(unique_labels):
        marker_color = col.rgb2hex(colors[i])
        marker_shape = markers[i % len(markers)]
        ax.scatter(X[y == label, 0], X[y == label, 1], s=s[y == label],
                   marker=marker_shape, c=marker_color, edgecolors='k', alpha=alpha)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)
    if title is not None:
        ax.set_title(title)
    if legend is not None:
        ax.legend(legend)


def plot_2d_classifier(ax: Axes, X: ndarray, y: ndarray, predict_function,
                       predict_args=None, predict_proba: bool = False, boundary_level=.5,
                       s=20, plot_data=True, alpha=.75, xlabel=None, ylabel=None, title=None, legend: str = None, colormap: str = 'viridis') -> None:
    xMin, xMax = X[:, 0].min() - .25, X[:, 0].max() + .25
    yMin, yMax = X[:, 1].min() - .25, X[:, 1].max() + .25
    xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, .05),
                               np.arange(yMin, yMax, .05))
    if predict_proba:
        zMesh = predict_function(np.c_[xMesh.ravel(), yMesh.ravel()])[:, 1]
    elif predict_args is None:
        zMesh = predict_function(np.c_[xMesh.ravel(), yMesh.ravel()])
    else:
        zMesh = predict_function(
            np.c_[xMesh.ravel(), yMesh.ravel()], predict_args)
    zMesh = zMesh.reshape(xMesh.shape)

    ax.contourf(xMesh, yMesh, zMesh, cmap=colormap,
                alpha=alpha, antialiased=True)
    if boundary_level is not None:
        ax.contour(xMesh, yMesh, zMesh, [
                   boundary_level], linewidths=3, colors='k')

    if plot_data:
        plot_2d_data(ax, X, y, s=s, xlabel=xlabel, ylabel=ylabel,
                     title=title, legend=legend, colormap=colormap)


if __name__ == '__main__':
    x = get_colors()
    X, y = make_moons(n_samples=100, noise=.15)

    plt.ion()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    svm = SVC(kernel='rbf', gamma=2.0, probability=True)
    svm.fit(X, y)
    plot_2d_classifier(ax, X, y, predict_function=svm.predict_proba,
                       predict_proba=True, xlabel='x', ylabel='y', title='Scatter plot test')
    fig.tight_layout()
