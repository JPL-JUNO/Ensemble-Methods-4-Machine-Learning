import matplotlib.pyplot as plt
from numpy import ndarray


def plot_train_vs_test(arg_space: ndarray, train_score: ndarray, test_score: ndarray, title=None, semilogx: bool = False) -> None:
    assert train_score.shape == test_score.shape, '训练得分与测试得分长度不同'
    assert arg_space.shape == train_score.shape
    fig = plt.figure()
    if semilogx:
        plt.semilogx(arg_space, train_score, linewidth=2,
                     marker='o', markersize=10, mfc='none', label='Train Score')
        plt.semilogx(arg_space, test_score, linewidth=2,
                     marker='s', markersize=10, mfc='none', label='Test Score')
    else:
        plt.plot(arg_space, train_score, linewidth=2,
                 marker='o', markersize=10, mfc='none', label='Train Score')
        plt.plot(arg_space, test_score, linewidth=2,
                 marker='s', markersize=10, mfc='none', label='Test Score')
    plt.legend()
    plt.ylabel('$R^2$ coefficient')
    if title:
        plt.title(title)
