"""
@Description: make dataset
@Author: Stephen CUI
@Time: 2023-04-14 10:58:07
"""

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=300, noise=.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

if __name__ == '__main__':
    pass
