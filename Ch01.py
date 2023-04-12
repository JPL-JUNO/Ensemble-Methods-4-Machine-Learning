from sklearn.datasets import make_friedman1

X, y = make_friedman1(n_samples=500,
                      n_features=15,
                      noise=.3,
                      random_state=23)
