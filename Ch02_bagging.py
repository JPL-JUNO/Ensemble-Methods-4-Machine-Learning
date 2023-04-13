import numpy as np

bag = np.random.choice(range(0, 50), size=50, replace=True)
np.sort(bag)

oob = np.setdiff1d(range(50), bag)
