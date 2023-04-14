import time
import os
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=300, noise=.25, random_state=0)

if not os.path.exists('Data/SequentialVsParallelBagging.pickle'):
    n_estimators_range = np.arange(50, 525, 50, dtype=int)
    n_range = len(n_estimators_range)
    n_runs = 10

    run_time_seq = np.zeros((n_runs, n_range))
    run_time_par = np.zeros((n_runs, n_range))

    base_estimator = DecisionTreeClassifier(max_depth=5)

    for r in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=100)

        for i, n_estimators in enumerate(n_estimators_range):
            start = time.time()
            bag_ens = BaggingClassifier(estimator=base_estimator,
                                        n_estimators=n_estimators,
                                        max_samples=100, oob_score=True, n_jobs=1)
            bag_ens.fit(X_train, y_train)

            run_time_seq[r, i] = time.time() - start

        for i, n_estimators in enumerate(n_estimators_range):
            start = time.time()
            bag_ens = BaggingClassifier(estimator=base_estimator,
                                        n_estimators=n_estimators,
                                        max_samples=100, oob_score=True, n_jobs=-1)
            bag_ens.fit(X_train, y_train)

            run_time_par[r, i] = time.time() - start

    results = (run_time_seq, run_time_par)

    with open('Data/SequentialVsParallelBagging.pickle', 'wb') as result_file:
        pickle.dump(results, result_file)
else:
    with open('Data/SequentialVsParallelBagging.pickle', 'rb') as result_file:
        (run_time_seq, run_time_par) = pickle.load(result_file)

# run_time_seq_adj = np.copy(run_time_seq)
# run_time_seq_a
run_time_seq_mean = np.nanmean(run_time_seq, axis=0)
run_time_par_mean = np.nanmean(run_time_par, axis=0)

fig = plt.figure(figsize=(6, 4))
plt.plot(n_estimators_range, run_time_par_mean, label='Parallel Bagging')
plt.plot(n_estimators_range, run_time_seq_mean, label='Sequential Bagging')
plt.legend()
plt.xlabel('Number of estimators', fontsize=16)
plt.ylabel('Run Time (msec.)', fontsize=16)
fig.tight_layout()
