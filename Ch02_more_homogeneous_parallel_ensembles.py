"""
@Description: More homogeneous parallel ensembles 
@Author: Stephen CUI
@Time: 2023-04-14 22:32:27 
"""
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

# random subspaces
bag_ens = BaggingClassifier(estimator=SVC(),
                            n_estimators=100, max_samples=1.0,
                            bootstrap=False,  # uses all training samples
                            max_features=.5, bootstrap_features=True)  # bootstrap samples 50% of feature

# random patches
bag_ens = BaggingClassifier(estimator=SVC(),
                            n_estimators=100, max_samples=.75, bootstrap=True,
                            max_features=.5, bootstrap_features=True)
