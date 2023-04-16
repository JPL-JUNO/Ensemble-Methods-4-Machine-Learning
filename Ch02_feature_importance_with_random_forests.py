"""
@Description: Feature importances with random forests
@Author: Stephen CUI
@Time: 2023-04-15 16:11:55
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = load_breast_cancer()
df = pd.DataFrame(data=dataset['data'],
                  columns=dataset['feature_names'])
df['diagnosis'] = dataset['target']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
cor = np.abs(df.corr())
sns.heatmap(cor, annot=False, cbar=False, cmap=plt.cm.Reds, ax=ax)
fig.tight_layout()

label_corr = cor.iloc[:, -1]
label_corr.sort_values(ascending=False)[1:11]  # remove correlation with self

X, y = dataset['data'], dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15)
n_features = X_train.shape[1]

rf = RandomForestClassifier(max_leaf_nodes=24, n_estimators=50, n_jobs=-1)
rf.fit(X_train, y_train)
err = 1 - accuracy_score(y_test, rf.predict(X_test))
print('Prediction Error = {0:4.2f}%'.format(err * 100))
# importance_threshold = 1 / n_features
importance_threshold = .02  # 个人觉得不应该用固定的值，应该将其设置为特征数量的函数
for i, (feature, importance) in enumerate(zip(dataset['feature_names'], rf.feature_importances_)):
    if importance > importance_threshold:
        # 4 表示的宽度
        print('[{0}] {1}(score={2:4.3f})'.format(i, feature, importance))


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
idx = np.arange(n_features)
imp = np.where(rf.feature_importances_ >= importance_threshold)
rest = np.setdiff1d(idx, imp)

plt.bar(idx[imp], rf.feature_importances_[imp], alpha=.65)
plt.bar(idx[rest], rf.feature_importances_[rest], alpha=.65)
for i, (feature, importance) in enumerate(zip(dataset['feature_names'], rf.feature_importances_)):
    if importance > importance_threshold:
        plt.text(i, .015, feature, ha='center', va='bottom',
                 rotation='vertical', fontsize=12, fontweight='bold')
    else:
        plt.text(i, .015, feature, ha='center', va='bottom',
                 rotation='vertical', fontsize=12, color='gray')
ax.xaxis.set_major_locator(plt.NullLocator())
# fig.axes[0].get_xaxis().set_visible(False)
plt.xlabel('Features for Breast Cancer Diagnosis', fontsize=16)
plt.ylabel('Feature Importance Score', fontsize=16)
fig.tight_layout()
