# coding:utf-8
import numpy as np
import os
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
import numpy as np
from numpy import *
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import time
import matplotlib.pyplot as plt
import mak
from sklearn.model_selection import GridSearchCV


def train_mak_xgb(X, Y):

    mahakil = mak.MAHAKIL()
    X, Y = mahakil.fit_sample(X, Y)

    kf = KFold(n_splits=10, shuffle=True)
    scores_max = 0
    scores_all = []
    clf_scores_max = 0
    start = time.time()
    for train, test in kf.split(X):
        # 遍历样本
        X_train, X_test = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        clf = XGBClassifier()
        params = [
            {'n_estimators': [10, 100, 1000, 10000]},
            {'max_depth': [6, 60, 600]},
            {'min_child_weight': [1, 2, 4, 8]},
            {'gamma': [1, 0.1, 0.01, 0.001]},
            {'subsample': [0.6, 0.7, 0.8]},
            {'colsample_btree': [0.6, 0.7, 0.8]},
            {'objective': ['reg:logistic',
                           'binary:logistic', 'binary:logitraw']}
        ]

        optimized_clf = GridSearchCV(estimator=clf,
                                     param_grid=params,
                                     n_jobs=4)
        optimized_clf = clf
        optimized_clf.fit(X_train, y_train)
        scores = optimized_clf.score(X_test, y_test)
        scores_all.append(scores)
        if scores_max < scores:
            clf_scores_max = optimized_clf
            scores_max = scores
        print("Accuracy: %0.8f。" % (scores))

    end = time.time()
    running_time = end - start
    print("耗时%0.2f秒。平均得分是：%0.2f%%" %
          (running_time, sum(scores_all)/len(scores_all)*100))

    return clf_scores_max


if __name__ == '__main__':

    feautres = np.random.uniform(low=2, high=8, size=(200, 8))
    targets = np.random.randint(0, 3, [200])

    train_x = feautres[:int(len(feautres)*0.8), :]
    test_x = feautres[int(len(feautres)*0.8):, :]
    train_y = targets[:int(len(targets)*0.8)]
    test_y = targets[int(len(targets)*0.8):]

    model = train_mak_xgb(train_x, train_y)
    pre_y = model.predict(test_x)
