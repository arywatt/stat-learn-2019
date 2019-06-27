from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from constants import *
from datetime import datetime
from joblib import dump, load
import numpy as np
import os
import time

# BEST_PARAMS


def base_model(X_train, Y_train, X_test, Y_test):

    # Instantiate classifier
    clf = KNeighborsClassifier()

    # fit the data
    start = time.time()
    clf.fit(X_train, Y_train)
    end = time.time() - start

    # Predict the test data
    pred = clf.predict(X_test)

    # Measure accuracy
    score = accuracy_score(Y_test, pred)
    print("MODEL    :  KNN    Time:   {}  Score:  {}".format(end,score))

    return score, end

#
# def search_best_params(X, y):
#     print(X.shape)
#     print(y.shape)
#     Cs = [0.001, 0.01, 0.1, 1, 10]
#     gammas = [0.001, 0.01, 0.1, 1]
#     kernels = ['rbf', 'linear', 'sigmoid', 'poly']
#     param_grid = {
#         'C': Cs,
#         'gamma': gammas,
#         'kernel': kernels,
#         'decision_function_shape': ['ovo']
#     }
#
#     grid_search = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=10)
#     grid_search.fit(X, y)
#
#     return grid_search.best_params_, grid_search.best_score_
#
#
# def best_model(X_train, Y_train, X_test, Y_test, params):
#     # Instantiate classifier
#     clf = SVC()
#
#     # Set params with best params we got with gridsearch
#     clf.set_params(**params)
#
#     # fit the data
#     clf.fit(X_train, Y_train)
#
#     # Predict the test data
#     pred = clf.predict(X_test)
#
#     # Measure accuracy
#     score = accuracy_score(Y_test, pred)
#     print("Score :", score)
#     save_model(clf)
#
#
# def save_model(model):
#     # create a folder to save model
#     model_name = 'svm' + datetime.now().strftime("%m_%d_%H:%M")
#     os.mkdir(OUTPUT_DATA + model_name)
#     save_dir = OUTPUT_DATA + model_name + '/'
#     dump(model, save_dir + model_name)
