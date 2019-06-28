from sklearn.ensemble import RandomForestClassifier
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
    clf = RandomForestClassifier()

    # fit the data
    start = time.time()
    clf.fit(X_train, Y_train)
    end = time.time() - start

    # Predict the test data
    pred = clf.predict(X_test)

    # Measure accuracy
    score = accuracy_score(Y_test, pred)
    print("RANDOM FOREST :  Time:   {}  Score:  {}".format(end, score))

    save_model(clf)

    return score,end
#
def search_best_params(X, y, param_grid):
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=10)
    grid_search.fit(X, y)

    return grid_search.best_params_, grid_search.best_score_


def save_model(model):
    # create a folder to save model
    model_name = 'rdf'
    os.mkdir(OUTPUT_DATA + model_name)
    save_dir = OUTPUT_DATA + model_name + '/'
    dump(model, save_dir + model_name)
