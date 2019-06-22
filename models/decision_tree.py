from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from constants import *
from datetime import datetime
from joblib import dump, load
import numpy as np
import os


# BEST_PARAMS


def base_model(X_train, Y_train, X_test, Y_test):
    # Instantiate classifier
    clf = DecisionTreeClassifier()

    # fit the data
    clf.fit(X_train, Y_train)

    # Predict the test data
    pred = clf.predict(X_test)

    # Measure accuracy
    score = accuracy_score(Y_test, pred)
    print("Score :", score)
