from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise import KNNWithZScore
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate


def base_model(X_train,Y_train,X_test,Y_test):
    # Instantiate classifier
    clf = SVC(gamma='scale', decision_function_shape='ovo')

    # fit the data
    clf.fit(X_train, Y_train)

    # Predict the test data
    pred = clf.predict(X_test)

    # Measure accuracy
    score = accuracy_score(Y_test, pred)
    print("Score :", score)

    print(sum(pred))


def search_best(X_train, y_train):
    kf = KFold(n_splits=5, random_state=0)
    all_algos = [NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline,
                 CoClustering, SVD, KNNWithZScore, SVDpp]

    for algo in all_algos:
        cross_validate(algo(), X_train, y_train, measures=['RMSE'], cv=kf, verbose=True, n_jobs=-1)
