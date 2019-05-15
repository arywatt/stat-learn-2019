import numpy as np 
import pandas as pd 

_DATAFOLDER = 'HAPT Data Set/'


## load data from files
def load_data():

    #Load train data
    X_train = pd.read_csv(_DATAFOLDER+'Train/X_train.txt',sep=' ')
    y_train = pd.read_csv(_DATAFOLDER+'Train/y_train.txt')

    #Load test data
    X_test = pd.read_csv(_DATAFOLDER+'Test/X_test.txt',sep=' ')
    y_test = pd.read_csv(_DATAFOLDER+'Test/y_test.txt')
    return X_train,y_train,X_test,y_test






def extract_data(X,Y,labels):
    sh = X.shape
    X = np.array(X).reshape(X.shape)
    Y = np.array(Y).reshape(Y.shape)
    # From Y extract only the data corresponding to labels chosen 
    # First get each each label's line index
    indexes = [x for  x in range(len(Y)) if Y[x][0] in set(labels)]

    # Now extract line from X corresponding to indexes found
    X_final = []
    Y_final = []

    for x in indexes:
        X_final.append(X[x])
        Y_final.append(Y[x])

    return np.array(X_final).reshape(len(indexes),sh[1]),np.array(Y_final).reshape(len(indexes),1)


