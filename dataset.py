import numpy as np 
import pandas as pd
import constants

_DATAFOLDER = 'HAPT Data Set/'


## load data from files
def load_data():

    #Load train data
    X_train = pd.read_csv(_DATAFOLDER+'Train/X_train.txt',sep=' ')
    y_train = pd.read_csv(_DATAFOLDER+'Train/y_train.txt')

    #Load test data
    X_test = pd.read_csv(_DATAFOLDER+'Test/X_test.txt',sep=' ')
    y_test = pd.read_csv(_DATAFOLDER+'Test/y_test.txt')
    return X_train.values, y_train.values, X_test.values, y_test.values






## Extract only record corresponding to  labels to study
## Takes in input data to use
# and labels choosen as an arry or integers
# returns X_selected , Y_selected

def extract_data(X,Y,labels):
    sh = X.shape
    #X = np.array(X).reshape(X.shape)
    #Y = np.array(Y).reshape(Y.shape)
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



def clean_record(filename, label):
    ff = open('data/' + filename + '.csv', 'r') # we open and read the file
    buffer = ""
    for line in ff.readlines()[1:]:

        tab = line.strip().split(',')
        if '' in tab:  ## We avoid lines with missing values
            continue
        else:
            # line.append(','+label)
            buffer = buffer + ','.join(line.split(','))
    ff2 = open('data2/' + filename + '.csv', 'w+')
    ff2.write(buffer)
    ff2.close
    ff.close


def process_record(filename, label,EXP_ID, BATCH_SIZE):
    ff = open('data2/' + filename + '.csv', 'r')    # we open and read the file
    lines = ff.readlines()[1:]
    buffer = ""
    count = 0
    for line in lines:
        count = count + 1
        if count % BATCH_SIZE == 0:
            EXP_ID += 1
        line = line.strip() + ',' + str(EXP_ID) + ',' + str(label) + '\n'
        buffer = buffer + line

    ff2 = open('data2/dataset.csv', 'w+')
    ff2.write(buffer)
    ff2.close
    ff.close




   





