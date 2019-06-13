import numpy as np 
import pandas as pd
from constants import *
from sklearn.model_selection import train_test_split
import os

files = [
    {'Laying':  LAYING},
    {'Running_1':  RUNNING},
    {'Running_2':  RUNNING},

    {'Sitting':  SITTING},
    {'Stairs_down_1':  WALKING_DOWNSTAIRS},
    {'Stairs_down_2':  WALKING_DOWNSTAIRS},
    {'Stairs_up_1':  WALKING_UPSTAIRS},
    {'Stairs_up_3':  WALKING_UPSTAIRS},
    {'Stairs_up_4':  WALKING_UPSTAIRS},
    {'Walking':  WALKING},
    {'Walking_2':  WALKING}
]


# load data from files
def load_hapt_data():

    # Load train data
    X_train = pd.read_csv(HAPT_DATASET_FOLDER+'Train/X_train.txt',sep=' ',header=None)
    y_train = pd.read_csv(HAPT_DATASET_FOLDER+'Train/y_train.txt',sep=' ',header=None)

    X_train.info()
    # Load test data
    X_test = pd.read_csv(HAPT_DATASET_FOLDER+'Test/X_test.txt',sep=' ',header=None)
    y_test = pd.read_csv(HAPT_DATASET_FOLDER+'Test/y_test.txt',sep=' ',header=None)
    return X_train.values, y_train.values, X_test.values, y_test.values


# This function loads the raw dataset
# and process it
# to obtain the final dataset to work on 
def load_data():
    # Load train data
    data = pd.read_csv(BASIC_DATASET, sep=',', names=['timestamp', 'tAccX', 'tAccY', 'tAccZ', 'EXP_ID', 'Label'])
    exp_total_number  = data['EXP_ID'].max(0)

    features = []
    for ID in range(exp_total_number+1):
        condition = data['EXP_ID'] == ID

        # retrieve experiment data
        experiment_data = data[condition].loc[:, ['tAccX', 'tAccY', 'tAccZ', 'Label']]

        # create features fro data
        features.append(create_features(experiment_data))
    data = np.array(features)
    X = data[:, :-1]
    y = data[:, -1]

    return X, y


# this function split data in train data and test data 
# level is the proportion on the test data 
# level must be between 0 and 1
def split_data(X, y, level=0.20, seed=42):
    return train_test_split(X, y, test_size=level, random_state=seed)
    

# This function take the data to work on 
# and ceate some features from it
# features are commputed for each experiment
# like min, max, mean, std, for each variable in an experiment
# then return a row will all the features created
# and the corresponding experiment Label

def create_features(data):

    # to store all our features
    experiment_features = []
    cols = ['tAccX', 'tAccY', 'tAccZ']

    # first we add basics features like min , max , mean,.....
    for col_name in cols:
        current_col = data[col_name]
        column_features = [np.min(current_col), np.max(current_col), np.std(current_col), np.mean(current_col)]
        experiment_features.extend(column_features)

    # then we also add all the advanced features created
    advanced_features = add_advanced_features(data)
    if len(advanced_features)!= 0:
        experiment_features.extend(advanced_features)

    # At the last position we add the corresponding experiment type
    experiment_features.append(int(np.mean(data['Label'])))
    return experiment_features


# this function goal is to add advanced features to the dataset
# one can define a function to create a particular feature
# and add it to the function list 

def add_advanced_features(data):
    advanced_features = []
    features_creations_functions = []
    for function in features_creations_functions:
        add_advanced_features.extend(function(data))

    return advanced_features


# Extract only record corresponding to  labels to study
# Takes in input data to use
# and labels choosen as an arry or integers
# returns X_selected , Y_selected

def extract_data(X, Y, labels):
    sh = X.shape
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

# The raw data is processed to remove empty values
# fileanme : name of the file 
# label : experiment type e.g Laying, running,etc . As defined in constants.py modules


def clean_record(filename, label):
    ff = open(EXPERIMENTS_RAWDATA + filename + '.csv', 'r') # we open and read the file
    buffer = ""
    for line in ff.readlines()[1:]:

        tab = line.strip().split(',')[:-1]
        if '' in tab:  ## We avoid lines with missing values
            continue
        else:
            # line.append(','+label)
            buffer = buffer + ','.join(tab) + '\n'
    ff2 = open(EXPERIMENTS_CLEANED_DATA + filename + '.csv', 'w+')
    ff2.write(buffer)
    ff2.close
    ff.close


# the cleaned data is processed to get the basic dataset 
# fileanme : name of the file 
# label : experiment type e.g Laying, running,etc . As defined in constants.py modules
# EXP_ID : ID of each experiment 
# numbers_of_records : Number  of records in each experiments


def process_record(filename, label,  exp_id, numbers_of_records):
    ff = open(EXPERIMENTS_CLEANED_DATA + filename + '.csv', 'r')    # we open and read the file
    lines = ff.readlines()[1:]
    buffer = ""
    count = 0
    for line in lines:
        count = count + 1
        if count % numbers_of_records == 0:
             exp_id += 1
        line = line.strip() + ',' + str( exp_id) + ',' + str(label) + ' \n'
        buffer = buffer + line

    ff2 = open(BASIC_DATASET, 'a')
    #ff2.write('timestamp,tAccX,tAccY,tAccZ,EXP_ID,Label \n')
    ff2.write(buffer)
    ff2.close
    ff.close
    return exp_id

# Once we put all the experiment files in the correct folder 
# we update the FILES variable to add all the file name and experiment type
# then we call this method
# it processes all the files, cleaning them, creating the basic dataset 
# then creating the final dataset with new features
# ID =  EXPERIMENT ID
# filelist : list of files to process 
# bacth : number of record to includ in each experiment 


def process_all_records(batch=50, ID= 0, filelist=files):

    # Delete old dataset if exist
    if os.path.isfile(BASIC_DATASET):
        os.remove(BASIC_DATASET)

    for elmt in filelist:
        for filename, label in elmt.items():
            clean_record(filename, label)


    for elmt in filelist:
        for filename, label in elmt.items():
            ID = process_record(filename, label, ID, 50)


