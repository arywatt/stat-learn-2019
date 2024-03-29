import numpy as np 
import pandas as pd
from constants import *
from sklearn.model_selection import train_test_split
from features import FEATURE_FUNCTIONS
from collections import defaultdict
import os

files = [
    {'Laying':  LAYING},
    {'Running_1':  RUNNING},
    {'Running_2':  RUNNING},
    {'Laying1': LAYING },
    {'Running':RUNNING },
    {'Sitting_1': SITTING},
    {'StairsDown': WALKING_DOWNSTAIRS},
    {'Stairsup':WALKING_UPSTAIRS },
    {'Sitting':  SITTING},
    {'Stairs_down_1':  WALKING_DOWNSTAIRS},
    {'Stairs_down_2':  WALKING_DOWNSTAIRS},
    {'Stairs_up_1':  WALKING_UPSTAIRS},
    {'Stairs_up_3':  WALKING_UPSTAIRS},
    {'Stairs_up_4':  WALKING_UPSTAIRS},
    {'Walking':  WALKING},
    {'Walking_1': WALKING},
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
def create_data():
    # Load train data
    data = pd.read_csv(BASIC_DATASET, sep=',', names=['timestamp', 'tAccX', 'tAccY', 'tAccZ', 'EXP_ID', 'Label'])
    exp_total_number  = data['EXP_ID'].max(0)

    features_dict = defaultdict(list)

    for ID in range(exp_total_number+1):
        condition = data['EXP_ID'] == ID

        features = []

        # retrieve experiment data
        experiment_data = data[condition].loc[:, ['tAccX', 'tAccY', 'tAccZ', 'Label']]

        # create features fro data
        features.extend(create_features(experiment_data))


        for key,value in features:
            features_dict[key].append(value)

    final_dataset = pd.DataFrame.from_dict(features_dict)
    final_dataset.to_csv('{}'.format(FEATURED_DATASET), index=False)


    # data = np.array(features)
    # X = data[:, :-1]
    # y = data[:, -1]
    #
    # return X, y


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

    # then we also add all the advanced features created
    advanced_features = add_advanced_features(data)
    if len(advanced_features)!= 0:
        experiment_features.extend(advanced_features)

    # At the last position we add the corresponding experiment type
    label_value = int(np.mean(data['Label']))
    experiment_features.append(('Label',label_value))
    return experiment_features


# this function goal is to add advanced features to the dataset
# one can define a function to create a particular feature
# and add it to the function list 

def add_advanced_features(data):
    advanced_features = []


    for function in FEATURE_FUNCTIONS:
        advanced_features.extend(function(data))

    return advanced_features

#
# # Extract only record corresponding to  labels to study
# # Takes in input data to use
# # and labels choosen as an arry or integers
# # returns X_selected , Y_selected
#
# def extract_data(X, Y, labels):
#     sh = X.shape
#     # From Y extract only the data corresponding to labels chosen
#     # First get each each label's line index
#     indexes = [x for  x in range(len(Y)) if Y[x][0] in set(labels)]
#
#     # Now extract line from X corresponding to indexes found
#     X_final = []
#     Y_final = []
#
#     for x in indexes:
#         X_final.append(X[x])
#         Y_final.append(Y[x])
#
#     return np.array(X_final).reshape(len(indexes),sh[1]),np.array(Y_final).reshape(len(indexes),1)

# The raw data is processed to remove empty values
# fileanme : name of the file 
# label : experiment type e.g Laying, running,etc . As defined in constants.py modules


def clean_record(filename, label):
    df = pd.read_csv(EXPERIMENTS_RAWDATA + filename+ '.csv')
    df = df.drop('CompassSensor', axis=1)
    df = df.interpolate(axis=0,limit_direction='both')  # we interpolate line with missing values
    #df = df.dropna()
    df.to_csv(EXPERIMENTS_CLEANED_DATA + filename+ '.csv', index=False)


# the cleaned data is processed to get the basic dataset
# fileanme : name of the file 
# label : experiment type e.g Laying, running,etc . As defined in constants.py modules
# EXP_ID : ID of each experiment 
# numbers_of_records : Number  of records in each experiments


def process_record(filename, label, exp_id, numbers_of_records):
    ff = open(EXPERIMENTS_CLEANED_DATA + filename + '.csv', 'r')  # we open and read the file
    lines = ff.readlines()[1:]
    buffer = ""
    count = 0
    for line in lines:
        count = count + 1
        if count % numbers_of_records == 0:
            exp_id += 1
        line = line.strip() + ',' + str(exp_id) + ',' + str(label) + ' \n'
        buffer = buffer + line

    ff2 = open(BASIC_DATASET, 'a')
    ff2.write(buffer)
    ff2.close
    # ff.close

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

    if os.path.isfile(FEATURED_DATASET):
        os.remove(FEATURED_DATASET)

    for elmt in filelist:
        for filename, label in elmt.items():
            clean_record(filename, label)


    for elmt in filelist:
        for filename, label in elmt.items():
            ID = process_record(filename, label, ID, batch)






def load_data():
    final_dataset = pd.read_csv(FEATURED_DATASET)
    return final_dataset.values[:, :-1], final_dataset.values[:, -1]

#process_all_records()  uncomment to process all the data files
create_data()  #uncomment to recreate the dataset