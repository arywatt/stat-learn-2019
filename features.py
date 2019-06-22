import numpy as np


# Here we define all the function that will create new feature from the data
# from each experiment


# Minimum of a column
def f_min(data):
    feature_name = 'min'
    experiment_features = []
    cols = ['tAccX', 'tAccY', 'tAccZ']

    for col_name in cols:
        current_col = data[col_name]
        column_feature = (feature_name + '-' + col_name, np.min(current_col))
        experiment_features.append(column_feature)

    return experiment_features


# Maximum of a column
def f_max(data):
    feature_name = 'max'
    experiment_features = []
    cols = ['tAccX', 'tAccY', 'tAccZ']

    for col_name in cols:
        current_col = data[col_name]
        column_feature = (feature_name + '-' + col_name, np.max(current_col))
        experiment_features.append(column_feature)

    return experiment_features


# Mean of a column
def f_mean(data):
    feature_name = 'mean'
    experiment_features = []
    cols = ['tAccX', 'tAccY', 'tAccZ']

    for col_name in cols:
        current_col = data[col_name]
        column_feature = (feature_name + '-' + col_name, np.mean(current_col))
        experiment_features.append(column_feature)

    return experiment_features


# std of a column
def f_std(data):
    feature_name = 'std'
    experiment_features = []
    cols = ['tAccX', 'tAccY', 'tAccZ']

    for col_name in cols:
        current_col = data[col_name]
        column_feature = (feature_name + '-' + col_name, np.std(current_col))
        experiment_features.append(column_feature)

    return experiment_features


def f_xyz_magnitude(data):
    feature_name = 'xyz-magnitude'
    experiment_features = []
    mg = np.linalg.norm(data.values,axis=1)





FEATURE_FUNCTIONS = [f_min, f_max, f_mean, f_std]
