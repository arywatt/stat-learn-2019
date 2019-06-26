import numpy as np
from utils  import *

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
    feature_name = 'xyz_mean_magnitude'
    experiment_features = []
    mg = np.mean(np.linalg.norm(data.values, axis=1))
    experiment_features.append((feature_name, mg))
    return experiment_features



def signal_peaks(data ,t_n=2):
    features = []
    cols = ['tAccX', 'tAccY', 'tAccZ']
    N = len(data)
    T = t_n / N
    f_s = 1 / T
    transformations = { 'fft': get_fft_values,'psd': get_psd_values}
    for colname in cols:
        col = data[colname]
        colpeaks = []
        for f_name, f in transformations.items():
            x,y =  f(col.values, T, N, f_s)

            print(colname+'_'+f_name, y)
            colpeaks = y[peaks(y)]
            print(colpeaks)
            print(len(colpeaks))
            features.append((colname+'_'+f_name+'_max_peak',np.max(colpeaks)))
            features.append((colname + '_'+f_name+'_min_peak', np.min(colpeaks)))
            features.append((colname + '_'+f_name+'_median_peak', np.median(colpeaks)))
            features.append((colname + '_'+f_name+'_std_peak', np.std(colpeaks)))

    return  features


FEATURE_FUNCTIONS = [f_min, f_max, f_mean, f_std,f_xyz_magnitude]
