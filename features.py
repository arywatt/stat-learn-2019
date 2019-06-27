import numpy as np
from utils import *


# Here we define all the function that will create new feature from the data
# from each experiment


# Minimum of a column
def fft_stats(data):
    feature_name = 'min'
    experiment_features = []
    cols = ['tAccX', 'tAccY', 'tAccZ']
    N = len(data)
    T = 3
    f_s = 1 / T

    for col_name in cols:
        current_col = data[col_name]
        ftt_x, ftt_y = get_fft_values(current_col.values, T, N, f_s)
        experiment_features.append(('fft_y_min' + '_' + col_name, np.min(ftt_y)))
        experiment_features.append(('fft_y_max' + '_' + col_name, np.max(ftt_y)))
        experiment_features.append(('fft_y_median' + '_' + col_name, np.median(ftt_y)))
        experiment_features.append(('fft_y_q25' + '_' + col_name, np.percentile(ftt_y, 25)))
        experiment_features.append(('fft_y_q75' + '_' + col_name, np.percentile(ftt_y, 75)))
        experiment_features.append(('fft_y_std' + '_' + col_name, np.std(ftt_y)))
        experiment_features.append(('fft_y_mean' + '_' + col_name, np.mean(ftt_y)))

    return experiment_features


def psd_stats(data):
    feature_name = 'min'
    experiment_features = []
    cols = ['tAccX', 'tAccY', 'tAccZ']
    N = len(data)
    T = 3
    f_s = 1 / T

    for col_name in cols:
        current_col = data[col_name]
        psd_x, psd_y = get_fft_values(current_col.values, T, N, f_s)
        experiment_features.append(('psd_y_min' + '_' + col_name, np.min(psd_y)))
        experiment_features.append(('psd_y_max' + '_' + col_name, np.max(psd_y)))
        experiment_features.append(('psd_y_median' + '_' + col_name, np.median(psd_y)))
        experiment_features.append(('psd_y_q25' + '_' + col_name, np.percentile(psd_y, 25)))
        experiment_features.append(('psd_y_q75' + '_' + col_name, np.percentile(psd_y, 75)))
        experiment_features.append(('psd_y_std' + '_' + col_name, np.std(psd_y)))
        experiment_features.append(('psd_y_mean' + '_' + col_name, np.mean(psd_y)))

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


def f_min(data):
    feature_name = 'min'
    experiment_features = []
    cols = ['tAccX', 'tAccY', 'tAccZ']

    for col_name in cols:
        current_col = data[col_name]
        column_feature = (feature_name + '-' + col_name, np.min(current_col))
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


def signal_peaks(data, f_s=17):
    features = []
    cols = ['tAccX', 'tAccY', 'tAccZ']
    N = len(data)
    T = 1 / f_s

    transformations = {'fft': get_fft_values, 'psd': get_psd_values}
    for colname in cols:
        col = data[colname]
        colpeaks = []
        for f_name, f in transformations.items():
            x_values, y_values = f(col.values, T, N, f_s)
            x_maxes, y_maxes = peaks(x_values, y_values,1)
            #if len(y_maxes < 1):
                #print(f_name+'_'+colname)
            for i in range(len(x_maxes)):
                features.append((f_name + '_' + colname + '_peak_x_' + str(i + 1), x_maxes[i]))

            for i in range(len(x_maxes)):
                features.append((f_name + '_' + colname + '_peak_y_' + str(i + 1), y_maxes[i]))

    #print(features)

    return features


#FEATURE_FUNCTIONS = [fft_stats, psd_stats, signal_peaks]
FEATURE_FUNCTIONS = [fft_stats, psd_stats,f_xyz_magnitude,f_mean,f_std,f_min,f_max]
