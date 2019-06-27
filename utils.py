import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.signal import welch , find_peaks,find_peaks_cwt
from scipy import signal


def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])
    return f_values, fft_values


def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values


def autocorr(x, mode='full'):
    result = np.correlate(x, x, mode=mode)
    return result[len(result) // 2:]


def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values



def peaks(x,y,n=3):
    indexes, properties = find_peaks(y)
    x_peaks = x[indexes]
    y_peaks = y[indexes]


    ind_max = n_max_indexes(y_peaks, n)
    print(len(ind_max))
    x_peaks_max = x_peaks[ind_max]
    y_peaks_max = y_peaks[ind_max]
    return x_peaks_max, y_peaks_max


def smooth_signal(x,order=3,fs=18):
    B, A = signal.butter(order, fs, output='ba', fs=fs)
    sm_data = signal.filtfilt(B, A, x.values)
    return sm_data


def n_max_indexes(arr, n):
    if len(arr) < n:
        n= len(arr)

    arr = np.array(arr)
    arr = arr.argsort()[-n:][::-1]
    return arr





# xx =[0.2699281 , 0.08034457, 0.09722889, 0.09450219 ,0.08091265, 0.0713773,0.06576681]
# print(n_max_indexes(xx,4))
#
# 1
# [0.2699281  0.08034457 0.09722889 0.09450219 0.08091265 0.0713773
#  0.06576681]
# 1
# [5.25976117e+00 5.59541314e-02 6.88101103e-03 2.10073592e-03
#  5.21228255e-04]
# 1
# [7.05316051 0.86129289 0.81499708 0.43252999 0.39468838 0.35680802
#  0.3159287 ]
# 1
# [3.00561853e+01 3.56268124e-01 5.29387452e-02 4.23766932e-02
#  1.39017203e-03 7.24542408e-03 2.45900652e-03]
# 1
# [1.76343561 1.29784118 0.68577224 0.69279991 0.50001448 0.30500105
#  0.27855459 0.21860216 0.14300843 0.12575188]
# 1
# [2.13488339 0.84982979 0.35427394 0.16611246 0.01696009 0.00491151]
# 1
# [0.73995259 0.78679329 0.2558439  0.13271267 0.12477101 0.13661038
#  0.08221931]
# 1
# [1.36805669e+00 2.97469282e-01 1.22843719e-01 8.04391575e-03
#  7.47953769e-03 2.44288500e-03 7.25365742e-04 2.69681996e-03]
# 1
# [1.25872201 0.6027085  0.46291448 0.34460416 0.03196348 0.07533869
#  0.04255486]