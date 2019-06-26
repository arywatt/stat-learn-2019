import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.signal import welch , find_peaks,find_peaks_cwt



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


# return peaks of a signal
# x : array representing signal
# n : number of peaks to return

def peaks(x, n=3):
    peaks = []
    indexes, _ = find_peaks(x)
    return indexes
