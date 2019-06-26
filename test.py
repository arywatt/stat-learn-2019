import numpy as np
import pandas as pd

from scipy.fftpack import fft
from scipy.signal import welch,find_peaks,find_peaks_cwt

import constants
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
from constants import *
from matplotlib import pyplot as plt

#df = pd.read_csv('running.csv', index_col='timestamp')
df = pd.read_csv('Stairs_up_5.csv', index_col='timestamp')
# print(df)
df = df.dropna()
df.info()



ind = df.index
x = df['AccX']
y = df['AccY']
z = df['AccZ']

xx = z



#print(N)

#n = 50


total_time = ( np.max(df.index.values) - np.min(df.index.values ) ) // 1000 + 1
print(total_time)

t_n = total_time
N = len(xx)
T = t_n / N
f_s = 1 / T

print(N,f_s,t_n,T)


#xx = x.interpolate(limit_direction='both')

#fttx = np.abs(fft(xx))

# ftty = np.abs(fftpack.fft(y, n))
#
# fttz = np.abs(fftpack.fft(z, n))

# plt.plot(ind,fttx)

#plt.plot(ind, fttx)

# plt.plot(ind,fttz,'z')
# plt.show()


# a = np.linalg.norm(df.values,axis=1)
#
# ind2 = ind  - ind[0]
#
# xx = z.interpolate(limit_direction='both')
#
# print(x.describe())
# print(xx.describe())

#xx.dropna()

# print(xx)
#
# print(fftpack.fft(xx[1:]))
#
# plt.plot(ind, xx)
# plt.show()

# for file in os.listdir(EXPERIMENTS_RAWDATA)[:1]:
#     print(file)
#     print(EXPERIMENTS_RAWDATA + file)
#     if  os.path.isfile(EXPERIMENTS_RAWDATA +file):
#
#         df = pd.read_csv(EXPERIMENTS_RAWDATA + file)
#         df = df.drop('CompassSensor',axis=1)
#         df = df.interpolate(axis=0,limit_direction='both')
#
#         df.to_csv(EXPERIMENTS_CLEANED_DATA + file, index=False)
#


def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])
    return f_values, fft_values





def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]


def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values




# plt.plot(df.index.values, x, linestyle='-', color='red')
#
#
# plt.plot(df.index.values, y, linestyle='-', color='blue')
#
# plt.plot(df.index.values, z, linestyle='-', color='green')


ff_values, psd_values = get_fft_values(x, T, N, f_s)
# plt.plot(ff_values, psd_values, linestyle='-', color='red')
#
# ff_values, psd_values = get_fft_values(y, T, N, f_s)
# plt.plot(ff_values, psd_values, linestyle='-', color='blue')
#
# ff_values, psd_values = get_fft_values(z, T, N, f_s)
# plt.plot(ff_values, psd_values, linestyle='-', color='green')
#
# print(len(ff_values),len(psd_values))

# t_values, autocorr_values = get_autocorr_values(x, T, N, f_s)
# plt.plot(t_values, autocorr_values, linestyle='-', color='red')
#
# t_values, autocorr_values = get_autocorr_values(y, T, N, f_s)
# plt.plot(t_values, autocorr_values, linestyle='-', color='blue')
#
# t_values, autocorr_values = get_autocorr_values(z, T, N, f_s)
# plt.plot(t_values, autocorr_values, linestyle='-', color='green')
#
#
# print(len(t_values), len(autocorr_values))




# ff_values, psd_values = get_psd_values(x, T, N, f_s)
# plt.plot(ff_values, psd_values, linestyle='-', color='red')
#
# ff_values, psd_values = get_psd_values(y, T, N, f_s)
# plt.plot(ff_values, psd_values, linestyle='-', color='blue')
#
# ff_values, psd_values = get_psd_values(z, T, N, f_s)
# plt.plot(ff_values, psd_values, linestyle='-', color='green')
#
# print(len(ff_values),len(psd_values))


# peaks, _ = find_peaks_cwt(xx)
#
# print(len(peaks))
# print(peaks)
# plt.plot(range(0,len(peaks)), xx.values[peaks])
# plt.show()
# 

ar = [1,2,3] + [3,2,1]
print(ar)
