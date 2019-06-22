import numpy as np
import pandas as pd

from scipy.fftpack import fft

import constants
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
from constants import *
from matplotlib import pyplot as plt

df = pd.read_csv('running.csv', index_col='timestamp')
# print(df)
df.info()

ind = df.index
x = df['AccX']
y = df['AccY']
z = df['AccZ']

n = 50


xx = x.interpolate(limit_direction='both')

fttx = np.abs(fft(xx))

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


t_n = 10
N = 1000
T = t_n / N
f_s = 1 / T

f_values, fft_values = get_fft_values(fttx, T, N, f_s)
plt.plot(f_values, fft_values, linestyle='-', color='blue')
plt.show()
