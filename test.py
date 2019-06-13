import numpy as np
import pandas as pd
from scipy import stats
import constants
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


from datetime import datetime

now = datetime.now().strftime("%m_%d_%H:%M")

print(now)