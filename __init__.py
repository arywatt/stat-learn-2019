from constants import *
import os

for folder in [OUTPUT_DATA, PROJECT_DATASET, EXPERIMENTS_CLEANED_DATA]:
    if not os.path.exists(folder):
        os.mkdir(folder)
