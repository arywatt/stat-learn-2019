from constants import *
from utils import *
from dataset import *
from features import *
import sys,os
import pandas as pd
import numpy as np
from joblib import dump, load


best_model_name = OUTPUT_DATA + 'rdf/rdf'
file = 'test.csv'
if(os.path.isfile(file)):
    df = pd.read_csv(file,sep=' ',names=['tAccX', 'tAccY', 'tAccZ'])
    df.info()
    features_dict = defaultdict(list)
    features = []
    # create features fro data
    features.extend(add_advanced_features(df))

    for key, value in features:
        features_dict[key].append(value)

    final_dataset = pd.DataFrame.from_dict(features_dict)
    final_dataset.info()

    loaded_model = load(open(best_model_name, 'rb'))
    predictions = loaded_model.predict(final_dataset)
    print(predictions)



else:
    print("Filename provided is not correct")
