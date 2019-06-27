import dataset
import constants
from keras.models import Model,model_from_json,save_model,load_model
from models import svm #neural_network as NN
from models import random_forest
from models import knn
from models import decision_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np



#Load data
# We can load hapt_data

# uncomment line below  to use it
#X_train, y_train, X_test, y_test = dataset.load_hapt_data()




# or load our personal data

# first we process the data
#dataset.process_all_records()
#dataset.create_data()

X, y = dataset.load_data()
X_train, X_test, y_train, y_test = dataset.split_data(X, y, level=0.33, seed=33000)


# Define Labels to use for analysis
LABELS = [constants.WALKING,
          constants.LAYING,
          constants.RUNNING,
          constants.SITTING,
          constants.WALKING_DOWNSTAIRS,
          constants.WALKING_UPSTAIRS
          ]


##  1- Using Support Vector Machines

## first we create a base model usin SVM

models = [svm.base_model, random_forest.base_model,knn.base_model,decision_tree.base_model]
for model in models:
    model(X_train,y_train,X_test,y_test)

# then we perform a model tuning to find the best params for
# our base model
# takes a lot of time

#best_params, best_score = SVM.search_best_params(X_train, y_train)


# best param found
# best_params = {
#     'gamma': 0.1,
#     'decision_function_shape': 'ovo',
#     'C': 0.001,
#     'kernel': 'rbf'
# }

## We test the best model obtained
#print('running svm best model ')
#SVM.best_model(X_train, y_train, X_test, y_test, best_params)



# 2- Using Neural Network
# print('Runnin NN')
# # We first encode the label
#
# encoder = OneHotEncoder(sparse=False,categories='auto')
# ar = np.array(y).reshape(len(y), 1)
# encoder.fit(ar)
#
#
# y_train = encoder.transform(y_train.reshape(len(y_train), 1))
# y_test = encoder.transform(y_test.reshape(len(y_test), 1))
#
# print('encoding done')
#
# units = y_train.shape[1]
#
# model = NN.base_model(units)
# trained_model = NN.train_model(model, X_train, y_train)
# NN.test_model(trained_model, X_test, y_test)
#





