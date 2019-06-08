import dataset
import constants
from keras.models import Model,model_from_json,save_model,load_model
from my_models import SVM_Model as SVM, NN_Model as NN


#Load provided data
X_train, y_train, X_test, y_test = dataset.load_data()


# Define Labels to use for analysis
LABELS = [constants.WALKING,
          constants.LAYING,
          constants.RUNNING,
          constants.SITTING,
          constants.WALKING_DOWNSTAIRS,
          constants.WALKING_UPSTAIRS
          ]

filename = [
    {'Laying': constants.LAYING},
    {'Running_1': constants.RUNNING},
    {'Running_2': constants.RUNNING},

    {'Sitting': constants.SITTING},
    {'Stairs_down_1': constants.WALKING_DOWNSTAIRS},
    {'Stairs_down_2': constants.WALKING_DOWNSTAIRS},
    {'Stairs_up_1': constants.WALKING_UPSTAIRS},
    {'Stairs_up_3': constants.WALKING_UPSTAIRS},
    {'Stairs_up_4': constants.WALKING_UPSTAIRS},
    {'Walking': constants.WALKING},
    {'Walking_2': constants.WALKING}
]





################################     Working on provided  dataSet    ###################################################################


#### Lets try to use all the dataset to create a model


##  1- Using Support Vector Machines
#SVM.base_model(X_train, y_train, X_test, y_test)
#SVM.search_best(X_train, y_train)

# 2- Using Neural Network

#'''

model = NN.base_model()
NN.train_model(model, X_train, y_train)

trained_model = load_model(NN.MODEL_PATH)
NN.test_model(trained_model, X_test, y_test)

#'''





### Selecting data to work on
#X_train_selected, y_train_selected = dataset.extract_data(X_train, y_test, LABELS)
#X_test_selected, y_test_selected = dataset.extract_data(X_test, y_test, LABELS)


#SVM.launch_model(X_train_selected,y_train_selected,X_test_selected,y_test_selected)
#NN_Model.launch_model(X_train_selected,y_train_selected,X_test_selected,y_test_selected)




############################################################  WORKING ON PERSONAL DATASET    ########################################






