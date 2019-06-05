from . import dataset
from . import constants

X_train,y_train,X_test,y_test = dataset.load_data()

# SelectLabels to use for analysis
#labels
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


