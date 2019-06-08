from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from contextlib import redirect_stdout
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback, LearningRateScheduler
from keras.layers import Convolution2D, Conv2D, MaxPooling2D,LeakyReLU
from keras.layers import Dropout, Activation, Flatten,Input, Dense
from keras.models import Model,model_from_json,save_model,load_model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# fix random seed for reproducibility
seed = 227
np.random.seed(seed)

## Constants

MODEL_BEST_WEIGHTS_PATH ='./data_out/nn_model/model_best.h5'
MODEL_SUMMARY_PATH = './data_out/nn_model/modelsummary.txt'
MODEL_LOG_PATH = './data_out/nn_model/training.log'
MODEL_JSON_PATH ='./data_out/nn_model/model.json'
MODEL_WEIGHTS_PATH ='./data_out/nn_model/model_weights.h5'
MODEL_PATH ='./data_out/nn_model/model.h5'



def decay(epoch):
   initial_lrate = 0.01
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   #lrate = max(lrate,0.00001)
   return lrate


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []



    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(decay(epoch))
        print(' lr:', decay(epoch))


### Call back list
# Check on model and save best weights
checkpointer = ModelCheckpoint(monitor='loss',
                               filepath=MODEL_BEST_WEIGHTS_PATH,
                               verbose=1, save_best_only=True,
                               save_weights_only=True, )
history = LossHistory()

# logger in csv file
csv_logger = CSVLogger(MODEL_LOG_PATH)

callbacks = [csv_logger, checkpointer, history]

def base_model():
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(units=13, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  callbacks=callbacks
                  )

    return model



def custom_model():


    return model


def train_model(model, X_train, Y_train):
    # Convert labels to categorical one-hot encoding
    one_hot_train_labels = np_utils.to_categorical(Y_train)

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(X_train,
              one_hot_train_labels,
              epochs=20,
              batch_size=32,
              #validation_data=(X_test, one_hot_test_labels)
              )

    model.summary()
    loss, accuracy = model.evaluate(X_train, one_hot_train_labels, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))


    model.save(MODEL_PATH)
    model.save_weights(MODEL_WEIGHTS_PATH)
    open(MODEL_JSON_PATH, 'w+').write(model.to_json())

    #plot_history(history)

def test_model(model, X_test, Y_test):
    # Convert labels to categorical one-hot encoding
    one_hot_test_labels = np_utils.to_categorical(Y_test)

    ## Load saved model

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  #callbacks=callbacks
                  )
    loss, accuracy = model.evaluate(X_test, one_hot_test_labels, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))



def run(X_train, Y_train,X_test,Y_test, model):
    train_model(X_train, Y_train, model)
    test_model(X_test,Y_train)



def plot_history(history):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()