from keras.models import Sequential
from contextlib import redirect_stdout
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback, LearningRateScheduler
from keras.layers import  MaxPooling2D, LeakyReLU,Dropout, Activation, Flatten, Input, Dense

from keras.models import Model, model_from_json
from keras.utils import np_utils
from sklearn.externals import joblib
import math
from matplotlib import pyplot as plt
from constants import *
import os
from datetime import datetime


# plt.style.use('ggplot')


# fix random seed for reproducibility
seed = 227
np.random.seed(seed)



def decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    # lrate = max(lrate,0.00001)
    return lrate


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(decay(epoch))
        print(' lr:', decay(epoch))


# return callbacks  Call back list
# Check on model and save best weights
def train_callbacks():
    checkpointer = ModelCheckpoint(monitor='loss',
                                   filepath=MODEL_BEST_WEIGHTS_PATH,
                                   verbose=1, save_best_only=True,
                                   save_weights_only=True, )

    # checkpoint_name = OUTPUT_DATA + 'weights-best.hdf5'
    # checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    # callbacks_list = [checkpoint]
    #
    history = LossHistory()

    # logger in csv file
    csv_logger = CSVLogger(MODEL_LOG_PATH)

    callbacks = [csv_logger, checkpointer, history]
    return callbacks


def base_model(units):
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(units=units, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  callbacks=train_callbacks()
                  )

    return model

def advanced_model(input_dim):
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal', input_dim=input_dim, activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network :
    NN_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  callbacks=train_callbacks())
    NN_model.summary()

    return NN_model


def train_model(model, X_train, Y_train):
    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit(X_train,
                        Y_train,
                        epochs=200,
                        batch_size=32,
                        # validation_data=(X_test, one_hot_test_labels)
                        )

    model.summary()
    loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))

    # Save the model
    save_model(model)
    return model

def save_model(model):
    # create a folder to save model
    model_name = 'nn'+ datetime.now().strftime("%m_%d_%H:%M")
    os.mkdir(OUTPUT_DATA + model_name)
    save_dir = OUTPUT_DATA + model_name + '/'

    model.save(save_dir + MODEL_PATH)
    model.save_weights(save_dir + MODEL_WEIGHTS_PATH)
    open(save_dir + MODEL_JSON_PATH, 'w+').write(model.to_json())
    with open(save_dir + MODEL_SUMMARY_PATH, 'w') as f:
        with redirect_stdout(f):
            model.summary()


def test_model(model, X_test, Y_test):
    # Convert labels to categorical one-hot encoding
    #one_hot_test_labels = np_utils.to_categorical(Y_test)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  # callbacks=callbacks
                  )
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


def run(X_train, Y_train, X_test, Y_test,model):
    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit(X_train,
                        Y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_test, Y_test),
                        verbose=False
                        )

    model.summary()
    loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    # Save the model
    save_model(model)


def plot_history(history):
    acc = history.history['acc']
    print(acc)
    # val_acc = history.history['val_acc']
    loss = history.history['loss']
    print(loss)
    # val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    # plt.plot(x, val_acc, 'r', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    # plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    # plt.plot(x, val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    plt.legend()
