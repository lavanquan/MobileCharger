import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import re

from keras.models import *
from keras.layers import *
from energy_predict.keras_model.seq2seqLSTM import *
from argparse import ArgumentParser
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from energy_predict.operation.train import *
from energy_predict.operation.test import *
from energy_predict.keras_model.seq2seqLSTM import *
from sklearn.utils import check_array
import numpy as np
import pandas as pd

BATCH_SIZE = 5000
LEARNING_RATE = 0.001
EMB_LEN = 512
HID_LEN = 128
OPTIMIZER = 'adam'
LOSS = 'mse'
METRICS = ['mse']
EPOCH = 100
current_dir = os.path.dirname(os.path.abspath(__file__))
PATH_STATE = os.path.join(current_dir, "train_model")
MODEL_PATH = os.path.join(PATH_STATE, "seqlstmnogroundmse_256_10000_200_24_8.hdf5")

CONSTB_NAME = "_best_model.hdf5"
PATH_OUTPUT = "output"


def load_model(model_path: str = MODEL_PATH, device: str = "0"):
    model_name = os.path.basename(model_path)

    arrays = model_name.split('.')[0].split('_')
    hidden_units = int(arrays[1])
    batch_size = int(arrays[2])
    epochs = int(arrays[3])
    input_dims = int(arrays[4])
    output_dims = int(arrays[5])

    with tf.device("/gpu:" + str(device)):
        model = seq_to_lstm_no_ground(4, input_dims, output_dims, hidden_units, optimizer=OPTIMIZER, loss=LOSS)
    model.load_weights(model_path)
    model.compile(optimizer=OPTIMIZER, loss=LOSS)
    return model


def predict(model, data, avg=4, nodes=1000, time_steps=120):
    """
    predict the average value of the next charging cycle
    data is defined as (number of sensors, number of steps). Ex: (120, 1000)
    :param model: load from the load_model function
    :param data: energy left of each step. type numpy.ndarray((120, 1000))
    :param avg: average time step per blocks
    :param nodes: number of nodes
    :param time_steps: number of time steps for inputting at a time
    :return:
    """
    x_test = np.zeros([int(time_steps/avg), nodes], dtype=float)
    set_point = 0
    j = 0
    for i in range(avg, time_steps, avg):
        x_test[j] = np.mean(data[set_point:i])
        set_point = i
        j += 1
    x_test = np.transpose(x_test)
    x_test = np.expand_dims(x_test, axis=2)
    y_predict = model.predict(x_test, verbose=0, batch_size=BATCH_SIZE)
    return np.mean(y_predict, axis=1)

#
# model = load_model()
# x_test, y_test, x2_test = read_X_Y_3(PATH_TEST, 120, 40, train=False, cache=True)
# y_predict = predict(model, x_test)
# print(y_predict)