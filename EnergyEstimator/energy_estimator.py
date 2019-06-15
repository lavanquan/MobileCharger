import os

import numpy as np

import common.configuration as Config


def data_preprocessing(nsteps, n_timesteps, data):
    pass


def load_data(data_name):
    data = np.genfromtxt(Config.DATA_PATH + data_name + '.csv')
    return data


def data_normalization(data):


def train():
    pass


def test():
    pass


if __name__ == '__main__':

    if os.path.isfile(Config.DATA_PATH + 'data.npy'):
        data = load_data('data.npy')
    else:
        data = data_preprocessing()

    train_set, valid_set, test_set =

    data, scaler = data_normalization(data)

    train()
