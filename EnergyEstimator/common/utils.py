import os.path as osp

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import configuration as Config


def data_splitting(data, split_ratio):
    data_size = data.shape[0]

    train_size = int(data_size * split_ratio[0])
    valid_size = int(data_size * split_ratio[1])

    return data[:train_size], data[train_size:train_size + valid_size], data[-(train_size + valid_size):]


def data_normalization(train_set):
    scaler = MinMaxScaler(copy=True)
    scaler.fit(train_set)

    return scaler


def data_preprocessing(raw_data):
    data_size = int(raw_data.shape[0] / Config.AVG_STEPS)

    data = np.zeros(shape=(data_size, raw_data.shape[1]))

    for i in range(data_size):
        avg_energy_consuming = (raw_data[i * Config.AVG_STEPS + Config.AVG_STEPS] - raw_data[
            i * Config.AVG_STEPS]) / Config.ENERGY_SEND_PERIOD
        data[i] = avg_energy_consuming

    np.save(Config.DATA_PATH + 'data', data)

    return data


def create_xy_set(data):
    data_x = np.zeros(
        shape=(data.shape[1] * (data.shape[0] - Config.N_TIMESTEPS), Config.N_TIMESTEPS, Config.N_FEATURES))
    data_y = np.zeros(shape=(data.shape[1] * (data.shape[0] - Config.N_TIMESTEPS), 1))

    i = 0
    for node_id in range(data.shape[1]):
        for ts in range(data.shape[0] - Config.N_TIMESTEPS):
            _x = data[ts:ts + Config.N_TIMESTEPS, node_id]

            data_x[i] = np.expand_dims(_x, axis=1)
            data_y[i] = data[ts + Config.N_TIMESTEPS, node_id]
            i += 1

    return data_x, data_y


def file_exist(file_name):
    return osp.isfile(file_name)
