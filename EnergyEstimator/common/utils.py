import os.path as osp

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import configuration as Config


def data_normalization(data):
    scaler = MinMaxScaler(copy=True)
    scaler.fit(data)

    return scaler


def data_preprocessing(raw_data, data_period=100.0):
    data = np.zeros(shape=(raw_data.shape[0] - 1, raw_data.shape[1]))

    for i in range(raw_data.shape[0] - 1):
        avg_energy_consuming = (raw_data[i] - raw_data[i + 1]) / data_period
        data[i] = avg_energy_consuming

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
