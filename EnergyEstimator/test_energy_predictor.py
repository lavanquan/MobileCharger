import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import common.configuration as Config
from EnergyEstimator import EnergyEstimator


def cv_test(raw_data):
    # Cross-validation test

    random_state = 2
    nfold = 3
    kf = KFold(n_splits=nfold, shuffle=False, random_state=random_state)

    cv_results = pd.DataFrame(index=range(kf.get_n_splits()), columns=['mae', 'r2'])
    _mae, _r2 = [], []

    k = 0
    for train_idx, test_idx in kf.split(raw_data):
        train_data, test_data = raw_data[train_idx], raw_data[test_idx]

        if not os.path.exists(Config.MODEL_SAVING_PATH + 'lstm-cv-{}/'.format(k)):
            os.makedirs(Config.MODEL_SAVING_PATH + 'lstm-cv-{}/'.format(k))

        lstm_predictor = EnergyEstimator(predictor='lstm', n_timestep=Config.N_TIMESTEPS,
                                         data_period=Config.ENERGY_SEND_PERIOD * 2,
                                         model_saving_path=Config.MODEL_SAVING_PATH + 'lstm-cv-{}/'.format(k))

        lstm_predictor.fit(train_data)

        mae, r2 = lstm_predictor.test(test_data)
        _mae.append(mae)
        _r2.append(r2)

        print "MAE: {} ---- R2: {}".format(_mae, _r2)
        k += 1

    cv_results['mae'] = _mae
    cv_results['r2'] = _r2
    cv_results.to_csv('lstm-cv-results.csv')


if __name__ == "__main__":

    raw_data = pd.read_csv(Config.DATA_PATH + 'log_file_noCharge_random.csv')

    raw_data = raw_data.values

    deleted_rows = [x for x in range(1, raw_data.shape[0], 2)]

    raw_data = np.delete(raw_data, deleted_rows, axis=0)

    train_data = raw_data[:int(raw_data.shape[0] * 0.8)]
    test_data = raw_data[int(raw_data.shape[0] * 0.8):]

    if not os.path.exists(Config.MODEL_SAVING_PATH + 'lstm-test/'):
        os.makedirs(Config.MODEL_SAVING_PATH + 'lstm-test/')

    lstm_predictor = EnergyEstimator(predictor='lstm', n_timestep=Config.N_TIMESTEPS,
                                     data_period=Config.ENERGY_SEND_PERIOD * 2,
                                     model_saving_path=Config.MODEL_SAVING_PATH + 'lstm-test/')

    lstm_predictor.fit(train_data)
    mae, r2 = lstm_predictor.test(test_data)

    print 'MAE: {} --- R2: {}'.format(mae, r2)
