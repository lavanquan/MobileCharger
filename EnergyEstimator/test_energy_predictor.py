import pandas as pd
from sklearn.model_selection import KFold

import common.configuration as Config
from EnergyEstimator import EnergyEstimator

raw_data = pd.read_csv(Config.DATA_PATH + 'log_file_noCharge_random.csv')

raw_data = raw_data.values

# Cross-validation test
lstm_predictor = EnergyEstimator(predictor='lstm', n_timestep=Config.N_TIMESTEPS, data_period=Config.ENERGY_SEND_PERIOD)

random_state = 2
nfold = 3
kf = KFold(n_splits=nfold, shuffle=False, random_state=random_state)

cv_results = pd.DataFrame(index=range(kf.get_n_splits()), columns=['mae', 'r2'])
for train_idx, test_idx in kf.split(raw_data):
    train_data, test_data = raw_data[train_idx], raw_data[test_idx]

    lstm_predictor.fit(train_data)

    mae, r2 = lstm_predictor.test(test_data)
    cv_results['mae'].append(mae)
    cv_results['r2'].append(r2)

cv_results.to_csv('lstm-cv-results.csv')
