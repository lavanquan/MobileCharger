import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import common.configuration as Config
from Models.LSTM_Model import lstm
from common.utils import data_preprocessing, create_xy_set, file_exist


def build_network():
    lstm_net = lstm(saving_path=Config.MODEL_SAVING_PATH,
                    input_shape=(Config.N_TIMESTEPS, Config.N_FEATURES),
                    hidden=Config.HIDDEN_UNIT,
                    drop_out=Config.DROP_OUT,
                    check_point=True)
    lstm_net.construct_n_to_one_model()
    lstm_net.plot_models()

    print lstm_net.model.summary()

    return lstm_net


def train(train_set, valid_set):
    print ('|--- Train average energy consumption predictor')

    lstm_net = build_network()

    if file_exist(lstm_net.saving_path + 'checkpoints/weights-{:02d}.hdf5'.format(Config.BEST_CHECKPOINT)):
        lstm_net.load_model_from_check_point(_from_epoch=Config.BEST_CHECKPOINT)
    else:
        train_x, train_y = create_xy_set(train_set)
        valid_x, valid_y = create_xy_set(valid_set)

        from_epoch = lstm_net.load_model_from_check_point()
        if from_epoch > 0:
            print('|--- Continue training forward model from epoch %i --- ' % from_epoch)
            training_fw_history = lstm_net.model.fit(x=train_x,
                                                     y=train_y,
                                                     batch_size=Config.BATCH_SIZE,
                                                     epochs=Config.N_EPOCH,
                                                     callbacks=lstm_net.callbacks_list,
                                                     validation_data=(valid_x, valid_y),
                                                     shuffle=True,
                                                     initial_epoch=from_epoch,
                                                     verbose=2)
        else:
            print('|--- Training new forward model.')

            training_fw_history = lstm_net.model.fit(x=train_x,
                                                     y=train_y,
                                                     batch_size=Config.BATCH_SIZE,
                                                     epochs=Config.N_EPOCH,
                                                     callbacks=lstm_net.callbacks_list,
                                                     validation_data=(valid_x, valid_y),
                                                     shuffle=True,
                                                     verbose=2)
        # Plot the training history
        if training_fw_history is not None:
            lstm_net.plot_training_history(training_fw_history)

    return lstm_net


def predict_energy_consumption(model, raw_data, scaler):
    n_series = raw_data.shape[1]

    assert raw_data.shape[0] == Config.N_TIMESTEPS * Config.AVG_STEPS

    data = data_preprocessing(raw_data)
    data_n = scaler.transform(data)

    data_n = data_n.T

    input = np.expand_dims(data_n, axis=2)

    pred = model.predict(input)

    pred = np.reshape(pred, newshape=(1, n_series))

    pred = scaler.inverse_transform(pred)

    return pred


def test(raw_test_set, scaler):
    lstm_net = build_network()
    if file_exist(lstm_net.saving_path + 'checkpoints/{weights-{:02d}.hdf5}'.format(Config.BEST_CHECKPOINT)):
        lstm_net.load_model_from_check_point(_from_epoch=Config.BEST_CHECKPOINT)
    else:
        raise RuntimeError('Model not found!')

    n_nodes = raw_test_set.shape[1]

    n_time_steps = int(raw_test_set.shape[0] / Config.AVG_STEPS)
    n_tests = n_time_steps - Config.N_TIMESTEPS

    test_set = data_preprocessing(raw_test_set)

    y_true = np.zeros((n_tests, n_nodes))
    y_pred = np.zeros((n_tests, n_nodes))

    for i in range(n_tests):
        raw_input = raw_test_set[i * Config.AVG_STEPS * Config.N_TIMESTEPS:
                                 i * Config.AVG_STEPS * Config.N_TIMESTEPS + Config.AVG_STEPS * Config.N_TIMESTEPS]
        pred = predict_energy_consumption(model=lstm_net.model, raw_data=raw_input, scaler=scaler)

        y_pred[i] = pred
        y_true[i] = test_set[i + n_tests]


if __name__ == '__main__':

    raw_data = pd.read_csv(Config.RAW_DATA_PATH + 'log_file_noCharge_random.csv')
    data = data_preprocessing(raw_data=raw_data.values)

    train_set = data[:int(data.shape[0]*0.8)]
    test_set = data[int(data.shape[0]*0.8):]

    train_x, train_y = create_xy_set(train_set)
    test_x, test_y = create_xy_set(test_set)

    train_x = train_x.squeeze(axis=2)
    # train_y.squeeze(axis=1)
    test_x = test_x.squeeze(axis=2)
    # test_y.squeeze(axis=1)

    model = XGBRegressor(n_jobs=4)
    from sklearn.model_selection import cross_validate

    cv_results = cross_validate(model, X=train_x, y=train_y, n_jobs=4,
                                scoring=("neg_mean_absolute_error", "neg_mean_squared_error"), return_train_score=True)

    print cv_results.keys()

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(cv_results['train_neg_mean_absolute_error'], label='train_mae')
    plt.plot(cv_results['test_neg_mean_absolute_error'], label='test_mae')
    plt.legend()
    plt.savefig('CV_plot.png')
