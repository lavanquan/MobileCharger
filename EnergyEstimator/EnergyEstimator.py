import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import common.configuration as Config
from common.utils import data_preprocessing, create_xy_set


class EnergyEstimator(object):
    """Implement of Energy predictor. Currently supported models are LSTM and XGBoost.

        Parameters
        ----------
        predictor: str
            The name of the based model. Default: "lstm"
        n_timestep: int
            The number of data fed to the model. Default: 20
        data_period: float
            The sampling period. Default: 5.0 second
    """

    def __init__(self, predictor='lstm', n_timestep=20, data_period=5.0, model_saving_path='./trained_models/'):
        assert predictor == 'lstm' or predictor == 'xgb'
        self.predictor = predictor

        assert n_timestep > 0
        self.n_timestep = n_timestep

        assert data_period > 0.0
        self.data_period = data_period

        assert os.path.isdir(model_saving_path)
        self.model_saving_path = model_saving_path

        self.model = None

    def fit(self, data):
        """
        Training energy predictor.
        :param data: (ndarray) The logged energy of sensor nodes. Shape = (#time-step, #nodes)
        :return:
        """
        data = data_preprocessing(raw_data=data, data_period=self.data_period)

        if self.predictor == 'xgb':
            self.model = self.__xgb_train(data)
        else:
            self.__lstm_train(data)

    def predict(self, data):
        """
        Predicting sensors' average energy consumption
        :param data: ndarray - Observed energy of sensors. Shape = (#time-step, #nodes)
        :return: Predicted value of energy consumption.
        """

        if self.model is None:
            if not os.path.isfile(self.model_saving_path + 'best_model.hdf5'):
                raise RuntimeError('Model needs to be trained first')
            else:
                self.__build_lstm_network()
                self.model.load_weights(self.model_saving_path, 'best_model.hdf5')
        if self.predictor == 'xgb':
            return self.__xgb_predict(data)
        else:
            return self.__lstm_predict(data)

    def test(self, data):
        """
        Only for model testing.
        :param data: ndarray - Observed energy of sensors. Shape = (#time-step, #nodes)
        :return:
        """

        if self.predictor == 'xgb':
            return self.__xgb_test(data)
        else:
            if not os.path.isfile(self.model_saving_path + 'best_model.hdf5'):
                raise RuntimeError('LSTM-based model needs to be trained first!')
            else:
                self.__build_lstm_network()
                print '|--- Load trained model'
                self.model.load_weights(self.model_saving_path, 'best_model.hdf5')

            return self.__lstm_test(data)

    def __xgb_predict(self, data):
        raise NotImplementedError('XGB_predict currently is not implemented!')

    def __prepare_lstm_input(self, data):
        input = np.zeros(shape=(data.shape[1], self.n_timestep, 1))

        input[:, :, 0] = data.T[:, -self.n_timestep:]
        return input

    def __lstm_predict(self, data):
        assert data.shape[0] >= self.n_timestep
        __lstm_input = self.__prepare_lstm_input(data)

        pred = self.model.predict(__lstm_input)
        pred = pred.squeeze(axis=1)
        return pred

    def __build_lstm_network(self):
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(Config.N_TIMESTEPS, 1)))
        self.model.add(Dropout(Config.DROP_OUT))
        self.model.add(Dense(32))
        self.model.add(Dense(1, name='output'))
        self.model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae'])

    def __xgb_test(self, data):
        train_x, train_y = create_xy_set(data)

        train_x = train_x.squeeze(axis=2)

        cv_results = cross_validate(self.model, X=train_x, y=train_y, n_jobs=4, cv=10,
                                    scoring=("neg_mean_absolute_error", "neg_mean_squared_error"),
                                    return_train_score=True)

        print cv_results.keys()

        plt.plot(cv_results['train_neg_mean_absolute_error'], label='train_mae')
        plt.plot(cv_results['test_neg_mean_absolute_error'], label='test_mae')
        plt.legend()
        plt.savefig('CV_plot.png')
        plt.close()

    def __train(self, train_data):
        print ('|--- Train average energy consumption predictor')

        self.__build_lstm_network()

        checkpoints = ModelCheckpoint(
            self.model_saving_path + "best_model.hdf5",
            monitor='val_loss',
            verbose=1,
            save_best_only=True)

        _training_fw_history = self.model.fit(x=train_data[0],
                                              y=train_data[1],
                                              batch_size=Config.BATCH_SIZE,
                                              epochs=Config.N_EPOCH,
                                              callbacks=[checkpoints],
                                              validation_data=(train_data[2], train_data[3]),
                                              shuffle=True,
                                              verbose=2)
        # Plot the training history
        if _training_fw_history is not None:
            self.__plot_training_history(_training_fw_history)

    def __plot_training_history(self, model_history):
        plt.plot(model_history.history['loss'], label='mse')
        plt.plot(model_history.history['val_loss'], label='val_mse')
        plt.savefig(self.model_saving_path + '[MSE]loss-val_loss.png')
        plt.legend()
        plt.close()

        plt.plot(model_history.history['val_loss'], label='val_mae')
        plt.legend()
        plt.savefig(self.model_saving_path + '[MSE]val_loss.png')
        plt.close()

        loss = np.array(model_history.history['loss'])
        val_loss = np.array(model_history.history['val_loss'])
        dump_model_history = pd.DataFrame(index=range(loss.size),
                                          columns=['epoch', 'loss', 'val_loss'])

        dump_model_history['epoch'] = range(loss.size)
        dump_model_history['loss'] = loss
        dump_model_history['val_loss'] = val_loss

        dump_model_history.to_csv(self.model_saving_path + 'training_history.csv', index=False)

    def __lstm_train(self, train_set):
        data_x, data_y = create_xy_set(train_set)

        X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True)

        self.__train((X_train, y_train, X_test, y_test))

    def __lstm_test(self, test_set):
        print '|--- Test lstm:'
        X_test, y_test = create_xy_set(test_set)

        pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_pred=pred, y_true=y_test)
        r2 = r2_score(y_true=y_test, y_pred=pred)

        return mae, r2

    def __xgb_train(self, train_data):
        train_x, train_y = create_xy_set(train_data)

        train_x = train_x.squeeze(axis=2)

        model = XGBRegressor(n_jobs=4)

        model.fit(train_x, train_y)

        return model
