import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import common.configuration as Config
from Models.LSTM_Model import lstm
from common.utils import data_preprocessing, create_xy_set
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import os

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

        self.__model = None

    def fit(self, data):
        """
        Training energy predictor.
        :param data: (ndarray) The logged energy of sensor nodes. Shape = (#time-step, #nodes)
        :return:
        """
        data = data_preprocessing(raw_data=data, data_period=self.data_period)

        if self.predictor == 'xgb':
            self.__model = self.__xgb_train(data)
        else:
            self.__model = self.__lstm_train(data)

    def predict(self, data):
        """
        Predicting sensors' average energy consumption
        :param data: ndarray - Observed energy of sensors. Shape = (#time-step, #nodes)
        :return: Predicted value of energy consumption.
        """

        if self.__model is None:
            if not os.path.isfile(self.model_saving_path + 'lstm/best_model.hdf5'):
                raise RuntimeError('Model needs to be trained first')
            else:
                self.__model = self.__build_network()
                self.__model.load_trained_model(self.__model.saving_path, 'best_model.hdf5')
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
            if not os.path.isfile(self.model_saving_path + 'lstm/best_model.hdf5'):
                raise RuntimeError('LSTM-based model needs to be trained first!')
            else:
                print '|--- Build and load trained LSTM model'
                self.__model = self.__build_network()
                self.__model.load_trained_model(self.__model.saving_path, 'best_model.hdf5')

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

        pred = self.__model.model.predict(__lstm_input)
        pred = pred.squeeze(axis=1)
        return pred

    def __build_network(self):
        lstm_net = lstm(saving_path=self.model_saving_path + 'lstm/',
                        input_shape=(self.n_timestep, 1),
                        hidden=Config.HIDDEN_UNIT,
                        drop_out=Config.DROP_OUT,
                        check_point=True)
        lstm_net.construct_n_to_one_model()
        lstm_net.plot_models()

        print lstm_net.model.summary()

        return lstm_net

    def __xgb_test(self, data):
        train_x, train_y = create_xy_set(data)

        train_x = train_x.squeeze(axis=2)

        cv_results = cross_validate(self.__model, X=train_x, y=train_y, n_jobs=4, cv=10,
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

        lstm_net = self.__build_network()

        training_fw_history = lstm_net.model.fit(x=train_data[0],
                                                 y=train_data[1],
                                                 batch_size=Config.BATCH_SIZE,
                                                 epochs=Config.N_EPOCH,
                                                 callbacks=lstm_net.callbacks_list,
                                                 validation_data=(train_data[2], train_data[3]),
                                                 shuffle=True,
                                                 verbose=2)
        # Plot the training history
        if training_fw_history is not None:
            lstm_net.plot_training_history(training_fw_history)

        return lstm_net

    def __lstm_train(self, train_set):
        data_x, data_y = create_xy_set(train_set)

        train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True)

        model = self.__train((train_x, train_y, valid_x, valid_y))

        return model

    def __lstm_test(self, test_set):
        print '|--- Test lstm:'
        test_x, test_y = create_xy_set(test_set)

        pred = self.__model.model.predict(test_x)

        mae = mean_absolute_error(y_pred=pred, y_true=test_y)
        r2 = r2_score(y_true=test_y, y_pred=pred)

        return mae, r2

    def __xgb_train(self, train_data):
        train_x, train_y = create_xy_set(train_data)

        train_x = train_x.squeeze(axis=2)

        model = XGBRegressor(n_jobs=4)

        model.fit(train_x, train_y)

        return model
