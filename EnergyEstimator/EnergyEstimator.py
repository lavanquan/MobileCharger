import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import common.configuration as Config
from Models.LSTM_Model import lstm
from common.utils import data_preprocessing, create_xy_set, file_exist
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate


class EnergyEstimator(object):
    def __init__(self, predictor='lstm'):
        self.predictor = predictor
        if self.predictor != 'lstm' or self.predictor != 'xgb':
            raise NotImplementedError('Predictor {} is not implemented'.format(predictor))

        self.__model = None

    def fit(self, data):
        """
        Training energy predictor.
        :param data: (ndarray) The logged energy of sensor node shape = (#time-step, #nodes)
        :return:
        """
        data = data_preprocessing(raw_data=data)

        if self.predictor == 'lstm':
            self.__model = self.__xgb_train(data)
        else:
            self.__model = self.__lstm_train(data)

    def predict(self, data):
        if self.__model is None:
            raise RuntimeError('Model needs to be trained first')

        if self.predictor == 'lstm':
            return self.__xgb_predict(data)
        else:
            return self.__lstm_predict(data)

    def __xgb_predict(self, data):
        pass

    def __lstm_predict(self, data):
        pass

    def __build_network(self):
        lstm_net = lstm(saving_path=Config.MODEL_SAVING_PATH + 'lstm-step-{}-hidden-{}/'.format(Config.N_TIMESTEPS,
                                                                                                Config.HIDDEN_UNIT),
                        input_shape=(Config.N_TIMESTEPS, Config.N_FEATURES),
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

        model = XGBRegressor(n_jobs=4)

        cv_results = cross_validate(model, X=train_x, y=train_y, n_jobs=4, cv=10,
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

        if file_exist(lstm_net.saving_path + 'checkpoints/weights-{:02d}.hdf5'.format(Config.BEST_CHECKPOINT)):
            lstm_net.load_model_from_check_point(_from_epoch=Config.BEST_CHECKPOINT)
        else:
            from_epoch = lstm_net.load_model_from_check_point()
            if from_epoch > 0:
                print('|--- Continue training forward model from epoch %i --- ' % from_epoch)
                training_fw_history = lstm_net.model.fit(x=train_data[0],
                                                         y=train_data[1],
                                                         batch_size=Config.BATCH_SIZE,
                                                         epochs=Config.N_EPOCH,
                                                         callbacks=lstm_net.callbacks_list,
                                                         validation_data=(train_data[2], train_data[3]),
                                                         shuffle=True,
                                                         initial_epoch=from_epoch,
                                                         verbose=2)
            else:
                print('|--- Training new forward model.')

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

    def __predict_energy_consumption(self, model, raw_data, scaler):
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

    def __test(self, raw_test_set, scaler):
        lstm_net = self.__build_network()
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
            pred = self.__predict_energy_consumption(model=lstm_net.model, raw_data=raw_input, scaler=scaler)

            y_pred[i] = pred
            y_true[i] = test_set[i + n_tests]

    def __lstm_train(self, train_set):
        data_x, data_y = create_xy_set(train_set)

        train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True)

        model = self.__train((train_x, train_y, valid_x, valid_y))

        return model

    def __lstm_test(self, test_set):
        print '|--- Test lstm:'
        test_x, test_y = create_xy_set(test_set)

        lstm_net = self.__build_network()
        if file_exist(lstm_net.saving_path + 'checkpoints/weights-{:02d}.hdf5'.format(Config.BEST_CHECKPOINT)):
            lstm_net.load_model_from_check_point(_from_epoch=Config.BEST_CHECKPOINT)
        else:
            raise RuntimeError('Model not found!')

        pred = lstm_net.model.predict(test_x)

        mae = mean_absolute_error(y_pred=pred, y_true=test_y)
        r2 = r2_score(y_true=test_y, y_pred=pred)

        print "Results: MAE: {} --- R2: {}".format(mae, r2)

    def __xgb_train(self, train_data):
        train_x, train_y = create_xy_set(train_data)

        train_x = train_x.squeeze(axis=2)

        model = XGBRegressor(n_jobs=4)

        model.fit(train_x, train_y)

        return model
