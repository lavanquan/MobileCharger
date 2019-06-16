import numpy as np

import common.configuration as Config
from Models.LSTM_Model import lstm
from common.utils import data_splitting, data_normalization, data_preprocessing, create_xy_set, file_exist


def build_network():
    lstm_net = lstm(saving_path=Config.MODEL_SAVING_PATH,
                    input_shape=(Config.N_FEATURES, Config.N_FEATURES),
                    hidden=Config.HIDDEN_UNIT,
                    drop_out=Config.DROP_OUT,
                    check_point=True)
    lstm_net.seq2seq_model_construction()
    lstm_net.plot_models()

    print lstm_net.model.summary()

    return lstm_net


def train(train_set, valid_set):
    print ('|--- Train average energy consumption predictor')

    lstm_net = build_network()

    if file_exist(lstm_net.saving_path + 'checkpoints/{weights-{:02d}.hdf5}'.format(Config.BEST_CHECKPOINT)):
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


def test():
    pass


if __name__ == '__main__':

    if file_exist(Config.DATA_PATH + 'data.npy'):
        data = np.load(Config.DATA_PATH + 'data.npy')
    else:
        raw_data = np.genfromtxt(Config.RAW_DATA_PATH + 'log_file_noCharge.csv', skip_header=1, delimiter=',')
        print (raw_data.shape)
        data = data_preprocessing(raw_data)

    splitting_ratio = (0.6, 0.2, 0.2)
    train_set, valid_set, test_set = data_splitting(data=data, split_ratio=splitting_ratio)

    n_train_set, n_valid_set, n_test_set, scaler = data_normalization(train_set, valid_set, test_set)

    train(n_train_set, n_valid_set)
