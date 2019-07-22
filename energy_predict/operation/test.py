from energy_predict.commons.data_handle import *
from energy_predict.keras_model.seq2seqLSTM import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error


def test_model(encoder_model, decoder_model, test_path: str, cache: bool, input_length: int, output_length: int, hidden_dim: int, batch):
    """
    :param test_path: path to training file
    :param cache: read from cached or calculate
    :param input_length: input
    :param output_length: length of output
    :param emd_len: embedding length
    :param hid_len: hidden layer length
    :param optimizer: optimize funtion (adam, grad des)
    :param loss_func: loss function
    :param batch: batch numbers
    :param epoch: number of epochs
    :param val: split validation from operation
    :return: model
    models will be storde in ./train_model folder
    """
    x_train_1, y_train, x_train_2 = read_X_Y_2(file_name=test_path, k=input_length, train=False, cache=cache)
    x_train_1 = np.expand_dims(x_train_1, axis=2)
    # x_train_2 = np.expand_dims(x_train_2, axis=2)
    for x in range(x_train_1.shape[0]):
        x_pred = np.expand_dims(x_train_1[x], axis=0)
        states_value = encoder_model.predict(x_pred)
        first_time_series = x_train_1[x]
        # print(first_time_series.shape)
        expected = []

        for i in range(y_train.shape[0]):
            # print(first_time_series)
            first_time_series = np.vstack((first_time_series[0], first_time_series))
            # print(first_time_series.shape)
            # print(first_time_series)
            first_time_series = np.expand_dims(first_time_series, axis=0)
            de_outputs, h_value, c_value = decoder_model.predict([first_time_series] + states_value)
            states_value = [h_value, c_value]
            expected.append(de_outputs[0])
            first_time_series = np.expand_dims(de_outputs[0], axis=2)
        expected = np.asarray(expected)
        print(expected.shape)
        print(y_train.shape)
        dist = mean_squared_error(expected, y_train)
        print(dist)









