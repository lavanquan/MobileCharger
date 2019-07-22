from energy_predict.commons.data_handle import *
from energy_predict.keras_model.seq2seqLSTM import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os


def train_model(train_path: str, cache: bool, input_length: int, output_length: int,emd_len: int, hid_len: int, optimizer, loss_func, batch,
                epoch, val=0.2, name="seq2seq", load=True):
    """

    :param train_path: path to training file
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

    file_path = name + "_best_model.hdf5"
    file_path = os.path.join("./train_model/", file_path)

    model, encoder_model, decoder_model = seq_to_lstm(4, input_length, output_length, emd_len,
                                                      hid_len)  # input_length == output_length
    # plot_model(model, "model.png", show_shapes=True)
    # plot_model(encoder_model, "en.png", show_shapes=True)
    # plot_model(decoder_model, "de.png", show_shapes=True)

    if load and os.path.isfile(file_path):
        model.load_weights(file_path)
        model.compile(optimizer=optimizer, loss=loss_func)
        return model, encoder_model, decoder_model
    model.compile(optimizer=optimizer, loss=loss_func)

    x_train_1, y_train, x_train_2= read_X_Y_2(file_name=train_path, k=input_length, train=True, cache=cache)
    x_train_1 = np.expand_dims(x_train_1, axis=2)
    x_train_2 = np.expand_dims(x_train_2, axis=2)

    checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                 save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=50)
    model.fit([x_train_1, x_train_2], y_train, batch_size=batch, epochs=epoch, validation_split=val,
              callbacks=[checkpoint, early_stop])

    model_test, encoder_model_test, decoder_model_test = seq_to_lstm(4, input_length, output_length, emd_len, hid_len)
    model_test.load_weights(file_path)
    model_test.compile(optimizer=optimizer, loss=loss_func)
    return model_test, encoder_model_test, decoder_model_test



