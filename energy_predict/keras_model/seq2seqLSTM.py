
import keras
from keras.utils import *
#

from keras.models import *
from keras.layers import *
from keras.activations import *


def seq_to_lstm(n_layers, s_input_length, t_input_length, emd_length, hidden_dim):
    encoder_inputs = Input(shape=(s_input_length, 1))
    first_dense_layer = Dense(emd_length)
    encoder_dense = first_dense_layer(encoder_inputs)
    encoder = LSTM(hidden_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_dense)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(t_input_length+1, 1))
    decoder = LSTM(hidden_dim, return_state=True, return_sequences=True)
    decoder_outputs, _, _, = decoder(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = Flatten()(decoder_outputs)
    second_dense_layer = Dense(t_input_length, activation=None)
    decoder_outputs = second_dense_layer(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    plot_model(model, 'model.png', show_shapes=True)

    # for testing
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(hidden_dim, ))
    decoder_state_input_c = Input(shape=(hidden_dim, ))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, d_state_h, d_state_c = decoder(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [d_state_h, d_state_c]
    decoder_outputs = Flatten()(decoder_outputs)
    decoder_outputs = second_dense_layer(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model


def seq_to_lstm2(n_layers, s_input_length, t_input_length, emd_length, hidden_dim):
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(s_input_length, 1)))
    model.add(RepeatVector(t_input_length))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(Flatten())
    model.add(Dense(t_input_length))
    # model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    plot_model(model, show_shapes=True, to_file='predict_lstm_autoencoder.png')
    return model


def seq_to_lstm_no_ground(n_layers, s_input_length, t_input_length, hidden_dim, optimizer='adam', loss='mse', lstm_activaton='relu'):
    # define model
    model = Sequential()
    model.add(LSTM(hidden_dim, activation=lstm_activaton, input_shape=(s_input_length, 1)))
    model.add(RepeatVector(t_input_length))
    model.add(LSTM(hidden_dim, activation=lstm_activaton, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(t_input_length))
    model.compile(optimizer=optimizer, loss=loss)
    # plot_model(model, show_shapes=True, to_file='seq2seqlstm_no_ground.png')
    return model

# from keras.models import Model
# from keras.layers import Input, LSTM, Dense, embeddings
# import numpy as np
#
#
# def seq_to_lstm(n_layers=4, hidden_dim=300, input_length=10000):
#     encoder_inputs = Input(shape=(input_length,), dtype='int32')
#
#     pass
#
#
# def seq2seq_model_builder(HIDDEN_DIM=300):
#     encoder_inputs = Input(shape=(MAX_LEN,), dtype='int32', )
#     encoder_embedding = embed_layer(encoder_inputs)
#     encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
#     encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
#
#     decoder_inputs = Input(shape=(MAX_LEN,), dtype='int32', )
#     decoder_embedding = embed_layer(decoder_inputs)
#     decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
#     decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
#
#     # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
#     outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
#     model = Model([encoder_inputs, decoder_inputs], outputs)
#
#     return model

