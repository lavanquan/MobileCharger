import fnmatch
import os

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json


def plot_training_history(saving_path, model_history):
    plt.plot(model_history.history['loss'], label='mse')
    plt.plot(model_history.history['val_loss'], label='val_mse')
    plt.savefig(saving_path + '[MSE]loss-val_loss.png')
    plt.legend()
    plt.close()

    plt.plot(model_history.history['val_loss'], label='val_mae')
    plt.legend()
    plt.savefig(saving_path + '[MSE]val_loss.png')
    plt.close()


class AbstractModel(object):

    def __init__(self, saving_path, early_stopping=False, check_point=False, **kwargs):
        self.saving_path = os.path.expanduser(saving_path)
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        self.callbacks_list = []

        self.checkpoints_path = self.saving_path + 'checkpoints/'

        if check_point:
            if not os.path.isdir(self.checkpoints_path):
                os.makedirs(self.checkpoints_path)
            self.checkpoints = ModelCheckpoint(
                self.checkpoints_path + "weights-{epoch:02d}.hdf5",
                monitor='val_loss', verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode='auto', period=1)
            self.callbacks_list = [self.checkpoints]
        if early_stopping:
            self.earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50,
                                           verbose=1, mode='auto')
            self.callbacks_list.append(self.earlystop)

    def save(self, model_json_filename='trained_model.json', model_weight_filename='trained_model.h5'):
        # Save model to dir + record_model/model_train_[%training_set].json

        model_json = self.model.to_json()
        with open(self.saving_path + model_json_filename, "w") as json_file:
            json_file.write(model_json)
            json_file.close()

        # Serialize weights to HDF5
        self.model.save_weights(self.saving_path + model_weight_filename)

    def load(self, model_json_file='trained_model.json', model_weight_file='trained_model.h5'):

        assert os.path.isfile(self.saving_path + model_json_file) & os.path.isfile(
            self.saving_path + model_weight_file)

        json_file = open(self.saving_path + model_json_file, 'r')
        model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(model_json)
        self.model.load_weights(self.saving_path + model_weight_file)

        return True

    def load_trained_model(self, path, weight_file):

        if not os.path.isfile(path + weight_file):
            print('   --> [RNN-load_weights_model] --- File %s not found ---' % (path + weight_file))
            return False
        else:
            print('   --> [RNN-load_weights_model] --- Load weights from ' + path + weight_file)
            self.model.load_weights(path + weight_file)
            return True

    def load_model_from_check_point(self, _from_epoch=None):

        if os.path.exists(self.checkpoints_path):
            list_files = fnmatch.filter(os.listdir(self.checkpoints_path), '*.hdf5')

            if len(list_files) == 0:
                print(
                        '|--- Found no weights file at %s---' % self.checkpoints_path)
                return -1

            list_files = sorted(list_files, key=lambda x: int(x.split('.')[0].split('-')[1]))

            weights_file_name = ''
            epoch = -1
            if _from_epoch:
                for _weights_file_name in list_files:
                    epoch = int(_weights_file_name.split('.')[0].split('-')[1])
                    if _from_epoch == epoch:
                        weights_file_name = _weights_file_name
                        break
            else:
                # Get the last check point
                weights_file_name = list_files[-1]
                epoch = int(weights_file_name.split('.')[0].split('-')[1])

            if self.load_trained_model(path=self.checkpoints_path, weight_file=weights_file_name):
                return epoch
            else:
                return -1
        else:
            print('----> [RNN-load_model_from_check_point] --- Models saving path dose not exist')
            return -1

    def plot_training_history(self, model_history):
        plot_training_history(saving_path=self.saving_path,
                              model_history=model_history)
