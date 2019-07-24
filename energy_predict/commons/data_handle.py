import os
import csv
import numpy as np


CUR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "data"
FULL_DATA1 = "log_file_noCharge.csv"
FULL_DATA2 = "log_file_noCharge_random.csv"
FILE_TRAIN = "train.csv"
FILE_VAL = "val.csv"
FILE_TEST = "test.csv"

CACHE_TRAIN_X = "tcx.npy"  # inputs for encoders for seq2seq lstm
CACHE_TRAIN_X_D = "tcd.npy"  # inputs for decoders for seq2seq lstm
CACHE_TRAIN_X_Y = "tcy.npy"  # outputs for seq2seq lstm

# test doesn't need decoders inputs
CACHE_TEST_X = "ttx.npy"
CACHE_TEST_Y = "tty.npy"

# cache folder name
CACHE_DIR = "cache"

PATH_FULL1 = os.path.join(CUR_PATH, DATA_DIR, FULL_DATA1)
PATH_FULL2 = os.path.join(CUR_PATH, DATA_DIR, FULL_DATA2)

PATH_TRAIN = os.path.join(CUR_PATH, DATA_DIR, FILE_TRAIN)
PATH_VAL = os.path.join(CUR_PATH, DATA_DIR, FILE_VAL)
PATH_TEST = os.path.join(CUR_PATH, DATA_DIR, FILE_TEST)

PATH_CACHE = os.path.join(CUR_PATH, CACHE_DIR)
if not os.path.isdir(PATH_CACHE):
    os.mkdir(PATH_CACHE)

PATH_CACHE_TRAIN_X = os.path.join(PATH_CACHE, CACHE_TRAIN_X)
PATH_CACHE_TRAIN_D = os.path.join(PATH_CACHE, CACHE_TRAIN_X_D)
PATH_CACHE_TRAIN_Y = os.path.join(PATH_CACHE, CACHE_TRAIN_X_Y)
# PATH_CACHE_TEST = os.path.join(PATH_CACHE, CACHE_TEST)
PATH_CACHE_TEST_X = os.path.join(PATH_CACHE, CACHE_TEST_X)
PATH_CACHE_TEST_Y = os.path.join(PATH_CACHE, CACHE_TEST_Y)
T = 5


def write_to_csv(file_name: str, data_list: list):
    f = open(file_name, "w")
    w = csv.writer(f)
    w.writerows(data_list)
    f.close()


def calculated_energy_used(energy_left, avg=1):
    n = energy_left.shape[0]
    l = energy_left.shape[1]
    energy_used = np.zeros([n-1, l])
    energy_block = np.zeros([int(n/avg), l])
    j = 0
    set_point = 0
    for i in range(1, n):
        energy_used[i-1] = (energy_left[i-1] - energy_left[i])/T
        if i % avg == 0:
            energy_block[j] = np.mean(energy_used[set_point:i])
            set_point = i
    return energy_block


def separate_train_val_test(file_name: str = PATH_FULL1, train_path: str = PATH_TRAIN,
                            test_path: str = PATH_TEST):
    """
    Separate traing and testing data
    :param file_name:
    :param train_path:
    :param test_path:
    :return:
    """
    with open(file_name, "r") as csv_file:
        r = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        reader = list(r)

        reader = np.asarray(reader[1:])
        reader = calculated_energy_used(energy_left=reader)

        n = reader.shape[0]
        train_no = int(n * 0.8)
        test_no = n + 1
        train_data = reader[:train_no]
        test_data = reader[train_no:]

        write_to_csv(train_path, train_data)
        write_to_csv(test_path, test_data)
    csv_file.close()


def separate_train_val_test2(file_name: str = PATH_FULL1, train_path: str = PATH_TRAIN,
                            test_path: str = PATH_TEST, avg: int = 1):
    """
    Separate traing and testing data
    :param file_name:
    :param train_path:
    :param test_path:
    :return:
    """
    with open(file_name, "r") as csv_file:
        r = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        reader = list(r)

        reader = np.asarray(reader[1:])
        reader = calculated_energy_used(energy_left=reader)

        n = reader.shape[0]
        train_no = int(n * 0.8)
        test_no = n + 1
        train_data = reader[:train_no]
        test_data = reader[train_no:]

        write_to_csv(train_path, train_data)
        write_to_csv(test_path, test_data)
    csv_file.close()


def read_X_Y(file_name: str=PATH_TRAIN, k: int = 10, n: int=1000):
    """
    Read data and define list pairs of X and Y for fitting into seq2seq LSTM
    data = [len - k + 1, 2, k, n]
    :param file_name:
    :param k: samples in a training
    :param n:
    :return:
    """
    with open(file_name, 'r') as csv_reader:
        f = csv.reader(csv_reader, quoting=csv.QUOTE_NONNUMERIC)
        list_data = list(f)
        len_data = len(list_data)
        number_of_pairs = len_data - k + 1 # o is the number of pair (X, Y)
        fit_data = np.zeros([number_of_pairs, 2, n * k])
        for i in range(number_of_pairs):
            start = i
            end_X = i + k

            end_Y = end_X + k
            if end_Y >= len_data:
                break
            fit_data[i][0] = np.concatenate(np.asarray(list_data[start:end_X]))
            fit_data[i][1] = np.concatenate(np.asarray(list_data[end_X: end_Y]))
            pass
        return fit_data
    pass


def read_X_Y1_Y2(file_name: str = PATH_TRAIN, k: int = 10):
    """
    read data and define list paris of X and Y for fitting into seq2swq LSTM
    this functions create a fit_data = ([X, Y1], Y2)
    :param file_name:
    :param k:
    :param n:
    :return:
    """
    with open(file_name, 'r') as csv_reader:
        f = csv.reader(csv_reader, quoting=csv.QUOTE_NONNUMERIC)
        list_data = list(f)
        len_data = len(list_data)
        n = len(list_data[0])
        number_of_pairs = len_data - k + 1 # o is the number of pair (X, Y)
        fit_data = np.zeros([number_of_pairs, 2, n * k])
        for i in range(number_of_pairs):
            start = i
            end_X = i + k
            end_Y = end_X + k
            if end_Y >= len_data:
                break
            fit_data[i][0] = np.concatenate(np.asarray(list_data[start:end_X]))
            fit_data[i][1] = np.concatenate(np.asarray(list_data[end_X:end_Y]))
            pass
        fit_in_decoders = np.vstack((fit_data[0][0], fit_data[:, 1]))
        return fit_data, fit_in_decoders, k, n


def read_X_Y_2(file_name: str=PATH_TRAIN, k: int=10, shuffle=True, train=True, cache=False):
    """
    Read fitted data, for the period of k time series
    :param file_name:
    :param k:
    :param shuffle:
    :param train: (boolean) true to indicate training false to indicate testing
    :param cache: True to read values, False to re-calculated
    :return: fit_x is the input array for neural network inputs
    fit_y is the output neurons for neural network
    fit_y2 is the input for decoders lstm
    n is the number of nodes
    number_of_pairs is the number of iterations
    """
    if not cache:
        with open(file_name, 'r') as csv_reader:
            f = csv.reader(csv_reader, quoting=csv.QUOTE_NONNUMERIC)
            list_data = list(f)
            list_data = np.asarray(list_data)
            len_data = list_data.shape[0]
            n = list_data.shape[1]
            number_of_pairs = len_data - 2 * k + 1  # o is the number of pair (X, Y) - 2k be
            fit_x = np.zeros([number_of_pairs * n, k], dtype=float)
            fit_y = np.zeros([number_of_pairs * n, k], dtype=float)
            fit_decoders = np.zeros([number_of_pairs * n, k + 1], dtype=float)
            count = 0
            for i in range(number_of_pairs):
                start = i
                end_X = i + k
                end_Y = end_X + k
                if end_Y > len_data:
                    print(str(i) + "-" + str(end_Y) + " - " + str(len_data))
                    continue
                for j in range(n):
                    fit_x[count] = list_data[start:end_X, j]
                    fit_y[count] = list_data[end_X:end_Y, j]
                    fit_decoders[count] = np.concatenate([[fit_x[count][0]], fit_y[count]])
                    count += 1
            if shuffle:
                a = np.arange(number_of_pairs * n)
                np.random.shuffle(a)
                fit_x = fit_x[a]
                fit_y = fit_y[a]
                fit_decoders = fit_decoders[a]

            if train:
                np.save(PATH_CACHE_TRAIN_X, fit_x)
                np.save(PATH_CACHE_TRAIN_D, fit_decoders)
                np.save(PATH_CACHE_TRAIN_Y, fit_y)
            else:
                np.save(PATH_CACHE_TEST_X, fit_x)
                np.save(PATH_CACHE_TEST_Y, fit_y)
    else:
        if train:
            if os.path.isfile(PATH_CACHE_TRAIN_X) or \
                    os.path.isfile(PATH_CACHE_TRAIN_Y) or \
                    os.path.isfile(PATH_CACHE_TRAIN_D):
                fit_x = np.load(PATH_CACHE_TRAIN_X)
                fit_y = np.load(PATH_CACHE_TRAIN_Y)
                fit_decoders = np.load(PATH_CACHE_TRAIN_D)
            else:
                fit_x, fit_y, fit_decoders = read_X_Y_2(file_name, k, shuffle, train, cache=False)
        else:
            if os.path.isfile(PATH_CACHE_TEST_X) or \
                    os.path.isfile(PATH_CACHE_TEST_Y):
                fit_x = np.load(PATH_CACHE_TEST_X)
                fit_y = np.load(PATH_CACHE_TEST_Y)
            else:
                fit_x, fit_y, fit_decoders = read_X_Y_2(file_name, k, shuffle, train, cache=False)
            fit_decoders = None
    return fit_x, fit_y, fit_decoders


def cache_name(path_cache: str, in_len, out_len):
    a = path_cache
    a = a.split(".")[0]
    a = a + "_" + str(in_len) + "_" + str(out_len) + ".npy"
    return a


def read_X_Y_3(file_name: str=PATH_TRAIN, in_len: int = 10, out_len: int = 10, shuffle=True, train=True, cache=False):
    """
    Read fitted data, for the period of k time series.
    allow two values output_length and input_lenght
    Allow the name of cache is stored based on in_len and out_len

    :param file_name:
    :param k:
    :param shuffle:
    :param train: (boolean) true to indicate training false to indicate testing
    :param cache: True to read values, False to re-calculated
    :param in_len: the input length
    :param out_len: the output length
    :return: fit_x is the input array for neural network inputs
    fit_y is the output neurons for neural network
    fit_y2 is the input for decoders lstm
    n is the number of nodes
    number_of_pairs is the number of iterations
    """
    path_cache_train_x = PATH_CACHE_TRAIN_X
    path_cache_train_y = PATH_CACHE_TRAIN_Y
    path_cache_train_d = PATH_CACHE_TRAIN_D

    path_cache_test_x = PATH_CACHE_TEST_X
    path_cache_test_y = PATH_CACHE_TEST_Y

    if train:
        path_cache_train_x = cache_name(PATH_CACHE_TRAIN_X, in_len, out_len)
        path_cache_train_y = cache_name(PATH_CACHE_TRAIN_Y, in_len, out_len)
        path_cache_train_d = cache_name(PATH_CACHE_TRAIN_D, in_len, out_len)
    else:
        path_cache_test_x = cache_name(PATH_CACHE_TEST_X, in_len, out_len)
        path_cache_test_y = cache_name(PATH_CACHE_TEST_Y, in_len, out_len)

    if not cache:
        with open(file_name, 'r') as csv_reader:
            f = csv.reader(csv_reader, quoting=csv.QUOTE_NONNUMERIC)
            list_data = list(f)
            list_data = np.asarray(list_data)
            len_data = list_data.shape[0]
            n = list_data.shape[1]
            number_of_pairs = len_data - (in_len + out_len) + 1  # o is the number of pair (X, Y) - 2k be
            fit_x = np.zeros([number_of_pairs * n, in_len], dtype=float)
            fit_y = np.zeros([number_of_pairs * n, out_len], dtype=float)
            fit_decoders = np.zeros([number_of_pairs * n, out_len + 1], dtype=float)
            count = 0
            for i in range(number_of_pairs):
                start = i
                end_X = i + in_len
                end_Y = end_X + out_len
                if end_Y > len_data:
                    print(str(i) + "-" + str(end_Y) + " - " + str(len_data))
                    continue
                for j in range(n):
                    fit_x[count] = list_data[start:end_X, j]
                    fit_y[count] = list_data[end_X:end_Y, j]
                    fit_decoders[count] = np.concatenate([[fit_x[count][0]], fit_y[count]])
                    count += 1
            if shuffle:
                a = np.arange(number_of_pairs * n)
                np.random.shuffle(a)
                fit_x = fit_x[a]
                fit_y = fit_y[a]
                fit_decoders = fit_decoders[a]

            if train:
                np.save(path_cache_train_x, fit_x)
                np.save(path_cache_train_d, fit_decoders)
                np.save(path_cache_train_y, fit_y)
            else:
                np.save(path_cache_test_x, fit_x)
                np.save(path_cache_test_y, fit_y)
    else:
        if train:
            if os.path.isfile(path_cache_train_x) or \
                    os.path.isfile(path_cache_train_y) or \
                    os.path.isfile(path_cache_train_d):
                fit_x = np.load(path_cache_train_x)
                fit_y = np.load(path_cache_train_y)
                fit_decoders = np.load(path_cache_train_d)
            else:
                fit_x, fit_y, fit_decoders = read_X_Y_3(file_name, in_len, out_len, shuffle, train, cache=False)
        else:
            if os.path.isfile(path_cache_test_x) or \
                    os.path.isfile(path_cache_test_y):
                fit_x = np.load(path_cache_test_x)
                fit_y = np.load(path_cache_test_y)
            else:
                fit_x, fit_y, fit_decoders = read_X_Y_3(file_name, in_len, out_len, shuffle, train, cache=False)
            fit_decoders = None
    return fit_x, fit_y, fit_decoders


# read_X_Y_3()

# read_X_Y_2()



# separate_train_val_test()
# fit_data = read_X_Y()
# a=5
