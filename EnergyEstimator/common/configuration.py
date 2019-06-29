DATA_PATH = '../Dataset/'
RAW_DATA_PATH = DATA_PATH + 'raw_data/'
MODEL_SAVING_PATH = 'trained_models/'

N_FOLD = 10
AVG_STEPS = 3
ENERGY_SEND_PERIOD = 5.0  # Time (in second)

# LSTM Configurations
N_TIMESTEPS = 100  # Number of data used as input
N_FEATURES = 1

HIDDEN_UNIT = 64
DROP_OUT = 0.5

N_EPOCH = 50
BATCH_SIZE = 256
BEST_CHECKPOINT = 50
