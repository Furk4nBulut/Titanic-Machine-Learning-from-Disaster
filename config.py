# Dataset configuration file

# Path to the dataset
DATASET_PATH = 'dataset/'

# Path to the training data
TRAINING_DATA_PATH = DATASET_PATH + "train.csv"

# Path to the testing data
TESTING_DATA_PATH = DATASET_PATH + "test.csv"


LOW_QUANTILE = 0.05
UP_QUANTILE = 0.90


CAT_THRESHOLD = 10
CAR_THRESHOLD = 20

CORRELATION_THRESHOLD = 0.60


CAT_LENGTH = 10

NUM_METHOD = "median"