import os
import numpy as np

from helper import *
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

def load_data(test=False):
    """
    Loads data from FTEST if test is True, otherwise from FTRAIN.
    Important that the files are in a `data` directory
    """
    FTRAIN = "training.csv"
    FTEST = "test.csv"
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))    # load dataframes

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    df = df.dropna()                            # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.    # scale pixel values to [0, 1] (Normalizing)
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1)                # return each images as 96 x 96 x 1

    if not test:                                # only FTRAIN has target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48                       # scale target coordinates to [-1, 1] (Normalizing)
        X, y = shuffle(X, y, random_state=42)   # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y
# Load training set
X_train, y_train = load_data()

# Setting the CNN architecture
my_model = get_my_CNN_model_architecture()

# Compiling the CNN model with an appropriate optimizer and loss and metrics
compile_my_CNN_model(my_model, optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Training the model
hist = train_my_CNN_model(my_model, X_train, y_train)

# train_my_CNN_model returns a History object. History.history attribute is a record of training loss values and metrics
# values at successive epochs, as well as validation loss values and validation metrics values (if applicable).

# Saving the model
save_my_CNN_model(my_model, 'my_model')
