import h5py
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data_spectrogram(path, flatten=False):
    file = h5py.File(path, 'r')
    data = np.array(file['data'])
    labels = np.array(file['labels'])
    if flatten:
        X = data.reshape((30000, 227*227))
    else:
        X = data

    y = labels
    file.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, shuffle=False)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test
