import h5py
import numpy as np
import tensorflow as tf
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
    X_train = X[0:18000]
    y_train = y[0:18000]
    
    X_val = X[18000:24000]
    y_val = y[18000:24000]
    
    X_test = X[24000:30000]
    y_test = y[24000:30000]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_data_waveform(path, flatten=False):
    file = h5py.File(path, 'r')
    data = np.array(file['data'])
    labels = np.array(file['labels'])
    if flatten:
        X = data.reshape((30000, 8000))
    else:
        X = data.reshape((30000, 1, 8000))
    y = labels
    file.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, shuffle=False)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


def return_generators(path):
    train = range(0, 18000)
    val = range(18000, 24000)
    test = range(24000, 30000)
    
    train_generator = DataGenerator(path, train)
    val_generator = DataGenerator(path, val)
    test_generator = DataGenerator(path, test)
    
    return (train_generator,val_generator,test_generator)

# Train (18000, 51529)
# Val   (6000, 51529)
# Test  (6000, 51529)

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path, indices, batch_size=32, num_classes=None, shuffle=False):
        self.batch_size = batch_size
        self.file = h5py.File(path, 'r')
        self.indices = indices
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) // self.batch_size)

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        #X = (self.file["data"])[batch,...]
        #y = (self.file['labels'])[batch,...]
        X = self.file["data",0:31,...]
        print(X.shape)
        y = self.file['labels',0:31,...]
        X = tf.transpose(X, [0, 2, 3, 1])
        #for i, id in enumerate(batch):
        #    X[i,] = (self.file["data"])[i,...]
        #    y[i] = (self.file['labels'])[i,...]

        return X, y