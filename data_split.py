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

def plot_confusion_matrix(cm,
                          target_names=[0,1,2,3,4,5,6,7,8,9],
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mp
    import itertools
    mp.rcParams.update({'font.size': 22})
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(11, 11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{0}".format(int(cm[i, j]*100)),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "white")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "white")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={0}%; misclass={1}%'.format(accuracy*100, misclass*100))
    plt.show()