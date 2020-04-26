import h5py
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import argparse

def merge_data(path, name, fileName):
    mypaths = []
    for i in range(60):
        mypaths.append('{}/{:02d}'.format(path, i + 1))

    onlyfiles = []
    for p in mypaths:
        temp = [join(p, f) for f in listdir(p) if isfile(join(p, f)) and f.startswith(name)]
        onlyfiles = onlyfiles + temp

    X=[]
    y=[]
    i=0
    for p in onlyfiles:
        file = h5py.File(p, 'r')
        data = np.array(file['data'])
        labels = np.array(file['label'])[0][0]
        X.append(data)
        y.append(labels)
        file.close()

    X = np.array(X)
    X = X[:,0,:,:,:]
    y = np.array(y)
    
    merged = h5py.File(f'merged/{fileName}.hdf5','w')
    merged['data'] = X
    merged['labels']=y
    merged.flush()
    merged.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-src', default=os.path.join(os.getcwd(), "../Data/processed_data"), help="Path to folder containing processed_data.")
    args = parser.parse_args()
    
    if not os.path.exists('merged'):
        os.makedirs('merged')

    print('Merging spectrogram data...')
    merge_data(args.source, 'AlexNet', 'spectrogram')
    print('...finished')
    print('Merging waveform data...')
    merge_data(args.source, 'AudioNet', 'waveform')
    print('...finished')

