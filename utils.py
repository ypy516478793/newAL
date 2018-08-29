import numpy as np
import h5py
import random
import scipy.ndimage

def loadData():

    data = h5py.File('./data/Lung_Nodule_2d.h5', 'r')
    X_train = data['X_train'][:]
    Y_train = data['Y_train'][:]
    X_valid = data['X_valid'][:]
    Y_valid = data['Y_valid'][:]
    data.close()

    X_train = np.reshape(X_train, (-1, X_train.shape[1] * X_train.shape[2]))
    X_valid = np.reshape(X_valid, (-1, X_valid.shape[1] * X_valid.shape[2]))

    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)
    X_train = (X_train - mean_train) / std_train

    mean_valid = np.mean(X_valid, axis=0)
    std_valid = np.std(X_valid, axis=0)
    X_valid = (X_valid - mean_valid) / std_valid

    Y_train = Y_train[:, np.newaxis]
    Y_valid = Y_valid[:, np.newaxis]

    return X_train, Y_train, X_valid, Y_valid

