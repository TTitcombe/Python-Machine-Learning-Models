import numpy as np
import scipy.linalg as spl
import scipy as sp
import cv2


def load_mnist_binary(number):
    f = open('../mnist_data/train-images.idx3-ubyte')
    loaded = np.fromfile(file=f, dtype=np.uint8)
    #Scale between -1 and 1
    X_train_ = loaded[16:].reshape((60000, 784)).astype(np.float32)
    X_train = np.zeros((60000,784))
    X_train[X_train_ >= 127.5] = 1

    f = open('../mnist_data/train-labels.idx1-ubyte')
    loaded = np.fromfile(file=f, dtype=np.uint8)
    labels_train = loaded[8:].reshape((60000)).astype(np.int32)

    newtrainX = []
    for idx in range(0,len(X_train)):
        if labels_train[idx] == number:
            newtrainX.append(X_train[idx])

    return np.array(newtrainX), len(X_train)
