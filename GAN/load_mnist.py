import numpy as np

def load_mnist(number):

	f = open('../train-images.idx3-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	X_train = loaded[16:].reshape((60000, 784)).astype(np.float32) /  127.5 - 1

	f = open('../train-labels.idx1-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	labels_train = loaded[8:].reshape((60000)).astype(np.int32)

	newtrainX = []
	for idx in range(0,len(X_train)):
		if labels_train[idx] == number:
			newtrainX.append(X_train[idx])

	return np.array(newtrainX), len(X_train)
