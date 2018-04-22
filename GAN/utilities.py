import numpy as np
import cv2

def save_samples(model, N_samples, path='./generated_im'):
    '''Function to save a selection of generated images in a given path'''
    anImg = gan.sample()
    for i in range(N_samples):
        path = path + str(i) + '.jpg'
        cv2.imwrite(path, anImg[i,:,:]*255)

def show_samples(samples, N_show=25):
    '''Creates a larger image of several sampled images side-by-side'''
    assert N_show <= samples.shape[0], "Can`t show more samples than batch size"
    width = samples.shape[1]
    height = samples.shape[2]
    tile_width = np.sqrt(N_show)
    if int(tile_width) < tile_width:
        tile_width = int(tile_width) + 1
    else:
        tile_width = int(tile_width)

    full_image = np.zeros((tile_width * width, tile_width * height))

    n_shown = 0
    while n_shown < N_show:
        for i in range(tile_width):
            for j in range(tile_width):
                im = samples[n_shown,:,:] * 255
                full_image[i*width:(i+1)*width, j*width:(j+1)*width] = im
                n_shown = n_shown + 1
    return full_image

def load_mnist(number):

	f = open('./mnist_data/train-images.idx3-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	X_train = loaded[16:].reshape((60000, 784)).astype(np.float32) /  127.5 - 1

	f = open('./mnist_data/train-labels.idx1-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	labels_train = loaded[8:].reshape((60000)).astype(np.int32)

	newtrainX = []
	for idx in range(0,len(X_train)):
		if labels_train[idx] == number:
			newtrainX.append(X_train[idx])

	return np.array(newtrainX), len(X_train)
