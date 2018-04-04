import numpy as np
import cv2

from gan import GAN

hyperparams = {}
hyperparams['lr'] = 0.001
hyperparams['decay'] = 0.95
hyperparams['epochs'] = 100
hyperparams['batchSize'] = 64


gan = GAN(784, [128],[128],100,1, hyperparams)
gan.train()

N_samples = 3 #must be less than batch size
anImg = gan.sample()
for i in range(N_samples):
    path = "./generated_im{}.jpg".format(i)
    cv2.imwrite(path, anImg[i,:,:]*255)
