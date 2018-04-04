import numpy as np

from gan import GAN

hyperparams = {}
hyperparams['lr'] = 0.001
hyperparams['decay'] = 0.95
hyperparams['epochs'] = 100
hyperparams['batchSize'] = 64


gan = GAN(784, [128],[128],100,1, hyperparams)
gan.train()
