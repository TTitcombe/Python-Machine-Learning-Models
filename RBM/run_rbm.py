import numpy as np

from rbm import RBM

from utils import load_mnist_binary

data, _ = load_mnist_binary(2)

rbm = RBM(784, 500, {})
rbm.train(data)
