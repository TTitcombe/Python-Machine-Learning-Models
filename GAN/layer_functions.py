import numpy as np

def linear(X, W, b):
    return np.dot(X,W) + b

def back_linear(X, W, b, dE):
    dW = np.dot(X.T,dE)
    dX = np.dot(dE, W.T)
    db = np.sum(dE,axis=0)
    return dX, dW, db

def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def back_sigmoid(input):
    o = sigmoid(input)
    return o * (1 - o)

def relu(input):
    x = input.copy()
    x[ x < 0] = 0.
    return x

def back_relu(input):
    x = np.ones(input.shape).astype(np.float32)
    x[input < 0] = 0.
    return x

def lrelu(input, alpha=0.01):
    x = input.copy()
    x[input < 0] = alpha * x[input < 0]
    return x

def back_lrelu(input, alpha = 0.01):
    x = np.ones(input.shape).astype(np.float32)
    x[input < 0] = alpha
    return x

def tanh(input):
    return np.tanh(input)

def back_tanh(input):
    t = tanh(input)
    return 1. - t**2
