import numpy as np
import cv2

class RBM(object):
    """An RBM Object"""

    def __init__(self,nInput, nHid, archs):
        self.nInput = nInput
        self.nHid = nHid
        self.archs = archs
        """Archs:
            learning rate
            batch size
            epochs
            k step
        """
        self.lr = archs.get('lr', 0.0001)
        self.batchSize = archs.get('batchSize', 64)
        self.epochs = archs.get('epochs', 100)
        self.k = archs.get('k', 1)

        weights, b, c = self.random_init(nInput, nHid)
        self.weights = weights
        self.b = b
        self.c = c

    @staticmethod
    def random_init(nInput, nHid, seed=None):
        if seed:
            np.random.seed(seed)
        weights = np.random.randn(nInput, nHid)
        b = np.zeros((nInput))
        c = np.zeros((nHid))
        return weights, b, c

    def train(self, x_train):
        batches = len(x_train) // self.batchSize
        for epoch in range(self.epochs):
            for batch in range(batches):
                dw = np.zeros((self.nInput,self.nHid))
                for i in range(self.batchSize):
                    x = x_train[batch*self.batchSize + i]
                    posX = self.sample(x,None,True)
                    posX_ = np.outer(x, posX)

                    #CD-1
                    newVis = self.sample(None,posX, False)
                    negX = self.sample(newVis,None, True)
                    negX_ = np.outer(newVis, negX)

                    dw += (posX_ - negX_)

                print('Epoch: {}; Batch: {}; Weight mean: {}'.format(epoch, batch, np.mean(self.weights)))

                dw = dw / self.batchSize
                self.weights = self.weights + self.lr * dw #gradient ascent
                image = self.sample_image()
                image = np.reshape(image, (28,28))
                cv2.imshow('Image', image)
                cv2.waitKey(1)


    def sample(self,input_data,hidden_data, hidden):
        '''Block gibbs sampling'''
        if hidden:
            newHid = np.matmul(input_data, self.weights)
            newHid = self.prob(input_data, newHid, 100.0,hidden)
            newLayer = newHid
        else:
            newInp = np.matmul(self.weights,hidden_data)
            newInp = self.prob(newInp, hidden_data, 100.0,hidden)
            newLayer = newInp

        newLayer[newLayer >= 0.5] = 1
        newLayer[newLayer < 0.5] = 0
        return newLayer

    def prob(self,inp,hid,T,hidden):
        '''Calculate prop of a variable = 1 given the other variables.
        inputs:
            inp | nodes in input layer, np array
            hid | nodes in input layer, np array
            T | boltzmann temperature
            hidden | bool, true if calculating hidden prop
        '''
        if hidden:
            exp = np.exp(-self.h(inp,hidden) * hid / T)
            return 1/(1+exp)
        else:
            exp = np.exp(-self.h(hid,hidden) * inp / T)
            return 1/(1+exp)

    def h(self,nodes, hid):
        '''Calculate a layer in the rbm.
        inputs:
            nodes | the nodes in the layer, either v or h
            hid | boolean value. True if calculating new hidden layer.
        '''
        if hid:
            return np.matmul(nodes, self.weights) #+ self.c
        else:
            return np.matmul(self.weights,nodes) #+ self.b

    def sample_image(self):
        image = self.generate()
        return image * 255

    def generate(self):
        newInputs = np.random.randint(0,2,self.nInput)
        for i in range(100):
            newHid = self.sample(newInputs,None,True)
            newInputs = self.sample(None,newHid, False)
        return newInputs
