import numpy as np
import cv2

class RBM(object):
    """An RBM Object"""

    def __init__(self,nVis, nHid, archs):
        self.nVis = nVis
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

        W, b, c = self.random_init(nVis, nHid)
        self.W = W
        self.b = b
        self.c = c

    @staticmethod
    def random_init(nVis, nHid, seed=None):
        if seed is None:
            seed = np.random.RandomState(42)
        W = np.asarray(seed.uniform(
        low=-4 * np.sqrt(6. / (nHid + nVis)),
        high=4 * np.sqrt(6. / (nHid + nVis)),
        size=(nVis, nHid)))
        b = np.zeros((nVis))
        c = np.zeros((nHid))
        return W, b, c

    def negative(self, hidden):
        new_vis, vis_binary = self.sample_nodes(hidden, False)
        new_hidden, hidden_binary = self.sample_nodes(vis_binary, True)

        return new_hidden, hidden_binary, new_vis, vis_binary

    def sample_nodes(self, nodes, hidden):
        #New function
        '''Sample a new layer given input nodes.
        hidden is a boolean value. If True, new hidden being calculated.'''
        if hidden:
            new_layer = np.matmul(nodes, self.W) + self.c
        else:
            new_layer = np.matmul(nodes, self.W.T) + self.b

        layer_binary = np.zeros(new_layer.shape)
        layer_binary[new_layer >= 0.5] = 1

        return new_layer, layer_binary

    def train(self,x_train, k=1):
            batches = len(x_train) // self.batchSize
            for epoch in range(self.epochs):
                for batch in range(batches):
                        input = x_train[batch*self.batchSize: (batch+1)*self.batchSize]

                        pos_h, hidden_binary = self.sample_nodes(input, True) #how to get hidden nodes

                        for i in range(k):
                            hidden, hidden_binary, vis, vis_binary = self.negative(hidden_binary)

                        #W update
                        pos_phase = np.dot(input.T, pos_h)
                        neg_phase = np.dot(vis_binary.T, hidden)

                        self.W += self.lr * (pos_phase - neg_phase)
                        self.b += self.lr * np.mean(input - vis_binary, axis=0)
                        self.c += self.lr * np.mean(pos_h - hidden, axis = 0)

                        print("Epoch: {}; Batch: {}".format(epoch, batch))
                        image = self.sample_image()
                        image = np.reshape(image, (28,28))
                        cv2.imshow('Image', image)
                        cv2.waitKey(1)

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def sample_image(self):
        image = self.generate()
        return image * 255

    def generate(self):
        newVis = np.random.randint(0,2,self.nVis)
        for i in range(100):
            _, newHid = self.sample_nodes(newVis, True)
            _, newVis = self.sample_nodes(newHid, False)
        return newVis
