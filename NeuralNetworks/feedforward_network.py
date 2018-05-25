import numpy as np

from layer_functions import *

class ANN():
    '''An artificial neural network object. Nice and basic. No convolutions,
    no recurrence.'''

    def __init__(self, input_dim, hidden_dims,output_dim, hypers, loss='softmax', seed=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dims
        self.N_layers = 1 + len(hidden_dims)
        self.output_dim = output_dim
        self.seed = seed
        if seed:
            np.random.seed(seed)

        self.lr = hypers.get('lr', 0.001)
        self.decay = hypers.get('decay', 1.)
        self.batchSize = hypers.get('batchSize', None)
        self.epochs = hypers.get('epochs', 100)

        hidden_activ = hypers.get('hidden_activation', 'relu')
        if hidden_activ.lower() == 'relu':
            self.act = relu
            self.back_act = back_relu
        elif hidden_activ.lower() in ('lrelu', 'leaky'):
            self.act = lrelu
            self.back_act = back_lrelu
        elif hidden_activ.lower() == 'tanh':
            self.act = tanh
            self.back_act = back_tanh
        else:
            raise NotImplementedError("The hidden activation function you \
                requested has not yet been implemented")

        final_act = hypers.get('final_activation', 'sigmoid')
        if final_act.lower() == 'sigmoid':
            self.final_act = sigmoid
            self.back_final_act = back_sigmoid
        elif final_act.lower() == 'tanh':
            self.final_act = tanh
            self.back_final_act = back_tanh
        else:
            raise NotImplementedError("Final activation function not implemented")

        self.loss = loss

        self.epsilon = 10e-8
        self._randomInit()

    def train(self, X, targets):
        '''
        Inputs:
            X | training data, N x d
            targets | y training labels, N x k

        d is input dimension, k is output dimension
        '''

        print("Beginning training...")

        if self.batchSize is None:
            self.batchSize = X.shape[0]

        N_train = X.shape[0]
		N_batch = N_train//self.batchSize

		for epoch in range(self.epochs):
			for step in range(N_batch):

				X_batch = X[step*self.batchSize:(1+step)*self.batchSize]
				if X_batch.shape[0] != self.batchSize:
					break

				#Feedforward
				logits, activations = self._feedforward(X_batch)

                loss = self._calcLoss(logits, activations, targets)

				#Backprop
				self._backprop(loss)

                if step % (self.batchSize // 5) == 0:
                    accuracy = None
				    print("Epoch: %d; Step: %d; Loss %d; Acc. %d"%(epoch, step, loss, np.mean(accuracy)))

			self.lr = self.lr * self.decay

    def test(self, X):
        '''
        Run X data through a forward pass of the network, to predict labels
        Input:
            X | test data, N x d
        Output:
            y | test predictions, N x k
        '''
        _, y = self._feedforward(X)

        return y

    def _calcLoss(self, logits, activations, targets):
        if self.loss.lower() == 'softmax':
            raise NotImplementedError("TODO")
        elif self.loss.lower() == 'cross_entropy':
            raise NotImplementedError("TODO")
        elif self.loss.lower() == 'mse':
            return np.sum((activations - targets)**2)
        else:
            raise NotImplementedError("Chosen loss function not implemented.")

    def _randomInit(self):
            archs = {}

            archs['W0'] = np.random.randn(self.input_dim,self.hidden_dim[0]).astype(np.float32) * np.sqrt(2.0/(self.input_dim))
            archs['b0'] = np.zeros(self.hidden_dim[0]).astype(np.float32)

            for i in range(1,self.N_layers-1):
                W = np.random.randn(self.hidden_dim[i-1], self.hidden_dim[i]).astype(np.float32) * np.sqrt(2.0/(self.hidden_dim[i-1]))
                b = np.zeros(self.hidden_dim[i]).astype(np.float32)
                archs['W{}'.format(i)] = W
                archs['b{}'.format(i)] = b
            archs['W{}'.format(self.N_layers-1)] = np.random.randn(self.hidden_dim[-1],self.output_dim).astype(np.float32) * np.sqrt(2.0/(self.hidden_dim[-1]))
            archs['b{}'.format(self.N_layers-1)] = np.zeros(self.output_dim).astype(np.float32)
            self.archs = archs

    def _feedforward(self, input):
        lin_store = {}
        act_store = {}

        if self.batchSize is None:
            self.batchSize = input.shape[0]

        act_store['0'] = np.reshape(input, (self.batchSize,-1))

        for i in range(1,self.N_layers):
            lin_store[str(i)] = act_store[str(i-1)].dot(self.archs['W{}'.format(i-1)]) + self.archs['b{}'.format(i-1)]
            act_store[str(i)] = self.act(lin_store[str(i)])

        lin_store[str(self.N_layers)] = act_store[str(self.N_layers-1)].dot(self.archs['W{}'.format(self.N_layers-1)]) + self.archs['b{}'.format(self.N_layers-1)]
        act_store[str(self.N_layers)] = self.final_act(lin_store[str(self.N_layers)])


        return lin_store[str(self.N_layers)], act_store[str(self.N_layers)]

    def _backprop(self, loss):


        dE = loss*self.back_final_act(self.lin_store[str(self.N_layers)])

        new_archs = self.archs.copy() #Create store of weights and biases
        for j in range(self.N_layers-1, -1, -1):
            input = self.act_store[str(j)]
            if j == 0:
                input = np.reshape(input, (self.batchSize,-1))

            prev_layer = np.expand_dims(input, axis=-1)

            dE_ = np.expand_dims(dE, axis=1)

            grad_W =  np.matmul(prev_layer,dE_)
            grad_b = dE

            #Update new archs (propagate error using old archs)
            new_archs['W{}'.format(j)] = new_archs['W{}'.format(j)] - self.lr*np.mean(grad_W, axis=0)
            new_archs['b{}'.format(j)] = new_archs['b{}'.format(j)] - self.lr*np.mean(grad_b, axis=0)

            if j > 0:
                dE = dE.dot(self.archs['W{}'.format(j)].T)
                dE = dE*back_lrelu(self.lin_store[str(j)])

        self.archs = new_archs
